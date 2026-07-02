
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import signal
import gc
import sys

from dataset.dataset import create_dataloaders
from models.unet import TinyDenoiser, count_parameters
from models.losses import TotalLoss
from models.ema import EMA

should_exit = False


def signal_handler(signum, frame):
    global should_exit
    print(f"\n信号 {signum}，保存后退出...")
    should_exit = True


signal.signal(signal.SIGINT, signal_handler)


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_checkpoint(state, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_metrics(output, target, metrics_list):
    results = {}
    for m in metrics_list:
        if m == 'psnr':
            mse = torch.mean((output - target) ** 2, dim=[1, 2, 3])
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
            results['psnr'] = psnr.mean().item()
        elif m == 'ssim':
            c1, c2 = 0.01 ** 2, 0.03 ** 2
            mu_x = torch.mean(output, dim=[1, 2, 3], keepdim=True)
            mu_y = torch.mean(target, dim=[1, 2, 3], keepdim=True)
            sx = torch.var(output, dim=[1, 2, 3], keepdim=True)
            sy = torch.var(target, dim=[1, 2, 3], keepdim=True)
            sxy = torch.mean((output - mu_x) * (target - mu_y), dim=[1, 2, 3], keepdim=True)
            ssim = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / (
                (mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2))
            results['ssim'] = ssim.mean().item()
    return results


def make_tex_edge_targets(y_32, albedo=None):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=y_32.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=y_32.device).view(1, 1, 3, 3)

    if albedo is not None:
        albedo_gray = albedo.mean(dim=1, keepdim=True)
        tex_x = F.conv2d(albedo_gray, sobel_x, padding=1)
        tex_y = F.conv2d(albedo_gray, sobel_y, padding=1)
        tex_target = (tex_x.abs() + tex_y.abs())
        B = tex_target.shape[0]
        tex_flat = tex_target.view(B, -1)
        threshold = torch.quantile(tex_flat, 0.7, dim=1, keepdim=True).view(B, 1, 1, 1)
        tex_target = tex_target * (tex_target > threshold).float()
    else:
        y_gray = y_32.mean(dim=1, keepdim=True)
        y_blur = F.avg_pool2d(y_gray, kernel_size=9, stride=1, padding=4)
        tex_target = (y_gray - y_blur).abs()

    tex_target = tex_target / (tex_target.max() + 1e-8)

    y_gray = y_32.mean(dim=1, keepdim=True)
    edge_x = F.conv2d(y_gray, sobel_x, padding=1)
    edge_y = F.conv2d(y_gray, sobel_y, padding=1)
    edge_target = (edge_x.abs() + edge_y.abs())
    edge_target = edge_target / (edge_target.max() + 1e-8)
    edge_target = edge_target - edge_target.mean()

    return tex_target.detach(), edge_target.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"配置: {args.config}")

    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_cfg = config['output']
    for d in [output_cfg['checkpoint_dir'], output_cfg['log_dir'], output_cfg['vis_dir']]:
        Path(d).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    print("加载数据...")
    train_loader, val_loader = create_dataloaders(config)

    model = TinyDenoiser(features=config['model'].get('features', [32, 48, 64])).to(device)

    hfb_mode = None
    hfb_net = None
    high_pass_fn = None
    use_drn = False

    if config['model'].get('use_hfb', False):
        from models.hfb import HFBNet, make_high_pass_fn

        use_drn_tex = config['model'].get('use_drn_tex', False)
        use_drn_edge = config['model'].get('use_drn_edge', False)
        use_drn = use_drn_tex or use_drn_edge

        hfb_net = HFBNet(
            use_tex_branch=use_drn_tex,
            use_edge_branch=use_drn_edge
        ).to(device)

        high_pass_fn = make_high_pass_fn()
        hfb_mode = config['model'].get('hfb_mode', 'step1')

    print(f"U-Net参数: {count_parameters(model):,}")
    if hfb_net is not None:
        print(f"HFB参数: {sum(p.numel() for p in hfb_net.parameters()):,}")
        print(f"总参数: {count_parameters(model) + sum(p.numel() for p in hfb_net.parameters()):,}")

    ema = EMA(model, decay=0.999)
    ema.to(device)
    criterion = TotalLoss(config).to(device)

    # ===== 修复1：HFB参数加入优化器 =====
    if hfb_net is not None:
        trainable_params = list(model.parameters()) + list(hfb_net.parameters())
    else:
        trainable_params = model.parameters()

    optimizer = optim.AdamW(
        trainable_params,
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=config['training']['lr_factor'],
        patience=config['training']['lr_patience'],
        min_lr=config['training']['min_lr']
    )

    writer = SummaryWriter(output_cfg['log_dir'])
    metrics_list = config.get('validation', {}).get('metrics', [])

    start_epoch = 0
    best_val_loss = float('inf')
    train_losses, val_losses, lrs = [], [], []
    no_improve = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        train_losses = ckpt.get('train_losses', [])
        val_losses = ckpt.get('val_losses', [])
        no_improve = ckpt.get('no_improve', 0)
        if 'hfb_state_dict' in ckpt and ckpt['hfb_state_dict'] is not None and hfb_net is not None:
            hfb_net.load_state_dict(ckpt['hfb_state_dict'])
        ema.to(device)

    total_epochs = config['training']['epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    accumulation = config['training'].get('gradient_accumulation', 1)
    early_stop_patience = config['validation'].get('early_stop_patience', 25)

    print(f"\n{'=' * 60}")
    print(f"训练: {start_epoch + 1}→{total_epochs} epochs")
    print(f"batch={config['training']['batch_size']}, lr={config['training']['lr']}")
    print(f"Albedo辅助训练 + 渐进丢弃 (0%→50%)")
    print(f"{'=' * 60}\n")

    sample_batch = next(iter(train_loader))
    use_albedo = 'albedo' in sample_batch and sample_batch['albedo'] is not None
    if use_albedo:
        print("检测到 Albedo 数据")
    else:
        print("未检测到 Albedo")

    for epoch in range(start_epoch + 1, total_epochs + 1):
        if should_exit:
            break

        # ===== 修复2：丢弃率降至0%→50% =====
        if epoch <= 10:
            drop_prob = 0.0
        elif epoch <= 25:
            drop_prob = 0.0 + (0.5 - 0.0) * (epoch - 10) / 15
        else:
            drop_prob = 0.5

        model.train()
        if hfb_net is not None:
            hfb_net.train()
        total_loss = 0
        nb = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        optimizer.zero_grad()

        if warmup_epochs > 0 and epoch <= warmup_epochs:
            lr = config['training']['lr'] * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        for bi, batch in enumerate(pbar):
            x = batch['input'].to(device)
            y = batch['target'].to(device)
            albedo = batch.get('albedo', None)
            if albedo is not None:
                albedo = albedo.to(device)

            pred = model(x)

            drn_tex_pred = None
            drn_tex_target = None
            drn_edge_pred = None
            drn_edge_target = None

            if hfb_net is not None and high_pass_fn is not None:
                I_high = high_pass_fn(x)

                # ===== 修复3：Albedo丢弃时跳过纹理损失 =====
                albedo_for_hfb = None
                albedo_dropped = False
                if albedo is not None:
                    if torch.rand(1).item() < drop_prob:
                        albedo_for_hfb = torch.zeros_like(albedo)
                        albedo_dropped = True
                    else:
                        albedo_for_hfb = albedo

                H_clean, H_tex, H_edge = hfb_net(I_high, albedo_for_hfb)
                pred = pred + H_clean

                if use_drn:
                    tex_target, edge_target = make_tex_edge_targets(y, albedo)
                    if H_tex is not None:
                        drn_tex_pred = H_tex
                        drn_tex_target = tex_target if not albedo_dropped else None
                    if H_edge is not None:
                        drn_edge_pred = H_edge
                        drn_edge_target = edge_target

            loss = criterion(pred, y, epoch, validation=False,
                           drn_tex_pred=drn_tex_pred, drn_tex_target=drn_tex_target,
                           drn_edge_pred=drn_edge_pred, drn_edge_target=drn_edge_target)
            loss = loss / accumulation
            loss.backward()

            if (bi + 1) % accumulation == 0 or (bi + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, config['training']['grad_clip'])
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation
            nb += 1
            postfix = {'loss': f'{loss.item() * accumulation:.4f}', 'drop': f'{drop_prob:.0%}'}
            if drn_tex_pred is not None and drn_tex_target is not None:
                with torch.no_grad():
                    mask_tex = (drn_tex_target > 0.05).float()
                    if mask_tex.sum() > 0:
                        tex_loss_val = (torch.abs(drn_tex_pred - drn_tex_target) * mask_tex).sum() / (mask_tex.sum() + 1e-8)
                        postfix['tex'] = f'{tex_loss_val.item():.4f}'
            if drn_edge_pred is not None and drn_edge_target is not None:
                with torch.no_grad():
                    edge_loss_val = F.l1_loss(drn_edge_pred, drn_edge_target).item()
                postfix['edge'] = f'{edge_loss_val:.4f}'
            pbar.set_postfix(postfix)

        ema.update()
        train_loss = total_loss / max(nb, 1)

        model.eval()
        if hfb_net is not None:
            hfb_net.eval()
        val_loss = 0
        val_metrics_all = {m: [] for m in metrics_list}
        with torch.no_grad():
            for batch in val_loader:
                x = batch['input'].to(device)
                y = batch['target'].to(device)
                pred = model(x)
                if hfb_net is not None and high_pass_fn is not None:
                    H_clean, _, _ = hfb_net(high_pass_fn(x), None)
                    pred = pred + H_clean
                val_loss += criterion(pred, y, epoch, validation=True).item()
                if metrics_list:
                    bm = compute_metrics(pred, y, metrics_list)
                    for k, v in bm.items():
                        val_metrics_all[k].append(v)

        val_loss /= max(len(val_loader), 1)
        val_metrics = {k: np.mean(v) for k, v in val_metrics_all.items() if v}

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lrs.append(current_lr)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'Metrics/{k}', v, epoch)

        print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"LR={current_lr:.2e} | "
              + ' | '.join(f"{k.upper()}={v:.4f}" for k, v in val_metrics.items()))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_shadow': ema.shadow, 'best_val_loss': best_val_loss,
                'train_losses': train_losses, 'val_losses': val_losses,
                'lrs': lrs, 'no_improve': no_improve,
                'hfb_state_dict': hfb_net.state_dict() if hfb_net is not None else None,
            }, Path(output_cfg['checkpoint_dir']) / 'best_model.pth')

            ema.apply_shadow()
            save_checkpoint({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'hfb_state_dict': hfb_net.state_dict() if hfb_net is not None else None,
            }, Path(output_cfg['checkpoint_dir']) / 'best_model_ema.pth')
            ema.restore()
            print(f"  ★ 新最佳模型! (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1

        save_every = output_cfg.get('save_every', 10)
        if epoch % save_every == 0:
            save_checkpoint({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_shadow': ema.shadow, 'best_val_loss': best_val_loss,
                'hfb_state_dict': hfb_net.state_dict() if hfb_net is not None else None,
            }, Path(output_cfg['checkpoint_dir']) / f'epoch_{epoch:03d}.pth')

        if no_improve >= early_stop_patience and epoch > warmup_epochs:
            print(f"\n早停: epoch {epoch}, {early_stop_patience} epochs 无改善")
            break

    save_checkpoint({
        'epoch': total_epochs, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_shadow': ema.shadow, 'best_val_loss': best_val_loss,
        'train_losses': train_losses, 'val_losses': val_losses, 'lrs': lrs,
        'hfb_state_dict': hfb_net.state_dict() if hfb_net is not None else None,
    }, Path(output_cfg['checkpoint_dir']) / 'final_model.pth')

    ema.apply_shadow()
    save_checkpoint({
        'epoch': total_epochs, 'model_state_dict': model.state_dict(),
        'best_val_loss': best_val_loss,
        'hfb_state_dict': hfb_net.state_dict() if hfb_net is not None else None,
    }, Path(output_cfg['checkpoint_dir']) / 'final_model_ema.pth')
    ema.restore()

    writer.close()
    print(f"\n训练完成！最佳 Val Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()