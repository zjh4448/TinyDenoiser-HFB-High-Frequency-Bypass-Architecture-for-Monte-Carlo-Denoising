# evaluate.py - 支持 HFB+DRN 区分度自动检测
import numpy as np
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from dataset.utils import read_exr_rgb, to_log_domain


def _compute_data_range(gt):
    dr = gt.max() - gt.min()
    return dr if dr > 0 else 1.0


def check_drn_discrimination(denoised_dir, gt_dir, config_path, model_path):
    """检测 DRN 纹理区分度"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    use_drn_tex = config['model'].get('use_drn_tex', False)
    if not use_drn_tex:
        print("\n[DRN Check] DRN not enabled, skipping discrimination check")
        return None

    from models.unet import TinyDenoiser
    from models.hfb import HFBNet, make_high_pass_fn

    use_drn_edge = config['model'].get('use_drn_edge', False)

    model = TinyDenoiser(features=config['model'].get('features', [32, 48, 64])).to(device)
    hfb_net = HFBNet(use_tex_branch=True, use_edge_branch=use_drn_edge, scale=0.5).to(device)
    high_pass_fn = make_high_pass_fn()

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    if 'hfb_state_dict' in ckpt and ckpt['hfb_state_dict'] is not None:
        hfb_net.load_state_dict(ckpt['hfb_state_dict'])
    hfb_net.eval()

    epsilon = config['data']['exr_epsilon']

    denoised_dir = Path(denoised_dir)
    gt_dir = Path(gt_dir)

    albedo_path = Path('dataset/train/sponza/view_0000_albedo.exr')
    if not albedo_path.exists():
        print("[DRN Check] Albedo file not found, skipping")
        return None

    albedo = read_exr_rgb(str(albedo_path))
    Albedo = torch.from_numpy(albedo).float().unsqueeze(0).to(device)

    ag = Albedo.mean(dim=1, keepdim=True)
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    atex = (F.conv2d(ag, sx, padding=1).abs() + F.conv2d(ag, sy, padding=1).abs())
    tex_mask = (atex > torch.quantile(atex.flatten(), 0.7)).float()

    exr_files = list(denoised_dir.rglob('**/spp_004.exr'))
    exr_files = [f for f in exr_files if 'ground_truth' not in f.name]

    if not exr_files:
        exr_files = list(denoised_dir.rglob('**/denoised_spp_004.exr'))

    if not exr_files:
        print("[DRN Check] No spp_004.exr files found")
        return None

    ratios_with = []
    ratios_without = []

    for exr_file in exr_files[:5]:
        img = read_exr_rgb(str(exr_file))
        I4 = torch.from_numpy(img).float().unsqueeze(0)
        I4_log = to_log_domain(I4, epsilon).to(device)
        I_high = high_pass_fn(I4_log)

        with torch.no_grad():
            _, H_tex, _ = hfb_net(I_high, Albedo[:, :, :I_high.shape[2], :I_high.shape[3]])

            mask_h = F.interpolate(tex_mask[:, :, :I_high.shape[2], :I_high.shape[3]],
                                   size=H_tex.shape[2:], mode='nearest')
            m_tex = mask_h[0, 0] > 0.5
            m_nontex = mask_h[0, 0] <= 0.5

            tex_in = H_tex.abs()[0, 0, m_tex].mean().item()
            tex_out = H_tex.abs()[0, 0, m_nontex].mean().item()
            ratios_with.append(tex_in / (tex_out + 1e-8))

            _, H_tex, _ = hfb_net(I_high, None)
            tex_in = H_tex.abs()[0, 0, m_tex].mean().item()
            tex_out = H_tex.abs()[0, 0, m_nontex].mean().item()
            ratios_without.append(tex_in / (tex_out + 1e-8))

    print(f"\n{'='*60}")
    print(f"DRN Texture Discrimination Check")
    print(f"{'='*60}")
    print(f"  Test files: {len(exr_files[:5])}")
    if ratios_with:
        print(f"  With Albedo: {np.mean(ratios_with):.1f}x")
        print(f"  Without Albedo: {np.mean(ratios_without):.1f}x")
        if np.mean(ratios_with) > 3.0 and np.mean(ratios_without) < 1.5:
            print(f"  -> Albedo supervision effective, but cannot transfer to pure RGB inference")
        elif np.mean(ratios_with) > 3.0 and np.mean(ratios_without) > 3.0:
            print(f"  -> Albedo supervision effective and successfully transferred")
        else:
            print(f"  -> Insufficient discrimination (texture and noise inseparable in high-pass domain)")

    return {'ratio_with_albedo': np.mean(ratios_with) if ratios_with else 0,
            'ratio_without_albedo': np.mean(ratios_without) if ratios_without else 0}


def evaluate_scene(denoised_dir, gt_dir, scenes, spp_list=None, denoised_pattern="denoised_{input_name}"):
    if spp_list is None:
        spp_list = [4, 64]

    denoised_dir = Path(denoised_dir)
    gt_dir = Path(gt_dir)
    results = {}

    for spp in spp_list:
        print(f"\n{'='*60}")
        print(f"Evaluating spp={spp}")
        print(f"{'='*60}")

        psnr_list = []
        ssim_list = []
        scene_results = {}

        input_name = f"spp_{spp:03d}.exr"
        denoised_name = denoised_pattern.format(input_name=input_name, spp=spp)

        for scene in scenes:
            denoised_path = denoised_dir / scene / denoised_name
            gt_path = gt_dir / scene / "ground_truth.exr"

            if not denoised_path.exists():
                print(f"  {scene}: SKIPPED ({denoised_path})")
                continue
            if not gt_path.exists():
                print(f"  {scene}: SKIPPED ({gt_path})")
                continue

            try:
                denoised = read_exr_rgb(str(denoised_path))
                gt = read_exr_rgb(str(gt_path))

                min_h = min(denoised.shape[1], gt.shape[1])
                min_w = min(denoised.shape[2], gt.shape[2])
                denoised = denoised[:, :min_h, :min_w]
                gt = gt[:, :min_h, :min_w]

                data_range = _compute_data_range(gt)

                psnr = peak_signal_noise_ratio(gt, denoised, data_range=data_range)
                ssim = structural_similarity(
                    gt.transpose(1, 2, 0),
                    denoised.transpose(1, 2, 0),
                    channel_axis=2,
                    data_range=data_range
                )

                psnr_list.append(psnr)
                ssim_list.append(ssim)
                scene_results[scene] = {'psnr': round(psnr, 2), 'ssim': round(ssim, 4)}
                print(f"  {scene}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")

            except Exception as e:
                print(f"  {scene}: ERROR ({e})")
                continue

        if psnr_list:
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            std_psnr = np.std(psnr_list)
            std_ssim = np.std(ssim_list)

            print(f"  {'-'*50}")
            print(f"  Avg PSNR: {avg_psnr:.2f} +- {std_psnr:.2f} dB")
            print(f"  Avg SSIM: {avg_ssim:.4f} +- {std_ssim:.4f}")

            results[spp] = {
                'avg_psnr': round(avg_psnr, 2),
                'std_psnr': round(std_psnr, 2),
                'avg_ssim': round(avg_ssim, 4),
                'std_ssim': round(std_ssim, 4),
                'scenes': scene_results
            }
        else:
            print(f"  No valid results for spp={spp}")
            results[spp] = None

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'SPP':<8} {'PSNR (dB)':<18} {'SSIM':<14} {'Scenes':<8}")
    print(f"{'-'*48}")

    for spp, res in results.items():
        if res:
            n_scenes = len(res['scenes'])
            print(f"  {spp:<6} {res['avg_psnr']:.2f} +- {res['std_psnr']:.2f}     "
                  f"{res['avg_ssim']:.4f} +- {res['std_ssim']:.4f}   "
                  f"{n_scenes}/{len(scenes)}")

    valid_results = {k: v for k, v in results.items() if v}
    if valid_results:
        best_psnr_spp = max(valid_results, key=lambda k: valid_results[k]['avg_psnr'])
        best_ssim_spp = max(valid_results, key=lambda k: valid_results[k]['avg_ssim'])
        print(f"\n  Best PSNR: spp={best_psnr_spp} ({valid_results[best_psnr_spp]['avg_psnr']:.2f} dB)")
        print(f"  Best SSIM: spp={best_ssim_spp} ({valid_results[best_ssim_spp]['avg_ssim']:.4f})")

    return results


def evaluate_baseline(gt_dir, scenes, spp_list=None):
    if spp_list is None:
        spp_list = [1, 2, 4, 8, 16, 32, 64]

    gt_dir = Path(gt_dir)
    print(f"\n{'='*60}")
    print(f"BASELINE: Raw vs GT")
    print(f"{'='*60}")

    baseline_results = {}

    for spp in spp_list:
        psnr_list = []
        ssim_list = []

        for scene in scenes:
            noisy_path = gt_dir / scene / f"spp_{spp:03d}.exr"
            gt_path = gt_dir / scene / "ground_truth.exr"

            if not noisy_path.exists() or not gt_path.exists():
                continue

            try:
                noisy = read_exr_rgb(str(noisy_path))
                gt = read_exr_rgb(str(gt_path))

                min_h = min(noisy.shape[1], gt.shape[1])
                min_w = min(noisy.shape[2], gt.shape[2])
                noisy = noisy[:, :min_h, :min_w]
                gt = gt[:, :min_h, :min_w]

                data_range = _compute_data_range(gt)

                psnr = peak_signal_noise_ratio(gt, noisy, data_range=data_range)
                ssim = structural_similarity(
                    gt.transpose(1, 2, 0), noisy.transpose(1, 2, 0),
                    channel_axis=2, data_range=data_range
                )

                psnr_list.append(psnr)
                ssim_list.append(ssim)

            except Exception:
                continue

        if psnr_list:
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            baseline_results[spp] = {
                'avg_psnr': round(avg_psnr, 2),
                'avg_ssim': round(avg_ssim, 4),
                'count': len(psnr_list)
            }

    print(f"\n{'SPP':<8} {'PSNR (dB)':<14} {'SSIM':<12} {'Samples':<8}")
    print(f"{'-'*42}")
    for spp, res in baseline_results.items():
        print(f"  {spp:<6} {res['avg_psnr']:.2f}          {res['avg_ssim']:.4f}      {res['count']}")

    return baseline_results


def print_comparison(denoised_results, baseline_results):
    if not denoised_results or not baseline_results:
        return

    print(f"\n{'='*60}")
    print(f"Comparison: Denoised vs Raw")
    print(f"{'='*60}")
    print(f"{'SPP':<8} {'Raw PSNR':<12} {'Den PSNR':<12} {'Gain':<10} {'Raw SSIM':<12} {'Den SSIM':<12}")
    print(f"{'-'*66}")

    for spp in sorted(set(list(denoised_results.keys()) + list(baseline_results.keys()))):
        d = denoised_results.get(spp, {})
        b = baseline_results.get(spp, {})

        if d and b:
            psnr_gain = d['avg_psnr'] - b['avg_psnr']
            ssim_gain = d['avg_ssim'] - b['avg_ssim']
            sign = '+' if psnr_gain >= 0 else ''
            print(f"  {spp:<6} {b['avg_psnr']:.2f}         {d['avg_psnr']:.2f}         "
                  f"{sign}{psnr_gain:.2f}       {b['avg_ssim']:.4f}        {d['avg_ssim']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate denoising results')
    parser.add_argument('--denoised_dir', type=str, default='denoised_results')
    parser.add_argument('--gt_dir', type=str, default='dataset/test')
    parser.add_argument('--scenes', type=str, nargs='+',
                        default=['front', 'right_high', 'left_low', 'far', 'angled'])
    parser.add_argument('--spp', type=int, nargs='+',
                        default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--pattern', type=str, default='denoised_{input_name}')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    denoised_results = evaluate_scene(
        args.denoised_dir, args.gt_dir, args.scenes, args.spp, args.pattern
    )

    baseline_results = None
    if args.baseline:
        baseline_results = evaluate_baseline(args.gt_dir, args.scenes, args.spp)
        print_comparison(denoised_results, baseline_results)

    if args.model and Path(args.model).exists():
        check_drn_discrimination(args.denoised_dir, args.gt_dir, args.config, args.model)
    elif args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config['model'].get('use_drn_tex', False):
            default_model = Path(config.get('output', {}).get('checkpoint_dir', '')) / 'best_model.pth'
            if default_model.exists():
                check_drn_discrimination(args.denoised_dir, args.gt_dir, args.config, str(default_model))


if __name__ == '__main__':
    main()