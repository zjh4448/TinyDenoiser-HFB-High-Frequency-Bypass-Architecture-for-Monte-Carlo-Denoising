# inference.py - 支持 HFB 模型
import torch
import numpy as np
from pathlib import Path
import argparse
import time
import yaml
import psutil
import os
from tqdm import tqdm

from models.unet import TinyDenoiser
from models.hfb import HFBNet, make_high_pass_fn
from dataset.utils import read_exr_rgb, write_exr_rgb, to_log_domain, from_log_domain


def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


class Denoiser:
    def __init__(self, model_path, config_path='config.yaml', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        model_config = self.config['model']
        features = model_config.get('features', [32, 48, 64])
        print(f"Model config: features={features}")

        self.model = TinyDenoiser(in_ch=3, out_ch=3, features=features).to(self.device)

        # HFB 支持
        self.use_hfb = model_config.get('use_hfb', False)
        self.hfb_net = None
        self.high_pass_fn = None

        if self.use_hfb:
            use_drn_tex = model_config.get('use_drn_tex', False)
            use_drn_edge = model_config.get('use_drn_edge', False)
            self.hfb_net = HFBNet(
                use_tex_branch=use_drn_tex,
                use_edge_branch=use_drn_edge
            ).to(self.device)
            self.high_pass_fn = make_high_pass_fn()

        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint - Epoch: {checkpoint.get('epoch', '?')}")
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        # 加载 HFB 权重
        if self.hfb_net is not None and 'hfb_state_dict' in checkpoint:
            if checkpoint['hfb_state_dict'] is not None:
                self.hfb_net.load_state_dict(checkpoint['hfb_state_dict'])
                print("Loaded HFB weights")
            else:
                print("Warning: use_hfb=True but hfb_state_dict is None in checkpoint")
        self.hfb_net.eval() if self.hfb_net is not None else None

        self.epsilon = self.config['data']['exr_epsilon']
        total_params = sum(p.numel() for p in self.model.parameters())
        if self.hfb_net is not None:
            total_params += sum(p.numel() for p in self.hfb_net.parameters())
        print(f"总参数: {total_params:,} ({total_params / 1000:.1f}K)")

        print("Warming up...")
        dummy = torch.randn(1, 3, 512, 512).to(self.device)
        for _ in range(3):
            _ = self.forward(dummy)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    def forward(self, x):
        out = self.model(x)
        if self.hfb_net is not None and self.high_pass_fn is not None:
            H_clean, _, _ = self.hfb_net(self.high_pass_fn(x), None)
            out = out + H_clean
        return out

    @torch.no_grad()
    def denoise(self, input_exr_path, output_exr_path=None):
        img = read_exr_rgb(input_exr_path)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        img_log = to_log_domain(img_tensor, self.epsilon).to(self.device)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        pred_log = self.forward(img_log)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        inference_time = (time.time() - start_time) * 1000

        pred_linear = from_log_domain(pred_log, self.epsilon)
        pred_linear = pred_linear.squeeze(0).cpu().numpy()

        if output_exr_path:
            write_exr_rgb(output_exr_path, pred_linear)

        return pred_linear, inference_time

    def batch_denoise(self, input_dir, output_dir, pattern="*.exr"):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exr_files = list(input_dir.rglob(pattern))
        exr_files = [f for f in exr_files if 'ground_truth' not in f.name]

        if not exr_files:
            print(f"No EXR files found in {input_dir}")
            return

        total_time = 0
        print(f"Found {len(exr_files)} EXR files")

        for exr_file in tqdm(exr_files, desc="Denoising"):
            rel_path = exr_file.relative_to(input_dir)
            output_path = output_dir / rel_path.parent / f"denoised_{rel_path.name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            _, infer_time = self.denoise(exr_file, output_path)
            total_time += infer_time

        avg_time = total_time / len(exr_files)
        print(f"\nBatch denoising completed!")
        print(f"   Average inference time: {avg_time:.2f}ms per image")
        print(f"   Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input {args.input} does not exist")
        return

    denoiser = Denoiser(model_path=args.model, config_path=args.config, device=args.device)

    if args.batch:
        output_dir = args.output if args.output else "denoised_results"
        denoiser.batch_denoise(args.input, output_dir)
    else:
        output_path = args.output if args.output else "denoised.exr"
        denoiser.denoise(args.input, output_path)
        print(f"\nDenoised image saved to {output_path}")


if __name__ == '__main__':
    main()