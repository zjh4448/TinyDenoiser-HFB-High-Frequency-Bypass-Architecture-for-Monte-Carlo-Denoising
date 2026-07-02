import torch
import numpy as np
import sys

sys.path.insert(0, '.')
from pathlib import Path
import yaml
import torch.nn.functional as F
from dataset.utils import read_exr_rgb, to_log_domain
from models.unet import TinyDenoiser
from models.hfb import HFBNet, make_high_pass_fn

device = torch.device('cuda')

# 加载配置和模型
with open('config_D_hfb_drn.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

ckpt = torch.load('output_n2n/sponza_hfb_drn_albedo_input/checkpoints/best_model.pth',
                  map_location=device, weights_only=True)

model = TinyDenoiser(features=[32, 48, 64]).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

hfb_net = HFBNet(use_tex_branch=True, use_edge_branch=True, scale=0.5).to(device)
hfb_net.load_state_dict(ckpt['hfb_state_dict'])
hfb_net.eval()

high_pass_fn = make_high_pass_fn()

# 加载测试数据
img_4 = read_exr_rgb('dataset/sponza_test/center/spp_004.exr')
albedo = read_exr_rgb('dataset/train/sponza/view_0000_albedo.exr')

I4 = torch.from_numpy(img_4).float().unsqueeze(0).to(device)
Albedo = torch.from_numpy(albedo).float().unsqueeze(0).to(device)

I4_log = to_log_domain(I4.cpu(), config['data']['exr_epsilon']).to(device)
I_high = high_pass_fn(I4_log)

# Albedo纹理mask（用于评估）
ag = Albedo.mean(dim=1, keepdim=True)
sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
atex = (F.conv2d(ag, sx, padding=1).abs() + F.conv2d(ag, sy, padding=1).abs())
tex_mask = (atex > torch.quantile(atex.flatten(), 0.7)).float()

# ============================================================
# 测试1：训练模式（有Albedo）— 模拟训练时的表现
# ============================================================
print("=" * 60)
print("Albedo辅助训练 — 纹理区分度验证")
print("=" * 60)

with torch.no_grad():
    H_clean, H_tex, H_edge = hfb_net(I_high, Albedo)  # 传入真实Albedo

    mask_h = F.interpolate(tex_mask, size=H_tex.shape[2:], mode='nearest')
    m_tex = mask_h[0, 0] > 0.5
    m_nontex = mask_h[0, 0] <= 0.5

    tex_in = H_tex.abs()[0, 0, m_tex].mean().item()
    tex_out = H_tex.abs()[0, 0, m_nontex].mean().item()
    ratio_with = tex_in / (tex_out + 1e-8)

    print(f"\n[训练模式] 有Albedo输入:")
    print(f"  纹理区H_tex均值: {tex_in:.6f}")
    print(f"  非纹理区H_tex均值: {tex_out:.6f}")
    print(f"  区分度: {ratio_with:.1f}x")
    print(f"  H_tex std: {H_tex.std():.6f}")

# ============================================================
# 测试2：推理模式（无Albedo）— 模拟推理时的表现
# ============================================================
with torch.no_grad():
    H_clean, H_tex, H_edge = hfb_net(I_high, None)  # Albedo=None

    tex_in = H_tex.abs()[0, 0, m_tex].mean().item()
    tex_out = H_tex.abs()[0, 0, m_nontex].mean().item()
    ratio_without = tex_in / (tex_out + 1e-8)

    print(f"\n[推理模式] 无Albedo输入:")
    print(f"  纹理区H_tex均值: {tex_in:.6f}")
    print(f"  非纹理区H_tex均值: {tex_out:.6f}")
    print(f"  区分度: {ratio_without:.1f}x")
    print(f"  H_tex std: {H_tex.std():.6f}")

# ============================================================
# 结论
# ============================================================
print(f"\n{'=' * 60}")
print("结论:")
print(f"{'=' * 60}")
print(f"  训练时区分度: {ratio_with:.1f}x")
print(f"  推理时区分度: {ratio_without:.1f}x")
if ratio_with > 1.5 and ratio_without < 1.2:
    print(f"  → Albedo监督有效，但无法迁移至纯RGB推理")
elif ratio_with > 1.5 and ratio_without > 1.5:
    print(f"  → Albedo监督有效且成功迁移")
else:
    print(f"  → 区分度不足")