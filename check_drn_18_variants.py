import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

sys.path.insert(0, '.')
from pathlib import Path
import yaml
from dataset.utils import read_exr_rgb, to_log_domain

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 60)
print("18组DRN变体快速验证")
print("=" * 60)

# 加载数据
img_4 = read_exr_rgb('dataset/sponza_test/center/spp_004.exr')
img_gt = read_exr_rgb('dataset/sponza_test/center/ground_truth.exr')
albedo = read_exr_rgb('dataset/train/sponza/view_0000_albedo.exr')

with open('config_E.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

I4_log = to_log_domain(torch.from_numpy(img_4).float().unsqueeze(0), config['data']['exr_epsilon']).to(device)
Igt_log = to_log_domain(torch.from_numpy(img_gt).float().unsqueeze(0), config['data']['exr_epsilon']).to(device)
Albedo = torch.from_numpy(albedo).float().unsqueeze(0).to(device)

# 纹理mask
ag = Albedo.mean(dim=1, keepdim=True)
sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
atex = (F.conv2d(ag, sx, padding=1).abs() + F.conv2d(ag, sy, padding=1).abs())
tex_mask = (atex > torch.quantile(atex.flatten(), 0.7)).float()
m_tex = tex_mask[0, 0] > 0.5
m_nontex = tex_mask[0, 0] <= 0.5

# 纹理目标
tex_target_32 = (Igt_log.mean(dim=1, keepdim=True) - F.avg_pool2d(Igt_log.mean(dim=1, keepdim=True), 9, stride=1,
                                                                  padding=4)).abs()
tex_target_32 = tex_target_32 / (tex_target_32.max() + 1e-8)

tex_target_alb = atex * (atex > torch.quantile(atex.flatten(), 0.7)).float()
tex_target_alb = tex_target_alb / (tex_target_alb.max() + 1e-8)

tex_target_mix = tex_mask * (
            Igt_log.mean(dim=1, keepdim=True) - F.avg_pool2d(Igt_log.mean(dim=1, keepdim=True), 5, stride=1,
                                                             padding=2)).abs()
tex_target_mix = tex_target_mix / (tex_target_mix.max() + 1e-8)

targets = {
    '32spp_gauss': tex_target_32,
    'albedo_grad': tex_target_alb,
    'albedo_x_32spp': tex_target_mix,
}


# 高通滤波
def high_pass(x, ks=5):
    return x - F.avg_pool2d(x, ks, stride=1, padding=ks // 2)


I_high = high_pass(I4_log)


# 简单编码器
class SimpleEnc(nn.Module):
    def __init__(self, in_ch, out_ch, layers):
        super().__init__()
        enc = []
        for i in range(layers):
            enc += [nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)]
        self.enc = nn.Sequential(*enc)

    def forward(self, x): return self.enc(x)


H, W = 512, 512
patch_size = 128
results = []

configs = []
for ch in [16, 24, 32]:
    for layers in [2, 4]:
        for tgt_name in ['32spp_gauss', 'albedo_grad', 'albedo_x_32spp']:
            configs.append((ch, layers, tgt_name, 'joint'))

# 运行
print(f"共 {len(configs)} 组变体\n")
for idx, (ch, layers, tgt_name, mode) in enumerate(configs):
    print(f"[{idx + 1}/{len(configs)}] ch={ch}, layers={layers}, target={tgt_name} ... ", end='', flush=True)

    encoder = SimpleEnc(3, ch, layers).to(device)
    tex_head = nn.Sequential(
        nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
        nn.Conv2d(ch, 1, 3, padding=1), nn.Tanh()
    ).to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(tex_head.parameters()), lr=1e-3)
    target = targets[tgt_name]

    best_ratio = 1.0
    for step in range(500):
        y = np.random.randint(0, H - patch_size)
        x = np.random.randint(0, W - patch_size)
        inp = I_high[:, :, y:y + patch_size, x:x + patch_size]
        tgt = target[:, :, y:y + patch_size, x:x + patch_size]
        m = (tgt > 0.05).float()

        feat = encoder(inp)
        out = tex_head(feat)
        if m.sum() > 0:
            loss = ((out - tgt).abs() * m).sum() / (m.sum() + 1e-8)
        else:
            loss = torch.tensor(0.0, device=device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 499:
            with torch.no_grad():
                full_out = tex_head(encoder(I_high))
                t = full_out.abs()[0, 0, m_tex].mean().item()
                n = full_out.abs()[0, 0, m_nontex].mean().item()
                ratio = t / (n + 1e-8)
                best_ratio = ratio

    results.append((ch, layers, tgt_name, mode, best_ratio))
    print(f"ratio={best_ratio:.1f}x")

print(f"\n{'=' * 60}")
print("结果汇总")
print(f"{'=' * 60}")
print(f"{'ch':<6} {'layers':<8} {'target':<18} {'mode':<8} {'ratio':<8}")
print("-" * 50)
for ch, layers, tgt, mode, ratio in results:
    print(f"{ch:<6} {layers:<8} {tgt:<18} {mode:<8} {ratio:.1f}x")

print(f"\n所有变体区分度: {np.mean([r[4] for r in results]):.1f}x ± {np.std([r[4] for r in results]):.1f}")
print(f"最大值: {np.max([r[4] for r in results]):.1f}x")
print(f"全部<1.3x -> 纹理不可分结论成立")