import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, '.')
import yaml
from dataset.dataset import create_dataloaders
from models.unet import TinyDenoiser

device = torch.device('cuda')

with open('config_E.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

train_loader, _ = create_dataloaders(config)

model = TinyDenoiser(features=[32, 48, 64]).to(device)
ckpt = torch.load('output_n2n/D/checkpoints/best_model.pth', map_location=device, weights_only=True)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=device).view(1,1,3,3)
sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=device).view(1,1,3,3)

def get_albedo_mask(albedo):
    ag = albedo.mean(dim=1, keepdim=True)
    tex = (F.conv2d(ag, sobel_x, padding=1).abs() + F.conv2d(ag, sobel_y, padding=1).abs())
    B = tex.shape[0]
    tf = tex.view(B, -1)
    th = torch.quantile(tf, 0.7, dim=1, keepdim=True).view(B, 1, 1, 1)
    return (tex > th).float()

tex_vars = []
nontex_vars = []

for bi, batch in enumerate(train_loader):
    if bi >= 20: break
    x = batch['input'].to(device)
    albedo = batch['albedo'].to(device)
    with torch.no_grad():
        base = model(x)
    mask = get_albedo_mask(albedo)

    # 局部方差（5×5 窗口）
    base_gray = base.mean(dim=1, keepdim=True)
    mean_5x5 = F.avg_pool2d(base_gray, 5, 1, 2)
    sq_mean_5x5 = F.avg_pool2d(base_gray ** 2, 5, 1, 2)
    local_var = sq_mean_5x5 - mean_5x5 ** 2

    tex_vars.append((local_var * mask).sum().item() / (mask.sum().item() + 1e-8))
    nontex_vars.append((local_var * (1 - mask)).sum().item() / ((1 - mask).sum().item() + 1e-8))

print(f"纹理区局部方差: {np.mean(tex_vars):.6f}")
print(f"非纹理区局部方差: {np.mean(nontex_vars):.6f}")
print(f"比值: {np.mean(tex_vars)/np.mean(nontex_vars):.1f}×")
if np.mean(tex_vars) > np.mean(nontex_vars) * 2:
    print("✅ 纹理区方差显著 > 非纹理区，NLVE 可行")
else:
    print("❌ 纹理区和非纹理区方差差异不够大")