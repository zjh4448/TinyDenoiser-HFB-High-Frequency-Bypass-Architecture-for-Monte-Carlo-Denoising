import torch
from models.unet import TinyDenoiser

model = TinyDenoiser(features=[32, 48, 64])
ckpt = torch.load('output_n2n/D/checkpoints/best_model.pth', map_location='cuda')
model.load_state_dict(ckpt['model_state_dict'])
model.cuda().eval()

dummy = torch.randn(1, 3, 512, 512).cuda()
with torch.no_grad():
    for _ in range(5):
        _ = model(dummy)
    torch.cuda.synchronize()

# 用 script 替代 trace
scripted = torch.jit.script(model)
scripted.save('denoiser_gpu.pt')
print('已导出 denoiser_gpu.pt (script版)')