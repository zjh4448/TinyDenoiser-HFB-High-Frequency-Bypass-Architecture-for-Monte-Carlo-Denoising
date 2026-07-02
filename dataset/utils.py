# dataset/utils.py
import numpy as np
import OpenEXR
import Imath
import torch
from pathlib import Path
import warnings


def read_exr_rgb(filepath):
    """读取EXR文件的RGB通道"""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"EXR file not found: {filepath}")

    exr_file = OpenEXR.InputFile(str(filepath))

    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = ['R', 'G', 'B']
    rgb_data = []

    for channel in channels:
        try:
            data = exr_file.channel(channel, pt)
            arr = np.frombuffer(data, dtype=np.float32)
            arr = arr.reshape((height, width))
            rgb_data.append(arr)
        except Exception as e:
            warnings.warn(f"Failed to read channel {channel}: {e}")
            rgb_data.append(np.zeros((height, width), dtype=np.float32))

    exr_file.close()
    img = np.stack(rgb_data, axis=0)
    return img.astype(np.float32)


def write_exr_rgb(filepath, img):
    """保存RGB图像为EXR文件"""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    H, W, C = img.shape

    header = OpenEXR.Header(W, H)
    header['channels'] = {
        c: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        for c in ['R', 'G', 'B']
    }

    rgb_data = {}
    for i, channel in enumerate(['R', 'G', 'B']):
        channel_data = img[:, :, i].astype(np.float32).tobytes()
        rgb_data[channel] = channel_data

    exr_file = OpenEXR.OutputFile(str(filepath), header)
    exr_file.writePixels(rgb_data)
    exr_file.close()


def to_log_domain(img, epsilon=1e-6):
    """线性域 → 对数域"""
    eps = float(epsilon)
    return torch.log(img + eps)


def from_log_domain(img_log, epsilon=1e-6):
    """对数域 → 线性域（不截断高光）"""
    eps = float(epsilon)
    log_eps = np.log(eps)
    img_log = torch.clamp(img_log, min=log_eps)
    result = torch.exp(img_log) - eps
    result = torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)
    return result
def visualize_tensor(img_tensor, method='percentile', percentile=99.0, gamma=2.2):
    """将 HDR 图像 tensor 转为可显示的 [0,1] 图像"""
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = img_tensor.copy()

    if img.ndim == 4:
        img = img[0]

    if method == 'percentile':
        vmax = np.percentile(img, percentile)
        if vmax > 0:
            img = np.clip(img, 0, vmax) / vmax
        else:
            img = np.clip(img, 0, 1)
        img = np.power(img, 1.0 / gamma)
    elif method == 'gamma':
        img_max = img.max()
        if img_max > 0:
            img = img / img_max
        img = np.power(np.clip(img, 0, 1), 1.0 / gamma)

    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)

    return np.clip(img, 0, 1)