# dataset/transforms.py
import torch
import random


class PairTransform:
    """对图像对应用相同的变换"""

    def __init__(self, config):
        self.config = config
        self.patch_size = config['training']['patch_size']
        aug = config.get('data_augmentation', {})
        self.hflip_prob = aug.get('hflip_prob', 0.5)
        self.vflip_prob = aug.get('vflip_prob', 0.5)
        self.rotate_90_prob = aug.get('rotate_90_prob', 0.3)
        self.color_jitter = aug.get('color_jitter', False)

        # 记录最近一次空间变换参数
        self._last_top = 0
        self._last_left = 0
        self._last_hflip = False
        self._last_vflip = False
        self._last_rot90 = 0

    def __call__(self, img_a, img_b):
        _, h, w = img_a.shape

        # 随机裁剪
        if h > self.patch_size and w > self.patch_size:
            top = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            self._last_top = top
            self._last_left = left
            img_a = img_a[:, top:top + self.patch_size, left:left + self.patch_size]
            img_b = img_b[:, top:top + self.patch_size, left:left + self.patch_size]
        else:
            self._last_top = 0
            self._last_left = 0

        # 随机水平翻转
        self._last_hflip = random.random() < self.hflip_prob
        if self._last_hflip:
            img_a = torch.flip(img_a, dims=[2])
            img_b = torch.flip(img_b, dims=[2])

        # 随机垂直翻转
        self._last_vflip = random.random() < self.vflip_prob
        if self._last_vflip:
            img_a = torch.flip(img_a, dims=[1])
            img_b = torch.flip(img_b, dims=[1])

        # 随机90度旋转
        self._last_rot90 = 0
        if random.random() < self.rotate_90_prob:
            self._last_rot90 = random.randint(1, 3)
            img_a = torch.rot90(img_a, self._last_rot90, dims=[1, 2])
            img_b = torch.rot90(img_b, self._last_rot90, dims=[1, 2])

        # 颜色扰动（只应用于 RGB，albedo 不走这里）
        if self.color_jitter:
            brightness = random.uniform(
                self.config['data_augmentation'].get('brightness_range', [0.8, 1.2])[0],
                self.config['data_augmentation'].get('brightness_range', [0.8, 1.2])[1]
            )
            img_a = img_a * brightness
            img_b = img_b * brightness

        return img_a, img_b

    def apply_spatial(self, img):
        """对 albedo 应用相同的空间变换（不做颜色扰动）"""
        _, h, w = img.shape

        # 同样的裁剪
        if h > self.patch_size and w > self.patch_size:
            img = img[:, self._last_top:self._last_top + self.patch_size,
                  self._last_left:self._last_left + self.patch_size]

        # 同样的翻转
        if self._last_hflip:
            img = torch.flip(img, dims=[2])
        if self._last_vflip:
            img = torch.flip(img, dims=[1])

        # 同样的旋转
        if self._last_rot90 > 0:
            img = torch.rot90(img, self._last_rot90, dims=[1, 2])

        return img


class ValidationTransform:
    """验证集：中心裁剪到 32 的倍数"""

    def __init__(self, config):
        self.patch_size = config['training']['patch_size']

    def __call__(self, img_a, img_b):
        _, h, w = img_a.shape

        new_h = h - (h % 32)
        new_w = w - (w % 32)

        if new_h < h or new_w < w:
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            img_a = img_a[:, top:top + new_h, left:left + new_w]
            img_b = img_b[:, top:top + new_h, left:left + new_w]

        return img_a, img_b