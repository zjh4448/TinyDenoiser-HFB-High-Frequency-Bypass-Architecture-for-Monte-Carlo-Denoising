# models/hfb.py - HFB + DRN 双分支版（Albedo 辅助训练 + 随机丢弃）
# 修复版：tex_head绕过shared，直接从feat_A+feat_B取特征
import torch
import torch.nn as nn
import torch.nn.functional as F


class HFBNet(nn.Module):
    """高频去噪网络 + DRN 纹理/边缘分支
    训练时可接收 Albedo 作为辅助输入，推理时 Albedo 全零。
    修复：tex_encoder直接从feat_A+feat_B取特征，绕过shared压缩瓶颈。
    """

    def __init__(self, use_tex_branch=False, use_edge_branch=False, scale=0.5):
        super().__init__()
        self.use_tex_branch = use_tex_branch
        self.use_edge_branch = use_edge_branch
        self.scale = scale

        # 编码器A：处理 HighPass(I_4)
        self.encoder_A = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 编码器B：处理 Albedo（仅训练时使用）
        self.encoder_B = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 共享主干（融合后3层，输入通道 16+16=32）
        self.shared = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 去噪输出头
        self.clean_head = nn.Sequential(
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh(),
        )

        # 纹理分支：独立编码器，直接从feat_A+feat_B取特征（绕过shared）
        if use_tex_branch:
            self.tex_encoder = nn.Sequential(
                nn.Conv2d(32, 24, 5, padding=2), nn.ReLU(inplace=True),
                nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(24, 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 3, padding=1), nn.Tanh(),
            )
        else:
            self.tex_encoder = None

        # 边缘分支
        if use_edge_branch:
            self.edge_head = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 3, padding=1),
                nn.Tanh(),
            )
        else:
            self.edge_head = None

    def forward(self, x, albedo=None):
        """
        x: HighPass(I_4), shape (B, 3, H, W)
        albedo: Albedo 图像, shape (B, 3, H, W) 或 None（推理时）
        """
        feat_A = self.encoder_A(x)

        if albedo is not None:
            feat_B = self.encoder_B(albedo)
        else:
            feat_B = torch.zeros_like(feat_A)

        # shared服务于clean_head
        feats = self.shared(torch.cat([feat_A, feat_B], dim=1))
        H_clean = self.clean_head(feats) * self.scale

        H_tex = None
        H_edge = None

        # 纹理分支直接从feat_A+feat_B取特征，绕过shared压缩
        if self.tex_encoder is not None:
            H_tex = self.tex_encoder(torch.cat([feat_A, feat_B], dim=1)) * self.scale

        if self.edge_head is not None:
            H_edge = self.edge_head(feats) * self.scale

        return H_clean, H_tex, H_edge


def make_high_pass_fn(kernel_size=5, sigma=1.0, channels=3):
    ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    g_2d = g[:, None] * g[None, :]
    kernel = g_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    def high_pass(x):
        k = kernel.to(x.device)
        blurred = F.conv2d(x, k, padding=kernel_size // 2, groups=x.shape[1])
        return x - blurred

    return high_pass