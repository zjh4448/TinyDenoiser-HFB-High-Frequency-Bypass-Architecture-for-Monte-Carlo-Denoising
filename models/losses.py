# models/losses.py - 修复版
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    """梯度损失，保持边缘"""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        C = pred.shape[1]
        sx = self.sobel_x.repeat(C, 1, 1, 1)
        sy = self.sobel_y.repeat(C, 1, 1, 1)
        gp_x = F.conv2d(pred, sx, padding=1, groups=C)
        gp_y = F.conv2d(pred, sy, padding=1, groups=C)
        gt_x = F.conv2d(target, sx, padding=1, groups=C)
        gt_y = F.conv2d(target, sy, padding=1, groups=C)
        gp = torch.abs(gp_x) + torch.abs(gp_y)
        gt = torch.abs(gt_x) + torch.abs(gt_y)
        return self.l1(gp, gt)


class MultiscaleL1Loss(nn.Module):
    """多尺度 L1"""

    def __init__(self, scales=(1, 2, 4), weights=(1.0, 0.5, 0.25)):
        super().__init__()
        self.scales = scales
        self.weights = weights
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        loss = 0.0
        for s, w in zip(self.scales, self.weights):
            if s == 1:
                loss += w * self.l1(pred, target)
            else:
                loss += w * self.l1(F.avg_pool2d(pred, s), F.avg_pool2d(target, s))
        return loss


class TotalLoss(nn.Module):
    """可配置的总损失，所有组件独立开关"""

    def __init__(self, config):
        super().__init__()
        loss_cfg = config['loss']

        self.use_channel_weight = loss_cfg.get('use_channel_weight', False)
        if self.use_channel_weight:
            weights = loss_cfg.get('channel_weights', [1.5, 1.0, 2.5])
            self.register_buffer('ch_weight', torch.tensor(weights).view(1, 3, 1, 1))
        else:
            self.register_buffer('ch_weight', torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1))

        self.use_positive_bias = loss_cfg.get('use_positive_bias', False)
        self.positive_bias_weight = loss_cfg.get('positive_bias_weight', 0.3)

        self.use_multiscale = loss_cfg.get('use_multiscale', False)
        if self.use_multiscale:
            scales = loss_cfg.get('multiscale_scales', [1, 2, 4])
            scale_weights = loss_cfg.get('multiscale_scale_weights', [1.0, 0.5, 0.25])
            self.multiscale = MultiscaleL1Loss(scales, scale_weights)
            self.ms_weight = loss_cfg.get('multiscale_weight', 0.1)
            self.ms_start = loss_cfg.get('multiscale_start_epoch', 5)
        else:
            self.multiscale = None

        self.use_gradient = loss_cfg.get('use_gradient', False)
        if self.use_gradient:
            self.gradient = GradientLoss()
            self.grad_weight = loss_cfg.get('gradient_weight', 0.05)
            self.grad_start = loss_cfg.get('gradient_start_epoch', 1)
        else:
            self.gradient = None

        self.use_drn_tex = loss_cfg.get('use_drn_tex', False)
        self.drn_tex_weight = loss_cfg.get('drn_tex_weight', 0.05)
        self.drn_tex_start = loss_cfg.get('drn_tex_start_epoch', 5)

        self.use_drn_edge = loss_cfg.get('use_drn_edge', False)
        self.drn_edge_weight = loss_cfg.get('drn_edge_weight', 0.05)
        self.drn_edge_start = loss_cfg.get('drn_edge_start_epoch', 5)

    def forward(self, pred, target, epoch, validation=False,
                drn_tex_pred=None, drn_tex_target=None,
                drn_edge_pred=None, drn_edge_target=None):
        loss = (torch.abs(pred - target) * self.ch_weight).mean()

        if validation:
            return loss

        if self.use_positive_bias:
            positive_bias = torch.clamp(pred - target, min=0)
            loss += self.positive_bias_weight * positive_bias.mean()

        if self.use_multiscale and epoch >= self.ms_start:
            loss += self.ms_weight * self.multiscale(pred, target)

        if self.use_gradient and epoch >= self.grad_start:
            loss += self.grad_weight * self.gradient(pred, target)

        # DRN 纹理损失（mask L1 + 非纹理区抑制）
        if self.use_drn_tex and epoch >= self.drn_tex_start \
                and drn_tex_pred is not None and drn_tex_target is not None:
            mask = (drn_tex_target > 0.05).float()
            if mask.sum() > 0:
                tex_loss = (torch.abs(drn_tex_pred - drn_tex_target) * mask).sum() / (mask.sum() + 1e-8)
                suppress_loss = (drn_tex_pred.abs() * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
                loss += self.drn_tex_weight * tex_loss + 0.25 * suppress_loss

        # DRN 边缘损失
        if self.use_drn_edge and epoch >= self.drn_edge_start \
                and drn_edge_pred is not None and drn_edge_target is not None:
            loss += self.drn_edge_weight * F.l1_loss(drn_edge_pred, drn_edge_target)

        return loss