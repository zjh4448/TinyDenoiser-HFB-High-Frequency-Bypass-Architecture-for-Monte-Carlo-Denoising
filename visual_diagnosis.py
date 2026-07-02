# visual_diagnosis.py - 全通道全面视觉诊断
import numpy as np
from dataset.utils import read_exr_rgb, visualize_tensor
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def diagnose(noisy_path, denoised_path, gt_path, output_path='visual_diagnosis.png'):
    """全通道全面视觉诊断"""

    noisy = read_exr_rgb(str(noisy_path))
    denoised = read_exr_rgb(str(denoised_path))
    gt = read_exr_rgb(str(gt_path))

    min_h = min(noisy.shape[1], denoised.shape[1], gt.shape[1])
    min_w = min(noisy.shape[2], denoised.shape[2], gt.shape[2])
    noisy = noisy[:, :min_h, :min_w]
    denoised = denoised[:, :min_h, :min_w]
    gt = gt[:, :min_h, :min_w]

    ch_names = ['R', 'G', 'B']
    results = {}

    print('=' * 70)
    print('全通道全面视觉诊断报告')
    print('=' * 70)

    # ====== 1. 全通道基础统计 ======
    print(f'\n{"="*70}')
    print('1. 全通道基础统计')
    print(f'{"="*70}')
    print(f"{'通道':<6} {'图像':<8} {'mean':<10} {'std':<10} {'min':<10} {'max':<10} {'median':<10}")
    print('-' * 60)
    for i, ch in enumerate(ch_names):
        for name, img in [('噪声', noisy[i]), ('降噪', denoised[i]), ('GT', gt[i])]:
            print(f'{ch:<6} {name:<8} {img.mean():<10.4f} {img.std():<10.4f} '
                  f'{img.min():<10.4f} {img.max():<10.4f} {np.median(img):<10.4f}')

    # ====== 2. 逐像素误差分析 ======
    print(f'\n{"="*70}')
    print('2. 逐像素误差分析 (vs GT)')
    print(f'{"="*70}')
    print(f"{'通道':<6} {'类型':<12} {'MAE':<10} {'MSE':<10} {'PSNR(dB)':<10} {'最大误差':<10}")
    print('-' * 60)
    for i, ch in enumerate(ch_names):
        for name, img in [('噪声', noisy[i]), ('降噪', denoised[i])]:
            diff = img - gt[i]
            mae = np.mean(np.abs(diff))
            mse = np.mean(diff ** 2)
            psnr = 20 * np.log10(gt[i].max() - gt[i].min()) - 10 * np.log10(mse + 1e-8)
            max_err = np.abs(diff).max()
            print(f'{ch:<6} {name:<12} {mae:<10.4f} {mse:<10.4f} {psnr:<10.2f} {max_err:<10.4f}')

    # ====== 3. 色偏分析 ======
    print(f'\n{"="*70}')
    print('3. 色偏分析')
    print(f'{"="*70}')
    gt_mean = np.array([gt[i].mean() for i in range(3)])
    noisy_mean = np.array([noisy[i].mean() for i in range(3)])
    denoised_mean = np.array([denoised[i].mean() for i in range(3)])

    gt_ratio = gt_mean / gt_mean.sum()
    noisy_ratio = noisy_mean / noisy_mean.sum()
    denoised_ratio = denoised_mean / denoised_mean.sum()

    print(f"{'通道':<6} {'GT占比':<10} {'噪声占比':<10} {'降噪占比':<10} "
          f"{'噪声偏差':<10} {'降噪偏差':<10}")
    print('-' * 60)
    for i, ch in enumerate(ch_names):
        print(f'{ch:<6} {gt_ratio[i]:<10.4f} {noisy_ratio[i]:<10.4f} {denoised_ratio[i]:<10.4f} '
              f'{noisy_ratio[i]-gt_ratio[i]:<+10.4f} {denoised_ratio[i]-gt_ratio[i]:<+10.4f}')

    # 色相分析
    print(f'\n--- 色相偏差 ---')
    for i, ch in enumerate(ch_names):
        n_dev = noisy_mean[i] - gt_mean[i]
        d_dev = denoised_mean[i] - gt_mean[i]
        print(f'  {ch}: 噪声均值偏差={n_dev:+.4f}, 降噪均值偏差={d_dev:+.4f} '
              f'({"变差" if abs(d_dev) > abs(n_dev) else "改善"})')

    # ====== 4. 异常像素检测（全通道） ======
    print(f'\n{"="*70}')
    print('4. 异常像素检测 (偏离GT超过阈值)')
    print(f'{"="*70}')

    for threshold in [0.05, 0.10, 0.20]:
        print(f'\n--- 阈值={threshold} ---')
        print(f"{'通道':<6} {'噪声异常%':<12} {'降噪异常%':<12} {'变化':<10}")
        print('-' * 42)
        for i, ch in enumerate(ch_names):
            noisy_abnormal = (np.abs(noisy[i] - gt[i]) > threshold).mean() * 100
            denoised_abnormal = (np.abs(denoised[i] - gt[i]) > threshold).mean() * 100
            change = denoised_abnormal - noisy_abnormal
            print(f'{ch:<6} {noisy_abnormal:<12.2f} {denoised_abnormal:<12.2f} {change:<+10.2f}')

    # ====== 5. 亮度分区分析 ======
    print(f'\n{"="*70}')
    print('5. 亮度分区分析')
    print(f'{"="*70}')
    luminance = np.mean(gt, axis=0)
    zones = {
        '暗部(<20%)': luminance < np.percentile(luminance, 20),
        '中间(20-80%)': (luminance >= np.percentile(luminance, 20)) & (luminance <= np.percentile(luminance, 80)),
        '亮部(>80%)': luminance > np.percentile(luminance, 80),
    }

    for zone_name, mask in zones.items():
        print(f'\n--- {zone_name} ({mask.mean()*100:.1f}%) ---')
        print(f"{'通道':<6} {'噪声MAE':<10} {'降噪MAE':<10} {'改善%':<10}")
        print('-' * 38)
        for i, ch in enumerate(ch_names):
            n_mae = np.abs(noisy[i][mask] - gt[i][mask]).mean()
            d_mae = np.abs(denoised[i][mask] - gt[i][mask]).mean()
            improvement = (n_mae - d_mae) / (n_mae + 1e-8) * 100
            print(f'{ch:<6} {n_mae:<10.4f} {d_mae:<10.4f} {improvement:<+10.1f}')

    # ====== 6. 边缘vs平滑 ======
    print(f'\n{"="*70}')
    print('6. 边缘 vs 平滑区域分析')
    print(f'{"="*70}')
    grad = np.abs(ndimage.sobel(gt[0])) + np.abs(ndimage.sobel(gt[1])) + np.abs(ndimage.sobel(gt[2]))
    edge_mask = grad > np.percentile(grad, 90)
    smooth_mask = grad < np.percentile(grad, 10)

    for region, mask in [('边缘(top10%)', edge_mask), ('平滑(bottom10%)', smooth_mask)]:
        print(f'\n--- {region} ({mask.mean()*100:.1f}%) ---')
        print(f"{'通道':<6} {'噪声MAE':<10} {'降噪MAE':<10} {'改善%':<10}")
        print('-' * 38)
        for i, ch in enumerate(ch_names):
            n_mae = np.abs(noisy[i][mask] - gt[i][mask]).mean()
            d_mae = np.abs(denoised[i][mask] - gt[i][mask]).mean()
            improvement = (n_mae - d_mae) / (n_mae + 1e-8) * 100
            print(f'{ch:<6} {n_mae:<10.4f} {d_mae:<10.4f} {improvement:<+10.1f}')

    # ====== 7. 误差相关性 ======
    print(f'\n{"="*70}')
    print('7. 误差空间分布')
    print(f'{"="*70}')
    for i, ch in enumerate(ch_names):
        noisy_err = noisy[i] - gt[i]
        denoised_err = denoised[i] - gt[i]
        # 误差正负分布
        n_pos = (noisy_err > 0.01).mean() * 100
        n_neg = (noisy_err < -0.01).mean() * 100
        d_pos = (denoised_err > 0.01).mean() * 100
        d_neg = (denoised_err < -0.01).mean() * 100
        print(f'\n  {ch}通道:')
        print(f'    噪声: 偏亮{n_pos:.1f}%, 偏暗{n_neg:.1f}%')
        print(f'    降噪: 偏亮{d_pos:.1f}%, 偏暗{d_neg:.1f}%')

    # ====== 8. 纹理保持 ======
    print(f'\n{"="*70}')
    print('8. 纹理保持分析 (5x5局部标准差)')
    print(f'{"="*70}')
    print(f"{'通道':<6} {'噪声':<10} {'降噪':<10} {'GT':<10} {'降噪/GT':<10}")
    print('-' * 50)
    for i, ch in enumerate(ch_names):
        n_std = ndimage.generic_filter(noisy[i], np.std, size=5).mean()
        d_std = ndimage.generic_filter(denoised[i], np.std, size=5).mean()
        g_std = ndimage.generic_filter(gt[i], np.std, size=5).mean()
        ratio = d_std / (g_std + 1e-8)
        print(f'{ch:<6} {n_std:<10.4f} {d_std:<10.4f} {g_std:<10.4f} {ratio:<10.2f}')

    # ====== 9. 综合评分 ======
    print(f'\n{"="*70}')
    print('9. 综合诊断结论')
    print(f'{"="*70}')
    issues = []

    for i, ch in enumerate(ch_names):
        # 均值偏差检查
        n_dev = noisy_mean[i] - gt_mean[i]
        d_dev = denoised_mean[i] - gt_mean[i]
        if abs(d_dev) > abs(n_dev) * 1.5:
            issues.append(f'🔴 {ch}通道: 降噪后均值偏差({d_dev:+.4f})大于噪声({n_dev:+.4f})，存在色偏')

        # 异常像素检查
        n_ab = (np.abs(noisy[i] - gt[i]) > 0.05).mean()
        d_ab = (np.abs(denoised[i] - gt[i]) > 0.05).mean()
        if d_ab > n_ab * 1.2:
            issues.append(f'🟡 {ch}通道: 异常像素从{n_ab*100:.1f}%增至{d_ab*100:.1f}%，过度修正')

    if not issues:
        print('✅ 未发现明显问题')
    else:
        for issue in issues:
            print(f'  {issue}')

    # ====== 10. 生成诊断图 ======
    noisy_vis = visualize_tensor(noisy)
    denoised_vis = visualize_tensor(denoised)
    gt_vis = visualize_tensor(gt)

    fig, axes = plt.subplots(3, 5, figsize=(22, 13))

    # --- 第1行：图像对比 ---
    axes[0, 0].imshow(noisy_vis)
    axes[0, 0].set_title('Noisy Input (4 spp)', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(denoised_vis)
    axes[0, 1].set_title('Denoised Output', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(gt_vis)
    axes[0, 2].set_title('Ground Truth (4096 spp)', fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')

    diff_enhanced = np.clip(np.abs(denoised_vis - gt_vis) * 3, 0, 1)
    axes[0, 3].imshow(diff_enhanced)
    axes[0, 3].set_title('|Diff| x3', fontsize=11, fontweight='bold')
    axes[0, 3].axis('off')

    # 误差改善热力图
    err_n = np.mean((noisy - gt) ** 2, axis=0)
    err_d = np.mean((denoised - gt) ** 2, axis=0)
    err_diff = err_n - err_d
    vmax_err = max(abs(err_diff).max(), 0.01)
    im = axes[0, 4].imshow(err_diff, cmap='RdBu_r', vmin=-vmax_err, vmax=vmax_err)
    axes[0, 4].set_title('MSE Reduction\n(Blue=Better)', fontsize=11)
    axes[0, 4].axis('off')
    plt.colorbar(im, ax=axes[0, 4], fraction=0.046)

    # --- 第2行：各通道偏差 ---
    for i, ch in enumerate(ch_names):
        bias = denoised[i] - gt[i]
        vmax = max(abs(bias).max(), 0.1)
        im = axes[1, i].imshow(bias, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1, i].set_title(f'{ch} Channel Bias\n(Denoised - GT)', fontsize=11)
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)

    # 偏亮偏暗分布
    for i, ch in enumerate(ch_names):
        err = denoised[i] - gt[i]
        bright_mask = err > 0.03
        dark_mask = err < -0.03
        overlay = np.zeros((*gt[i].shape, 3))
        overlay[bright_mask] = [1, 0.5, 0]  # 橙色=偏亮
        overlay[dark_mask] = [0, 0.3, 1]     # 蓝色=偏暗
        axes[1, 3+i].imshow(gt_vis * 0.5 + overlay * 0.5)
        axes[1, 3+i].set_title(f'{ch} Over/Under\n(Orange=Bright, Blue=Dark)', fontsize=10)
        axes[1, 3+i].axis('off')

    # --- 第3行：区域分析 ---
    for i, ch in enumerate(ch_names):
        n_mae_map = np.abs(noisy[i] - gt[i])
        d_mae_map = np.abs(denoised[i] - gt[i])
        improvement = (n_mae_map - d_mae_map) / (n_mae_map + 1e-8) * 100
        im = axes[2, i].imshow(improvement, cmap='RdYlGn', vmin=-100, vmax=100)
        axes[2, i].set_title(f'{ch} Improvement %\n(Green=Better)', fontsize=11)
        axes[2, i].axis('off')
        plt.colorbar(im, ax=axes[2, i], fraction=0.046)

    # 边缘掩码
    axes[2, 3].imshow(edge_mask, cmap='gray')
    axes[2, 3].set_title('Edge Regions', fontsize=11)
    axes[2, 3].axis('off')

    # 亮度分区
    axes[2, 4].imshow(luminance, cmap='viridis')
    axes[2, 4].set_title('Luminance (GT)', fontsize=11)
    axes[2, 4].axis('off')

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'\n诊断图已保存: {output_path}')
    return results


def main():
    parser = argparse.ArgumentParser(description='Full visual diagnosis')
    parser.add_argument('--noisy', type=str, default='dataset/test/front/spp_004.exr')
    parser.add_argument('--denoised', type=str, default='denoised_n2n/front/denoised_spp_004.exr')
    parser.add_argument('--gt', type=str, default='dataset/test/front/ground_truth.exr')
    parser.add_argument('--output', type=str, default='visual_diagnosis.png')
    args = parser.parse_args()

    diagnose(args.noisy, args.denoised, args.gt, args.output)


if __name__ == '__main__':
    main()