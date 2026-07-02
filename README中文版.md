# TinyDenoiser HFB 高频旁路架构实验指令

## 环境
- Python 3.12
- PyTorch 2.5.1+cu121
- CUDA 12.1

## 环境配置

### 一键安装（推荐）

双击项目根目录下的 `setup.bat`，自动完成虚拟环境创建和所有依赖安装。

### 手动安装

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境 (Windows)
venv\Scripts\activate

# 3. 安装 PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 安装其他依赖
pip install -r requirements.txt
```

---
数据集使用https://github.com/zjh4448/From-L1-Degradation-to-Shallow-Emergence的releases中存放的数据生成代码生成，使用Sponza场景训练,sponza_test、test、train、val均放在dataset目录下
## 第4.2节：HFB 基础架构训练与验证

### HFB 基础训练（Sponza，无 DRN，纯 RGB）

```bash
python train.py --config config_D_hfb.yaml
```

训练输出目录：`output_n2n/sponza_hfb/`

### HFB 推理（Sponza 测试集）

```bash
python inference.py --model output_n2n/sponza_hfb/checkpoints/best_model.pth --config config_D_hfb.yaml --input dataset/sponza_test --output denoised_hfb --batch
```

### HFB 基准测试（FLOPs + 内存 + 速度）

```bash
python inference.py --model output_n2n/sponza_hfb/checkpoints/best_model.pth --config config_D_hfb.yaml --input dataset/sponza_test/center/spp_004.exr --benchmark
```

### HFB 评估

```bash
# Sponza 评估
python evaluate.py --denoised_dir denoised_hfb --gt_dir dataset/sponza_test --spp 4 --scenes center left right up corner

# 带 baseline 对比
python evaluate.py --denoised_dir denoised_hfb --gt_dir dataset/sponza_test --spp 4 --scenes center left right up corner --baseline
```

### HFB 与基线对比

用消融实验 D 的 baseline 结果（31.29 dB）与 HFB 结果对比，验证 +0.20 dB PSNR 增益。

---

## 第4.3节：DRN 纹理恢复分支实验

### DRN 完整训练（Sponza，Albedo 辅助）

```bash
python train.py --config config_D_hfb_drn.yaml
```

训练输出目录：`output_n2n/sponza_hfb_drn_albedo_input/`

### DRN 推理

```bash
python inference.py --model output_n2n/sponza_hfb_drn_albedo_input/checkpoints/best_model.pth --config config_D_hfb_drn.yaml --input dataset/sponza_test --output denoised_hfb_drn --batch
```

### DRN 评估（含自动区分度检测）

```bash
python evaluate.py --denoised_dir denoised_hfb_drn --gt_dir dataset/sponza_test --spp 4 --scenes center left right up corner --config config_D_hfb_drn.yaml --model output_n2n/sponza_hfb_drn_albedo_input/checkpoints/best_model.pth
```

`evaluate.py` 会自动检测 `use_drn_tex: true` 并输出：
- 有 Albedo 区分度
- 无 Albedo 区分度

### DRN 18 变体快速验证

```bash
python check_drn_18_variants.py
```

遍历 18 种配置（3 通道 × 2 层数 × 3 监督信号），每种训练 500 步，输出纹理区分度。

预期结果：所有变体区分度 ≈ 1.0×。

### DRN 区分度单独检测

```bash
python check_drn_discrimination.py
```

加载 `sponza_hfb_drn_albedo_input/checkpoints/best_model.pth`，分别测试：
- 训练模式（有 Albedo）：预期区分度 6.9×
- 推理模式（无 Albedo）：预期区分度 1.0×

### 高通域空间统计分析

```bash
python check_highpass_statistics.py
```

对 4 spp 高通滤波图像，分纹理/非纹理区域计算：
- 局部标准差比（k=3, 7, 11）：预期 ≈ 1.0×
- 梯度强度比：预期 ≈ 1.0×
- 空间自相关比：预期 ≈ 1.1×

结论：所有指标比值均 ≈ 1.0–1.1×，高通域内纹理与噪声无统计差异。

---

## 双重交叉验证证据链

| 实验 | 脚本 | 预期结果 |
|------|------|----------|
| DRN 18 变体快速验证 | `check_drn_18_variants.py` | 所有变体区分度 ≈ 1.0× |
| DRN 完整训练区分度 | `check_drn_discrimination.py` | 训练 6.9× / 推理 1.0× |
| 高通域局部方差比 | `check_highpass_statistics.py` | k=3,7,11 均 ≈ 1.0× |
| 高通域梯度强度比 | `check_highpass_statistics.py` | ≈ 1.0× |
| 高通域空间自相关比 | `check_highpass_statistics.py` | ≈ 1.1× |

**结论：4 spp 下高斯高通滤波将纹理信号与 MC 噪声均转化为不可区分的高频随机波动，纹理与噪声在物理层面不可分。**

---

