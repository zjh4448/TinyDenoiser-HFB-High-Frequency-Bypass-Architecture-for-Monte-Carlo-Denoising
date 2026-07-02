# TinyDenoiser HFB High-Frequency Bypass Architecture Experiment Instructions

## Environment
- Python 3.12
- PyTorch 2.5.1+cu121
- CUDA 12.1

## Environment Setup

### One-Click Installation (Recommended)

Double-click `setup.bat` in the project root directory to automatically create the virtual environment and install all dependencies.

### Manual Installation

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment (Windows)
venv\Scripts\activate

# 3. Install PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install other dependencies
pip install -r requirements.txt

```

---
Dataset generated using the code from https://github.com/zjh4448/From-L1-Degradation-to-Shallow-Emergence releases. Sponza scene for training. sponza_test, test, train, val are placed under the dataset directory.
## Section 4.2: HFB Base Architecture Training and Validation

### HFB Base Training (Sponza, No DRN, Pure RGB)

```bash
python train.py --config config_D_hfb.yaml
```

Training output directory: `output_n2n/sponza_hfb/`

### HFB Inference (Sponza Test Set)

```bash
python inference.py --model output_n2n/sponza_hfb/checkpoints/best_model.pth --config config_D_hfb.yaml --input dataset/sponza_test --output denoised_hfb --batch
```

### HFB Benchmark (FLOPs + Memory + Speed)

```bash
python inference.py --model output_n2n/sponza_hfb/checkpoints/best_model.pth --config config_D_hfb.yaml --input dataset/sponza_test/center/spp_004.exr --benchmark
```

### HFB Evaluation

```bash
# Sponza evaluation
python evaluate.py --denoised_dir denoised_hfb --gt_dir dataset/sponza_test --spp 4 --scenes center left right up corner

# With baseline comparison
python evaluate.py --denoised_dir denoised_hfb --gt_dir dataset/sponza_test --spp 4 --scenes center left right up corner --baseline
```

---

## Section 4.3: DRN Texture Restoration Branch Experiments

### DRN Full Training (Sponza, Albedo-Assisted)

```bash
python train.py --config config_D_hfb_drn.yaml
```

Training output directory: `output_n2n/sponza_hfb_drn_albedo_input/`

### DRN Inference

```bash
python inference.py --model output_n2n/sponza_hfb_drn_albedo_input/checkpoints/best_model.pth --config config_D_hfb_drn.yaml --input dataset/sponza_test --output denoised_hfb_drn --batch
```

### DRN Evaluation (with Automatic Discrimination Check)

```bash
python evaluate.py --denoised_dir denoised_hfb_drn --gt_dir dataset/sponza_test --spp 4 --scenes center left right up corner --config config_D_hfb_drn.yaml --model output_n2n/sponza_hfb_drn_albedo_input/checkpoints/best_model.pth
```

`evaluate.py` will automatically detect `use_drn_tex: true` and output:
- Discrimination ratio with Albedo
- Discrimination ratio without Albedo

### DRN 18-Variant Rapid Validation

```bash
python check_drn_18_variants.py
```

Iterates through 18 configurations (3 channel sizes × 2 layer counts × 3 supervision signal types), trains each for 500 steps, and outputs texture discrimination ratios.

Expected result: All variants yield discrimination ratios ≈ 1.0×.

### DRN Discrimination Standalone Check

```bash
python check_drn_discrimination.py
```

Loads `sponza_hfb_drn_albedo_input/checkpoints/best_model.pth` and separately tests:
- Training mode (with Albedo): expected discrimination ratio 6.9×
- Inference mode (without Albedo): expected discrimination ratio 1.0×

### High-Pass Domain Spatial Statistical Analysis

```bash
python check_highpass_statistics.py
```

Computes the following metrics on 4 spp high-pass filtered images, partitioned into texture vs. non-texture regions:
- Local standard deviation ratio (k=3, 7, 11): expected ≈ 1.0×
- Gradient magnitude ratio: expected ≈ 1.0×
- Spatial autocorrelation ratio: expected ≈ 1.1×

Conclusion: All metric ratios are approximately 1.0–1.1×, indicating no statistical difference between texture and noise in the high-pass domain.

---

## Dual Cross-Validation Evidence Chain

| Experiment | Script | Expected Result |
|------------|--------|-----------------|
| DRN 18-variant rapid validation | `check_drn_18_variants.py` | All variants ≈ 1.0× |
| DRN full training discrimination | `check_drn_discrimination.py` | Training 6.9× / Inference 1.0× |
| High-pass local variance ratio | `check_highpass_statistics.py` | k=3,7,11 all ≈ 1.0× |
| High-pass gradient magnitude ratio | `check_highpass_statistics.py` | ≈ 1.0× |
| High-pass spatial autocorrelation ratio | `check_highpass_statistics.py` | ≈ 1.1× |

**Conclusion: Under 4 spp, Gaussian high-pass filtering converts both texture signals and MC noise into indistinguishable high-frequency random fluctuations. Texture and noise are physically inseparable at the signal level.**

---

