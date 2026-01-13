# ForensicVision

Multi-Method Document Forgery Detection System

---

## 1. Introduction

ForensicVision is a document forgery detection system for image forensics. It focuses on document-like images such as ID cards, receipts, and contracts, and aims to:

- decide whether an image has been tampered with (image-level genuine / tampered classification), and
- localize the forged regions at pixel level with intuitive visualizations.

The current implementation is built on the CASIA 2.0 image tampering dataset. It uses a semantic segmentation model (e.g., Unet++ with EfficientNet-B3 backbone) and combines RGB inputs with Error Level Analysis (ELA) to enhance sensitivity to subtle manipulations.

**Key capabilities:**
- Image-level genuineness / tampering decision
- Pixel-level tampering localization and heatmap visualization
- End-to-end scripts for training, evaluation, and inference, with rich visual outputs

---

## 2. Features

- **Multi-modal input**
  - Fuse raw RGB image with its ELA noise map to better detect low-contrast tampering.
- **Pixel-level localization**
  - Output continuous probability maps in [0, 1] and binary masks via thresholding.
- **Image-level decision**
  - Derive “tampered / authentic” decision from predicted tampered area statistics.
- **Training optimizations**
  - Mixed precision (AMP), gradient accumulation, learning rate scheduling, etc.
- **Evaluation toolkit**
  - Pixel-level metrics: IoU, F1, Precision, Recall, Accuracy
  - Image-level confusion matrix and F1
  - Automatic search of the best probability threshold w.r.t. F1
- **Inference & visualization**
  - Single-image and batch prediction scripts
  - Multi-panel visualizations (original, ELA, heatmap, mask)
- **Result logging**
  - Training log in `training_log.csv`
  - Checkpoints and split indices in `checkpoints/`
  - Curves, visualizations, and reports in `results/`

---

## 3. Project Structure

```text
ForensicVision/
├── README.md                 # Project description (English, default)
├── README_cn.md              # Chinese README
├── training_log.csv          # Per-epoch training & validation logs
├── checkpoints/
│   ├── best_model.pth        # Best-performing model weights on validation set
│   └── split_idx.pth         # Train/validation split indices
├── data/
│   ├── Au/                   # Authentic (untampered) images
│   ├── Tp/                   # Tampered images
│   └── CASIA2/               # Ground-truth tampering masks
├── results/
│   ├── chart_loss_curve.png  # Training/validation loss curves
│   ├── chart_f1_curve.png    # Validation F1 curves
│   ├── epoch_*_val.png       # Validation visualizations per epoch
│   ├── epoch_*_val_preview.png
│   ├── evaluation_confusion_matrix.png
│   ├── final_pred_*.png      # Sample prediction visualizations
│   └── final_report.txt      # Final evaluation report (if generated)
├── scripts/
│   ├── train.py              # Training entry script
│   ├── evaluate.py           # Evaluation entry script
│   └── predict.py            # Inference / prediction entry script
└── src/
    ├── config.py             # Global configuration (paths, hyper-parameters, thresholds)
    ├── dataset.py            # Dataset definitions & data loading logic
    ├── ela.py                # ELA (Error Level Analysis) utilities
    ├── model.py              # Model definition (e.g., Unet++ + EfficientNet-B3)
    ├── utils.py              # Helper functions (metrics, visualization, etc.)
    └── __pycache__/          # Python bytecode cache (can be ignored)
```

---

## 4. Environment & Dependencies

### 4.1 Recommended environment

- Python: 3.10 (3.93.11 should also work)
- OS: Linux / WSL / Windows (GPU recommended)
- GPU (optional): NVIDIA GPU with a CUDA version compatible with your PyTorch install

This repository provides both a Conda environment file and a pip requirements file:

- `environment.yml`: recommended way to create a full Conda environment (Python + PyTorch + CUDA + project deps)
- `requirements.txt`: pure Python package list, useful for `pip` / virtualenv setups

### 4.2 Main dependencies

See `requirements.txt` for the full list. Key packages include:

- Deep learning & data processing:
  - `torch`, `torchvision`, `torchaudio`
  - `segmentation_models_pytorch`
  - `numpy`, `pandas`, `scipy`, `numexpr`
- Data augmentation & image processing:
  - `albumentations`, `albucore`
  - `opencv-python` (or `opencv-python-headless` on servers)
- Visualization & analysis:
  - `matplotlib`, `seaborn`
  - `scikit-learn`, `tqdm`
- Misc / utilities:
  - `PyYAML`, `pydantic`, `typing_extensions`

### 4.3 Installation (option A: Conda, recommended)

```bash
# Create and activate Conda environment
conda env create -f environment.yml
conda activate forensicvision

# If environment.yml only installs base packages, you can additionally run:
# pip install -r requirements.txt
```

The provided `environment.yml` assumes CUDA 12.1 for GPU acceleration via `pytorch-cuda=12.1`.  
If your GPU / driver uses a different CUDA version, please follow the official PyTorch installation guide
(https://pytorch.org/) and adjust the environment file or install commands accordingly.

### 4.4 Installation (option B: Python venv + pip)

If you prefer not to use Conda, you can use a standard Python virtual environment.  
You are responsible for installing a compatible PyTorch build (CPU or GPU) yourself.

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 1) Install PyTorch first (CPU or GPU) according to https://pytorch.org/
#    Example (Linux, CUDA 12.1, pip):
#    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2) Install project dependencies
pip install -r requirements.txt
```

> Note: Paths on WSL look like `/home/username/projects/ForensicVision`,
> which is different from Windows drive letter paths.

---

## 5. Dataset Preparation

### 5.1 CASIA 2.0 dataset (from Kaggle)

This project is built on the CASIA 2.0 image tampering dataset.  
A curated version is available on Kaggle:

- **CASIA 2.0 Image Tampering Detection Dataset** (by `divg07`)  
  https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset

Please download the dataset from the official source or the above Kaggle link,  
follow the dataset license and terms of use (usually research-only),  
and then organize it under the `data/` directory of this project.

### 5.2 Directory layout

Organize files as follows (relative to project root):

```text
data/
├── Au/         # Authentic images, e.g., Au_ani_00001.jpg
├── Tp/         # Tampered images corresponding to Au
└── CASIA2/     # Ground-truth masks aligned with Tp (e.g., with _gt suffix)
```

Path names are configured in `src/config.py` and can be adjusted if your layout differs.

---

## 6. Training

In short:

```bash
python scripts/train.py
```

- Ensure data are correctly placed under `data/` (or update paths in `config.py`).
- Tune key hyper-parameters in `src/config.py` as needed (batch size, epochs, LR, image size, num_workers, etc.).
- The best model will be saved as `checkpoints/best_model.pth` and logs to `training_log.csv`.

---

## 7. Evaluation

After training:

```bash
python scripts/evaluate.py
```

- Reloads the same validation split via `split_idx.pth`.
- Loads `checkpoints/best_model.pth` and evaluates at multiple thresholds.
- Saves quantitative results and plots under `results/` (e.g., confusion matrix, final report).

---

## 8. Inference / Prediction

Basic usage examples:

```bash
# Single image
python scripts/predict.py \
    -i data/Tp/Tp_S_NNN_S_B_nat00090_nat00090_11110.jpg \
    -o results/preds_single

# Batch prediction (e.g., randomly select 50 images from a folder)
python scripts/predict.py \
    -i data/Tp \
    -o results/preds_batch \
    --limit 50 \
    --seed 42
```

Outputs include multi-panel visualizations (original, ELA, heatmap, mask) stored under the chosen output directory.

---

## 9. Visualization & Analysis

- Training curves: `chart_loss_curve.png`, `chart_f1_curve.png`
- Epoch-wise validation previews: `epoch_*_val*.png`
- Final qualitative examples: `final_pred_*.png`
- Quantitative summary: `evaluation_confusion_matrix.png`, `final_report.txt` (if generated)

These files help diagnose overfitting / underfitting and understand the model’s behavior qualitatively and quantitatively.

---

## 10. Troubleshooting

- **CUDA Out of Memory**: reduce `BATCH_SIZE`, increase gradient accumulation steps, or switch to CPU.
- **DataLoader issues on Windows / WSL**: try setting `NUM_WORKERS = 0`.
- **Missing data or checkpoints**: double-check `data/` directory, `config.py` paths, and that `best_model.pth` exists in `checkpoints/`.

---

