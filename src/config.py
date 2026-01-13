# -*- coding: utf-8 -*-
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
config.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
    Configuration file for dataset paths and training hyperparameters.
    
CONTENTS
    Functions  - None
    Classes    - None
    
NOTES
    Dependencies  - os, torch
    Limitations   - None

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Author: You Boyao | Created: 2026/1/11
"""

# ── Imports ──────────────────────────────────────────────────────

# Standard library
import os
# Third party
import torch
# Local application


# ====== Path Configuration ======
# Root directory for the dataset. Adjust this if your data is located elsewhere.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
DATA_ROOT = os.path.join(PROJECT_DIR, 'data')

print(f"DEBUG: Project Root detected at: {PROJECT_DIR}")
print(f"DEBUG: Data Root set to: {DATA_ROOT}")

# Directories for specific data categories based on your provided structure
# Au: Contains authentic images (.jpg)
AU_DIR = os.path.join(DATA_ROOT, 'Au')

# Tp: Contains tampered images (.tif, .png, .jpg)
TP_DIR = os.path.join(DATA_ROOT, 'Tp')

# CASIA2: Contains ground truth masks (.png), usually with '_gt' suffix
MASK_DIR = os.path.join(DATA_ROOT, 'CASIA2')

# ====== Training Hyperparameters ======
IMAGE_SIZE = (512, 512)

# Smaller batch to avoid OOM/crash on higher memory usage steps
BATCH_SIZE = 4

# Gradient accumulation: effective_batch = BATCH_SIZE * GRAD_ACCUM_STEPS
GRAD_ACCUM_STEPS = 4

LR = 1e-4
EPOCHS = 30

# Loss / stability controls
POS_WEIGHT_CLAMP_MAX = 10.0
MAX_GRAD_NORM = 1.0

# Validation / inference thresholds
VAL_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
INFER_THRESHOLD = 0.50

# Image-level tamper decision: minimum number of predicted tampered pixels
IMG_TAMPER_MIN_PIXELS = 100

# Number of worker threads for data loading.
# Set to 0 if you are on Windows to avoid multiprocessing errors.
NUM_WORKERS = 8

# Random seed for reproducibility
SEED = 42

# Computation device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Print device information
if DEVICE == 'cuda':
    print(f"✅ Accelerating with: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ WARNING: CUDA not found. Training will be slow on CPU.")

