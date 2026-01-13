# -*- coding: utf-8 -*-
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
dataset.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
    Custom Dataset class for CASIA v2.0 Image Tampering Detection.
    
CONTENTS
    Functions  - None
    Classes    - CASIADataset
    
NOTES
    Dependencies  - OpenCV, NumPy, PyTorch, Albumentations, PIL
    Limitations   - Assumes specific directory structure for CASIA v2 dataset.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Author: You Boyao | Created: 2026/1/11
"""

# ── Imports ──────────────────────────────────────────────────────

# Standard library
import os
import glob

# Third party
import cv2
import numpy as np
import torch
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from PIL import ImageFile

# Local application
from src import config
from src.ela import convert_to_ela

# Enable tolerance for truncated/corrupted image files to prevent crashes
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CASIADataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        self.image_paths = []

        # Define Normalization (ImageNet stats)
        mean_6ch = (0.485, 0.456, 0.406, 0.485, 0.456, 0.406)
        std_6ch = (0.229, 0.224, 0.225, 0.229, 0.224, 0.225)

        # Define Transforms
        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.Normalize(mean=mean_6ch, std=std_6ch, max_pixel_value=255.0),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
                A.Normalize(mean=mean_6ch, std=std_6ch, max_pixel_value=255.0),
                ToTensorV2(),
            ])

        # --- STEP 1: Pre-build Mask Index (The "Deterministic" Approach) ---
        # Scan CASIA2 folder once and build a lookup dictionary.
        # Logic: Clean the mask filename to get the core ID, map it to the full path.
        print(f"[{mode.upper()}] Indexing masks from {config.MASK_DIR}...")
        self.mask_index = {}
        mask_files = glob.glob(os.path.join(config.MASK_DIR, '*.png'))

        for m_path in mask_files:
            fname = os.path.basename(m_path)
            # Remove extension (.png)
            name_no_ext = os.path.splitext(fname)[0]

            # Remove '_gt' suffix if present to get the "Core Name"
            # Example: "Tp_..._12345_gt" -> "Tp_..._12345"
            if name_no_ext.endswith('_gt'):
                core_name = name_no_ext[:-3]
            else:
                core_name = name_no_ext

            self.mask_index[core_name] = m_path

        # --- STEP 2: Load Authentic Images (Au) ---
        # User confirmed Au contains .jpg and .bmp
        au_files = []
        au_files.extend(glob.glob(os.path.join(config.AU_DIR, '*.jpg')))
        au_files.extend(glob.glob(os.path.join(config.AU_DIR, '*.bmp'))) # Added .bmp support

        for p in au_files:
            self.image_paths.append({'path': p, 'type': 'authentic'})

        # --- STEP 3: Load Tampered Images (Tp) ---
        # User confirmed Tp contains .jpg and .tif
        tp_files = []
        tp_files.extend(glob.glob(os.path.join(config.TP_DIR, '*.jpg')))
        tp_files.extend(glob.glob(os.path.join(config.TP_DIR, '*.tif')))

        # Filter: Only add Tampered images if we actually HAVE a corresponding mask.
        # This prevents the "Missing Mask" error during training entirely.
        valid_tp_count = 0
        skipped_tp_count = 0

        for p in tp_files:
            fname = os.path.basename(p)
            core_name = os.path.splitext(fname)[0]

            # Check if this image exists in our mask index
            if core_name in self.mask_index:
                self.image_paths.append({'path': p, 'type': 'tampered'})
                valid_tp_count += 1
            else:
                # If not found in index, we simply don't add it to the dataset.
                skipped_tp_count += 1

        print(f"[{mode.upper()}] Dataset initialized.")
        print(f"   > Authentic: {len(au_files)}")
        print(f"   > Tampered (Matched): {valid_tp_count}")
        print(f"   > Tampered (Skipped - No Mask): {skipped_tp_count}")
        print(f"   > Total Samples: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        item = self.image_paths[idx]
        img_path = item['path']
        img_type = item['type']

        # 1. Read Image
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ CRITICAL ERROR: Cannot read image: {img_path}")
            # Return a dummy tensor to avoid crashing
            dummy_img = torch.zeros((6, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))
            dummy_mask = torch.zeros((1, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))
            return dummy_img, dummy_mask
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_origin, w_origin = image.shape[:2]

        # 2. Get Mask (Direct Lookup)
        if img_type == 'authentic':
            mask = np.zeros((h_origin, w_origin), dtype=np.float32)
        else:
            fname = os.path.basename(img_path)
            core_name = os.path.splitext(fname)[0]
            mask_path = self.mask_index[core_name]

            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask_img is None:
                # Rare case: file exists but unreadable
                return self.__getitem__((idx + 1) % len(self.image_paths))

            # Force resize mask to match image
            if mask_img.shape[:2] != (h_origin, w_origin):
                mask_img = cv2.resize(mask_img, (w_origin, h_origin), interpolation=cv2.INTER_NEAREST)

            mask = mask_img.astype(np.float32) / 255.0

        # 3. Generate ELA
        try:
            ela_pil = convert_to_ela(img_path)
            ela = np.array(ela_pil)
            if ela.shape[:2] != (h_origin, w_origin):
                ela = cv2.resize(ela, (w_origin, h_origin))
        except Exception:
            return self.__getitem__((idx + 1) % len(self.image_paths))

        # 4. Stack & Augment
        combined_img = np.dstack((image, ela))

        try:
            augmented = self.transform(image=combined_img, mask=mask)
            input_tensor = augmented['image']
            mask_tensor = augmented['mask'].unsqueeze(0)
            return input_tensor, mask_tensor
        except ValueError:
            return self.__getitem__((idx + 1) % len(self.image_paths))
