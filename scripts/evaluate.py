# -*- coding: utf-8 -*-
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
evaluate.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
    Entry point for evaluating the trained tampering localization model
    on the CASIA v2.0 validation set.
    
CONTENTS
    Functions  - evaluate_model
    Classes    - None
    
NOTES
    Dependencies  - PyTorch, NumPy, Matplotlib, Seaborn, scikit-learn, tqdm
    Limitations   - None

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Author: You Boyao | Created: 2026/1/12
"""

# ── Imports ──────────────────────────────────────────────────────
# Standard library
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Third party
import torch
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

# Local application


# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
sys.path.append(root_path)

from src import config
from src.dataset import CASIADataset
from src.model import build_model

def evaluate_model():
    # 1. Setup Environment
    device = config.DEVICE
    print(f"[INFO] Running evaluation on {device}...")

    # 2. Load Dataset (Replicate the Train/Val Split)
    print("[INFO] Loading dataset...")
    full_dataset = CASIADataset(mode='val')  # no augmentation

    split_path = os.path.join(root_path, 'checkpoints', 'split_idx.pth')
    if os.path.exists(split_path):
        split = torch.load(split_path, map_location='cpu')
        val_idx = split['val_idx']
        val_dataset = Subset(full_dataset, val_idx)
        print(f"[INFO] Loaded val split from {split_path} | Val: {len(val_dataset)}")
    else:
        # fallback: deterministic split (should rarely happen)
        print("[WARN] split_idx.pth not found, falling back to deterministic split by SEED.")
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        generator = torch.Generator().manual_seed(config.SEED)
        _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)


    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    print(f"[INFO] Evaluating on {len(val_dataset)} validation images (Unseen Data).")

    # 3. Load Model
    model = build_model().to(device)
    checkpoint_path = os.path.join(root_path, 'checkpoints', 'best_model.pth')

    if not os.path.exists(checkpoint_path):
        print("[ERROR] No checkpoint found. Please run training first.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("[INFO] Best model loaded successfully.")

    thr_list = config.VAL_THRESHOLDS
    pixel_stats = {thr: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0} for thr in thr_list}
    img_preds_by_thr = {thr: [] for thr in thr_list}
    img_labels = []

    print("[INFO] Starting batch evaluation...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            bsz = images.size(0)
            for i in range(bsz):
                is_tampered_gt = 1 if masks[i].max() > 0.5 else 0
                img_labels.append(is_tampered_gt)

            for thr in thr_list:
                preds = (probs > thr).float()

                preds_flat = preds.view(-1)
                masks_flat = masks.view(-1)

                tp = (preds_flat * masks_flat).sum().item()
                fp = (preds_flat * (1 - masks_flat)).sum().item()
                fn = ((1 - preds_flat) * masks_flat).sum().item()
                tn = ((1 - preds_flat) * (1 - masks_flat)).sum().item()

                pixel_stats[thr]["tp"] += tp
                pixel_stats[thr]["fp"] += fp
                pixel_stats[thr]["fn"] += fn
                pixel_stats[thr]["tn"] += tn

                for i in range(bsz):
                    pred_area = preds[i].sum().item()
                    is_tampered_pred = 1 if pred_area > config.IMG_TAMPER_MIN_PIXELS else 0
                    img_preds_by_thr[thr].append(is_tampered_pred)

    if len(img_labels) == 0:
        print("[ERROR] Validation loader returned no samples.")
        return

    epsilon = 1e-7
    best_thr = None
    best_pixel_f1 = -1.0

    print("\n" + "=" * 40)
    print("       FINAL EVALUATION REPORT       ")
    print("=" * 40)
    print("\n[Per-threshold Pixel Metrics]")

    for thr in thr_list:
        stats = pixel_stats[thr]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        tn = stats["tn"]

        pixel_precision = tp / (tp + fp + epsilon)
        pixel_recall = tp / (tp + fn + epsilon)
        pixel_f1 = 2 * (pixel_precision * pixel_recall) / (pixel_precision + pixel_recall + epsilon)
        pixel_iou = tp / (tp + fp + fn + epsilon)
        pixel_acc = (tp + tn) / (tp + tn + fp + fn + epsilon)

        print(f"[Threshold={thr:.2f}] IoU={pixel_iou:.4f} | F1={pixel_f1:.4f} | P={pixel_precision:.4f} | R={pixel_recall:.4f} | Acc={pixel_acc:.4f}")

        if pixel_f1 > best_pixel_f1:
            best_pixel_f1 = pixel_f1
            best_thr = thr

    if best_thr is None:
        print("[ERROR] No thresholds evaluated.")
        return

    best_stats = pixel_stats[best_thr]
    pixel_tp = best_stats["tp"]
    pixel_fp = best_stats["fp"]
    pixel_fn = best_stats["fn"]
    pixel_tn = best_stats["tn"]

    pixel_precision = pixel_tp / (pixel_tp + pixel_fp + epsilon)
    pixel_recall = pixel_tp / (pixel_tp + pixel_fn + epsilon)
    pixel_f1 = 2 * (pixel_precision * pixel_recall) / (pixel_precision + pixel_recall + epsilon)
    pixel_iou = pixel_tp / (pixel_tp + pixel_fp + pixel_fn + epsilon)
    pixel_acc = (pixel_tp + pixel_tn) / (pixel_tp + pixel_tn + pixel_fp + pixel_fn + epsilon)

    print(f"\n[BEST] threshold={best_thr:.2f} by Pixel F1(Dice)={best_pixel_f1:.4f}")
    print(f"\n[Pixel-Level Segmentation Metrics @ {best_thr:.2f}]")
    print(f"  > IoU (Intersection over Union): {pixel_iou:.4f}")
    print(f"  > F1-Score (Dice)              : {pixel_f1:.4f}")
    print(f"  > Precision                    : {pixel_precision:.4f}")
    print(f"  > Recall (Sensitivity)         : {pixel_recall:.4f}")
    print(f"  > Pixel Accuracy               : {pixel_acc:.4f}")

    img_labels_arr = np.array(img_labels)
    best_img_preds = np.array(img_preds_by_thr[best_thr])
    img_acc = np.mean(img_labels_arr == best_img_preds)
    img_f1 = f1_score(img_labels_arr, best_img_preds, zero_division=0)
    img_cm = confusion_matrix(img_labels_arr, best_img_preds)

    print(f"\n[Image-Level Detection Metrics @ {best_thr:.2f}]")
    print("Confusion Matrix:\n", img_cm)
    print(f"  > Accuracy (Fake/Real)         : {img_acc:.4f}")
    print(f"  > F1-Score (Fake/Real)         : {img_f1:.4f}")

    cm = np.array([[pixel_tn, pixel_fp], [pixel_fn, pixel_tp]])
    annot = np.array([
        [f"{int(pixel_tn):,}", f"{int(pixel_fp):,}"],
        [f"{int(pixel_fn):,}", f"{int(pixel_tp):,}"]
    ])

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                     xticklabels=['Pred: Authentic', 'Pred: Tampered'],
                     yticklabels=['True: Authentic', 'True: Tampered'])
    plt.title(f'Pixel-Level Confusion Matrix (thr={best_thr:.2f})')
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')

    cbar = ax.collections[0].colorbar if ax.collections else None
    if cbar is not None:
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        cbar.ax.yaxis.set_major_formatter(formatter)

    cm_path = os.path.join(root_path, 'results', 'evaluation_confusion_matrix.png')
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"\n[INFO] Confusion Matrix saved to {cm_path}")

    report_path = os.path.join(root_path, 'results', 'final_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("=== ForensicVision Evaluation Report ===\n")
        f.write(f"Best Threshold: {best_thr:.2f}\n")
        f.write(f"Pixel IoU:       {pixel_iou:.4f}\n")
        f.write(f"Pixel F1:        {pixel_f1:.4f}\n")
        f.write(f"Pixel Precision: {pixel_precision:.4f}\n")
        f.write(f"Pixel Recall:    {pixel_recall:.4f}\n")
        f.write(f"Image Accuracy:  {img_acc:.4f}\n")
        f.write(f"Image F1:        {img_f1:.4f}\n")
    print(f"[INFO] Full report saved to {report_path}")

if __name__ == '__main__':
    evaluate_model()
