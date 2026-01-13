# -*- coding: utf-8 -*-
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
train.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
    Entry point for training tampering localization model.
    
CONTENTS
    Functions  - train, save_plots
    Classes    - None
    
NOTES
    Dependencies  - PyTorch, segmentation_models_pytorch, Matplotlib, tqdm, numpy
    Limitations   - Designed for training Unet++ on CASIA dataset with dynamic pos_weight BCE loss.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Author: You Boyao | Created: 2026/1/11
"""

# ── Imports ──────────────────────────────────────────────────────

# Standard library
import sys
import os
import csv

# Third party
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import segmentation_models_pytorch as smp
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Subset

# Local application
from src import config
from src.dataset import CASIADataset
from src.model import build_model
from src.utils import calculate_metrics, visualize_prediction

# --- Path System Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
sys.path.append(root_path)


def save_plots(history, save_dir):
    """
    Helper function to generate training charts based on history dictionary.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # Set style
    plt.style.use('ggplot')

    # 1. Plot Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue', alpha=0.6)
    plt.plot(epochs, history['val_loss'], label='Val Loss', color='red', linewidth=2)
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'chart_loss_curve.png'))
    plt.close()

    # 2. Plot F1 Score Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_f1'], label='Val F1 Score', color='green', linewidth=2, marker='o')
    plt.title('Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'chart_f1_curve.png'))
    plt.close()

def train():
    # --- 1. Setup Logging & Data Structures ---
    # Create directories
    if not os.path.exists('./checkpoints'): os.makedirs('./checkpoints')
    if not os.path.exists('./results'): os.makedirs('./results')

    # Initialize CSV logging
    log_file = open('training_log.csv', 'w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_IoU', 'Val_F1', 'Best_Thr', 'Learning_Rate'])


    # History dictionary for plotting
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    # --- 2. Initialize Dataset ---
    print(f"[INFO] Initializing dataset (Resolution: {config.IMAGE_SIZE})...")

    # Split 80/20
    print(f"[INFO] Initializing dataset (Resolution: {config.IMAGE_SIZE})...")

    # ----- Reproducible split indices -----
    # Use a non-aug dataset to generate stable indices
    index_dataset = CASIADataset(mode='val')
    g = torch.Generator().manual_seed(config.SEED)

    indices = torch.randperm(len(index_dataset), generator=g).tolist()
    train_size = int(0.8 * len(indices))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    # Train uses augmentation, Val uses no augmentation
    train_dataset = Subset(CASIADataset(mode='train'), train_idx)
    val_dataset = Subset(CASIADataset(mode='val'), val_idx)

    # Save split for evaluate.py to use the exact same val set
    torch.save({"train_idx": train_idx, "val_idx": val_idx}, "./checkpoints/split_idx.pth")
    print(f"[INFO] Split saved to ./checkpoints/split_idx.pth | Train: {len(train_idx)}, Val: {len(val_idx)}")


    # DataLoaders (Keep num_workers=0 for stability on WSL)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0),
        prefetch_factor=2 if config.NUM_WORKERS > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    # --- 3. Build Model & Loss ---
    print(f"[INFO] Building Unet++ (EfficientNet-B3) on {config.DEVICE}...")
    model = build_model().to(config.DEVICE)

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = smp.losses.DiceLoss(mode='binary')
    criterion_focal = smp.losses.FocalLoss(mode='binary')

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    # AMP Scaler
    scaler = GradScaler('cuda')

    best_val_f1 = 0.0

    print("[INFO] Starting training...")

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss_accum = 0

        # --- Training Loop ---
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")

        optimizer.zero_grad(set_to_none=True)
        for step, (images, masks) in enumerate(loop):
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            with autocast('cuda'):
                outputs = model(images)

                # --- Dynamic pos_weight for sparse foreground ---
                pos = masks.sum()
                neg = masks.numel() - pos
                pos_weight = (neg / (pos + 1e-7)).clamp(1.0, config.POS_WEIGHT_CLAMP_MAX)

                bce = F.binary_cross_entropy_with_logits(outputs, masks, pos_weight=pos_weight)
                dice = criterion_dice(outputs, masks)
                focal = criterion_focal(outputs, masks)

                # Gradient accumulation (divide loss)
                loss = (0.3 * bce + 0.4 * dice + 0.3 * focal) / config.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            # Step optimizer only every GRAD_ACCUM_STEPS
            if (step + 1) % config.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Log the "real" loss (multiply back)
            train_loss_accum += loss.item() * config.GRAD_ACCUM_STEPS

            loop.set_postfix(loss=(loss.item() * config.GRAD_ACCUM_STEPS))

        avg_train_loss = train_loss_accum / len(train_loader)

        # Update Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # --- Validation Loop ---
        model.eval()
        val_loss_accum = 0.0

        thr_list = config.VAL_THRESHOLDS
        thr_val_iou = [0.0 for _ in thr_list]
        thr_val_f1  = [0.0 for _ in thr_list]

        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                images = images.to(config.DEVICE)
                masks = masks.to(config.DEVICE)

                with autocast('cuda'):
                    outputs = model(images)

                    if i == 0:
                        probs = torch.sigmoid(outputs)
                        print("[DEBUG] probs min/mean/max:",
                              probs.min().item(), probs.mean().item(), probs.max().item())
                        print("[DEBUG] gt positive ratio:", masks.mean().item())
                        print("[DEBUG] pred>0.5 ratio:", (probs > 0.5).float().mean().item())


                    loss = (0.3 * criterion_bce(outputs, masks) +
                            0.4 * criterion_dice(outputs, masks) +
                            0.3 * criterion_focal(outputs, masks))

                # accumulate val loss
                val_loss_accum += loss.item()

                # accumulate metrics for each threshold
                for t_i, thr in enumerate(thr_list):
                    iou_t, f1_t = calculate_metrics(outputs, masks, thr=thr)
                    thr_val_iou[t_i] += iou_t
                    thr_val_f1[t_i]  += f1_t

                # Save first batch visualization
                if i == 0:
                    visualize_prediction(
                        images, masks, outputs,
                        save_path=f"./results/epoch_{epoch+1}_val_preview.png"
                    )


        avg_thr_iou = [x / len(val_loader) for x in thr_val_iou]
        avg_thr_f1  = [x / len(val_loader) for x in thr_val_f1]

        best_idx = int(np.argmax(avg_thr_f1))
        best_thr = thr_list[best_idx]
        avg_val_iou = avg_thr_iou[best_idx]
        avg_val_f1  = avg_thr_f1[best_idx]
        avg_val_loss = val_loss_accum / len(val_loader)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f}")
        print(f"[VAL] Best threshold: {best_thr:.2f} | Val IoU: {avg_val_iou:.4f} | Val F1: {avg_val_f1:.4f}")

        # --- Logging & Plotting ---
        # 1. Update History List
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(avg_val_f1)

        # 2. Write to CSV
        writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_val_iou, avg_val_f1, best_thr, current_lr])
        log_file.flush()

        # 3. Generate/Update Plots
        save_plots(history, './results')

        # --- Save Best Model ---
        if avg_val_f1 > best_val_f1:
            print(f">>> Improved F1 ({best_val_f1:.4f} -> {avg_val_f1:.4f}). Saving model...")
            best_val_f1 = avg_val_f1
            torch.save(model.state_dict(), "./checkpoints/best_model.pth")

        print("-" * 50)

    log_file.close()
    print("[INFO] Training Complete. Logs saved to training_log.csv")

if __name__ == '__main__':
    train()
