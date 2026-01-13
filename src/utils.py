# -*- coding: utf-8 -*-
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
utils.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
    Provides utility functions for calculating evaluation metrics
    and visualizing model predictions.
    
CONTENTS
    Functions  - calculate_metrics, visualize_prediction
    Classes    - None
    
NOTES
    Dependencies  - torch, matplotlib
    Limitations   - None

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Author: You Boyao | Created: 2026/1/11
"""

# ── Imports ──────────────────────────────────────────────────────

# Standard library

# Third party
import torch
import matplotlib.pyplot as plt

# Local application


def calculate_metrics(pred_logits, masks, thr=0.5, eps=1e-7):
    """
    Calculate IoU and Dice(F1) at a given threshold.
    pred_logits: (B,1,H,W) raw logits
    masks:      (B,1,H,W) float mask in {0,1}
    """
    probs = torch.sigmoid(pred_logits)
    preds = (probs > thr).float()

    intersection = (preds * masks).sum()
    total = preds.sum() + masks.sum()
    union = total - intersection

    iou = (intersection + eps) / (union + eps)
    f1 = (2 * intersection + eps) / (total + eps)
    return iou.item(), f1.item()

def visualize_prediction(images, masks, preds, save_path="result.png"):
    """
    Visualizes a single sample: Original Image, Ground Truth, and Prediction.
    Note: Since input is 6 channels, we only visualize the RGB part (first 3 channels).
    """
    # Move to CPU and detach
    img = images[0][:3, :, :].cpu().permute(1, 2, 0).numpy() # Take first 3 channels (RGB)
    mask = masks[0].cpu().squeeze().numpy()
    pred = torch.sigmoid(preds[0]).cpu().squeeze().detach().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Original Image
    ax[0].imshow(img)
    ax[0].set_title("Input RGB Image")
    ax[0].axis('off')

    # 2. Ground Truth
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Ground Truth Mask")
    ax[1].axis('off')

    # 3. Prediction
    ax[2].imshow(pred, cmap='gray')
    ax[2].set_title("Model Prediction")
    ax[2].axis('off')

    plt.savefig(save_path)
    plt.close()
