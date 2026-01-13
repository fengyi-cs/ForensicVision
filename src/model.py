# -*- coding: utf-8 -*-
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
model.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
    Defines the deep learning model architecture for image segmentation tasks.
    
CONTENTS
    Functions  - build_model
    Classes    - None
    
NOTES
    Dependencies  - segmentation_models_pytorch
    Limitations   - None

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Author: You Boyao | Created: 2026/1/11
"""

# ── Imports ──────────────────────────────────────────────────────

# Standard library

# Third party
import segmentation_models_pytorch as smp

# Local application


def build_model():
    """
    Kaggle-style Architecture: Unet++ with EfficientNet Encoder.

    Why this is better:
    1. Unet++: Dense skip connections capture multi-scale artifacts better than standard U-Net.
    2. EfficientNet-B3: Much stronger feature extractor than ResNet18, optimal for fine-grained details.
    """
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3", # Strong encoder
        encoder_weights="imagenet",
        in_channels=6,                  # RGB + ELA
        classes=1,
        activation=None
    )

    return model
