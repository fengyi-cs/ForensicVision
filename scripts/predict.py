# -*- coding: utf-8 -*-
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
predict.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
    Entry point for running tampering localization inference.

CONTENTS
    Functions  - CLI parsing, preprocessing helpers, inference, visualization
    Classes    - None

NOTES
    Dependencies  - PyTorch, Albumentations, OpenCV, Matplotlib
    Limitations   - Designed for single-model inference only

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Author: You Boyao | Created: 2026/1/11
"""

# ── Imports ──────────────────────────────────────────────────────

# Standard library
import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Third party
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Local application
current_dir = Path(__file__).resolve().parent
root_path = current_dir.parent
sys.path.append(str(root_path))

from src import config
from src.model import build_model
from src.ela import convert_to_ela

# ── Globals ──────────────────────────────────────────────────────

_ALLOWED_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
_TRANSFORM: Optional[A.Compose] = None


# ── Helper functions ─────────────────────────────────────────────

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Undo normalization on a CHW tensor and return an RGB image in [0, 1]."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std) + mean
    return np.clip(img, 0, 1)


def get_inference_transform() -> A.Compose:
    """Create (or reuse) the Albumentations pipeline for inference."""
    global _TRANSFORM
    if _TRANSFORM is None:
        mean_6ch = (0.485, 0.456, 0.406, 0.485, 0.456, 0.406)
        std_6ch = (0.229, 0.224, 0.225, 0.229, 0.224, 0.225)
        _TRANSFORM = A.Compose([
            A.Resize(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
            A.Normalize(mean=mean_6ch, std=std_6ch, max_pixel_value=255.0),
            ToTensorV2(),
        ])
    return _TRANSFORM


def prepare_image_tensors(image_path: Path, transform: A.Compose) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load RGB + ELA stacks, apply preprocessing, and return batched tensor and channel splits."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ela = np.array(convert_to_ela(str(image_path)))
    if ela.shape[:2] != image.shape[:2]:
        ela = cv2.resize(ela, (image.shape[1], image.shape[0]))

    combined_img = np.dstack((image, ela))
    augmented = transform(image=combined_img)
    image_tensor = augmented["image"]
    batched_tensor = image_tensor.unsqueeze(0).to(config.DEVICE)
    rgb_tensor = image_tensor[:3, :, :]
    ela_tensor = image_tensor[3:, :, :]
    return batched_tensor, rgb_tensor, ela_tensor


def visualize_prediction(
    rgb_tensor: torch.Tensor,
    ela_tensor: torch.Tensor,
    pred_mask: np.ndarray,
    save_path: Optional[Path],
    show: bool,
) -> None:
    """Render inference panels and optionally save/show the figure."""
    fig, ax = plt.subplots(1, 4, figsize=(22, 6))

    ax[0].imshow(denormalize_image(rgb_tensor))
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    ax[1].imshow(denormalize_image(ela_tensor))
    ax[1].set_title("ELA Noise Map")
    ax[1].axis("off")

    ax[2].imshow(pred_mask, cmap="jet", vmin=0, vmax=1)
    ax[2].set_title("Prediction Probability")
    ax[2].axis("off")

    binary_mask = (pred_mask > config.INFER_THRESHOLD).astype(float)
    ax[3].imshow(binary_mask, cmap="gray")
    ax[3].set_title("Final Binary Mask")
    ax[3].axis("off")

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    if save_path:
        fig.savefig(save_path)
        print(f"✅ Saved result to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def predict_single_image(
    image_path: Path,
    model: torch.nn.Module,
    transform: A.Compose,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> bool:
    """Run inference for one image and handle optional visualization output."""
    try:
        batched_tensor, rgb_tensor, ela_tensor = prepare_image_tensors(image_path, transform)
    except ValueError as exc:
        print(f"[WARN] {exc}")
        return False

    with torch.inference_mode():
        logits = model(batched_tensor)
        pred_mask = torch.sigmoid(logits).squeeze().cpu().numpy()

    if save_path or show:
        visualize_prediction(rgb_tensor, ela_tensor, pred_mask, save_path, show)
    return True


def collect_image_paths(input_path: Path, limit: Optional[int], seed: Optional[int]) -> List[Path]:
    """Resolve the list of images to run inference on."""
    input_path = input_path.expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        return [input_path]

    candidates = [p for p in input_path.iterdir() if p.suffix.lower() in _ALLOWED_SUFFIXES]
    if not candidates:
        return []

    if seed is not None:
        random.seed(seed)
    random.shuffle(candidates)

    if limit is not None:
        return candidates[:limit]
    return candidates


def load_trained_model(checkpoint_path: Path) -> torch.nn.Module:
    """Instantiate the model architecture and load checkpoint weights."""
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Please train the model first."
        )

    print("Loading model from checkpoint...")
    model = build_model().to(config.DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded!")
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tampering localization inference.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Image file or directory. Defaults to config.TP_DIR when omitted.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=str(root_path / "checkpoints" / "best_model.pth"),
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(root_path / "results"),
        help="Directory to store visualization grids.",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=5,
        help="Maximum number of images to sample when the input is a directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed used when sampling images from a directory.",
    )
    parser.add_argument(
        "--disable-save",
        action="store_true",
        help="Skip writing visualization PNGs to disk.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display matplotlib windows for each prediction (requires GUI).",
    )
    return parser.parse_args()


# ── Main entry ───────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    try:
        model = load_trained_model(Path(args.checkpoint))
    except FileNotFoundError as exc:
        print(exc)
        return

    transform = get_inference_transform()
    input_root = Path(args.input) if args.input else Path(config.TP_DIR)

    try:
        images = collect_image_paths(input_root, None if input_root.is_file() else args.limit, args.seed)
    except FileNotFoundError as exc:
        print(exc)
        return

    if not images:
        print(f"No images found in {input_root}.")
        return

    output_dir = None if args.disable_save else Path(args.output_dir)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running inference on {len(images)} image(s)...")
    successes = 0
    for idx, image_path in enumerate(images):
        save_path = None
        if output_dir:
            save_path = output_dir / f"final_pred_{idx}_{image_path.name}.png"
        if predict_single_image(image_path, model, transform, save_path=save_path, show=args.show):
            successes += 1

    print(f"Completed {successes}/{len(images)} predictions.")


if __name__ == "__main__":
    main()
