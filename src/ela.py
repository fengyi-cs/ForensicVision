# -*- coding: utf-8 -*-
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ela.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PURPOSE
    Provides functionality to perform Error Level Analysis (ELA) on images.
    
CONTENTS
    Functions  - convert_to_ela
    Classes    - None
    
NOTES
    Dependencies  - PIL (Pillow)
    Limitations   - None

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Author: You Boyao | Created: 2026/1/11
"""

# ── Imports ──────────────────────────────────────────────────────

# Standard library
import io
# Third party
from PIL import Image, ImageChops, ImageEnhance
# Local application


def convert_to_ela(image_path, quality=90):
    """
    Generates an Error Level Analysis (ELA) image.
    Refactored to use in-memory BytesIO to prevent multi-threading file conflicts.
    """
    try:
        # 1. Open original image
        original = Image.open(image_path).convert('RGB')

        # 2. Save to memory buffer instead of disk
        # This fixes the race condition when NUM_WORKERS > 0
        buffer = io.BytesIO()
        original.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)  # Rewind the file pointer to the beginning

        # 3. Read back from memory
        resaved = Image.open(buffer)

        # 4. Calculate difference
        ela_image = ImageChops.difference(original, resaved)

        # 5. Enhance visibility
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1

        scale = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        return ela_image

    except Exception as e:
        print(f"Error processing ELA for {image_path}: {e}")
        # Return a black image in case of error to prevent crash
        return Image.new('RGB', (256, 256), (0, 0, 0))
