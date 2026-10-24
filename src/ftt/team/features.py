"""Colour features (v1): mean colour of unmasked pixels in Lab. Embeddings are a stretch (v2)."""

from __future__ import annotations

import cv2
import numpy as np


def colour_feature(crop_bgr: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Mean (L, a, b) of the pixels where ``keep_mask`` is True; NaNs if no pixel is kept.

    Float input in [0, 1] makes OpenCV return true Lab (L 0-100, a/b about -128..127)
    instead of the 8-bit version with offsets.
    """
    if not keep_mask.any():
        return np.full(3, np.nan)
    lab = cv2.cvtColor(crop_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    return lab[keep_mask].mean(axis=0)
