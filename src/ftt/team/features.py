"""Colour features (v1): mean colour of unmasked pixels in Lab. Embeddings are a stretch (v2)."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import pandas as pd

from ftt.team.crops import grass_mask, torso_crop


def colour_feature(crop_bgr: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Mean (L, a, b) of the pixels where ``keep_mask`` is True; NaNs if no pixel is kept.

    Float input in [0, 1] makes OpenCV return true Lab (L 0-100, a/b about -128..127)
    instead of the 8-bit version with offsets.
    """
    if not keep_mask.any():
        return np.full(3, np.nan)
    lab = cv2.cvtColor(crop_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    return lab[keep_mask].mean(axis=0)


def box_features(boxes: pd.DataFrame, frame_loader: Callable[[int], np.ndarray], cfg: dict):
    """Lab colour of every box's torso. ``boxes`` has frame, x1, y1, x2, y2 (extras ignored).

    ``frame_loader(frame_no)`` returns the BGR image (a folder of JPGs, a video reader, ...).
    Returns frame, x1..y2, L, a, b with the INDEX of the input row it came from, so callers can
    join labels back without matching floats. Empty or all-grass crops are dropped.
    """
    rows, index = [], []
    for fr, d in boxes.groupby("frame", sort=True):
        img = frame_loader(int(fr))
        for i, r in zip(d.index, d.itertuples(), strict=True):
            box = np.array([r.x1, r.y1, r.x2, r.y2])
            crop = torso_crop(img, box, cfg)
            if crop.size == 0:
                continue
            f = colour_feature(crop, grass_mask(crop, cfg))
            if not np.isnan(f).any():
                rows.append((int(fr), *box, *f))
                index.append(i)
    cols = ["frame", "x1", "y1", "x2", "y2", "L", "a", "b"]
    return pd.DataFrame(rows, columns=cols, index=index).astype({"frame": "int64"})
