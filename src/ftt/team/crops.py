"""Torso crop + grass mask. Torso = upper ~15-50% of box height (see configs/team.yaml)."""

from __future__ import annotations

import cv2
import numpy as np


def torso_crop(frame: np.ndarray, box_xyxy: np.ndarray, cfg: dict) -> np.ndarray:
    """Slice the shirt band out of a player box: rows y_top..y_bottom of the height,
    with x_trim cut off each side. Pure arithmetic, no model. Returns a view of ``frame``."""
    x1, y1, x2, y2 = box_xyxy
    w, h = x2 - x1, y2 - y1
    c = cfg["crop"]
    top = int(round(y1 + c["y_top"] * h))
    bottom = int(round(y1 + c["y_bottom"] * h))
    left = int(round(x1 + c["x_trim"] * w))
    right = int(round(x2 - c["x_trim"] * w))
    # Boxes can poke outside the image at the frame edge; clamp so slicing never wraps.
    img_h, img_w = frame.shape[:2]
    top, bottom = max(top, 0), min(bottom, img_h)
    left, right = max(left, 0), min(right, img_w)
    return frame[top:bottom, left:right]


def grass_mask(crop_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Boolean mask, True where the pixel is NOT grass (HSV green range removed)."""
    g = cfg["grass_mask"]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Grass = green hue AND saturated enough AND bright enough. The s/v floors keep white,
    # grey and dark pixels (which have unstable hue) from being called grass.
    is_grass = (h >= g["h"][0]) & (h <= g["h"][1]) & (s >= g["s_min"]) & (v >= g["v_min"])
    return ~is_grass
