"""Torso crop + grass mask. Torso = upper ~15-50% of box height (see configs/team.yaml)."""

from __future__ import annotations

import numpy as np


def torso_crop(frame: np.ndarray, box_xyxy: np.ndarray, cfg: dict) -> np.ndarray:
    raise NotImplementedError


def grass_mask(crop_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Boolean mask, True where the pixel is NOT grass (HSV green range removed)."""
    raise NotImplementedError
