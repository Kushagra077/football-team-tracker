"""Colour features (v1): mean colour of unmasked pixels in Lab. Embeddings are a stretch (v2)."""

from __future__ import annotations

import numpy as np


def colour_feature(crop_bgr: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    raise NotImplementedError
