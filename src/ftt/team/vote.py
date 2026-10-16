"""Rolling per-track majority vote (window ~25 frames) with a lock once enough frames agree."""

from __future__ import annotations

import numpy as np


def vote_labels(
    track_ids: np.ndarray,
    frames: np.ndarray,
    labels: np.ndarray,
    window: int = 25,
    lock_min_agree: int = 15,
) -> np.ndarray:
    raise NotImplementedError
