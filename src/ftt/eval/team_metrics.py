"""Team-assignment metrics. KMeans ids are arbitrary, so every score is permutation-invariant."""

from __future__ import annotations

import numpy as np


def best_permutation_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Accuracy of 0/1 labels, taking the better of the two possible cluster-to-team namings."""
    if len(pred) == 0:
        return float("nan")
    agree = float((pred == true).mean())
    return max(agree, 1.0 - agree)


def per_track_accuracy(pred: np.ndarray, true: np.ndarray, track_ids: np.ndarray) -> float:
    """Share of tracks whose majority label is right, under ONE naming for the whole clip.

    The naming is chosen once per clip (not per track), otherwise every track could pick its own.
    """
    if len(pred) == 0:
        return float("nan")
    if (pred == true).mean() < 0.5:
        pred = 1 - pred
    ok = [(pred[track_ids == t] == true[track_ids == t]).mean() > 0.5 for t in np.unique(track_ids)]
    return float(np.mean(ok))
