"""Team-assignment metrics. KMeans ids are arbitrary, so every score is permutation-invariant."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


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


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, 4) x (m, 4) xyxy boxes -> (n, m) IoU."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def match_to_gt(boxes: np.ndarray, gt_boxes: np.ndarray, min_iou: float) -> np.ndarray:
    """For each detection, the index of its matched GT player (one-to-one) or -1."""
    out = np.full(len(boxes), -1)
    if len(boxes) == 0 or len(gt_boxes) == 0:
        return out
    iou = iou_matrix(boxes, gt_boxes)
    rows, cols = linear_sum_assignment(-iou)
    for r, c in zip(rows, cols, strict=True):
        if iou[r, c] >= min_iou:
            out[r] = c
    return out
