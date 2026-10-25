"""Colour features of a player's torso: mean Lab colour (``mean_lab``) or a Lab colour histogram
(``hist_lab``), chosen by ``features.kind`` in configs/team.yaml. Embeddings are a stretch (v2)."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import pandas as pd

from ftt.team.crops import grass_mask, torso_crop

# Histogram bins: L in 4, a and b in 6 each, over ranges that cover shirt colours (a coarse
# palette; 3x5x5 to 5x8x8 gave the same accuracy on the dev clips, see DD-032).
HIST_RANGE = ((0, 100), (-60, 80), (-70, 80))
HIST_BINS = (4, 6, 6)
FRAME_COLS = ["frame", "x1", "y1", "x2", "y2"]


def lab_pixels(crop_bgr: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Lab values (n, 3) of the pixels where ``keep_mask`` is True.

    Float input in [0, 1] makes OpenCV return true Lab (L 0-100, a/b about -128..127)
    instead of the 8-bit version with offsets.
    """
    lab = cv2.cvtColor(crop_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    return lab[keep_mask]


def mean_feature(px: np.ndarray) -> np.ndarray:
    """Mean (L, a, b); NaNs if there are no pixels."""
    return px.mean(axis=0) if len(px) else np.full(3, np.nan)


def histogram_feature(px: np.ndarray) -> np.ndarray:
    """Share of pixels per Lab colour bin, square-rooted. A striped shirt keeps two peaks instead
    of averaging into a colour neither stripe has; leftover grass sits in its own bins. NaNs if
    there are no pixels.

    Counts are divided by ALL pixels, but ``np.histogramdd`` drops pixels outside ``HIST_RANGE``
    (very saturated colours), so the shares can sum to a little under 1. On the demo clips that
    is at most 0.2% of shirt pixels, so Euclidean distance here is only approximately the
    Hellinger distance between true probability histograms."""
    n_bins = int(np.prod(HIST_BINS))
    if not len(px):
        return np.full(n_bins, np.nan)
    h, _ = np.histogramdd(px, bins=HIST_BINS, range=HIST_RANGE)
    return np.sqrt(h.ravel() / len(px))


def colour_feature(crop_bgr: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Mean (L, a, b) of the pixels where ``keep_mask`` is True; NaNs if no pixel is kept."""
    return mean_feature(lab_pixels(crop_bgr, keep_mask))


MEAN_COLS = ["L", "a", "b"]
HIST_COLS = [f"h{i}" for i in range(int(np.prod(HIST_BINS)))]


def feature_kind(cfg: dict) -> str:
    kind = cfg.get("features", {}).get("kind", "mean_lab")
    if kind not in ("mean_lab", "hist_lab"):
        raise ValueError(f"features.kind must be mean_lab or hist_lab, got {kind!r}")
    return kind


def feature_names(cfg: dict) -> list[str]:
    """Columns the team clustering uses (``features.kind``)."""
    return HIST_COLS if feature_kind(cfg) == "hist_lab" else MEAN_COLS


def extra_names(cfg: dict) -> list[str]:
    """Columns ``box_features`` returns: always the mean Lab (the colour-outlier rule measures
    distances with it), plus the histogram when that is the clustering feature."""
    return MEAN_COLS + (HIST_COLS if feature_kind(cfg) == "hist_lab" else [])


def box_features(boxes: pd.DataFrame, frame_loader: Callable[[int], np.ndarray], cfg: dict):
    """Colour feature of every box's torso (``features.kind``). ``boxes`` has frame, x1..y2.

    ``frame_loader(frame_no)`` returns the BGR image (a folder of JPGs, a video reader, ...).
    Returns frame, x1..y2 and ``extra_names(cfg)`` with the INDEX of the input row it came from,
    so callers can join labels back without matching floats. Empty or all-grass crops are dropped.
    """
    hist = feature_kind(cfg) == "hist_lab"
    rows, index = [], []
    for fr, d in boxes.groupby("frame", sort=True):
        img = frame_loader(int(fr))
        for i, r in zip(d.index, d.itertuples(), strict=True):
            box = np.array([r.x1, r.y1, r.x2, r.y2])
            crop = torso_crop(img, box, cfg)
            if crop.size == 0:
                continue
            px = lab_pixels(crop, grass_mask(crop, cfg))
            f = mean_feature(px)
            if hist and not np.isnan(f).any():
                f = np.concatenate([f, histogram_feature(px)])
            if not np.isnan(f).any():
                rows.append((int(fr), *box, *f))
                index.append(i)
    cols = [*FRAME_COLS, *extra_names(cfg)]
    # One float array, one block: building the wide histogram table from tuples and converting
    # its dtype every frame cost ~3x the whole colour step.
    out = pd.DataFrame(
        np.array(rows, dtype="float64").reshape(-1, len(cols)), columns=cols, index=index
    )
    out["frame"] = out["frame"].astype("int64")
    return out
