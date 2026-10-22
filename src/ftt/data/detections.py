"""GT-as-detections and cached A1 detections in ONE identical format.

This module is the reason the oracle-vs-real comparison is valid. Both sources
return a DataFrame with exactly ``DET_COLUMNS``; trackers never know which they got.

Class policy (written down before any number is computed): evaluate PERSONS only
(player, goalkeeper, referee). Ball is A4's job; ``other``/staff are excluded.
A1 has 5 classes (see ``A1_CLASS_NAMES``); the id mapping must be explicit here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from ftt.data.sn_tracking import load_persons_gt

DET_COLUMNS: Final[list[str]] = ["frame", "x1", "y1", "x2", "y2", "conf", "cls"]
# frame is 1-indexed (MOT convention) everywhere in this package.

# From football-detect-serve/configs/train.yaml expected_classes.
A1_CLASS_NAMES: Final[dict[int, str]] = {
    0: "ball", 1: "goalkeeper", 2: "player", 3: "referee", 4: "other",
}  # fmt: skip
KEEP_CLASSES: Final[tuple[str, ...]] = ("player", "goalkeeper", "referee")
# The `cls` column holds A1's class ids, the same numbers YOLO outputs, for both sources.
CLASS_IDS: Final[dict[str, int]] = {name: i for i, name in A1_CLASS_NAMES.items()}
KEEP_IDS: Final[frozenset[int]] = frozenset(CLASS_IDS[name] for name in KEEP_CLASSES)


def validate_detections(df: pd.DataFrame) -> pd.DataFrame:
    """The gatekeeper both sources pass through. Fails loud on anything a tracker should not see.

    Returns the frame with canonical dtypes, sorted by frame and reindexed from 0.
    """
    if list(df.columns) != DET_COLUMNS:
        raise ValueError(f"detections must have columns {DET_COLUMNS}, got {list(df.columns)}")
    if df.isna().any().any():
        raise ValueError("detections contain NaN values")
    if len(df):
        if df["frame"].min() < 1:
            raise ValueError("frames are 1-indexed; got frame < 1")
        if not df["conf"].between(0, 1).all():
            raise ValueError("conf must be within [0, 1]")
        bad = set(df["cls"].astype(int)) - KEEP_IDS
        if bad:
            raise ValueError(f"cls ids {sorted(bad)} are not in {sorted(KEEP_IDS)} (persons only)")

    out = df.astype(
        {
            "frame": "int64",
            "cls": "int64",
            **{c: "float64" for c in ["x1", "y1", "x2", "y2", "conf"]},
        }
    )
    return out.sort_values("frame", kind="stable").reset_index(drop=True)


def gt_as_detections(clip_dir: Path) -> pd.DataFrame:
    """Ground-truth boxes as detections: conf = 1.0, classes mapped to KEEP_CLASSES.

    Track ids are dropped on purpose: a detector does not know them.
    """
    gt = load_persons_gt(clip_dir)
    dets = pd.DataFrame(
        {
            "frame": gt["frame"],
            "x1": gt["x1"],
            "y1": gt["y1"],
            "x2": gt["x2"],
            "y2": gt["y2"],
            "conf": 1.0,
            "cls": gt["cls"].map(CLASS_IDS),
        }
    )
    return validate_detections(dets)


def detector_name(weights: Path) -> str:
    """The detector's name is its weights file name (``football_detection_v3``).

    Cache and result folders are keyed by it, so a new detector can never silently reuse
    another detector's cached boxes (the cache script skips clips that already exist).
    """
    return Path(weights).stem


def cached_path(cache_dir: Path, detector: str, clip_id: str) -> Path:
    """Where ``detect/run_detector.py`` saves a clip's detections for one detector."""
    return Path(cache_dir) / "detections" / detector / f"{clip_id}.parquet"


def load_cached_detections(cache_dir: Path, detector: str, clip_id: str) -> pd.DataFrame:
    """A1 detections cached by ``detect/run_detector.py`` (conf threshold ~0.1), same format."""
    path = cached_path(cache_dir, detector, clip_id)
    if not path.exists():
        raise FileNotFoundError(f"no cached detections from {detector} for {clip_id} at {path}")
    return validate_detections(pd.read_parquet(path))
