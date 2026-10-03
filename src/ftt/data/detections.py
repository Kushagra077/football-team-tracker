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

DET_COLUMNS: Final[list[str]] = ["frame", "x1", "y1", "x2", "y2", "conf", "cls"]
# frame is 1-indexed (MOT convention) everywhere in this package.

# From football-detect-serve/configs/train.yaml expected_classes.
A1_CLASS_NAMES: Final[dict[int, str]] = {
    0: "ball", 1: "goalkeeper", 2: "player", 3: "referee", 4: "other",
}  # fmt: skip
KEEP_CLASSES: Final[tuple[str, ...]] = ("player", "goalkeeper", "referee")


def gt_as_detections(clip_dir: Path) -> pd.DataFrame:
    """Ground-truth boxes as detections: conf = 1.0, classes mapped to KEEP_CLASSES."""
    raise NotImplementedError


def load_cached_detections(cache_dir: Path, clip_id: str) -> pd.DataFrame:
    """A1 detections cached by ``detect/run_detector.py`` (conf threshold ~0.1), same format."""
    raise NotImplementedError
