"""A2's output schema (``tracks.parquet``) and its adapter to B1's contract.

A2 emits the shared program columns plus image-space columns. ``x_pitch`` /
``y_pitch`` / ``jersey_number`` are explicitly NULL (never 0: 0 is the centre
spot and a silent zero corrupts every downstream metric).

OPEN ISSUE (see design_decision.md DD-003): B1's ``ftm.schema.validate()``
requires non-null ``x_pitch``/``y_pitch`` and also needs ``period``, ``is_gk``,
``ball_state`` and ``team`` in {home, away, ball}. ``to_b1()`` is where that
mismatch is resolved once the decision is made.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

FPS: Final[int] = 25  # SoccerNet-Tracking clips are 25 fps; timestamp = frame_id / FPS

# Shared program columns (spec step 7).
SHARED_COLUMNS: Final[list[str]] = [
    "frame_id", "timestamp", "track_id", "team", "jersey_number",
    "x_pitch", "y_pitch", "is_ball",
]  # fmt: skip
# A2-specific image-space columns.
IMAGE_COLUMNS: Final[list[str]] = ["x1", "y1", "x2", "y2", "role", "conf"]
COLUMNS: Final[list[str]] = SHARED_COLUMNS + IMAGE_COLUMNS

DTYPES: Final[dict[str, str]] = {
    "frame_id": "int64",
    "timestamp": "float64",
    "track_id": "int64",      # TODO(DD-003): B1 wants string
    "team": "Int8",           # TODO: cluster id {0,1} vs left/right vs home/away: decide with B1 mapping
    "jersey_number": "Int16",  # nullable, always null in A2
    "x_pitch": "float64",     # NaN until A3
    "y_pitch": "float64",     # NaN until A3
    "is_ball": "bool",
    "x1": "float32", "y1": "float32", "x2": "float32", "y2": "float32",
    "role": "category",
    "conf": "float32",
}  # fmt: skip


def empty_frame() -> pd.DataFrame:
    """Zero-row frame with the exact A2 columns and dtypes."""
    raise NotImplementedError


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Assert A2's own contract: columns, dtypes, pitch/jersey columns all-null, ids unique per frame."""
    raise NotImplementedError


def to_b1(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt an A2 frame to B1's canonical frame (period, is_gk, ball_state, home/away, string ids)."""
    raise NotImplementedError
