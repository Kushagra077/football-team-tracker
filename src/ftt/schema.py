"""The output schema (``tracks.parquet``) and its adapter to the metrics project's contract.

This project emits the shared columns plus image-space columns. ``x_pitch`` /
``y_pitch`` / ``jersey_number`` are explicitly NULL (never 0: 0 is the centre
spot and a silent zero corrupts every downstream metric).

DECIDED (design_decision.md DD-003): standalone for now. The metrics project's
``ftm.schema.validate()`` requires non-null ``x_pitch``/``y_pitch`` and also needs
``period``, ``is_gk``, ``ball_state`` and ``team`` in {home, away, ball}; that contract
is revisited at integration time, so ``to_b1()`` stays a stub until then.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

FPS: Final[int] = 25  # SoccerNet-Tracking clips are 25 fps; timestamp = frame_id / FPS

# Shared program columns.
SHARED_COLUMNS: Final[list[str]] = [
    "frame_id", "timestamp", "track_id", "team", "jersey_number",
    "x_pitch", "y_pitch", "is_ball",
]  # fmt: skip
# Image-space columns.
IMAGE_COLUMNS: Final[list[str]] = ["x1", "y1", "x2", "y2", "role", "conf"]
COLUMNS: Final[list[str]] = SHARED_COLUMNS + IMAGE_COLUMNS
ROLES: Final[tuple[str, ...]] = ("player", "goalkeeper", "referee")  # the ball is out of scope here
ALWAYS_NULL: Final[tuple[str, ...]] = ("jersey_number", "x_pitch", "y_pitch")  # until pitch coords
# Only `team` may be null among the rest: referees and undecided goalkeepers have no team.
NULLABLE: Final[tuple[str, ...]] = ("team", *ALWAYS_NULL)


class SchemaError(ValueError):
    """Raised by ``validate()`` on the first violated rule. Never merely warns."""


DTYPES: Final[dict[str, str]] = {
    "frame_id": "int64",
    "timestamp": "float64",
    "track_id": "int64",      # tracker id; the metrics project wants a string (DD-003)
    "team": "Int8",           # KMeans cluster 0/1 (arbitrary per clip); null: referee, undecided GK
    "jersey_number": "Int16",  # nullable, always null here
    "x_pitch": "float64",     # NaN until pitch coordinates exist
    "y_pitch": "float64",     # NaN until pitch coordinates exist
    "is_ball": "bool",
    "x1": "float32", "y1": "float32", "x2": "float32", "y2": "float32",
    "role": "category",
    "conf": "float32",
}  # fmt: skip


def _role_dtype() -> pd.CategoricalDtype:
    return pd.CategoricalDtype(categories=ROLES)


def empty_frame() -> pd.DataFrame:
    """Zero-row frame with the exact output columns and dtypes."""
    cols = {c: pd.Series(dtype=DTYPES[c]) for c in COLUMNS if c != "role"}
    cols["role"] = pd.Series(dtype=_role_dtype())
    return pd.DataFrame(cols)[COLUMNS]


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Cast a nearly-right frame to the exact dtypes (presentation only, no semantic fixes).

    Missing pitch/jersey columns are filled with nulls, never zeros.
    """
    out = df.copy()
    for c in ALWAYS_NULL:
        if c not in out:
            out[c] = np.nan
    for c in COLUMNS:
        out[c] = out[c].astype(_role_dtype() if c == "role" else DTYPES[c])
    return out[COLUMNS]


def validate(df: pd.DataFrame, fps: float = FPS) -> pd.DataFrame:
    """Assert our own contract (DD-003: nothing external involved). Returns df, columns in order.

    Columns and dtypes match; pitch/jersey columns are ALL null (a zero is the centre spot);
    only ``team`` may otherwise be null; no ball rows; one row per (frame_id, track_id);
    ``timestamp == frame_id / fps``; boxes have positive size; conf in [0, 1].
    """
    missing, extra = set(COLUMNS) - set(df.columns), set(df.columns) - set(COLUMNS)
    if missing or extra:
        raise SchemaError(f"columns differ: missing {sorted(missing)}, extra {sorted(extra)}")
    df = df[COLUMNS]
    for c in COLUMNS:
        if str(df[c].dtype) != DTYPES[c]:
            raise SchemaError(f"column {c} has dtype {df[c].dtype}, expected {DTYPES[c]}")
    if set(df["role"].cat.categories) != set(ROLES):
        raise SchemaError(
            f"role categories are {list(df['role'].cat.categories)}, expected {ROLES}"
        )

    for c in ALWAYS_NULL:
        if df[c].notna().any():
            raise SchemaError(f"{c} must be null until pitch coordinates exist (never 0)")
    for c in COLUMNS:
        if c not in NULLABLE and df[c].isna().any():
            raise SchemaError(f"column {c} has nulls but must be fully populated")

    if not df["team"].dropna().isin([0, 1]).all():
        raise SchemaError("team must be 0, 1 or null (cluster ids), found other values")
    if df["is_ball"].any():
        raise SchemaError("The ball is not tracked here: is_ball must be all False")
    if df.duplicated(["frame_id", "track_id"]).any():
        raise SchemaError("duplicate (frame_id, track_id) rows")
    if not np.allclose(df["timestamp"], df["frame_id"] / fps):
        raise SchemaError(f"timestamp must equal frame_id / {fps}")
    if (df["frame_id"] < 1).any():
        raise SchemaError("frame_id is 1-indexed; found < 1")
    if ((df["x2"] <= df["x1"]) | (df["y2"] <= df["y1"])).any():
        raise SchemaError("boxes need x2 > x1 and y2 > y1")
    if not df["conf"].between(0, 1).all():
        raise SchemaError("conf must be within [0, 1]")
    return df


def to_b1(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt an output frame to the tactical-metrics project's canonical frame.

    Adds period, is_gk, ball_state; maps team to home/away; converts ids to strings.
    """
    raise NotImplementedError
