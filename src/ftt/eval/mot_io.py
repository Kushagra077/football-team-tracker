"""Write tracks / GT in MOT format for TrackEval. Frames are 1-INDEXED. Persons only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MOT_COLUMNS = ["frame", "track_id", "x", "y", "w", "h", "conf", "cls", "vis", "unused"]
REQUIRED = ["frame", "track_id", "x1", "y1", "x2", "y2"]


def write_mot(tracks: pd.DataFrame, out_path: Path) -> None:
    """Write ``frame, id, x, y, w, h, conf, -1, -1, -1`` rows, one per box.

    ``tracks`` needs frame (1-indexed, as ``load_gt`` returns it), track_id, x1..y2 and
    optionally conf (defaults to 1). An empty frame writes an empty file, so a clip with no
    predictions still evaluates (as all misses) instead of silently dropping out.
    """
    missing = [c for c in REQUIRED if c not in tracks.columns]
    if missing:
        raise ValueError(f"tracks is missing columns: {missing}")
    if len(tracks) and tracks["frame"].min() < 1:
        raise ValueError("MOT frames are 1-indexed; got frame < 1")

    out = pd.DataFrame(
        {
            "frame": tracks["frame"].astype(int),
            "track_id": tracks["track_id"].astype(int),
            "x": tracks["x1"].astype(float),
            "y": tracks["y1"].astype(float),
            "w": (tracks["x2"] - tracks["x1"]).astype(float),
            "h": (tracks["y2"] - tracks["y1"]).astype(float),
            "conf": tracks["conf"].astype(float) if "conf" in tracks else 1.0,
            "cls": -1,
            "vis": -1,
            "unused": -1,
        }
    ).sort_values(["frame", "track_id"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, header=False, index=False, float_format="%.2f")


def empty_mot() -> pd.DataFrame:
    types = {"frame": int, "track_id": int, "x": float, "y": float, "w": float, "h": float}
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in types.items()})


def read_mot(path: Path) -> pd.DataFrame:
    """Read a MOT txt back as frame, track_id, x, y, w, h (empty file -> empty frame)."""
    path = Path(path)
    if path.stat().st_size == 0:
        return empty_mot()
    df = pd.read_csv(path, header=None, names=MOT_COLUMNS, usecols=range(6))
    return df.astype({"frame": int, "track_id": int}).reset_index(drop=True)


def box_iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xywh boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(
        np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0:1], b[None, :, 0]), 0, None
    )
    ih = np.clip(
        np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1:2], b[None, :, 1]), 0, None
    )
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
