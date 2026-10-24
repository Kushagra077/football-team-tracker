"""Build A2's table from tracker output and write/read ``tracks.parquet`` (see ``schema.py``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ftt.data.detections import A1_CLASS_NAMES
from ftt.schema import FPS, ROLES, coerce, validate


def build_tracks(tracked: pd.DataFrame, team) -> pd.DataFrame:
    """Tracker rows + one team label per row -> A2 table (not yet validated).

    ``tracked`` has frame, track_id, x1, y1, x2, y2, conf, cls (A1 class ids). ``team`` is
    aligned with it: 0 / 1, or NA where undecided. Referees never get a team. Pitch and jersey
    columns come out null (``coerce`` fills them with NaN, never 0).
    """
    role = tracked["cls"].map(A1_CLASS_NAMES)
    bad = sorted(set(role.dropna()) - set(ROLES)) + (["unmapped"] if role.isna().any() else [])
    if bad:
        raise ValueError(f"tracked rows with classes outside {ROLES}: {bad}")
    team = pd.array(np.asarray(team, dtype="float64"), dtype="Int8")  # NaN -> NA
    team[(role == "referee").to_numpy()] = pd.NA
    out = pd.DataFrame(
        {
            "frame_id": tracked["frame"].to_numpy(),
            "timestamp": tracked["frame"].to_numpy() / FPS,
            "track_id": tracked["track_id"].to_numpy(),
            "team": team,
            "is_ball": False,
            "x1": tracked["x1"].to_numpy(),
            "y1": tracked["y1"].to_numpy(),
            "x2": tracked["x2"].to_numpy(),
            "y2": tracked["y2"].to_numpy(),
            "role": role.to_numpy(),
            "conf": tracked["conf"].to_numpy(),
        }
    )
    return coerce(out).sort_values(["frame_id", "track_id"]).reset_index(drop=True)


def export_tracks(tracks: pd.DataFrame, out_path: Path) -> Path:
    """Validate against ``schema.validate`` then write parquet. Pitch/jersey columns stay null."""
    validate(tracks)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_parquet(out_path, index=False)
    return out_path


def read_tracks(path: Path) -> pd.DataFrame:
    """Read ``tracks.parquet`` back and re-validate: a file that does not round-trip is a bug."""
    return validate(pd.read_parquet(path))
