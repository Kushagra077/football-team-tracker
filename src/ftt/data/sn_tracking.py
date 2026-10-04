"""Load SoccerNet-Tracking clips: frames, MOT ground truth, ``gameinfo.ini`` roles.

MOT ground-truth rows: ``frame, id, x, y, w, h, conf, -1, -1, -1`` with
**1-indexed frames** (the classic off-by-one source). ``gameinfo.ini`` maps each
tracklet id to a role string, e.g. ``player team left``, ``goalkeeper team right``,
``referee``, ``side referee``, ``staff``, ``ball``. That mapping is the free
team-assignment ground truth.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import re
from typing import NamedTuple

class Role(NamedTuple):
    cls : str # player, goalkeeper, referee, staff, ball
    side : str | None # left/right for players and goalkeepers, None for others
    tag : str # jersey number or letter after the ';'

TRACKLET_RE = re.compile(r"^trackletID_(\d+)\s*=\s*(.+)$")

def list_clips(data_root: Path, split: str) -> list[str]:
    """Clip ids available under ``data_root`` for a split."""
    raise NotImplementedError


def load_gt(clip_dir: Path) -> pd.DataFrame:
    """MOT gt.txt as a frame: frame (1-indexed), track_id, x1, y1, x2, y2, conf."""
    clip_dir = Path(clip_dir)
    gt_path = clip_dir / "gt" / "gt.txt"
    df = pd.read_csv(gt_path, header=None,
        names=["frame", "track_id", "x1", "y1", "w", "h", "conf"], usecols=range(7))
    df["frame"] = df["frame"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df["x1"] = df["x1"].astype(float)
    df["y1"] = df["y1"].astype(float)
    df["x2"] = df["x1"] + df["w"]
    df["y2"] = df["y1"] + df["h"]
    df = df[["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]]
    df.sort_values(by=["frame", "track_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_roles(clip_dir: Path) -> dict[int, str]:
    """Parse ``gameinfo.ini`` into {track_id: role string}."""
    raise NotImplementedError


def role_to_team(role: str) -> str | None:
    """'player team left' -> 'left'; non-team roles (referee, staff, ball) -> None."""
    raise NotImplementedError
