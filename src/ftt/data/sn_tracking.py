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


def list_clips(data_root: Path, split: str) -> list[str]:
    """Clip ids available under ``data_root`` for a split."""
    raise NotImplementedError


def load_gt(clip_dir: Path) -> pd.DataFrame:
    """MOT gt.txt as a frame: frame (1-indexed), track_id, x1, y1, x2, y2, conf."""
    raise NotImplementedError


def load_roles(clip_dir: Path) -> dict[int, str]:
    """Parse ``gameinfo.ini`` into {track_id: role string}."""
    raise NotImplementedError


def role_to_team(role: str) -> str | None:
    """'player team left' -> 'left'; non-team roles (referee, staff, ball) -> None."""
    raise NotImplementedError
