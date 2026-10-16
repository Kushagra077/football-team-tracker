"""Load SoccerNet-Tracking clips: frames, MOT ground truth, ``gameinfo.ini`` roles.

MOT ground-truth rows: ``frame, id, x, y, w, h, conf, -1, -1, -1`` with
**1-indexed frames** (the classic off-by-one source). ``gameinfo.ini`` maps each
tracklet id to a role string, e.g. ``player team left``, ``goalkeeper team right``,
``referee``, ``side referee``, ``other``, ``ball``. That mapping is the free
team-assignment ground truth.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pandas as pd


class Role(NamedTuple):
    cls: str  # player, goalkeeper, referee , ball, other
    side: str | None  # left/right for players and goalkeepers, None for others
    tag: str  # jersey number or letter after the ';'


TRACKLET_RE = re.compile(r"^trackletID_(\d+)\s*=\s*(.+)$")

# Spellings in gameinfo.ini that differ from our class names.
CLS_ALIASES = {"goalkeepers": "goalkeeper"}
VALID_CLS = {"player", "goalkeeper", "referee", "ball", "other"}


def _parse_role(text: str) -> Role:
    """'goalkeepers team left;1' -> Role('goalkeeper', 'left', '1')"""
    role_text, _, tag = text.partition(";")
    words = role_text.strip().lower().split()
    tag = tag.strip()

    first = words[0]
    cls = CLS_ALIASES.get(first, first)
    if cls not in VALID_CLS:
        raise ValueError(f"unknown role in gameinfo.ini: {text!r}")

    side = None
    if cls in ("player", "goalkeeper"):
        side = words[-1]
        if side not in ("left", "right"):
            raise ValueError(f"unexpected side in role text: {text!r}")

    return Role(cls, side, tag)

SPLITS = ("train", "val", "test")

def list_clips(data_root: Path, split: str) -> list[str]:
    """Clip ids for a split, sorted. 'val' = every 5th sorted train clip (A1's rule)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    folder = "test" if split == "test" else "train"
    clips = sorted(
        p.name for p in (Path(data_root) / folder).iterdir()
        if p.is_dir() and p.name.startswith("SNMOT-")
    )

    if split == "test":
        return clips

    val = clips[4::5]
    if split == "val":
        return val
    return [c for c in clips if c not in val]


def load_gt(clip_dir: Path) -> pd.DataFrame:
    """MOT gt.txt as a frame: frame (1-indexed), track_id, x1, y1, x2, y2, conf."""
    clip_dir = Path(clip_dir)
    gt_path = clip_dir / "gt" / "gt.txt"
    df = pd.read_csv(
        gt_path,
        header=None,
        names=["frame", "track_id", "x1", "y1", "w", "h", "conf"],
        usecols=range(7),
    )
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


def load_roles(clip_dir: Path) -> dict[int, Role]:
    """Parse ``gameinfo.ini`` into {track_id: Role}."""
    roles = {}
    for line in (Path(clip_dir) / "gameinfo.ini").read_text().splitlines():
        m = TRACKLET_RE.match(line.strip())
        if m:
            roles[int(m.group(1))] = _parse_role(m.group(2))
    return roles


def role_to_team(role: Role) -> str | None:
    """Ground-truth team of a tracklet: 'left' or 'right'; None for referee, ball, other.

    Players and goalkeepers carry a side (see ``_parse_role``); everyone else has None.
    """
    return role.side
