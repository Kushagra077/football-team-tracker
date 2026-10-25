"""Make frame- and pixel-counted settings mean the same thing on any video (upload path only).

The settings were chosen on SoccerNet: 25 fps, 1920 px wide. "25 frames" is 1 s there but 0.4 s at
60 fps, and "150 px" is 8% of the width there but 12% of a 1280 px frame. Everything is scaled
from that reference, so a 25 fps, 1920 px video gets exactly the settings in the configs and the
SoccerNet evaluation scripts (which never call this) are untouched.
"""

from __future__ import annotations

import copy
import math

REF_FPS = 25.0
REF_WIDTH = 1920.0


def clean_fps(fps: float | None) -> float:
    """OpenCV reports 0 or nan for some files; fall back to the reference rate."""
    if fps is None or not math.isfinite(fps) or fps <= 0:
        return REF_FPS
    return float(fps)


def _frames(n: int, fps: float) -> int:
    return max(1, round(n * fps / REF_FPS))


def scale_team_cfg(cfg: dict, fps: float | None, width: float) -> dict:
    """A copy of the team config with frame counts scaled by fps and pixel gaps by frame width."""
    fps, px = clean_fps(fps), width / REF_WIDTH
    out = copy.deepcopy(cfg)
    out["cluster"]["fit_frames"] = _frames(cfg["cluster"]["fit_frames"], fps)
    out["vote"]["window"] = _frames(cfg["vote"]["window"], fps)
    if cfg["vote"]["lock_min_agree"] is not None:
        out["vote"]["lock_min_agree"] = _frames(cfg["vote"]["lock_min_agree"], fps)
    for key in ("min_team_gap", "min_margin"):
        out["goalkeeper"][key] = cfg["goalkeeper"][key] * px
    return out


def scale_tracker_cfg(tcfg: dict, fps: float | None) -> dict:
    """A copy of a tracker config whose memory (``track_buffer``, in frames) spans the same time."""
    out = dict(tcfg)
    if "track_buffer" in out:
        out["track_buffer"] = _frames(out["track_buffer"], clean_fps(fps))
    return out
