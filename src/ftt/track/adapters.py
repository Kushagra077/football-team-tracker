"""Wrappers that make ByteTrack / BoT-SORT satisfy ``base.Tracker``. Built from configs/*.yaml."""

from __future__ import annotations

from ftt.track.base import Tracker


def build_tracker(config_path: str) -> Tracker:
    """Factory: read a tracker config and return the matching adapter."""
    raise NotImplementedError
