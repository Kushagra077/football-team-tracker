"""Thin wrapper over TrackEval (use the SoccerNet-Tracking devkit's bundled, configured copy).

Returns HOTA, DetA, AssA, IDF1, ID switches. Sanity checks that must pass before any
tracker is trusted: GT-as-prediction -> HOTA 100 / IDSW 0; GT with IDs shuffled
every 50 frames -> HOTA drops and IDSW spikes.
"""

from __future__ import annotations

from pathlib import Path


def evaluate(gt_dir: Path, pred_dir: Path) -> dict[str, float]:
    raise NotImplementedError
