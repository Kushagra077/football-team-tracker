"""Write tracks / GT in MOT format for TrackEval. Frames are 1-INDEXED. Persons only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_mot(tracks: pd.DataFrame, out_path: Path) -> None:
    raise NotImplementedError
