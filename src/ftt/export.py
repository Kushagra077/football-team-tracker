"""Write tracks to ``tracks.parquet`` in the A2 schema (see ``schema.py``)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_tracks(tracks: pd.DataFrame, out_path: Path) -> Path:
    """Validate against ``schema.validate`` then write parquet. Pitch/jersey columns stay null."""
    raise NotImplementedError
