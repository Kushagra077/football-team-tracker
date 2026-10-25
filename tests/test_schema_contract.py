"""tracks.parquet follows A2's own schema contract.

DD-003: A2 is standalone for now. The B1 contract (``ftm.schema.validate()``) is revisited at
integration time, after A3 supplies pitch coordinates, so nothing here imports B1.
The file is read back with plain pandas/pyarrow too, so the contract is checked on what is
really stored, not only through our own reader.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ftt.data.detections import CLASS_IDS
from ftt.export import build_tracks, export_tracks
from ftt.schema import COLUMNS, DTYPES, FPS


def _export(tmp_path):
    tracked = pd.DataFrame(
        {
            "frame": [1, 1, 2, 2],
            "track_id": [1, 2, 1, 2],
            "x1": [10.0, 200.0, 12.0, 202.0], "y1": [20.0, 30.0, 22.0, 32.0],
            "x2": [50.0, 240.0, 52.0, 242.0], "y2": [120.0, 130.0, 122.0, 132.0],
            "conf": [0.9, 0.8, 0.9, 0.8],
            "cls": [CLASS_IDS["player"], CLASS_IDS["referee"]] * 2,
        }
    )  # fmt: skip
    return export_tracks(build_tracks(tracked, team=[0, np.nan, 0, np.nan]), tmp_path / "t.parquet")


def test_tracks_parquet_matches_a2_schema(tmp_path):
    path = _export(tmp_path)
    df = pd.read_parquet(path)
    assert list(df.columns) == COLUMNS
    for col, dtype in DTYPES.items():
        assert str(df[col].dtype) == dtype, col
    assert not df.duplicated(["frame_id", "track_id"]).any()
    assert np.allclose(df["timestamp"], df["frame_id"] / FPS)


def test_pitch_and_jersey_columns_are_null_not_zero(tmp_path):
    path = _export(tmp_path)
    table = pq.read_table(path)  # stored values, read without our own reader
    for col in ("x_pitch", "y_pitch", "jersey_number"):
        values = table[col].to_pandas()
        assert values.isna().all(), f"{col} must be all null, found {values.dropna().tolist()}"
