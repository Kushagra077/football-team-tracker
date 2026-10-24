import numpy as np
import pandas as pd
import pytest

from ftt.data.detections import CLASS_IDS
from ftt.export import build_tracks, export_tracks, read_tracks
from ftt.schema import FPS, SchemaError, validate


def _tracked() -> pd.DataFrame:
    """Frame 1: a player, a goalkeeper, a referee. Frame 2: the same player again."""
    c = CLASS_IDS
    return pd.DataFrame(
        {
            "frame": [1, 1, 1, 2],
            "track_id": [3, 4, 5, 3],
            "x1": [10.0, 200.0, 400.0, 12.0], "y1": [20.0, 30.0, 40.0, 22.0],
            "x2": [50.0, 240.0, 440.0, 52.0], "y2": [120.0, 130.0, 140.0, 122.0],
            "conf": [0.9, 0.8, 0.7, 0.9],
            "cls": [c["player"], c["goalkeeper"], c["referee"], c["player"]],
        }
    )  # fmt: skip


def test_build_tracks_maps_roles_time_and_nulls():
    out = build_tracks(_tracked(), team=[0, np.nan, 1, 0])
    validate(out)
    assert out["role"].astype(str).tolist() == ["player", "goalkeeper", "referee", "player"]
    assert np.allclose(out["timestamp"], out["frame_id"] / FPS)
    assert out[["jersey_number", "x_pitch", "y_pitch"]].isna().all().all()
    assert not out["is_ball"].any()


def test_referees_never_get_a_team_even_if_one_is_passed():
    out = build_tracks(_tracked(), team=[0, 1, 1, 0])
    assert out.loc[out["role"] == "referee", "team"].isna().all()
    assert out.loc[out["role"] == "player", "team"].tolist() == [0, 0]


def test_undecided_goalkeeper_stays_null():
    out = build_tracks(_tracked(), team=[0, np.nan, np.nan, 0])
    assert out.loc[out["role"] == "goalkeeper", "team"].isna().all()


def test_unknown_class_is_rejected():
    bad = _tracked().assign(cls=[CLASS_IDS["ball"], 2, 3, 2])
    with pytest.raises(ValueError, match="outside"):
        build_tracks(bad, team=[0, 0, 0, 0])


def test_parquet_round_trip_keeps_dtypes_and_nulls(tmp_path):
    built = build_tracks(_tracked(), team=[0, np.nan, np.nan, 0])
    path = export_tracks(built, tmp_path / "sub" / "tracks.parquet")
    back = read_tracks(path)  # re-validates, so dtype drift would raise here
    pd.testing.assert_frame_equal(built, back)


def test_export_refuses_an_invalid_table(tmp_path):
    bad = build_tracks(_tracked(), team=[0, 0, 0, 0]).assign(x_pitch=0.0)  # a zero, not a null
    with pytest.raises(SchemaError):
        export_tracks(bad, tmp_path / "tracks.parquet")
    assert not (tmp_path / "tracks.parquet").exists()
