import numpy as np
import pandas as pd
import pytest

from ftt.schema import COLUMNS, FPS, SchemaError, coerce, empty_frame, validate


def _frame(**overrides) -> pd.DataFrame:
    """Two valid rows (frame 1 and 2, track 7), then apply per-column overrides."""
    df = pd.DataFrame(
        {
            "frame_id": [1, 2],
            "timestamp": [1 / FPS, 2 / FPS],
            "track_id": [7, 7],
            "team": [0, 0],
            "is_ball": [False, False],
            "x1": [10.0, 12.0], "y1": [20.0, 22.0], "x2": [50.0, 52.0], "y2": [120.0, 122.0],
            "role": ["player", "player"],
            "conf": [0.9, 0.8],
        }
    )  # fmt: skip
    for k, v in overrides.items():
        df[k] = v
    return coerce(df)


def test_empty_frame_is_valid_and_has_exact_columns():
    df = empty_frame()
    assert list(df.columns) == COLUMNS and len(df) == 0
    validate(df)


def test_valid_frame_passes_and_pitch_is_null_not_zero():
    out = validate(_frame())
    assert out[["jersey_number", "x_pitch", "y_pitch"]].isna().all().all()


@pytest.mark.parametrize("bad", [2, -1, 127])
def test_team_labels_must_be_zero_one_or_null(bad):
    with pytest.raises(SchemaError, match="team"):
        validate(_frame(team=pd.array([0, bad], dtype="Int8")))
    validate(_frame(team=pd.array([0, 1], dtype="Int8")))  # both clusters are fine


def test_team_may_be_null_for_referees():
    validate(_frame(team=pd.array([None, None], dtype="Int8"), role=["referee", "referee"]))


@pytest.mark.parametrize("col", ["x_pitch", "y_pitch"])
def test_zero_pitch_coordinate_is_rejected(col):
    with pytest.raises(SchemaError, match="null"):
        validate(_frame(**{col: [0.0, 0.0]}))


def test_jersey_number_must_stay_null():
    with pytest.raises(SchemaError, match="jersey_number"):
        validate(_frame(jersey_number=pd.array([10, 10], dtype="Int16")))


def test_missing_and_extra_columns_are_rejected():
    with pytest.raises(SchemaError, match="missing"):
        validate(_frame().drop(columns=["conf"]))
    with pytest.raises(SchemaError, match="extra"):
        validate(_frame().assign(surprise=1))


def test_wrong_dtype_is_rejected():
    bad = _frame()
    bad["track_id"] = bad["track_id"].astype("float64")
    with pytest.raises(SchemaError, match="track_id"):
        validate(bad)


def test_duplicate_frame_track_rows_are_rejected():
    with pytest.raises(SchemaError, match="duplicate"):
        validate(_frame(frame_id=[1, 1], timestamp=[1 / FPS, 1 / FPS]))


def test_timestamp_must_be_frame_over_fps():
    with pytest.raises(SchemaError, match="timestamp"):
        validate(_frame(timestamp=[1.0, 2.0]))


def test_ball_rows_are_rejected():
    with pytest.raises(SchemaError, match="ball"):
        validate(_frame(is_ball=[True, False]))


def test_null_in_a_required_column_is_rejected():
    with pytest.raises(SchemaError, match="conf"):
        validate(_frame().assign(conf=np.array([0.9, np.nan], dtype="float32")))


def test_degenerate_box_and_bad_conf_are_rejected():
    with pytest.raises(SchemaError, match="boxes"):
        validate(_frame(x2=[10.0, 52.0]))
    with pytest.raises(SchemaError, match="conf"):
        validate(_frame(conf=[1.5, 0.8]))
