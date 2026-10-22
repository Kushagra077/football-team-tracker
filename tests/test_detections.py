"""The shared detections format: GT and cached YOLO boxes must be indistinguishable to a tracker."""

from pathlib import Path

import pandas as pd
import pytest

from ftt.data.detections import (
    CLASS_IDS,
    DET_COLUMNS,
    cached_path,
    detector_name,
    gt_as_detections,
    load_cached_detections,
    validate_detections,
)
from ftt.data.sn_tracking import load_persons_gt

CLIP = Path("data/raw_meta") / "test" / "SNMOT-116"
needs_data = pytest.mark.skipif(not CLIP.exists(), reason="raw_meta not available (gitignored)")


def good() -> pd.DataFrame:
    return pd.DataFrame(
        [(2, 10, 10, 50, 90, 0.4, 2), (1, 100, 20, 140, 110, 0.9, 3)], columns=DET_COLUMNS
    )


def test_class_ids_follow_a1():
    assert (CLASS_IDS["goalkeeper"], CLASS_IDS["player"], CLASS_IDS["referee"]) == (1, 2, 3)


def test_validate_sorts_by_frame_and_keeps_columns():
    out = validate_detections(good())
    assert list(out.columns) == DET_COLUMNS
    assert out["frame"].tolist() == [1, 2]
    assert out.index.tolist() == [0, 1]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.drop(columns=["conf"]),
        lambda d: d.assign(frame=[0, 1]),  # 0-indexed frames
        lambda d: d.assign(conf=[0.5, 1.5]),
        lambda d: d.assign(cls=[0, 2]),  # ball is not a person class
        lambda d: d.assign(cls=[2, 4]),  # neither is "other"
        lambda d: d.assign(x1=[None, 10.0]),
    ],
)
def test_validate_rejects_bad_input(mutate):
    with pytest.raises(ValueError):
        validate_detections(mutate(good()))


def test_cached_round_trip(tmp_path):
    path = cached_path(tmp_path, "det_a", "SNMOT-999")
    path.parent.mkdir(parents=True)
    good().to_parquet(path)
    back = load_cached_detections(tmp_path, "det_a", "SNMOT-999")
    assert list(back.columns) == DET_COLUMNS and len(back) == 2


def test_detector_name_is_the_weights_stem():
    assert detector_name(Path("../models/football_detection_v3.pt")) == "football_detection_v3"


def test_another_detector_cannot_see_a_cached_clip(tmp_path):
    """The reason for keying the cache by detector: a new detector must not reuse old boxes."""
    path = cached_path(tmp_path, "football_detection_v3", "SNMOT-999")
    path.parent.mkdir(parents=True)
    good().to_parquet(path)
    assert cached_path(tmp_path, "football_detection_v6", "SNMOT-999") != path
    with pytest.raises(FileNotFoundError):
        load_cached_detections(tmp_path, "football_detection_v6", "SNMOT-999")


def test_missing_cache_fails_loud(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cached_detections(tmp_path, "det_a", "SNMOT-000")


@needs_data
def test_gt_as_detections_snmot116():
    dets = gt_as_detections(CLIP)
    assert list(dets.columns) == DET_COLUMNS
    # Counted by hand from the raw files (see test_sn_tracking): 11460 rows minus 724 ball rows.
    assert len(dets) == 11460 - 724
    assert (dets["conf"] == 1.0).all()
    assert set(dets["cls"]) == {1, 2, 3}
    assert dets["frame"].min() == 1


@needs_data
def test_gt_boxes_match_the_loader():
    dets = gt_as_detections(CLIP)
    gt = load_persons_gt(CLIP)
    assert (
        dets[["x1", "y1", "x2", "y2"]].to_numpy().tolist()
        == gt[["x1", "y1", "x2", "y2"]].to_numpy().tolist()
    )
