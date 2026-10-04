"""Tests for the SoccerNet-Tracking loader, against one real clip (SNMOT-116).

Expected values below were read off the raw files by hand, not produced by the code under test.
"""

from pathlib import Path

import pytest

from ftt.data.sn_tracking import load_gt

CLIP = Path("data/raw_meta/test/SNMOT-116")

# data/raw_meta is gitignored, so CI will not have it: skip instead of failing.
pytestmark = pytest.mark.skipif(not CLIP.exists(), reason="raw_meta not available (gitignored)")


@pytest.fixture(scope="module")
def gt():
    return load_gt(CLIP)


def test_load_gt_shape_and_frames(gt):
    assert gt.shape == (11460, 7)
    assert list(gt.columns) == ["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]
    assert gt["frame"].min() == 1  # MOT frames are 1-indexed
    assert gt["frame"].max() == 750
    assert gt["track_id"].nunique() == 27


def test_load_gt_first_row(gt):
    # Raw line 1 of gt.txt is "1,1,136,520,51,135,1,-1,-1,-1": x, y, w, h = 136, 520, 51, 135.
    first = gt.iloc[0]
    assert first["frame"] == 1
    assert first["track_id"] == 1
    assert (first["x1"], first["y1"], first["x2"], first["y2"]) == (136, 520, 187, 655)


def test_load_gt_boxes_valid(gt):
    assert (gt["x2"] > gt["x1"]).all()
    assert (gt["y2"] > gt["y1"]).all()


def test_load_gt_sorted_by_frame_then_track(gt):
    # Trackers consume frames in order, so the loader must not depend on file order.
    assert gt.equals(gt.sort_values(["frame", "track_id"]).reset_index(drop=True))
