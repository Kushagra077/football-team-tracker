"""Tests for the SoccerNet-Tracking loader, against one real clip (SNMOT-116).

Expected values below were read off the raw files by hand, not produced by the code under test.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from ftt.data.sn_tracking import (
    Role,
    list_clips,
    load_gt,
    load_persons_gt,
    load_roles,
    role_to_team,
)

RAW_META = Path("data/raw_meta")
CLIP = RAW_META / "test" / "SNMOT-116"

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


def test_list_clips_val_matches_the_detector_val():
    # Expected ids come from the detector's own val image filenames (dataset/images/val in
    # football-detect-serve), not from list_clips: it held out every 5th sorted train clip.
    expected = [
        "SNMOT-064", "SNMOT-069", "SNMOT-074", "SNMOT-098", "SNMOT-103", "SNMOT-108",
        "SNMOT-113", "SNMOT-153", "SNMOT-158", "SNMOT-163", "SNMOT-168",
    ]  # fmt: skip
    assert list_clips(RAW_META, "val") == expected


def test_list_clips_partition():
    train, val, test = (list_clips(RAW_META, s) for s in ("train", "val", "test"))
    assert (len(train), len(val), len(test)) == (46, 11, 49)
    # No clip in two splits, and together they cover all 106 clips.
    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)
    assert len(set(train) | set(val) | set(test)) == 106


def test_load_roles_snmot116():
    # Expected values read by hand from data/raw_meta/test/SNMOT-116/gameinfo.ini.
    roles = load_roles(CLIP)
    assert len(roles) == 27
    assert roles[4] == Role("referee", None, "main")
    assert roles[5] == Role("player", "left", "A")
    assert roles[19] == Role("goalkeeper", "left", "1")  # file spells it "goalkeepers"
    assert roles[20] == Role("ball", None, "1")
    assert roles[25] == Role("referee", None, "side bottom")


def test_role_counts_over_all_clips():
    # Totals come from an independent count of the raw role strings in all 106 gameinfo.ini
    # files (done before load_roles existed), not from load_roles itself.
    counts = Counter()
    for clip in RAW_META.glob("*/SNMOT-*"):
        for role in load_roles(clip).values():
            counts[(role.cls, role.side)] += 1
    assert counts == {
        ("player", "left"): 1062,
        ("player", "right"): 1058,
        ("goalkeeper", "left"): 66,  # the plural "goalkeepers" spelling
        ("goalkeeper", "right"): 76,
        ("referee", None): 227,
        ("ball", None): 139,
        ("other", None): 10,
    }


def test_role_to_team_snmot116():
    # Tracklets read by hand from SNMOT-116's gameinfo.ini.
    roles = load_roles(CLIP)
    assert role_to_team(roles[5]) == "left"  # player team left
    assert role_to_team(roles[3]) == "right"  # player team right
    assert role_to_team(roles[19]) == "left"  # goalkeepers team left: keepers have a team
    assert role_to_team(roles[4]) is None  # referee
    assert role_to_team(roles[25]) is None  # side referee
    assert role_to_team(roles[20]) is None  # ball


def test_role_to_team_only_none_for_non_team_roles():
    # Over all clips: exactly the player/goalkeeper tracklets have a team, nobody else does.
    for clip in RAW_META.glob("*/SNMOT-*"):
        for role in load_roles(clip).values():
            has_team = role_to_team(role) is not None
            assert has_team == (role.cls in ("player", "goalkeeper"))


def test_load_persons_gt_snmot116():
    # Counted from the raw files: 11460 rows, of which 724 belong to tracklet 20 (the ball);
    # 27 tracklets minus the ball leaves 26. SNMOT-116 has no "other" tracklets.
    persons = load_persons_gt(CLIP)
    assert len(persons) == 11460 - 724
    assert persons["track_id"].nunique() == 26
    assert 20 not in set(persons["track_id"])
    assert set(persons["cls"]) == {"player", "goalkeeper", "referee"}


def test_load_persons_gt_attaches_class_and_team():
    persons = load_persons_gt(CLIP)
    by_track = persons.drop_duplicates("track_id").set_index("track_id")
    assert by_track.loc[5, "cls"] == "player" and by_track.loc[5, "team"] == "left"
    assert by_track.loc[3, "team"] == "right"
    assert by_track.loc[19, "cls"] == "goalkeeper" and by_track.loc[19, "team"] == "left"
    # A missing team is stored as NaN in a DataFrame column (not None), so test with isna.
    assert by_track.loc[4, "cls"] == "referee" and pd.isna(by_track.loc[4, "team"])


def test_load_persons_gt_keeps_frame_order():
    persons = load_persons_gt(CLIP)
    keys = persons[["frame", "track_id"]]
    assert keys.equals(keys.sort_values(["frame", "track_id"]).reset_index(drop=True))
