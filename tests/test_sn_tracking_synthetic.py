"""The SoccerNet-Tracking loader on a tiny made-up clip, so it is tested in CI too.

``test_sn_tracking.py`` checks the real files but skips without the (gitignored) data. The files
here follow the same layout: ``gt/gt.txt`` in MOT format and ``gameinfo.ini`` with one role line
per tracklet. Expected values are written out by hand.
"""

import pandas as pd
import pytest

from ftt.data.sn_tracking import Role, list_clips, load_gt, load_persons_gt, load_roles

GT = """\
2,1,100,200,30,60,1,-1,-1,-1
1,1,90,190,30,60,1,-1,-1,-1
1,2,400,210,28,58,1,-1,-1,-1
1,3,10,10,8,8,1,-1,-1,-1
1,4,300,50,20,50,1,-1,-1,-1
"""

INI = """\
[Sequence]
name=SNMOT-900
num_tracklets=4
trackletID_1= player team left;7
trackletID_2= goalkeepers team right;1
trackletID_3= ball;1
trackletID_4= referee;main
"""


@pytest.fixture
def clip(tmp_path):
    (tmp_path / "gt").mkdir()
    (tmp_path / "gt" / "gt.txt").write_text(GT)
    (tmp_path / "gameinfo.ini").write_text(INI)
    return tmp_path


def test_load_gt_converts_xywh_to_corners_and_sorts(clip):
    gt = load_gt(clip)
    assert list(gt["frame"]) == [1, 1, 1, 1, 2]  # file order was not frame order
    first = gt.iloc[0]
    assert (first["track_id"], first["x1"], first["y1"], first["x2"], first["y2"]) == (
        1,
        90,
        190,
        120,
        250,
    )


def test_load_roles_handles_the_goalkeepers_spelling(clip):
    roles = load_roles(clip)
    assert roles[1] == Role("player", "left", "7")
    assert roles[2] == Role("goalkeeper", "right", "1")
    assert roles[3] == Role("ball", None, "1")
    assert roles[4] == Role("referee", None, "main")


def test_load_persons_gt_drops_the_ball_and_attaches_team(clip):
    persons = load_persons_gt(clip)
    assert len(persons) == 4  # 5 rows minus the ball's one row
    assert 3 not in set(persons["track_id"])
    by_track = persons.drop_duplicates("track_id").set_index("track_id")
    assert by_track.loc[1, "team"] == "left"
    assert by_track.loc[2, "cls"] == "goalkeeper" and by_track.loc[2, "team"] == "right"
    assert pd.isna(by_track.loc[4, "team"])  # a referee has no team


def test_unknown_role_is_an_error(clip):
    (clip / "gameinfo.ini").write_text(INI.replace("referee;main", "mascot;main"))
    with pytest.raises(ValueError, match="unknown role"):
        load_roles(clip)


def test_list_clips_val_is_every_fifth_train_clip(tmp_path):
    for i in range(1, 11):
        (tmp_path / "train" / f"SNMOT-{i:03d}").mkdir(parents=True)
    (tmp_path / "test" / "SNMOT-500").mkdir(parents=True)
    assert list_clips(tmp_path, "val") == ["SNMOT-005", "SNMOT-010"]
    assert list_clips(tmp_path, "train") == [f"SNMOT-{i:03d}" for i in (1, 2, 3, 4, 6, 7, 8, 9)]
    assert list_clips(tmp_path, "test") == ["SNMOT-500"]
    with pytest.raises(ValueError):
        list_clips(tmp_path, "dev")
