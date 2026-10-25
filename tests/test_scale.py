import math

from ftt.scale import clean_fps, scale_team_cfg, scale_tracker_cfg

CFG = {
    "cluster": {"fit_frames": 50, "fit_min_players": 3},
    "vote": {"window": 25, "lock_min_agree": None},
    "goalkeeper": {"min_team_gap": 150, "min_margin": 60},
    "pitch_view": {"max_box_height": 0.2},
}


def test_reference_video_keeps_the_config_exactly():
    assert scale_team_cfg(CFG, 25.0, 1920) == CFG
    assert scale_tracker_cfg({"track_buffer": 30, "match_thresh": 0.9}, 25.0) == {
        "track_buffer": 30,
        "match_thresh": 0.9,
    }


def test_frame_counts_follow_fps_and_pixels_follow_width():
    out = scale_team_cfg(CFG, 50.0, 1280)
    assert out["vote"]["window"] == 50 and out["cluster"]["fit_frames"] == 100
    assert math.isclose(out["goalkeeper"]["min_team_gap"], 100.0)
    assert math.isclose(out["goalkeeper"]["min_margin"], 40.0)
    assert out["cluster"]["fit_min_players"] == 3  # a count of players, not of time
    assert out["pitch_view"] == CFG["pitch_view"]  # already a fraction of the frame
    assert scale_tracker_cfg({"track_buffer": 30}, 12.5)["track_buffer"] == 15


def test_input_is_not_modified_and_a_lock_scales_with_the_window():
    cfg = {**CFG, "vote": {"window": 25, "lock_min_agree": 15}}
    out = scale_team_cfg(cfg, 50.0, 1920)
    assert cfg["vote"] == {"window": 25, "lock_min_agree": 15}
    assert out["vote"] == {"window": 50, "lock_min_agree": 30}


def test_missing_or_zero_fps_falls_back_to_25():
    for bad in (0, -1, None, float("nan"), float("inf")):
        assert clean_fps(bad) == 25.0
        assert scale_team_cfg(CFG, bad, 1920)["vote"]["window"] == 25


def test_counts_never_drop_below_one():
    assert scale_team_cfg(CFG, 0.5, 1920)["vote"]["window"] == 1
