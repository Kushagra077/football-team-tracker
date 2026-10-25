import numpy as np
import pandas as pd
import pytest

from ftt.team.assign import label_from_features
from ftt.team.features import FRAME_COLS, extra_names, histogram_feature, lab_pixels, mean_feature

RED, BLUE, YELLOW = (0, 0, 220), (220, 0, 0), (0, 255, 255)  # BGR; yellow fits neither team


def _feature_row(colour, kind):
    px = lab_pixels(np.full((4, 4, 3), colour, np.uint8), np.ones((4, 4), bool))
    row = list(mean_feature(px))
    return row + list(histogram_feature(px)) if kind == "hist_lab" else row


@pytest.mark.parametrize("kind", ["mean_lab", "hist_lab"])
def test_a_colour_far_from_both_teams_gets_no_team_whatever_the_clustering_feature(kind):
    cfg = {
        "cluster": {"fit_frames": 3, "fit_min_players": 3},
        "vote": {"window": 5, "lock_min_agree": None},
        "features": {"kind": kind},
    }
    colours = [RED] * 10 + [BLUE] * 10 + [YELLOW]  # track i has colour i in every frame
    rows = [(fr, 0.0, 0.0, 1.0, 1.0, *_feature_row(c, kind)) for fr in (1, 2, 3) for c in colours]
    feats = pd.DataFrame(rows, columns=[*FRAME_COLS, *extra_names(cfg)])
    tracks = pd.DataFrame({"frame": feats["frame"], "track_id": list(range(len(colours))) * 3})
    out = label_from_features(tracks, feats, cfg, max_colour_ratio=0.5)
    team = out.groupby("track_id")["team"].agg(lambda s: set(s))
    assert team[20] == {-1}  # the yellow one
    assert team[0] == team[9] and team[10] == team[19] and team[0] != team[10]
