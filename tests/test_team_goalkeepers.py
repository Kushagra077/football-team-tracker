import numpy as np
import pandas as pd

from ftt.team.assign import label_goalkeepers
from ftt.team.cluster import assign_goalkeepers

KW = {"min_team_gap": 150.0, "min_margin": 60.0}


def test_keeper_goes_to_nearest_team():
    teams = np.array([600.0, 1400.0])
    out = assign_goalkeepers(np.array([300.0, 1700.0]), teams, **KW)
    assert out.tolist() == [0, 1]


def test_keeper_midway_is_unknown():
    out = assign_goalkeepers(np.array([1000.0]), np.array([600.0, 1400.0]), **KW)
    assert out.tolist() == [-1]


def test_bunched_teams_are_all_unknown():
    out = assign_goalkeepers(np.array([300.0, 1700.0]), np.array([900.0, 1000.0]), **KW)
    assert out.tolist() == [-1, -1]


def test_works_when_team_order_is_swapped():
    out = assign_goalkeepers(np.array([300.0]), np.array([1400.0, 600.0]), **KW)
    assert out.tolist() == [1]


_CFG = {"goalkeeper": {"min_team_gap": 150.0, "min_margin": 60.0}}


def _players(x_by_team):
    rows = [(t, x - 20, x + 20) for t, xs in x_by_team.items() for x in xs]
    return pd.DataFrame(rows, columns=["team", "x1", "x2"])


def test_label_goalkeepers_decides_per_track_from_median_x():
    players = _players({0: [550, 600, 650], 1: [1350, 1400, 1450]})
    # Track 5 mostly stands at x=300 (one outlier frame); track 6 at x=1700.
    keepers = pd.DataFrame(
        {
            "track_id": [5, 5, 5, 6, 6],
            "x1": [280, 280, 1500, 1680, 1680],
            "x2": [320, 320, 1540, 1720, 1720],
        }
    )
    out = label_goalkeepers(keepers, players, _CFG).set_index("track_id")["team"]
    assert out.to_dict() == {5: 0, 6: 1}


def test_label_goalkeepers_is_null_when_teams_overlap_or_no_keepers():
    keepers = pd.DataFrame({"track_id": [5], "x1": [280], "x2": [320]})
    bunched = _players({0: [900, 950], 1: [1000, 1050]})
    assert label_goalkeepers(keepers, bunched, _CFG)["team"].isna().all()
    assert len(label_goalkeepers(keepers.iloc[0:0], bunched, _CFG)) == 0
