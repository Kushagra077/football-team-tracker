import numpy as np

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
