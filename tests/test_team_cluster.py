import numpy as np
import pytest

from ftt.team.cluster import (
    TooFewColoursError,
    choose_fit_frames,
    far_from_both_teams,
    fit_team_clusters,
)


def test_two_separated_blobs_are_split_cleanly():
    rng = np.random.default_rng(0)
    a = rng.normal([80, 0, -5], 2, size=(30, 3))
    b = rng.normal([65, -18, 20], 2, size=(30, 3))
    labels = fit_team_clusters(np.vstack([a, b])).labels_
    # Ids are arbitrary, so check grouping, not the numbers.
    assert len(set(labels[:30])) == 1 and len(set(labels[30:])) == 1
    assert labels[0] != labels[30]


def test_fit_is_repeatable():
    x = np.random.default_rng(1).normal(size=(60, 3))
    assert (fit_team_clusters(x).labels_ == fit_team_clusters(x).labels_).all()


def test_fit_frames_skip_frames_without_a_pitch_view():
    # Frames 1-3 show one player (a close-up); frames 4+ show five. Fit on the pitch frames.
    box_frames = np.array([1, 2, 3] + [f for f in range(4, 10) for _ in range(5)])
    out = choose_fit_frames(box_frames, fit_n=4, min_players=3)
    assert out.tolist() == [4, 5, 6, 7]


def test_fit_frames_match_the_first_frames_on_a_normal_clip():
    box_frames = np.repeat(np.arange(1, 101), 6)
    assert choose_fit_frames(box_frames, 50, 3).tolist() == list(range(1, 51))


def test_far_from_both_teams_flags_a_third_colour():
    centres = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])  # gap 10
    feats = np.array([[1.0, 0, 0], [9.0, 0, 0], [5.0, 0, 0], [5.0, 9.0, 0]])
    # nearest distances: 1, 1, 5, ~10.3 -> ratios 0.1, 0.1, 0.5, 1.03
    out = far_from_both_teams(feats, centres, max_ratio=0.7)
    assert out.tolist() == [False, False, False, True]


@pytest.mark.parametrize("n", [0, 1])
def test_fewer_colours_than_teams_is_a_clear_error_not_a_sklearn_one(n):
    with pytest.raises(TooFewColoursError, match="need at least 2"):
        fit_team_clusters(np.zeros((n, 3)))


def test_two_colours_are_enough_to_fit():
    assert len(fit_team_clusters(np.array([[0.0, 0, 0], [50, 10, 10]])).cluster_centers_) == 2
