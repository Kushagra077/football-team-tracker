import numpy as np

from ftt.team.cluster import choose_fit_frames, fit_team_clusters


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
