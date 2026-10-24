import numpy as np

from ftt.team.cluster import fit_team_clusters


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
