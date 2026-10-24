import numpy as np

from ftt.eval.team_metrics import best_permutation_accuracy, per_track_accuracy


def test_accuracy_ignores_cluster_naming():
    true = np.array([0, 0, 1, 1])
    assert best_permutation_accuracy(np.array([0, 0, 1, 1]), true) == 1.0
    assert best_permutation_accuracy(np.array([1, 1, 0, 0]), true) == 1.0  # names swapped


def test_accuracy_floor_is_half():
    # Random labels cannot score below 50% once both namings are tried.
    pred = np.array([0, 1, 0, 1])
    assert best_permutation_accuracy(pred, np.array([0, 0, 1, 1])) == 0.5


def test_per_track_uses_majority_and_one_global_naming():
    true = np.array([0, 0, 0, 1, 1, 1])
    tid = np.array([7, 7, 7, 9, 9, 9])
    pred = np.array([1, 1, 0, 0, 0, 0])  # naming swapped; track 7 has 2/3 right after the swap
    assert per_track_accuracy(pred, true, tid) == 1.0


def test_per_track_counts_a_wrong_track():
    true = np.array([0, 0, 0, 0, 1, 1])
    tid = np.array([1, 1, 2, 2, 3, 3])
    pred = np.array([0, 0, 1, 1, 1, 1])  # track 2 is wrong in every crop
    assert abs(per_track_accuracy(pred, true, tid) - 2 / 3) < 1e-9
