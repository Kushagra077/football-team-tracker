import numpy as np

from ftt.eval.team_metrics import (
    best_permutation_accuracy,
    iou_matrix,
    match_to_gt,
    per_track_accuracy,
)


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


def test_iou_of_identical_and_disjoint_boxes():
    a = np.array([[0.0, 0, 10, 10]])
    b = np.array([[0.0, 0, 10, 10], [20, 20, 30, 30]])
    assert iou_matrix(a, b).tolist() == [[1.0, 0.0]]


def test_match_to_gt_is_one_to_one_and_respects_min_iou():
    gt = np.array([[0.0, 0, 10, 10], [50, 50, 60, 60]])
    dets = np.array([[1.0, 1, 10, 10], [2, 2, 10, 10], [200, 200, 210, 210]])
    # two boxes overlap gt 0 but only the better one may take it; the far box matches nothing
    assert match_to_gt(dets, gt, min_iou=0.5).tolist() == [0, -1, -1]
    assert match_to_gt(dets[:0], gt, min_iou=0.5).tolist() == []
