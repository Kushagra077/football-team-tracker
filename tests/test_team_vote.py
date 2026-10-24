import numpy as np
import pytest

from ftt.team.vote import vote_labels


def _one_track(labels):
    n = len(labels)
    return np.zeros(n, int), np.arange(1, n + 1), np.array(labels)


def test_isolated_flip_is_outvoted():
    ids, fr, lab = _one_track([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    out = vote_labels(ids, fr, lab, window=25, lock_min_agree=None)
    assert out.tolist() == [0] * 10


def test_tie_keeps_previous_label():
    ids, fr, lab = _one_track([0, 0, 1, 1])  # frame 4: window has two 0s and two 1s
    out = vote_labels(ids, fr, lab, window=25, lock_min_agree=None)
    assert out.tolist() == [0, 0, 0, 0]


def test_lock_ignores_later_contrary_labels():
    ids, fr, lab = _one_track([0] * 5 + [1] * 20)
    out = vote_labels(ids, fr, lab, window=25, lock_min_agree=5)
    assert out.tolist() == [0] * 25  # locked to 0 after five agreeing frames


def test_old_frames_leave_the_window():
    # Without a lock, a long run of 1s must eventually take over from earlier 0s.
    ids, fr, lab = _one_track([0] * 5 + [1] * 40)
    out = vote_labels(ids, fr, lab, window=10, lock_min_agree=None)
    assert out[-1] == 1


def test_no_lock_means_a_full_window_of_agreement_still_does_not_freeze():
    # 25 agreeing labels with window 25 would trigger lock_min_agree=25; None must not.
    ids, fr, lab = _one_track([0] * 30 + [1] * 40)
    assert vote_labels(ids, fr, lab, window=25, lock_min_agree=None)[-1] == 1
    assert vote_labels(ids, fr, lab, window=25, lock_min_agree=25)[-1] == 0


def test_tracks_are_independent_and_output_follows_input_order():
    ids = np.array([1, 2, 1, 2, 1, 2])
    fr = np.array([3, 3, 2, 2, 1, 1])  # deliberately not sorted by frame
    lab = np.array([0, 1, 0, 1, 0, 1])
    out = vote_labels(ids, fr, lab, window=25, lock_min_agree=None)
    assert out.tolist() == [0, 1, 0, 1, 0, 1]


def test_frame_gaps_count_as_time_not_as_rows():
    # Two labels 100 frames apart: the first has left the window, so the newest wins alone.
    ids = np.array([0, 0])
    out = vote_labels(ids, np.array([1, 101]), np.array([0, 1]), window=25, lock_min_agree=None)
    assert out.tolist() == [0, 1]


def test_lock_larger_than_window_is_rejected():
    with pytest.raises(ValueError):
        vote_labels(np.array([0]), np.array([1]), np.array([0]), window=10, lock_min_agree=15)
