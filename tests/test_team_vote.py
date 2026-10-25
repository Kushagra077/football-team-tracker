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


def test_default_never_locks_like_the_config():
    ids = np.zeros(40, int)
    fr = np.arange(1, 41)
    lab = np.array([0] * 20 + [1] * 20)
    assert vote_labels(ids, fr, lab)[-1] == 1  # a lock after 15 agreeing frames would keep it at 0


def test_centred_vote_fixes_the_frames_after_a_jump_that_a_causal_vote_gets_wrong():
    # One track whose true team switches at frame 31 (an id jump): 30 frames of 0, then 30 of 1.
    ids, fr = np.zeros(60, int), np.arange(1, 61)
    lab = np.array([0] * 30 + [1] * 30)
    causal = vote_labels(ids, fr, lab, window=25)
    centred = vote_labels(ids, fr, lab, window=25, centred=True)
    assert (causal != lab).sum() >= 10  # the causal vote stays wrong for about half a window
    assert (centred != lab).sum() <= 2  # the centred one switches within a frame or two of the jump


def test_centred_vote_removes_a_single_stray_label_and_respects_gaps_in_frames():
    ids = np.zeros(7, int)
    fr = np.array([1, 2, 3, 4, 5, 6, 7])
    lab = np.array([0, 0, 0, 1, 0, 0, 0])
    assert vote_labels(ids, fr, lab, window=5, centred=True).tolist() == [0] * 7
    # frames 1 and 100 are far apart: neither sees the other
    far = vote_labels(np.zeros(2, int), np.array([1, 100]), np.array([0, 1]), 25, centred=True)
    assert far.tolist() == [0, 1]


def test_centred_vote_is_aligned_with_input_rows_and_cannot_lock():
    ids = np.array([1, 0, 1, 0])
    fr = np.array([2, 1, 1, 2])  # unsorted, two tracks interleaved
    lab = np.array([1, 0, 1, 0])
    assert vote_labels(ids, fr, lab, window=25, centred=True).tolist() == [1, 0, 1, 0]
    with pytest.raises(ValueError, match="cannot lock"):
        vote_labels(ids, fr, lab, window=25, lock_min_agree=5, centred=True)
