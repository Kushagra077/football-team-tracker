import pandas as pd

from ftt.team.gate import pitch_view_frames


def _rows(frame, n, height):
    return [(frame, 100.0, 100.0 + height, 2) for _ in range(n)]


def test_only_frames_with_enough_small_players_pass():
    rows = _rows(1, 5, 100) + _rows(2, 2, 100) + _rows(3, 5, 400) + _rows(4, 4, 150)
    t = pd.DataFrame(rows, columns=["frame", "y1", "y2", "cls"])
    out = pitch_view_frames(t, frame_height=1000, min_players=3, max_box_height=0.20)
    # frame 2: too few players; frame 3: median 40% of the frame (a close-up)
    assert out.tolist() == [1, 4]


def test_non_players_do_not_count():
    t = pd.DataFrame([(1, 0.0, 50.0, 3)] * 5, columns=["frame", "y1", "y2", "cls"])
    assert len(pitch_view_frames(t, 1000, 3, 0.2)) == 0
