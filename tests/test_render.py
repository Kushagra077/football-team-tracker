import cv2
import numpy as np
import pandas as pd

from ftt.render import render_video


def _table():
    return pd.DataFrame(
        {
            "frame_id": [1, 1, 3],
            "track_id": [1, 2, 1],
            "team": pd.array([0, pd.NA, 1], dtype="Int8"),
            "x1": [10.0, 60.0, 12.0], "y1": [10.0, 10.0, 12.0],
            "x2": [40.0, 90.0, 42.0], "y2": [80.0, 80.0, 82.0],
            "role": ["player", "referee", "player"],
        }
    )  # fmt: skip


def test_render_writes_every_frame_and_draws_on_annotated_ones(tmp_path):
    src = tmp_path / "in.mp4"
    vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 25, (128, 128))
    for _ in range(4):
        vw.write(np.zeros((128, 128, 3), np.uint8))
    vw.release()
    out = render_video(src, _table(), tmp_path / "out.mp4", fps=25)
    cap = cv2.VideoCapture(str(out))
    frames = [cap.read()[1] for _ in range(4)]
    assert cap.read()[0] is False  # exactly 4 frames
    assert frames[0].max() > 100  # frame 1 has drawings
    assert frames[1].max() < 30  # frame 2 has none (lossy codec: allow tiny noise)
