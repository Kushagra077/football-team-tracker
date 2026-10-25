from types import SimpleNamespace

import cv2
import numpy as np

from ftt.data.detections import CLASS_IDS
from ftt.video import track_video

CFG = {
    "crop": {"y_top": 0.15, "y_bottom": 0.50, "x_trim": 0.20},
    "grass_mask": {"h": [34, 42], "s_min": 40, "v_min": 40},
}


class _T:  # just enough of a torch tensor for the code under test
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def int(self):
        return _T(self.a.astype(int))


def _result(img, boxes, ids, cls):
    if ids is None:
        return SimpleNamespace(orig_img=img, boxes=SimpleNamespace(id=None))
    b = SimpleNamespace(xyxy=_T(boxes), id=_T(ids), conf=_T([0.9] * len(ids)), cls=_T(cls))
    return SimpleNamespace(orig_img=img, boxes=b)


class _FakeModel:
    def __init__(self, results):
        self.results, self.kwargs = results, None

    def track(self, **kwargs):
        self.kwargs = kwargs
        return iter(self.results)


def _tiny_video(path, n=3):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 64))
    for _ in range(n):
        vw.write(np.zeros((64, 64, 3), np.uint8))
    vw.release()


def test_track_video_keeps_boxes_and_player_colours_only(tmp_path):
    video = tmp_path / "v.mp4"
    _tiny_video(video)
    white = np.full((200, 200, 3), 255, np.uint8)
    c = CLASS_IDS
    player_box, ref_box = [10.0, 20.0, 90.0, 120.0], [100.0, 20.0, 180.0, 120.0]
    model = _FakeModel(
        [
            _result(white, [player_box, ref_box], [3, 5], [c["player"], c["referee"]]),
            _result(white, None, None, None),  # nothing tracked in frame 2
            _result(white, [player_box], [3], [c["player"]]),
        ]
    )
    out = track_video(model, video, tmp_path / "t.yaml", CFG, device="cpu")
    assert out.tracks["frame"].tolist() == [1, 1, 3]
    assert out.tracks["track_id"].tolist() == [3, 5, 3]
    # colours only for PLAYER rows, indexed by their row in `tracks` (rows 0 and 2)
    assert out.feats.index.tolist() == [0, 2]
    assert (out.feats["L"] > 99).all()
    assert out.n_frames == 3 and out.size == (64, 64)
    # the settings that matter for parity with our cached-box pipeline are passed through
    assert model.kwargs["conf"] == 0.1 and model.kwargs["stream"] is True
    assert model.kwargs["classes"] == [c["goalkeeper"], c["player"], c["referee"]]


def test_track_video_stops_at_max_frames(tmp_path):
    video = tmp_path / "v.mp4"
    _tiny_video(video, n=5)
    white = np.full((200, 200, 3), 255, np.uint8)
    box = [10.0, 20.0, 90.0, 120.0]
    results = [_result(white, [box], [3], [CLASS_IDS["player"]]) for _ in range(5)]
    model = _FakeModel(results)
    out = track_video(model, video, tmp_path / "t.yaml", CFG, device="cpu", max_frames=2)
    assert out.tracks["frame"].tolist() == [1, 2] and out.n_frames == 2
