from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from ftt.data.detections import CLASS_IDS
from ftt.pipeline import NoPitchViewError, process_video

ROOT = Path(__file__).parents[1]
W, H, N = 400, 300, 6
RED, BLUE = (0, 0, 220), (220, 0, 0)  # BGR shirts, nothing like pitch green
BOXES = [[20, 100, 50, 160], [80, 100, 110, 160], [200, 100, 230, 160], [260, 100, 290, 160]]


class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def int(self):
        return _T(self.a.astype(int))


class _FakeModel:
    """Two red and two blue players in every frame, tracked with fixed ids 1..4."""

    def __init__(self, n_players=4, shirts=None):
        self.n_players = n_players
        self.shirts = shirts or [RED, RED, BLUE, BLUE]

    def track(self, **kwargs):
        def frames():
            for _ in range(N):
                img = np.full((H, W, 3), 128, np.uint8)
                for (x1, y1, x2, y2), colour in zip(BOXES, self.shirts, strict=True):
                    img[y1:y2, x1:x2] = colour
                k = self.n_players
                b = SimpleNamespace(
                    xyxy=_T(np.array(BOXES[:k], float)),
                    id=_T(np.arange(1, k + 1)),
                    conf=_T([0.9] * k),
                    cls=_T([CLASS_IDS["player"]] * k),
                )
                yield SimpleNamespace(orig_img=img, boxes=b)

        return frames()


def _video(path):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
    for _ in range(N):
        vw.write(np.full((H, W, 3), 128, np.uint8))
    vw.release()


def _run(tmp_path, model, kind=None, **kw):
    video = tmp_path / "v.mp4"
    _video(video)
    cfg = yaml.safe_load((ROOT / "configs" / "team.yaml").read_text())
    if kind:
        cfg["features"]["kind"] = kind
    return process_video(
        video, tmp_path / "out", model=model, team_cfg=cfg, tracker_cfg={"track_buffer": 30},
        device="cpu", log=lambda _m: None, **kw,
    )  # fmt: skip


@pytest.mark.parametrize("kind", ["mean_lab", "hist_lab"])
def test_process_video_splits_two_colours_into_two_teams_and_writes_files(tmp_path, kind):
    out = _run(tmp_path, _FakeModel(), kind=kind)
    by_track = out.table.groupby("track_id")["team"].agg(lambda s: set(s.dropna()))
    assert all(len(v) == 1 for v in by_track)  # one team per track
    assert by_track[1] == by_track[2] and by_track[3] == by_track[4]
    assert by_track[1] != by_track[3]  # red and blue end up on different teams
    assert out.parquet.exists() and out.annotated.exists()
    assert out.n_frames == N and out.pitch_view_share == 1.0


def test_process_video_respects_max_frames_and_no_render(tmp_path):
    out = _run(tmp_path, _FakeModel(), max_frames=3, render=False)
    assert out.table["frame_id"].max() == 3 and out.n_frames == 3
    assert out.annotated is None


def test_process_video_without_a_wide_view_raises(tmp_path):
    with pytest.raises(NoPitchViewError):
        _run(tmp_path, _FakeModel(n_players=2))  # fewer than 3 players in every frame


@pytest.mark.parametrize("bad_fps", [0.0, float("nan"), -1.0])
def test_unreadable_fps_falls_back_to_25_everywhere(tmp_path, monkeypatch, bad_fps):
    """The container reports no usable fps: scaling, timestamps and the drawn video all use 25."""
    real_capture = cv2.VideoCapture

    class _NoFps:
        def __init__(self, *args):
            self._cap = real_capture(*args)

        def get(self, prop):
            return bad_fps if prop == cv2.CAP_PROP_FPS else self._cap.get(prop)

        def __getattr__(self, name):
            return getattr(self._cap, name)

    monkeypatch.setattr(cv2, "VideoCapture", _NoFps)
    out = _run(tmp_path, _FakeModel())
    assert out.fps == 25.0
    assert np.isfinite(out.table["timestamp"]).all()
    assert np.allclose(out.table["timestamp"], out.table["frame_id"] / 25)
    assert out.annotated.exists()


def test_players_with_only_grass_coloured_torsos_raise_a_clear_error(tmp_path):
    """Four players in every frame (a pitch view), but every torso crop is grass: no colours."""
    hsv_grass = np.uint8([[[38, 200, 200]]])  # inside the grass_mask hue/saturation/value range
    grass = tuple(int(v) for v in cv2.cvtColor(hsv_grass, cv2.COLOR_HSV2BGR)[0, 0])
    with pytest.raises(NoPitchViewError, match="cannot tell two teams apart"):
        _run(tmp_path, _FakeModel(shirts=[grass] * 4))
