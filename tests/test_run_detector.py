"""Detector runner: frame-name parsing, plus a 3-frame smoke run (needs weights + frames)."""

from pathlib import Path

import pandas as pd
import pytest

from ftt.data.detections import DET_COLUMNS, KEEP_IDS
from ftt.detect.run_detector import frame_number, load_detector, run_detector

WEIGHTS = Path("../football-detect-serve/models/football_detection_v3.pt")
FRAMES = Path("../football-detect-serve/dataset/images/test")
CLIP = "SNMOT-116"


def test_frame_number_reads_the_filename():
    assert frame_number(Path("x/SNMOT-116_000042.jpg")) == 42
    assert frame_number(Path("SNMOT-001_000750.jpg")) == 750


def test_frame_number_rejects_other_names():
    with pytest.raises(ValueError):
        frame_number(Path("frame42.png"))


@pytest.mark.skipif(
    not (WEIGHTS.exists() and (FRAMES / f"{CLIP}_000001.jpg").exists()),
    reason="detector weights / frames not available (outside this repo)",
)
def test_smoke_three_frames(tmp_path):
    paths = [FRAMES / f"{CLIP}_{i:06d}.jpg" for i in (1, 2, 3)]
    out = run_detector(paths, load_detector(WEIGHTS), tmp_path / "d.parquet", device="cpu")
    dets = pd.read_parquet(out)
    assert list(dets.columns) == DET_COLUMNS
    assert set(dets["frame"]) == {1, 2, 3}
    assert set(dets["cls"]) <= KEEP_IDS
    assert (dets["conf"] >= 0.1).all()
    # GT has ~14 persons per frame in this clip (PROGRESS.md); low conf adds some extras.
    per_frame = dets.groupby("frame").size()
    assert per_frame.between(8, 80).all(), per_frame.to_dict()
    # Boxes are in original 1920x1080 pixel coordinates, not the 640 network size.
    assert dets["x2"].max() > 640


class _Weights:
    """Stands in for a loaded YOLO: only the attributes ``detector_imgsz`` reads."""

    def __init__(self, model_args=None, overrides=None):
        self.model = type("M", (), {"args": model_args})() if model_args is not None else None
        self.overrides = overrides or {}


def test_detector_imgsz_reads_the_training_size():
    from ftt.detect.run_detector import detector_imgsz

    assert detector_imgsz(_Weights({"imgsz": 1280})) == 1280  # v6
    assert detector_imgsz(_Weights({"imgsz": 960})) == 960  # v7, v8
    assert detector_imgsz(_Weights({"imgsz": 640})) == 640  # v3


def test_detector_imgsz_fallbacks():
    from ftt.detect.run_detector import detector_imgsz

    assert detector_imgsz(_Weights(overrides={"imgsz": 960})) == 960  # no model args
    assert detector_imgsz(_Weights({"imgsz": [736, 1280]})) == 1280  # (h, w) pair: the long side
    assert detector_imgsz(_Weights({})) == 640  # nothing recorded: the default
    assert detector_imgsz(object()) == 640  # a plain fake model
