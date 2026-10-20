"""Every adapter obeys ``track/base.Tracker`` (input (N,6), output (M,7), reset() works)."""

from pathlib import Path

import numpy as np
import pytest

from ftt.track.adapters import build_tracker

CONFIGS = Path(__file__).parents[1] / "configs"
NAMES = ["bytetrack", "botsort_nogmc", "botsort_gmc"]


def moving_dets(t: int) -> np.ndarray:
    """Two boxes moving right at 3 px/frame, conf 1.0, cls 0."""
    return np.array(
        [
            [100 + 3 * t, 100, 140 + 3 * t, 200, 1.0, 0],
            [400 + 3 * t, 300, 440 + 3 * t, 400, 1.0, 0],
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize("name", NAMES)
def test_adapters_obey_interface(name):
    tracker = build_tracker(CONFIGS / f"{name}.yaml")
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ids_per_run = []
    for _ in range(2):  # second pass proves reset() gives a clean slate
        tracker.reset()
        out = None
        for t in range(10):
            out = tracker.update(moving_dets(t), frame)
            assert out.ndim == 2 and out.shape[1] == 7
        assert len(out) == 2
        ids_per_run.append(sorted(out[:, 4].astype(int)))
    assert ids_per_run[0] == ids_per_run[1]  # same ids again after reset


@pytest.mark.parametrize("name", NAMES)
def test_empty_frame(name):
    tracker = build_tracker(CONFIGS / f"{name}.yaml")
    out = tracker.update(np.empty((0, 6), dtype=np.float32), np.zeros((64, 64, 3), np.uint8))
    assert out.shape == (0, 7)


def test_gmc_requires_frame():
    with pytest.raises(ValueError):
        build_tracker(CONFIGS / "botsort_gmc.yaml").update(moving_dets(0), None)
