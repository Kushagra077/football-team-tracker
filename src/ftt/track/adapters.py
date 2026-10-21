"""Wrappers that make Ultralytics' ByteTrack / BoT-SORT satisfy ``base.Tracker``.

Built from configs/*.yaml. Ultralytics' trackers take a results-like object (``conf``, ``cls``,
``xywh``, boolean indexing) and return rows ``[x1, y1, x2, y2, id, conf, cls, det_idx]``; we hide
that behind the (N,6) -> (M,7) contract so the ablation just swaps config files.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from ultralytics.trackers import BOTSORT, OCSORT, BYTETracker, DeepOCSORT, FASTTracker

from ftt.track.base import Tracker

TRACKERS = {
    "bytetrack": BYTETracker,
    "botsort": BOTSORT,
    "ocsort": OCSORT,
    "deepocsort": DeepOCSORT,
    "fasttrack": FASTTracker,
}


class _Dets:
    """Minimal stand-in for ultralytics ``Boxes``: just what the trackers read."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls
        w, h = xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]
        self.xywh = np.stack([xyxy[:, 0] + w / 2, xyxy[:, 1] + h / 2, w, h], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> _Dets:
        return _Dets(self.xyxy[idx], self.conf[idx], self.cls[idx])


class UltralyticsTracker(Tracker):
    def __init__(self, cfg: dict):
        cfg = dict(cfg)
        self.name = cfg.pop("name")
        kind = cfg["tracker_type"]
        if kind not in TRACKERS:
            raise ValueError(f"tracker_type must be one of {sorted(TRACKERS)}, got {kind!r}")
        self._args = SimpleNamespace(**cfg)
        self.needs_frame = cfg.get("gmc_method", "none") != "none" or cfg.get("with_reid", False)
        self._tracker = TRACKERS[kind](self._args)

    def update(self, dets: np.ndarray, frame: np.ndarray | None) -> np.ndarray:
        if self.needs_frame and frame is None:
            raise ValueError(f"{self.name} needs the BGR frame (GMC / ReID)")
        dets = np.asarray(dets, dtype=np.float32).reshape(-1, 6)
        out = self._tracker.update(_Dets(dets[:, :4], dets[:, 4], dets[:, 5]), frame)
        out = np.asarray(out, dtype=np.float32).reshape(-1, 8)
        return out[:, :7]  # drop the detection index

    def reset(self) -> None:
        self._tracker.reset()


def build_tracker(config_path: str | Path) -> Tracker:
    """Factory: read a tracker config and return the matching adapter."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    return UltralyticsTracker(cfg)
