"""Run the detector ONCE per clip and cache to ``cache/detections/<detector>/<clip>.parquet``.

Runs on the best local device (``ftt.device.default_device``: cuda, mps or cpu).
Use a LOW confidence threshold (~0.1): ByteTrack's second association pass needs low-confidence
boxes, and a detector thresholded at 0.5 silently disables it. Keep player/goalkeeper/referee;
drop ball/other.
Weights path comes from ``configs/paths.yaml``; never copy weights into this repo.

Defaults mirror how the detector itself is evaluated (imgsz 640, NMS IoU 0.7, max 300 detections
per image).
Ultralytics resizes each frame for the network and maps boxes back to original pixel coordinates.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from ftt.data.detections import DET_COLUMNS, DETECTOR_CLASS_NAMES, KEEP_IDS, validate_detections
from ftt.device import default_device

FRAME_RE = re.compile(r"_(\d+)\.jpg$")


def frame_number(path: Path) -> int:
    """``SNMOT-116_000042.jpg`` -> 42 (1-indexed, as in the filename)."""
    m = FRAME_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"cannot read a frame number from {Path(path).name!r}")
    return int(m.group(1))


def load_detector(weights: Path) -> YOLO:
    """Load the weights once and check the class ids are the expected ones, so ids can't shift."""
    model = YOLO(str(weights))
    names = {int(k): v for k, v in dict(model.names).items()}
    if names != DETECTOR_CLASS_NAMES:
        raise ValueError(f"{Path(weights).name} classes {names} != expected {DETECTOR_CLASS_NAMES}")
    return model


def run_detector(
    frame_paths: Sequence[Path],
    model: YOLO,
    out_path: Path,
    conf: float = 0.1,
    iou: float = 0.7,
    imgsz: int = 640,
    max_det: int = 300,
    device: str = default_device(),
    batch: int = 16,
) -> Path:
    """Detect persons in ``frame_paths`` with a loaded ``model``; save a validated parquet."""
    rows = []
    for start in range(0, len(frame_paths), batch):
        chunk = [Path(p) for p in frame_paths[start : start + batch]]
        results = model.predict(
            [str(p) for p in chunk],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=device,
            verbose=False,
        )
        for path, res in zip(chunk, results, strict=True):
            frame = frame_number(path)
            boxes = res.boxes
            for (x1, y1, x2, y2), score, cls in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.conf.cpu().numpy(),
                boxes.cls.cpu().numpy().astype(int),
                strict=True,
            ):
                if cls in KEEP_IDS:
                    rows.append((frame, x1, y1, x2, y2, score, cls))

    dets = validate_detections(pd.DataFrame(rows, columns=DET_COLUMNS))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dets.to_parquet(out_path)
    return out_path
