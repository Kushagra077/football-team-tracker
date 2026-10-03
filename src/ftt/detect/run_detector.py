"""Run the A1 detector ONCE per clip and cache to ``cache/detections/<clip>.parquet``.

Runs on Kaggle (GPU). Use a LOW confidence threshold (~0.1): ByteTrack's second
association pass needs low-confidence boxes, and a detector thresholded at 0.5
silently disables it. Keep player/goalkeeper/referee; drop ball/other.
Weights path comes from ``configs/paths.yaml``; never copy weights into this repo.
"""

from __future__ import annotations

from pathlib import Path


def run_detector(clip_dir: Path, weights: Path, out_path: Path, conf: float = 0.1) -> Path:
    raise NotImplementedError
