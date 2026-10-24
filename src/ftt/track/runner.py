"""Feed one clip's detections through a tracker, frame by frame."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ftt.track.base import Tracker

COLS = ["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "cls"]


def run_clip(tracker: Tracker, dets: pd.DataFrame, n_frames: int, frame_loader) -> pd.DataFrame:
    """Run ``tracker`` over frames 1..n_frames; ``dets`` is in the ftt.data.detections format.

    ``frame_loader(f)`` is only called for trackers that need the image (GMC / ReID).
    """
    tracker.reset()
    by_frame = dict(tuple(dets.groupby("frame")))
    rows = []
    for f in range(1, n_frames + 1):
        d = by_frame.get(f)
        arr = np.empty((0, 6), np.float32)
        if d is not None:
            arr = d[["x1", "y1", "x2", "y2", "conf", "cls"]].to_numpy(np.float32)
        frame = frame_loader(f) if tracker.needs_frame else None
        for x1, y1, x2, y2, tid, conf, cls in tracker.update(arr, frame):
            rows.append((f, int(tid), x1, y1, x2, y2, conf, int(cls)))
    return pd.DataFrame(rows, columns=COLS)


def tracks_for_clip(
    cache_path: Path,
    tracker: Tracker,
    dets_loader: Callable[[], pd.DataFrame],
    n_frames: int,
    frame_loader,
) -> pd.DataFrame:
    """Tracker output for one clip, cached as parquet (delete the file to re-track)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    tracks = run_clip(tracker, dets_loader(), n_frames, frame_loader)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_parquet(cache_path)
    return tracks
