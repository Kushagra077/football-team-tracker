"""Feed one clip's detections through a tracker, frame by frame."""

from __future__ import annotations

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
