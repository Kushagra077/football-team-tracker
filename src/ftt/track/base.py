"""The tracker interface. Every tracker consumes the same detections format."""

from __future__ import annotations

import abc

import numpy as np


class Tracker(abc.ABC):
    """``update(dets, frame) -> tracks``, called once per frame in order.

    ``dets``  : float array (N, 6) = [x1, y1, x2, y2, conf, cls]
    ``frame`` : BGR image (H, W, 3); needed by GMC / ReID, ignored by pure-motion trackers
    returns   : float array (M, 7) = [x1, y1, x2, y2, track_id, conf, cls]

    ``needs_frame`` says whether ``update`` needs the image (GMC / ReID); the runner only loads
    frames for trackers that set it. The Ultralytics adapter in ``adapters.py`` is checked
    against the pinned library version (see ``pyproject.toml``).
    """

    needs_frame: bool = False

    @abc.abstractmethod
    def update(self, dets: np.ndarray, frame: np.ndarray | None) -> np.ndarray: ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear all state so one instance can be reused across clips."""
