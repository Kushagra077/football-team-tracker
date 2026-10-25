"""Pass 1 for an uploaded video: detect + track with ``model.track(stream=True)``.

Frames stream one at a time; for each tracked player box the torso colour is computed right
away from that frame and the frame is dropped, so a long video never sits in memory. Pass 2
(``team.assign``) turns the colours into team labels once the whole video has been seen.
On one val clip's frames it gives the same tracks as the cached-box evaluation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

from ftt.data.detections import CLASS_IDS
from ftt.scale import clean_fps
from ftt.team.features import FRAME_COLS, box_features, extra_names

COLS = ["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "cls"]
PERSON_IDS = [CLASS_IDS["goalkeeper"], CLASS_IDS["player"], CLASS_IDS["referee"]]


@dataclass
class TrackedVideo:
    tracks: pd.DataFrame  # every tracked person box, columns COLS, frame 1-indexed
    feats: pd.DataFrame  # ``box_features`` of the PLAYER rows, indexed like ``tracks``
    fps: float
    n_frames: int
    size: tuple[int, int]  # (width, height)


def video_info(source: Path) -> tuple[float, int, tuple[int, int]]:
    """(fps, declared frame count, (width, height)). The one place fps is read: a bad value
    (0, nan, negative) becomes 25 here, so scaling, timestamps and rendering all agree."""
    cap = cv2.VideoCapture(str(source))
    try:
        fps = clean_fps(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()
    return fps, n, size


def track_video(
    model,
    source: Path,
    tracker_yaml: Path,
    cfg: dict,
    *,
    conf: float = 0.1,
    iou: float = 0.7,
    imgsz: int = 640,
    max_det: int = 300,
    device: str = "mps",
    max_frames: int | None = None,
) -> TrackedVideo:
    """Run ``model.track`` over a video file; keep boxes + player torso colours, not frames.

    ``conf`` must stay low (0.1): the tracker's second association pass needs weak boxes.
    ``max_frames`` stops after that many frames (a demo cap).
    """
    fps, n_declared, size = video_info(source)
    stream = model.track(
        source=str(source),
        stream=True,
        tracker=str(tracker_yaml),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        device=device,
        classes=PERSON_IDS,
        verbose=False,
    )
    rows, feats, n_seen = [], [], 0
    for n_seen, result in enumerate(stream, start=1):  # noqa: B007
        if max_frames is not None and n_seen > max_frames:
            n_seen = max_frames
            break
        b = result.boxes
        if b is None or b.id is None:
            continue
        first = len(rows)
        for (x1, y1, x2, y2), tid, score, cls in zip(
            b.xyxy.cpu().numpy(),
            b.id.int().cpu().numpy(),
            b.conf.cpu().numpy(),
            b.cls.int().cpu().numpy(),
            strict=True,
        ):
            rows.append((n_seen, int(tid), x1, y1, x2, y2, score, int(cls)))
        here = pd.DataFrame(rows[first:], columns=COLS, index=range(first, len(rows)))
        players = here[here["cls"] == CLASS_IDS["player"]]
        if len(players):
            # The loader hands back the frame we are holding right now; it is not kept.
            feats.append(box_features(players, lambda _f, img=result.orig_img: img, cfg))
    tracks = pd.DataFrame(rows, columns=COLS)
    feats = pd.concat(feats) if feats else pd.DataFrame(columns=[*FRAME_COLS, *extra_names(cfg)])
    return TrackedVideo(
        tracks=tracks, feats=feats, fps=fps, n_frames=n_seen or n_declared, size=size
    )
