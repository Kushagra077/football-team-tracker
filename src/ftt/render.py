"""Draw the tracks onto the source video: an ellipse at each person's feet plus a small label.

Team 0 / team 1 get two colours chosen NOT to look like common kits (orange, cyan); people
without a team (referees, undecided keepers, players with no usable crop) are drawn white.
Ids are the tracker's, so an id switch is visible as a label change.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

TEAM_BGR = {0: (0, 140, 255), 1: (255, 200, 0)}  # orange, cyan
NO_TEAM_BGR = (255, 255, 255)
ROLE_TAG = {"goalkeeper": "GK ", "referee": "REF ", "player": ""}


def draw_frame(img, rows: pd.DataFrame):
    """Annotate one BGR frame in place with that frame's rows of the output table."""
    for r in rows.itertuples():
        colour = TEAM_BGR.get(int(r.team), NO_TEAM_BGR) if pd.notna(r.team) else NO_TEAM_BGR
        cx, foot = int((r.x1 + r.x2) / 2), int(r.y2)
        w = max(int(r.x2 - r.x1), 8)
        cv2.ellipse(img, (cx, foot), (w // 2 + 4, w // 5 + 2), 0, -45, 235, colour, 3)
        label = f"{ROLE_TAG[str(r.role)]}{r.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = foot + w // 5 + 6
        cv2.rectangle(img, (cx - tw // 2 - 3, top), (cx + tw // 2 + 3, top + th + 8), colour, -1)
        cv2.putText(img, label, (cx - tw // 2, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)  # fmt: skip
    return img


def render_video(video: Path, table: pd.DataFrame, out_path: Path, fps: float) -> Path:
    """Write ``out_path`` (mp4) with the table's boxes drawn on every frame of ``video``."""
    by_frame = dict(tuple(table.groupby("frame_id")))
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for frame_no in range(1, 10**9):
            ok, img = cap.read()
            if not ok:
                break
            rows = by_frame.get(frame_no)
            writer.write(draw_frame(img, rows) if rows is not None else img)
    finally:
        cap.release()
        writer.release()
    return out_path
