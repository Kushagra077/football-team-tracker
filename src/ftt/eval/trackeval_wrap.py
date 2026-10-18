"""Thin wrapper over TrackEval's metric classes (HOTA, CLEAR, Identity).

Reads one MOT txt per clip (``<clip>.txt``) from a GT dir and a prediction dir and returns
HOTA, DetA, AssA, IDF1, ID switches pooled over all GT clips. We call the metric classes
directly rather than TrackEval's dataset loader: same maths, none of its directory layout.

Sanity checks that must pass before any tracker is trusted: GT-as-prediction -> HOTA 100 /
IDSW 0; GT with IDs shuffled every 50 frames -> HOTA drops and IDSW spikes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from trackeval.metrics import CLEAR, HOTA, Identity

from ftt.eval.mot_io import box_iou_xywh, empty_mot, read_mot

IOU_THRESHOLD = 0.5  # CLEAR / Identity matching threshold (HOTA sweeps its own alphas)


def _clip_data(gt, pred) -> dict:
    """Build TrackEval's per-sequence dict. Ids are remapped to 0..n-1; frames are 1-indexed."""
    n_frames = int(
        max(gt["frame"].max() if len(gt) else 0, pred["frame"].max() if len(pred) else 0)
    )
    gt_map = {i: k for k, i in enumerate(sorted(gt["track_id"].unique()))}
    pr_map = {i: k for k, i in enumerate(sorted(pred["track_id"].unique()))}
    gt_by = dict(tuple(gt.groupby("frame")))
    pr_by = dict(tuple(pred.groupby("frame")))
    empty = np.empty((0, 4))

    gt_ids, pr_ids, sims = [], [], []
    for f in range(1, n_frames + 1):
        g, p = gt_by.get(f), pr_by.get(f)
        gb = g[["x", "y", "w", "h"]].to_numpy(float) if g is not None else empty
        pb = p[["x", "y", "w", "h"]].to_numpy(float) if p is not None else empty
        gt_ids.append(
            g["track_id"].map(gt_map).to_numpy(int) if g is not None else np.empty(0, int)
        )
        pr_ids.append(
            p["track_id"].map(pr_map).to_numpy(int) if p is not None else np.empty(0, int)
        )
        sims.append(box_iou_xywh(gb, pb))

    return {
        "num_timesteps": n_frames,
        "num_gt_ids": len(gt_map),
        "num_tracker_ids": len(pr_map),
        "num_gt_dets": len(gt),
        "num_tracker_dets": len(pred),
        "gt_ids": gt_ids,
        "tracker_ids": pr_ids,
        "similarity_scores": sims,
    }


def evaluate(gt_dir: Path, pred_dir: Path) -> dict[str, float]:
    """Pooled metrics over every ``*.txt`` in ``gt_dir``. A clip missing from ``pred_dir``
    counts as an empty prediction (all misses), never as skipped."""
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    gt_files = sorted(gt_dir.glob("*.txt"))
    if not gt_files:
        raise FileNotFoundError(f"no GT .txt files in {gt_dir}")

    metrics = {
        "HOTA": HOTA(),
        "CLEAR": CLEAR({"THRESHOLD": IOU_THRESHOLD, "PRINT_CONFIG": False}),
        "Identity": Identity({"THRESHOLD": IOU_THRESHOLD, "PRINT_CONFIG": False}),
    }
    per_clip: dict[str, list] = {k: [] for k in metrics}
    for gt_file in gt_files:
        gt = read_mot(gt_file)
        pred_file = pred_dir / gt_file.name
        pred = read_mot(pred_file) if pred_file.exists() else empty_mot()
        data = _clip_data(gt, pred)
        for name, m in metrics.items():
            per_clip[name].append(m.eval_sequence(data))

    res = {
        name: m.combine_sequences(dict(enumerate(per_clip[name]))) for name, m in metrics.items()
    }
    return {
        "HOTA": 100 * float(np.mean(res["HOTA"]["HOTA"])),
        "DetA": 100 * float(np.mean(res["HOTA"]["DetA"])),
        "AssA": 100 * float(np.mean(res["HOTA"]["AssA"])),
        "IDF1": 100 * float(res["Identity"]["IDF1"]),
        "IDSW": float(res["CLEAR"]["IDSW"]),
    }
