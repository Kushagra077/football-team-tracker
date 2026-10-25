"""Permutation-invariant team accuracy.

Reports per-detection and per-track accuracy per clip. ``--source gt`` uses ground-truth player
boxes (an upper bound); ``--source real`` uses the cached detector boxes. Real boxes carry no team
label, so each is matched one-to-one to a GT player by IoU and scored against that player's team;
unmatched boxes (false positives) are counted but not scored. KMeans never sees a label.
Per-track here = per GT person (loose: a majority vote).
Tracker ids and voting are scored in eval_team_vote.py.

    uv run python scripts/eval_team.py --split val --source real
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from ftt.data.detections import (
    CLASS_IDS,
    detector_name,
    gt_as_detections,
    load_cached_detections,
)
from ftt.data.sn_tracking import frame_path, list_clips, load_persons_gt
from ftt.eval.team_metrics import best_permutation_accuracy, per_track_accuracy
from ftt.team.cluster import choose_fit_frames, fit_team_clusters
from ftt.team.features import FRAME_COLS, box_features, feature_names

ROOT = Path(__file__).parents[1]
PLAYER = CLASS_IDS["player"]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, 4) x (m, 4) xyxy boxes -> (n, m) IoU."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def match_to_gt(boxes: np.ndarray, gt_boxes: np.ndarray, min_iou: float) -> np.ndarray:
    """For each detection, the index of its matched GT player (one-to-one) or -1."""
    out = np.full(len(boxes), -1)
    if len(boxes) == 0 or len(gt_boxes) == 0:
        return out
    iou = iou_matrix(boxes, gt_boxes)
    rows, cols = linear_sum_assignment(-iou)
    for r, c in zip(rows, cols, strict=True):
        if iou[r, c] >= min_iou:
            out[r] = c
    return out


def features_for(dets, frames, clip, split, cfg, images_root):
    """(frame, box x4, Lab x3) per player detection of the given frames, as one array."""
    sub = dets[dets["frame"].isin(list(frames))]
    feats = box_features(
        sub, lambda f: cv2.imread(str(frame_path(images_root, split, clip, f))), cfg
    )
    return feats[[*FRAME_COLS, *feature_names(cfg)]].to_numpy(dtype="float64")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--source", default="real", choices=["gt", "real"])
    ap.add_argument("--conf", type=float, default=0.5, help="min detector confidence (real only)")
    ap.add_argument("--stride", type=int, default=25, help="score every Nth frame")
    ap.add_argument("--min-iou", type=float, default=0.5)
    args = ap.parse_args()

    paths = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text())
    cfg = yaml.safe_load((ROOT / "configs" / "team.yaml").read_text())
    meta_root = ROOT / paths["meta_root"]
    images_root = ROOT / paths["detector_images"]
    folder = "test" if args.split == "test" else "train"
    detector = detector_name(paths["detector_weights"])
    fit_n = cfg["cluster"]["fit_frames"]

    results = {}
    for clip in list_clips(meta_root, args.split):
        clip_dir = meta_root / folder / clip
        gt = load_persons_gt(clip_dir)
        gt = gt[gt["cls"] == "player"]
        if args.source == "gt":
            dets = gt_as_detections(clip_dir)
        else:
            dets = load_cached_detections(ROOT / paths["cache_dir"], detector, clip)
            dets = dets[dets["conf"] >= args.conf]
        dets = dets[dets["cls"] == PLAYER]

        # Fit on the first fit_n frames: every player detection, labels never used.
        fit_frames = choose_fit_frames(
            dets["frame"].to_numpy(), fit_n, cfg["cluster"]["fit_min_players"]
        )
        fit = features_for(dets, fit_frames, clip, args.split, cfg, images_root)
        km = fit_team_clusters(fit[:, 5:])  # columns after frame, x1..y2 = the clustering feature

        ev = features_for(dets, range(1, 751, args.stride), clip, args.split, cfg, images_root)
        pred_all = km.predict(ev[:, 5:])
        pred, true, tid, n_fp = [], [], [], 0
        for fr in np.unique(ev[:, 0]):
            m = ev[:, 0] == fr
            g = gt[gt["frame"] == int(fr)].reset_index(drop=True)
            idx = match_to_gt(ev[m, 1:5], g[["x1", "y1", "x2", "y2"]].to_numpy(), args.min_iou)
            n_fp += int((idx < 0).sum())
            for p, j in zip(pred_all[m], idx, strict=True):
                if j >= 0:
                    pred.append(p)
                    true.append(0 if g.loc[j, "team"] == "left" else 1)
                    tid.append(g.loc[j, "track_id"])
        pred, true, tid = np.array(pred), np.array(true), np.array(tid)
        det_acc = best_permutation_accuracy(pred, true)
        trk_acc = per_track_accuracy(pred, true, tid)
        results[clip] = {
            "scored": len(pred),
            "unmatched": n_fp,
            "per_detection": det_acc,
            "per_track": trk_acc,
        }
        print(
            f"{clip}  scored {len(pred):4d}  unmatched {n_fp:3d}  "
            f"per-detection {det_acc:6.1%}  per-track {trk_acc:6.1%}"
        )

    mean_det = float(np.mean([r["per_detection"] for r in results.values()]))
    mean_trk = float(np.mean([r["per_track"] for r in results.values()]))
    print(f"MEAN  per-detection {mean_det:.1%}  per-track {mean_trk:.1%}")
    print("clips < 80% per-detection:", [c for c, r in results.items() if r["per_detection"] < 0.8])

    out_dir = ROOT / paths["reports_dir"] / detector
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"team_{args.source}_{args.split}.json"
    out.write_text(
        json.dumps({"args": vars(args), "mean": [mean_det, mean_trk], "clips": results}, indent=2)
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
