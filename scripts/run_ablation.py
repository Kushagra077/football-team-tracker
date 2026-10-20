"""Step 3-4: 3 tracker configs x {oracle, A1} detection sources -> reports/tracking.json.

Step 3 (this version) runs the oracle source only: GT person boxes, conf = 1.0, so every
point lost is the tracker's fault, not the detector's.

    uv run python scripts/run_ablation.py --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from ftt.data.sn_tracking import frame_path, list_clips, load_persons_gt
from ftt.eval.mot_io import write_mot
from ftt.eval.trackeval_wrap import evaluate
from ftt.track.adapters import build_tracker

ROOT = Path(__file__).parents[1]
CONFIGS = ["bytetrack", "botsort_nogmc", "botsort_gmc"]
COLS = ["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]


def run_clip(tracker, gt: pd.DataFrame, frame_loader) -> pd.DataFrame:
    """Feed one clip's GT boxes through ``tracker`` frame by frame; return its tracks."""
    tracker.reset()
    by_frame = dict(tuple(gt.groupby("frame")))
    rows = []
    for f in range(1, int(gt["frame"].max()) + 1):
        g = by_frame.get(f)
        dets = np.empty((0, 6), np.float32)
        if g is not None:
            boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
            dets = np.column_stack([boxes, np.ones(len(g)), np.zeros(len(g))]).astype(np.float32)
        frame = frame_loader(f) if tracker.needs_frame else None
        for x1, y1, x2, y2, tid, conf, _ in tracker.update(dets, frame):
            rows.append((f, int(tid), x1, y1, x2, y2, conf))
    return pd.DataFrame(rows, columns=COLS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (smoke test)")
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    args = ap.parse_args()

    paths = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text())
    meta_root = ROOT / paths["meta_root"]
    images_root = (ROOT / paths["a1_images"]).resolve()
    folder = "test" if args.split == "test" else "train"
    clips = list_clips(meta_root, args.split)[: args.limit]

    out_dir = ROOT / paths["reports_dir"] / "oracle" / args.split
    gt_by_clip = {c: load_persons_gt(meta_root / folder / c) for c in clips}
    for c, gt in gt_by_clip.items():
        write_mot(gt, out_dir / "gt" / f"{c}.txt")

    results = {}
    for name in args.configs:
        tracker = build_tracker(ROOT / "configs" / f"{name}.yaml")
        for c, gt in gt_by_clip.items():

            def load(f, c=c):
                return cv2.imread(str(frame_path(images_root, args.split, c, f)))

            write_mot(run_clip(tracker, gt, load), out_dir / name / f"{c}.txt")
        results[name] = evaluate(out_dir / "gt", out_dir / name)
        print(name, {k: round(v, 1) for k, v in results[name].items()}, flush=True)

    report = {"source": "oracle", "split": args.split, "clips": len(clips), "results": results}
    (ROOT / paths["reports_dir"]).mkdir(exist_ok=True)
    out = ROOT / paths["reports_dir"] / f"tracking_oracle_{args.split}.json"
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
