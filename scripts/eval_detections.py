"""Step 4: detection-only DetA of the cached A1 boxes vs GT persons, at several confidence cut-offs.

Association is meaningless here, so each box just gets its index within its frame as an id (unique
ids per box would make TrackEval's IDF1 allocate a matrix with thousands of rows). Only DetA is
read. Class is ignored: persons vs persons. The GT itself goes through the same path first; it
must score DetA 100 or the pipeline is wrong.

    uv run python scripts/eval_detections.py --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from ftt.data.detections import gt_as_detections, load_cached_detections
from ftt.data.sn_tracking import list_clips
from ftt.eval.mot_io import write_mot
from ftt.eval.trackeval_wrap import evaluate

ROOT = Path(__file__).parents[1]


def as_mot(dets: pd.DataFrame) -> pd.DataFrame:
    """Detections -> MOT rows with a per-frame index as track_id."""
    out = dets.copy()
    out["track_id"] = out.groupby("frame").cumcount() + 1
    return out[["frame", "track_id", "x1", "y1", "x2", "y2", "conf"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--confs", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    args = ap.parse_args()

    paths = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text())
    meta_root = ROOT / paths["meta_root"]
    folder = "test" if args.split == "test" else "train"
    clips = list_clips(meta_root, args.split)
    out_dir = ROOT / paths["reports_dir"] / "det_only" / args.split

    for c in clips:
        write_mot(as_mot(gt_as_detections(meta_root / folder / c)), out_dir / "gt" / f"{c}.txt")

    # Sanity: the GT, pushed through the same path, must be a perfect detector.
    for c in clips:
        write_mot(as_mot(gt_as_detections(meta_root / folder / c)), out_dir / "sanity" / f"{c}.txt")
    sanity = evaluate(out_dir / "gt", out_dir / "sanity")
    assert sanity["DetA"] > 99.9, f"GT-as-detections should score DetA 100, got {sanity['DetA']}"
    print(f"sanity (GT as detections): DetA {sanity['DetA']:.1f}")

    cached = {c: load_cached_detections(ROOT / paths["cache_dir"], c) for c in clips}
    results = {}
    for thr in args.confs:
        for c, dets in cached.items():
            write_mot(as_mot(dets[dets["conf"] >= thr]), out_dir / f"conf{thr}" / f"{c}.txt")
        results[thr] = evaluate(out_dir / "gt", out_dir / f"conf{thr}")["DetA"]
        n = sum((d["conf"] >= thr).sum() for d in cached.values())
        print(
            f"conf >= {thr:<4}: DetA {results[thr]:5.1f}   ({n / len(clips) / 750:.1f} boxes/frame)"
        )

    (ROOT / paths["reports_dir"]).mkdir(exist_ok=True)
    out = ROOT / paths["reports_dir"] / f"det_only_{args.split}.json"
    out.write_text(json.dumps({"split": args.split, "DetA_by_conf": results}, indent=2))


if __name__ == "__main__":
    main()
