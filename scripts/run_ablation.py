"""Step 3-4: 3 tracker configs x {oracle, real} detection sources.

Results go to reports/tracking_<source>_<split>.json.

--source oracle : GT person boxes, conf = 1.0, so every point lost is the tracker's fault.
--source real   : A1's cached detections (scripts/cache_detections.py), conf >= 0.1.
Both come through the same table format (ftt.data.detections), so the tracker cannot tell which
it got. Both are scored against the same GT persons.

    uv run python scripts/run_ablation.py --source oracle --split val
    uv run python scripts/run_ablation.py --source real --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml

from ftt.data.detections import gt_as_detections, load_cached_detections
from ftt.data.sn_tracking import frame_path, list_clips, load_persons_gt
from ftt.eval.mot_io import write_mot
from ftt.eval.trackeval_wrap import evaluate
from ftt.track.adapters import UltralyticsTracker
from ftt.track.runner import run_clip

ROOT = Path(__file__).parents[1]
CONFIGS = ["bytetrack", "botsort_nogmc", "botsort_gmc"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="oracle", choices=["oracle", "real"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (smoke test)")
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. match_thresh=0.9")
    ap.add_argument("--tag", default="", help="suffix for result names, required with --set")
    args = ap.parse_args()
    overrides = {k: yaml.safe_load(v) for k, _, v in (item.partition("=") for item in args.set)}
    if overrides and not args.tag:
        ap.error("--set needs --tag, so tuned results do not overwrite the untuned ones")

    paths = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text())
    meta_root = ROOT / paths["meta_root"]
    images_root = (ROOT / paths["a1_images"]).resolve()
    folder = "test" if args.split == "test" else "train"
    clips = list_clips(meta_root, args.split)[: args.limit]

    out_dir = ROOT / paths["reports_dir"] / args.source / args.split
    gt_by_clip = {c: load_persons_gt(meta_root / folder / c) for c in clips}
    for c, gt in gt_by_clip.items():
        write_mot(gt, out_dir / "gt" / f"{c}.txt")

    if args.source == "oracle":
        dets_by_clip = {c: gt_as_detections(meta_root / folder / c) for c in clips}
    else:
        cache_dir = ROOT / paths["cache_dir"]
        dets_by_clip = {c: load_cached_detections(cache_dir, c) for c in clips}

    results = {}
    for config in args.configs:
        name = f"{config}__{args.tag}" if args.tag else config  # label for output dirs / results
        cfg = {**yaml.safe_load((ROOT / "configs" / f"{config}.yaml").read_text()), **overrides}
        tracker = UltralyticsTracker(cfg)
        for c, gt in gt_by_clip.items():

            def load(f, c=c):
                return cv2.imread(str(frame_path(images_root, args.split, c, f)))

            n_frames = int(gt["frame"].max())
            tracks = run_clip(tracker, dets_by_clip[c], n_frames, load)
            write_mot(tracks, out_dir / name / f"{c}.txt")
        results[name] = evaluate(out_dir / "gt", out_dir / name)
        print(name, {k: round(v, 1) for k, v in results[name].items()}, flush=True)

    (ROOT / paths["reports_dir"]).mkdir(exist_ok=True)
    out = ROOT / paths["reports_dir"] / f"tracking_{args.source}_{args.split}.json"
    if out.exists():  # a partial --configs run must not erase the other trackers' results
        results = {**json.loads(out.read_text())["results"], **results}
    report = {"source": args.source, "split": args.split, "clips": len(clips), "results": results}
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
