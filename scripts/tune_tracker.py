"""Step 4 tuning: run a tracker on REAL detections with config overrides; log the scores.

One knob at a time (the others stay at the config's values). Tune on val only; the test split
is for one final run. Results append to reports/tuning/<split>.jsonl and a run already in the
log is skipped, so an interrupted sweep resumes.

    uv run python scripts/tune_tracker.py --trackers bytetrack --set match_thresh=0.9
    uv run python scripts/tune_tracker.py --trackers bytetrack botsort_gmc --sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import yaml

from ftt.data.detections import load_cached_detections
from ftt.data.sn_tracking import frame_path, list_clips, load_persons_gt
from ftt.eval.mot_io import write_mot
from ftt.eval.trackeval_wrap import evaluate
from ftt.track.adapters import UltralyticsTracker
from ftt.track.runner import run_clip

ROOT = Path(__file__).parents[1]
ALL_TRACKERS = [
    "bytetrack", "botsort_nogmc", "botsort_gmc", "ocsort_byte",
    "deepocsort_gmc", "fasttrack", "ocsort", "deepocsort",
]  # fmt: skip
# One-at-a-time grid: every other knob stays at the config's value.
GRID = {
    "track_high_thresh": [0.2, 0.25, 0.3, 0.7],
    "new_track_thresh": [0.4, 0.7],
    "track_buffer": [15, 60],
    "match_thresh": [0.7, 0.9, 0.95],
}


def parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        out[key] = yaml.safe_load(value)  # "0.9" -> 0.9, "15" -> 15, "true" -> True
    return out


def run_key(tracker: str, overrides: dict) -> str:
    return tracker + ":" + json.dumps(overrides, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackers", nargs="+", required=True, help="config names, or 'all'")
    ap.add_argument("--split", default="val", choices=["val"], help="tuning is on val only")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. match_thresh=0.9")
    ap.add_argument("--sweep", action="store_true", help="baseline + the one-at-a-time grid")
    args = ap.parse_args()
    trackers = ALL_TRACKERS if args.trackers == ["all"] else args.trackers

    paths = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text())
    meta_root = ROOT / paths["meta_root"]
    images_root = (ROOT / paths["a1_images"]).resolve()
    clips = list_clips(meta_root, args.split)
    work = ROOT / paths["reports_dir"] / "tuning" / args.split
    log_path = ROOT / paths["reports_dir"] / "tuning" / f"{args.split}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    gt = {c: load_persons_gt(meta_root / "train" / c) for c in clips}
    for c, g in gt.items():
        write_mot(g, work / "gt" / f"{c}.txt")
    dets = {c: load_cached_detections(ROOT / paths["cache_dir"], c) for c in clips}

    done = set()
    if log_path.exists():
        done = {run_key(r["tracker"], r["overrides"]) for r in map(json.loads, log_path.open())}

    plans = []
    for name in trackers:
        if args.sweep:
            plans.append((name, {}))
            plans += [(name, {k: v}) for k, vals in GRID.items() for v in vals]
        else:
            plans.append((name, parse_set(args.set)))

    for name, overrides in plans:
        if run_key(name, overrides) in done:
            print(f"{name} {overrides}: already in log, skipping")
            continue
        cfg = {**yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text()), **overrides}
        tracker = UltralyticsTracker(cfg)
        out = work / "tmp" / name
        t = time.perf_counter()
        for c in clips:

            def load(f, c=c):
                return cv2.imread(str(frame_path(images_root, args.split, c, f)))

            tracks = run_clip(tracker, dets[c], int(gt[c]["frame"].max()), load)
            write_mot(tracks, out / f"{c}.txt")
        scores = evaluate(work / "gt", out)
        seconds = round(time.perf_counter() - t, 1)
        rounded = {k: round(v, 2) for k, v in scores.items()}
        record = {"tracker": name, "overrides": overrides, "seconds": seconds, **rounded}
        with log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(
            f"{name:15} {str(overrides):28} HOTA {scores['HOTA']:5.1f}  "
            f"IDF1 {scores['IDF1']:5.1f}  IDSW {scores['IDSW']:4.0f}  ({seconds}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
