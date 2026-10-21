"""Step 4: two-fold check of the tuned tracker settings (val clips only).

Split the val clips into folds A and B (alternating sorted order), run each tracker with the
baseline config and the tuned overrides once over ALL clips, then score fold A, fold B and all
clips from the same output files. A tuned setting passes for a tracker if it beats the baseline
on BOTH folds. This is an early warning, not a clean test: the settings were chosen looking at
all of the val clips. The test split is for one final run.

    uv run python scripts/check_folds.py
    uv run python scripts/check_folds.py --trackers bytetrack --set track_high_thresh=0.25
"""

from __future__ import annotations

import argparse
import json
import shutil
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
TUNED = {"track_high_thresh": 0.25, "match_thresh": 0.9}


def parse_set(items: list[str]) -> dict:
    return {k: yaml.safe_load(v) for k, _, v in (item.partition("=") for item in items)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackers", nargs="+", default=["all"], help="config names, or 'all'")
    ap.add_argument(
        "--set", nargs="*", default=None, help="tuned overrides (default: shared setting)"
    )
    args = ap.parse_args()
    trackers = ALL_TRACKERS if args.trackers == ["all"] else args.trackers
    tuned = TUNED if args.set is None else parse_set(args.set)

    paths = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text())
    meta_root = ROOT / paths["meta_root"]
    images_root = (ROOT / paths["a1_images"]).resolve()
    clips = list_clips(meta_root, "val")
    folds = {"all": clips, "A": clips[0::2], "B": clips[1::2]}
    print(f"fold A ({len(folds['A'])}): {' '.join(c[-3:] for c in folds['A'])}")
    print(f"fold B ({len(folds['B'])}): {' '.join(c[-3:] for c in folds['B'])}")

    work = ROOT / paths["reports_dir"] / "folds"
    gt = {c: load_persons_gt(meta_root / "train" / c) for c in clips}
    for name, members in folds.items():  # one GT dir per fold, so evaluate() scores that fold only
        shutil.rmtree(work / f"gt_{name}", ignore_errors=True)
        for c in members:
            write_mot(gt[c], work / f"gt_{name}" / f"{c}.txt")
    dets = {c: load_cached_detections(ROOT / paths["cache_dir"], c) for c in clips}

    tuning_log = ROOT / paths["reports_dir"] / "tuning" / "val.jsonl"
    logged = {}
    if tuning_log.exists():
        for r in map(json.loads, tuning_log.open()):
            logged[(r["tracker"], json.dumps(r["overrides"], sort_keys=True))] = r["HOTA"]

    results = {}
    for name in trackers:
        base_cfg = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
        for label, overrides in [("base", {}), ("tuned", tuned)]:
            tracker = UltralyticsTracker({**base_cfg, **overrides})
            out = work / f"{name}__{label}"
            for c in clips:

                def load(f, c=c):
                    return cv2.imread(str(frame_path(images_root, "val", c, f)))

                tracks = run_clip(tracker, dets[c], int(gt[c]["frame"].max()), load)
                write_mot(tracks, out / f"{c}.txt")
            results[(name, label)] = {f: evaluate(work / f"gt_{f}", out) for f in folds}

            # Regression: the all-clips score must match what the tuning sweep logged.
            key = (name, json.dumps(overrides, sort_keys=True))
            if key in logged:
                got = results[(name, label)]["all"]["HOTA"]
                assert abs(got - logged[key]) < 0.01, (
                    f"{name} {label}: {got} != logged {logged[key]}"
                )
            print(f"  ran {name} / {label}", flush=True)

    print(f"\ntuned = {tuned}")
    print(f"{'tracker':15} {'all':>13} {'fold A':>13} {'fold B':>13}   verdict")
    print(f"{'':15} {'base>tuned':>13} {'base>tuned':>13} {'base>tuned':>13}")
    summary = {}
    for name in trackers:
        cells, gains = [], {}
        for f in folds:
            b, t = results[(name, "base")][f]["HOTA"], results[(name, "tuned")][f]["HOTA"]
            gains[f] = t - b
            cells.append(f"{b:5.1f}>{t:5.1f}")
        ok = gains["A"] > 0 and gains["B"] > 0
        summary[name] = {"gain": {f: round(g, 2) for f, g in gains.items()}, "pass": ok}
        print(f"{name:15} {cells[0]:>13} {cells[1]:>13} {cells[2]:>13}   "
              f"{'PASS' if ok else 'FAIL'} (A {gains['A']:+.1f}, B {gains['B']:+.1f})")  # fmt: skip

    out_json = ROOT / paths["reports_dir"] / "folds_val.json"
    out_json.write_text(json.dumps({"tuned": tuned, "folds": folds, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
