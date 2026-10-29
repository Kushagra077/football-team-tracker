"""Collect per-detector tracking and team scores into ``results/detector_comparison.json``.

Reads what ``run_ablation.py`` (BoT-SORT+GMC, shared tuned setting) and ``eval_team_vote.py`` wrote
under ``reports/<detector>/``. A split a detector was not run on is left out. Selection is on val;
the test split was run once, for the detector chosen on val.

    uv run python scripts/compare_detectors.py football_detection_v3 football_detection_v6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ftt.paths import load_paths

ROOT = Path(__file__).parents[1]
TRACKER = "botsort_gmc__tuned"
METRICS = ["HOTA", "DetA", "AssA", "IDF1", "IDSW"]


def collect(reports: Path, detector: str) -> dict:
    out = {}
    for split in ("val", "test"):
        track_file = reports / detector / f"tracking_real_{split}.json"
        team_file = reports / detector / f"team_vote_{split}.json"
        if not track_file.exists() or TRACKER not in json.loads(track_file.read_text())["results"]:
            continue
        row = {
            m: round(json.loads(track_file.read_text())["results"][TRACKER][m], 1) for m in METRICS
        }
        if team_file.exists():
            team = json.loads(team_file.read_text())
            row["team_acc_vote"] = round(100 * team["mean_acc"]["vote"], 1)
            row["team_acc_raw"] = round(100 * team["mean_acc"]["raw"], 1)
            row["label_flips_vote"] = team["total_flips"]["vote"]
        out[split] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("detectors", nargs="+", help="names such as football_detection_v6")
    args = ap.parse_args()
    reports = ROOT / load_paths(ROOT)["reports_dir"]
    table = {d: collect(reports, d) for d in args.detectors}
    out = ROOT / "results" / "detector_comparison.json"
    out.write_text(json.dumps({"tracker": TRACKER, "detectors": table}, indent=2))
    print(f"{'detector':24}{'split':6}" + "".join(f"{m:>7}" for m in METRICS) + f"{'team%':>7}")
    for d, splits in table.items():
        for split, r in splits.items():
            print(
                f"{d:24}{split:6}"
                + "".join(f"{r[m]:7.1f}" for m in METRICS)
                + f"{r.get('team_acc_vote', float('nan')):7.1f}"
            )
    print("wrote", out)


if __name__ == "__main__":
    main()
