"""Goalkeeper heuristic vs ground truth, in isolation from colour clustering.

Team positions come from GT team labels (not KMeans), so this scores only the keeper rule.
Two windows: the fit window (first N frames) and the whole clip. Output: correct / wrong /
unknown counts per keeper, plus a picture of where teams and keepers stand.

    uv run python scripts/eval_goalkeepers.py --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ftt.data.sn_tracking import list_clips, load_persons_gt  # noqa: E402
from ftt.paths import load_paths  # noqa: E402
from ftt.team.cluster import assign_goalkeepers  # noqa: E402

ROOT = Path(__file__).parents[1]
TEAM_COLOUR = {"left": "tab:blue", "right": "tab:green"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "goalkeepers.png")
    args = ap.parse_args()

    paths = load_paths(ROOT)
    cfg = yaml.safe_load((ROOT / "configs" / "team.yaml").read_text())
    gk_cfg = cfg["goalkeeper"]
    meta_root = ROOT / paths["meta_root"]
    folder = "test" if args.split == "test" else "train"
    clips = list_clips(meta_root, args.split)
    windows = {f"first {cfg['cluster']['fit_frames']} frames": cfg["cluster"]["fit_frames"],
               "whole clip": 10_000}  # fmt: skip

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, (wname, last) in zip(axes, windows.items(), strict=True):
        counts = {"correct": 0, "wrong": 0, "unknown": 0}
        for row, clip in enumerate(clips):
            gt = load_persons_gt(meta_root / folder / clip)
            gt = gt[gt["frame"] <= last].assign(cx=lambda d: (d["x1"] + d["x2"]) / 2)
            pl = gt[gt["cls"] == "player"]
            team_x = np.array([pl[pl["team"] == t]["cx"].median() for t in ("left", "right")])
            gks = (
                gt[gt["cls"] == "goalkeeper"]
                .groupby("track_id")
                .agg(cx=("cx", "median"), team=("team", "first"))
            )
            ax.scatter(team_x, [row, row], s=420, c=list(TEAM_COLOUR.values()), alpha=0.35)
            if len(gks) == 0 or np.isnan(team_x).any():
                continue
            pred = assign_goalkeepers(
                gks["cx"].to_numpy(), team_x, gk_cfg["min_team_gap"], gk_cfg["min_margin"]
            )
            for (_, g), p in zip(gks.iterrows(), pred, strict=True):
                true = 0 if g["team"] == "left" else 1
                status = "unknown" if p < 0 else ("correct" if p == true else "wrong")
                counts[status] += 1
                edge = {"correct": "k", "wrong": "red", "unknown": "grey"}[status]
                ax.scatter(g["cx"], row, marker="^", s=130, c=TEAM_COLOUR[g["team"]],
                           edgecolors=edge, linewidths=2.5, zorder=3)  # fmt: skip
        ax.set_title(f"{wname}: {counts}", fontsize=10)
        ax.set_xlim(0, 1920)
        ax.set_xlabel(
            "image x (px)  |  big dot = team's median x, triangle = keeper (true team colour)"
        )
        ax.set_yticks(range(len(clips)), [c[-3:] for c in clips])
        ax.invert_yaxis()
        print(f"{wname:18s} {counts}")
    fig.suptitle(
        "Goalkeeper heuristic. Triangle outline: black = right team, red = wrong, grey = unknown"
    )
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=100)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
