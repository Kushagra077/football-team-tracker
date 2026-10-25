"""README figures, drawn from result files only (no video pixels, nothing typed in by hand).

1. ``tracker_benchmark.svg``: oracle vs real HOTA per tracker on the test split
   (``reports/tracking_oracle_test.json``, ``reports/<detector>/tracking_real_test.json``).
2. ``tracking_trails.svg``: a few seconds of tracked players from a ``process_video`` output
   (``tracks.parquet``), drawn as trails coloured by team. Positions only, no footage.

    uv run python scripts/make_figures.py --trails path/to/tracks.parquet --start 100 --frames 125
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ftt.data.detections import detector_name  # noqa: E402

ROOT = Path(__file__).parents[1]
OUT = ROOT / "reports" / "figures"

# Same colours as the annotated video (ftt/render.py): team 0 orange, team 1 cyan, none white.
BG, GRID, TEXT, MUTED = "#0f1b2d", "#22324a", "#e8edf3", "#8a9bb0"
ORANGE, CYAN, WHITE, GREEN = "#ff9f1c", "#2ec4e6", "#e8edf3", "#3ddc84"
NAMES = {
    "botsort_gmc": "BoT-SORT + GMC",
    "botsort_nogmc": "BoT-SORT",
    "deepocsort_gmc": "Deep OC-SORT + GMC",
    "ocsort_byte": "OC-SORT + BYTE",
    "bytetrack": "ByteTrack",
    "ocsort": "OC-SORT",
    "fasttrack": "FastTracker",
}


def _style(ax) -> None:
    ax.set_facecolor(BG)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def benchmark(detector: str) -> Path:
    oracle = json.loads((ROOT / "reports" / "tracking_oracle_test.json").read_text())["results"]
    real = json.loads((ROOT / "reports" / detector / "tracking_real_test.json").read_text())[
        "results"
    ]
    rows = sorted(
        ((NAMES[k], oracle[f"{k}__tuned"]["HOTA"], real[f"{k}__tuned"]["HOTA"]) for k in NAMES),
        key=lambda r: r[2],
    )
    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor=BG)
    _style(ax)
    for y, (_name, o, r) in enumerate(rows):
        ax.plot([r, o], [y, y], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        ax.scatter(o, y, s=90, color=BG, edgecolor=WHITE, lw=2, zorder=2)
        ax.scatter(r, y, s=90, color=GREEN, zorder=3)
        ax.text(o + 0.8, y, f"{o:.1f}", va="center", color=MUTED, fontsize=9)
        ax.text(r - 0.8, y, f"{r:.1f}", va="center", ha="right", color=GREEN, fontsize=9)
        ax.text((o + r) / 2, y + 0.3, f"-{o - r:.1f}", ha="center", color=MUTED, fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=TEXT, fontsize=10)
    ax.set_xlim(48, 90)
    ax.set_xlabel("HOTA (higher is better)", color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.scatter([], [], s=60, color=BG, edgecolor=WHITE, lw=2, label="oracle: ground-truth boxes")
    ax.scatter([], [], s=60, color=GREEN, label="real: my detector's boxes")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TEXT)
    ax.set_title(
        "SoccerNet-Tracking test, 49 clips: what each tracker loses to the detector",
        color=TEXT, fontsize=11, loc="left", pad=12,
    )  # fmt: skip
    out = OUT / "tracker_benchmark.svg"
    fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out


def trails(tracks_path: Path, start: int, n_frames: int) -> Path:
    t = pd.read_parquet(tracks_path)
    t = t[(t["frame_id"] >= start) & (t["frame_id"] < start + n_frames) & (t["role"] != "referee")]
    t = t.assign(cx=(t["x1"] + t["x2"]) / 2, fy=t["y2"]).sort_values("frame_id")
    w, h = float(t["x2"].max()), float(t["y2"].max())
    fig, ax = plt.subplots(figsize=(9, 9 * h / w * 0.85), facecolor=BG)
    _style(ax)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for tid, g in t.groupby("track_id"):
        team = g["team"].mode()
        colour = WHITE if team.empty else (ORANGE, CYAN)[int(team.iloc[0])]
        ax.plot(g["cx"], g["fy"], color=colour, lw=2, alpha=0.85, solid_capstyle="round")
        last = g.iloc[-1]
        ax.scatter(last["cx"], last["fy"], s=40, color=colour, zorder=3)
        ax.text(last["cx"] + 8, last["fy"] - 8, str(tid), color=colour, fontsize=8)
    ax.set_xlim(0, w)
    ax.set_ylim(h, t["fy"].min() - 40)  # image coordinates: y grows downwards
    ax.set_title(
        f"{n_frames / 25:.0f} s of tracked players (feet positions), coloured by team, tracker ids",
        color=TEXT, fontsize=11, loc="left",
    )  # fmt: skip
    out = OUT / "tracking_trails.svg"
    fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trails", type=Path, help="a process_video tracks.parquet")
    ap.add_argument("--start", type=int, default=1, help="first frame of the trails window")
    ap.add_argument("--frames", type=int, default=125, help="trails window length in frames")
    args = ap.parse_args()
    paths = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    print("wrote", benchmark(detector_name(paths["detector_weights"])))
    if args.trails:
        print("wrote", trails(args.trails, args.start, args.frames))


if __name__ == "__main__":
    main()
