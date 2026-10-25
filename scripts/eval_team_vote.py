"""Per-frame team labels vs voted labels, on REAL tracker output.

Pipeline per clip: cached detector boxes -> tracker (BoT-SORT + GMC, tuned shared setting)
-> torso colour of every tracked player box -> KMeans (fit on frames 1..N, no labels) ->
per-frame label -> rolling vote per tracker id. Tracked boxes are matched to GT players
(IoU >= 0.5) only to get a true team for scoring; unmatched boxes are not scored.
Gate: voted labels flip less than raw labels without lowering accuracy.

Tracks are cached under cache/tracks/<detector>/ (delete the file to re-track).

    uv run python scripts/eval_team_vote.py --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from ftt.data.detections import CLASS_IDS, detector_name, load_cached_detections
from ftt.data.sn_tracking import frame_path, list_clips, load_persons_gt
from ftt.eval.team_metrics import match_to_gt
from ftt.team.assign import label_players
from ftt.team.vote import vote_labels
from ftt.track.adapters import UltralyticsTracker
from ftt.track.config import load_tracker_cfg
from ftt.track.runner import tracks_for_clip

ROOT = Path(__file__).parents[1]
PLAYER = CLASS_IDS["player"]


def flips_per_track(track_ids: np.ndarray, frames: np.ndarray, labels: np.ndarray) -> int:
    """Total label changes between consecutive observations of the same track."""
    total = 0
    for tid in np.unique(track_ids):
        lab = labels[track_ids == tid][np.argsort(frames[track_ids == tid], kind="stable")]
        total += int((np.diff(lab) != 0).sum())
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--tracker", default="botsort_gmc")
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (smoke test)")
    ap.add_argument(
        "--centred-windows", type=int, nargs="*", default=[],
        help="also score a centred vote (looks both ways) with these extra window sizes",
    )
    args = ap.parse_args()

    paths = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text())
    cfg = yaml.safe_load((ROOT / "configs" / "team.yaml").read_text())
    meta_root = ROOT / paths["meta_root"]
    images_root = (ROOT / paths["detector_images"]).resolve()
    folder = "test" if args.split == "test" else "train"
    detector = detector_name(paths["detector_weights"])
    window = cfg["vote"]["window"]
    tcfg = load_tracker_cfg(ROOT / "configs", args.tracker)
    tracker = UltralyticsTracker(tcfg)
    track_dir = ROOT / paths["cache_dir"] / "tracks" / detector
    track_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for clip in list_clips(meta_root, args.split)[: args.limit]:
        gt = load_persons_gt(meta_root / folder / clip)
        n_frames = int(gt["frame"].max())

        def loader(f, c=clip):
            return cv2.imread(str(frame_path(images_root, args.split, c, f)))

        tracks = tracks_for_clip(
            track_dir / f"{clip}__{args.tracker}__tuned.parquet",
            tracker,
            lambda c=clip: load_cached_detections(ROOT / paths["cache_dir"], detector, c),
            n_frames,
            loader,
        )
        tracks = tracks[tracks["cls"] == PLAYER].reset_index(drop=True)
        fcols = label_players(tracks, loader, cfg)
        raw = fcols["raw"].to_numpy()

        # True team per tracked box via GT match (one-to-one per frame).
        gt_players = gt[gt["cls"] == "player"]
        true = np.full(len(fcols), -1)
        for fr, idx in fcols.groupby("frame").indices.items():
            g = gt_players[gt_players["frame"] == fr].reset_index(drop=True)
            m = match_to_gt(
                fcols.loc[idx, ["x1", "y1", "x2", "y2"]].to_numpy(),
                g[["x1", "y1", "x2", "y2"]].to_numpy(),
                0.5,
            )
            for i, j in zip(idx, m, strict=True):
                if j >= 0:
                    true[i] = 0 if g.loc[j, "team"] == "left" else 1

        tid, fr = fcols["track_id"].to_numpy(), fcols["frame"].to_numpy()
        variants = {
            "raw": raw,
            "vote": fcols["team"].to_numpy(),  # the config setting (centred or causal)
            "vote_causal": vote_labels(tid, fr, raw, window),  # looks back only
            "vote_lock15": vote_labels(tid, fr, raw, window, 15),  # reference: the lock we dropped
        }
        for w in args.centred_windows:
            variants[f"vote_centred_{w}"] = vote_labels(tid, fr, raw, w, centred=True)
        scored = true >= 0
        # One naming per clip, taken from the raw labels, so all variants are compared fairly.
        flip_names = (raw[scored] == true[scored]).mean() < 0.5
        row = {"scored": int(scored.sum()), "tracks": int(len(np.unique(tid)))}
        for name, v in variants.items():
            v_named = 1 - v if flip_names else v
            row[name] = {
                "acc": float((v_named[scored] == true[scored]).mean()),
                "flips": flips_per_track(tid, fr, v),
            }
        # Tracker ids that cover both teams (ID swaps): voting cannot fix these.
        mixed = sum(len(set(true[(tid == t) & scored])) > 1 for t in np.unique(tid))
        row["mixed_team_tracks"] = int(mixed)
        results[clip] = row
        print(
            f"{clip} scored {row['scored']:5d} tracks {row['tracks']:3d} mixed {mixed}  "
            + "  ".join(
                f"{n}: acc {row[n]['acc']:.1%} flips {row[n]['flips']:4d}" for n in variants
            ),
            flush=True,
        )

    mean = {n: float(np.mean([r[n]["acc"] for r in results.values()])) for n in variants}
    flips = {n: int(sum(r[n]["flips"] for r in results.values())) for n in variants}
    print("MEAN acc", {k: f"{v:.1%}" for k, v in mean.items()}, " TOTAL flips", flips)
    out_dir = ROOT / paths["reports_dir"] / detector
    out = out_dir / f"team_vote_{args.split}.json"
    out.write_text(
        json.dumps(
            {"args": vars(args), "mean_acc": mean, "total_flips": flips, "clips": results}, indent=2
        )
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
