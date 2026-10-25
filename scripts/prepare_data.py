"""Print, don't assume.

Clip counts per split, frames per clip, GT tracklets per clip, role distribution, average
track length, tracks that leave and re-enter, average simultaneous players, and the
OVERLAP between the detector's training clips and our evaluation clips.

Run from anywhere: ``uv run python scripts/prepare_data.py``.
"""

from __future__ import annotations

import configparser
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ftt.data.sn_tracking import SPLITS, list_clips, load_persons_gt, load_roles

ROOT = Path(__file__).resolve().parents[1]


def read_seqinfo(clip_dir: Path) -> configparser.SectionProxy:
    """The [Sequence] section of seqinfo.ini (seqLength, frameRate, imWidth, ...)."""
    cfg = configparser.ConfigParser()
    cfg.read(clip_dir / "seqinfo.ini")
    return cfg["Sequence"]


def clip_dir_for(meta_root: Path, split: str, clip: str) -> Path:
    # val clips live in train/ on disk; only the split name differs.
    return meta_root / ("test" if split == "test" else "train") / clip


def clip_stats(clip_dir: Path) -> dict:
    """One clip -> one row of facts, persons only (DD-002)."""
    frames = int(read_seqinfo(clip_dir)["seqLength"])
    persons = load_persons_gt(clip_dir)

    track_len = persons.groupby("track_id").size()
    # A track "leaves and re-enters" if its sorted frame numbers ever jump by more than 1.
    has_gap = persons.groupby("track_id")["frame"].agg(
        lambda f: bool((np.diff(np.sort(f.to_numpy())) > 1).any())
    )

    # Divide by seqLength, not by groupby("frame"): a frame with zero players has no rows,
    # so a groupby mean would skip it and overstate the average.
    n_players = int((persons["cls"] == "player").sum())

    return {
        "clip": clip_dir.name,
        "frames": frames,
        "max_gt_frame": int(persons["frame"].max()),
        "n_tracks": int(persons["track_id"].nunique()),
        "avg_track_len": float(track_len.mean()),
        "n_gap_tracks": int(has_gap.sum()),
        "avg_players": n_players / frames,
        "avg_persons": len(persons) / frames,
    }


def frames_per_clip(detector_images: Path, split: str) -> Counter:
    """{clip: number of detector-project images} for one split. Names are <clip>_<frame:06d>.jpg."""
    return Counter(p.name.split("_")[0] for p in (detector_images / split).glob("SNMOT-*.jpg"))


def main() -> None:
    paths = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text())
    meta_root = ROOT / paths["meta_root"]
    detector_images = (ROOT / paths["detector_images"]).resolve()

    pd.set_option("display.width", 120)
    clips_by_split = {s: list_clips(meta_root, s) for s in SPLITS}

    # --- per-split statistics -------------------------------------------------------------
    all_rows = []
    for split in SPLITS:
        rows = [clip_stats(clip_dir_for(meta_root, split, c)) for c in clips_by_split[split]]
        df = pd.DataFrame(rows).assign(split=split)
        all_rows.append(df)

        roles = Counter(
            role.cls
            for c in clips_by_split[split]
            for role in load_roles(clip_dir_for(meta_root, split, c)).values()
        )
        print(f"\n=== {split}: {len(df)} clips ===")
        print(f"frames/clip      : {sorted(set(df['frames']))}")
        print(
            f"tracklets/clip   : mean {df['n_tracks'].mean():.1f}  "
            f"(min {df['n_tracks'].min()}, max {df['n_tracks'].max()})"
        )
        print(f"avg track length : {df['avg_track_len'].mean():.0f} frames")
        print(
            f"gap tracks       : {df['n_gap_tracks'].sum()} of {df['n_tracks'].sum()} "
            f"({df['n_gap_tracks'].sum() / df['n_tracks'].sum():.1%})"
        )
        print(
            f"players/frame    : {df['avg_players'].mean():.2f}  "
            f"persons/frame: {df['avg_persons'].mean():.2f}"
        )
        print(f"roles (tracklets): {dict(roles.most_common())}")

    stats = pd.concat(all_rows, ignore_index=True)
    bad = stats[stats["max_gt_frame"] > stats["frames"]]
    print(f"\nclips with GT beyond seqLength: {list(bad['clip']) or 'none'}")

    # --- frames on disk (the detector project's copy is our frame source) ------------------
    frames_on_disk = {s: frames_per_clip(detector_images, s) for s in SPLITS}
    on_disk = sum(frames_on_disk.values(), Counter())
    expected = dict(zip(stats["clip"], stats["frames"], strict=True))
    short = {c: (on_disk.get(c, 0), n) for c, n in expected.items() if on_disk.get(c, 0) != n}
    print(f"clips whose frame count != seqLength: {short or 'none'}")

    # --- leakage check ---------------------------------------------------------------------
    detector_clips = {s: set(frames_on_disk[s]) for s in SPLITS}
    eval_clips = set(clips_by_split["val"]) | set(clips_by_split["test"])
    overlap = detector_clips["train"] & eval_clips

    print("\n=== leakage check ===")
    for s in SPLITS:
        print(
            f"detector {s:5}: {len(detector_clips[s])} clips   "
            f"evaluation {s:5}: {len(clips_by_split[s])} clips   "
            f"identical: {detector_clips[s] == set(clips_by_split[s])}"
        )
    print(f"detector train ∩ evaluation (val+test): {sorted(overlap) or 'none'}")
    print(
        "note: our val = the detector's val, which it used to pick its checkpoint -> val numbers "
        "may be slightly optimistic; final numbers come from test only (DD-007)."
    )

    # --- gate ------------------------------------------------------------------------------
    eval_stats = stats[stats["split"] != "train"]
    print("\n=== gate ===")
    print(f"avg simultaneous players (val+test): {eval_stats['avg_players'].mean():.2f}")
    print(f"overlap count: {len(overlap)}")


if __name__ == "__main__":
    main()
