"""Step 4: run A1's detector once per clip and cache the boxes (cache/detections/<clip>.parquet).

uv run python scripts/cache_detections.py --split val
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from ftt.data.detections import cached_path
from ftt.data.sn_tracking import list_clips
from ftt.detect.run_detector import frame_number, load_detector, run_detector

ROOT = Path(__file__).parents[1]


def clip_frames(images_root: Path, split: str, clip: str) -> list[Path]:
    """All frames of a clip in order. Fails loud unless they are exactly 1..N with no gaps."""
    frames = sorted((images_root / split).glob(f"{clip}_*.jpg"))
    numbers = [frame_number(p) for p in frames]
    if not frames or numbers != list(range(1, len(frames) + 1)):
        raise ValueError(f"{clip}: frames are missing or not numbered 1..N ({len(frames)} found)")
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--force", action="store_true", help="re-run clips that are already cached")
    args = ap.parse_args()

    paths = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text())
    weights = (ROOT / paths["a1_weights"]).resolve()
    images_root = (ROOT / paths["a1_images"]).resolve()
    cache_dir = ROOT / paths["cache_dir"]
    clips = list_clips(ROOT / paths["meta_root"], args.split)[: args.limit]

    model = load_detector(weights)  # once, not per clip
    print(f"{weights.name} on {args.device}, conf {args.conf}, {len(clips)} {args.split} clips")
    start = time.perf_counter()
    for clip in clips:
        out = cached_path(cache_dir, clip)
        if out.exists() and not args.force:
            print(f"  {clip}: cached, skipping")
            continue
        t = time.perf_counter()
        frames = clip_frames(images_root, args.split, clip)
        run_detector(frames, model, out, conf=args.conf, device=args.device)
        print(f"  {clip}: {len(frames)} frames in {time.perf_counter() - t:.1f}s", flush=True)
    print(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
