"""One command for an uploaded video: boxes + ids + teams -> tracks.parquet and an annotated mp4.

Pass 1: ``model.track(stream=True)`` with the tuned BoT-SORT+GMC setting (DD-018/024), torso
colours taken from each frame as it streams past. Pass 2: KMeans on pitch-view frames, rolling
vote, goalkeeper rule, validated parquet. Then the video is read again and drawn on. The work
is done by ``ftt.pipeline.process_video`` (the Hugging Face demo calls the same function).
Outputs go to ``reports/<detector>/process_video/<video name>/`` (gitignored).

    uv run python scripts/process_video.py path/to/clip.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ftt.data.detections import detector_name
from ftt.detect.run_detector import load_detector
from ftt.pipeline import NoPitchViewError, process_video

ROOT = Path(__file__).parents[1]
TUNED = {"track_high_thresh": 0.25, "match_thresh": 0.9}  # the shared tuned setting (DD-018)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--tracker", default="botsort_gmc")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--no-render", action="store_true", help="skip the annotated video")
    args = ap.parse_args()

    paths = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text())
    team_cfg = yaml.safe_load((ROOT / "configs" / "team.yaml").read_text())
    tracker_cfg = {
        **yaml.safe_load((ROOT / "configs" / f"{args.tracker}.yaml").read_text()),
        **TUNED,
    }
    weights = (ROOT / paths["detector_weights"]).resolve()
    out_dir = (
        ROOT / paths["reports_dir"] / detector_name(weights) / "process_video" / args.video.stem
    )
    try:
        process_video(
            args.video,
            out_dir,
            model=load_detector(weights),
            team_cfg=team_cfg,
            tracker_cfg=tracker_cfg,
            device=args.device,
            render=not args.no_render,
        )
    except NoPitchViewError as e:
        raise SystemExit(str(e)) from e
    print("outputs in", out_dir)


if __name__ == "__main__":
    main()
