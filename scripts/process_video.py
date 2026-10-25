"""One command for an uploaded video: boxes + ids + teams -> tracks.parquet and an annotated mp4.

Pass 1: ``model.track(stream=True)`` with the tuned BoT-SORT+GMC setting (DD-018/024), torso
colours taken from each frame as it streams past. Pass 2: KMeans on pitch-view frames, rolling
vote, goalkeeper rule, validated parquet. Then the video is read again and drawn on.
Outputs go to ``reports/<detector>/process_video/<video name>/`` (gitignored).

    uv run python scripts/process_video.py path/to/clip.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from ftt.data.detections import detector_name
from ftt.detect.run_detector import load_detector
from ftt.export import build_tracks, export_tracks, read_tracks
from ftt.render import render_video
from ftt.scale import scale_team_cfg, scale_tracker_cfg
from ftt.team.assign import label_from_features, team_for_tracks
from ftt.team.gate import pitch_view_frames, small_boxes
from ftt.video import track_video, video_info

ROOT = Path(__file__).parents[1]
PLAYER_CLS = 2
TUNED = {"track_high_thresh": 0.25, "match_thresh": 0.9}  # the shared tuned setting (DD-018)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--tracker", default="botsort_gmc")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--no-render", action="store_true", help="skip the annotated video")
    args = ap.parse_args()

    paths = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text())
    cfg = yaml.safe_load((ROOT / "configs" / "team.yaml").read_text())
    weights = (ROOT / paths["a1_weights"]).resolve()
    out_dir = (
        ROOT / paths["reports_dir"] / detector_name(weights) / "process_video" / args.video.stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Settings are counted in frames / pixels of a 25 fps, 1920 px video: rescale for this one.
    src_fps, _, (src_w, _) = video_info(args.video)
    cfg = scale_team_cfg(cfg, src_fps, src_w)
    tcfg = {**yaml.safe_load((ROOT / "configs" / f"{args.tracker}.yaml").read_text()), **TUNED}
    tcfg = scale_tracker_cfg(tcfg, src_fps)
    tcfg.pop("name", None)  # model.track reads a plain Ultralytics tracker yaml
    tracker_yaml = out_dir / f"{args.tracker}_tuned.yaml"
    tracker_yaml.write_text(yaml.safe_dump(tcfg))

    t0 = time.time()
    tv = track_video(load_detector(weights), args.video, tracker_yaml, cfg, device=args.device)
    t1 = time.time()
    w, h = tv.size
    n_ids = tv.tracks["track_id"].nunique()
    print(f"pass 1: {tv.n_frames} frames at {tv.fps:g} fps, {w}x{h}, {len(tv.tracks)} boxes, "
          f"{n_ids} tracker ids, {t1 - t0:.0f}s")  # fmt: skip
    pv = cfg["pitch_view"]
    view = pitch_view_frames(tv.tracks, tv.size[1], pv["min_players"], pv["max_box_height"])
    small = small_boxes(tv.tracks, tv.size[1], pv["max_box_height"])
    players = tv.tracks[(tv.tracks["cls"] == PLAYER_CLS) & tv.tracks["frame"].isin(view) & small]
    if players.empty:
        raise SystemExit("no wide pitch view with players found: is this football footage?")
    print(f"pitch-view frames (teams labelled only here): {len(view)} of {tv.n_frames} "
          f"({len(view) / tv.n_frames:.0%})")  # fmt: skip

    labelled = label_from_features(
        players,
        tv.feats.loc[players.index.intersection(tv.feats.index)],
        cfg,
        max_colour_ratio=pv["max_colour_ratio"],
    )
    table = build_tracks(tv.tracks, team_for_tracks(tv.tracks, labelled, cfg, view), fps=tv.fps)
    parquet = export_tracks(table, out_dir / "tracks.parquet", fps=tv.fps)
    back = read_tracks(parquet, fps=tv.fps)
    t2 = time.time()
    roles = back["role"].value_counts().to_dict()
    teams = back["team"].value_counts(dropna=False).to_dict()
    print(f"pass 2: wrote {parquet.name}: {len(back)} rows, roles {roles}, team {teams}, "
          f"{t2 - t1:.0f}s")  # fmt: skip

    if not args.no_render:
        video_out = render_video(args.video, back, out_dir / "annotated.mp4", tv.fps)
        print(f"render: wrote {video_out.name}, {time.time() - t2:.0f}s")
    print("outputs in", out_dir)


if __name__ == "__main__":
    main()
