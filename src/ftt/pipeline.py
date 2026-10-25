"""One call for an uploaded video: detect + track + team labels -> table, parquet, annotated mp4.

``scripts/process_video.py`` and the Hugging Face demo both call this, so the demo runs exactly
the code that is tested here. Settings come in as plain dicts (the unscaled configs); they are
scaled to the video's frame rate and width inside (``ftt.scale``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from ftt.device import default_device
from ftt.export import build_tracks, export_tracks, read_tracks
from ftt.render import render_video
from ftt.scale import scale_team_cfg, scale_tracker_cfg
from ftt.team.assign import label_from_features, team_for_tracks
from ftt.team.cluster import TooFewColoursError
from ftt.team.gate import pitch_view_frames, small_boxes
from ftt.video import track_video, video_info

PLAYER_CLS = 2


class NoPitchViewError(ValueError):
    """The video has no wide view with enough players: probably not football footage."""


@dataclass
class ProcessedVideo:
    table: pd.DataFrame  # the validated output table (same as tracks.parquet)
    parquet: Path
    annotated: Path | None  # None when ``render=False``
    fps: float
    n_frames: int
    size: tuple[int, int]  # (width, height)
    pitch_view_share: float  # share of frames where team colours are trusted


def process_video(
    video: Path,
    out_dir: Path,
    *,
    model,
    team_cfg: dict,
    tracker_cfg: dict,
    device: str = default_device(),
    render: bool = True,
    max_frames: int | None = None,
    log: Callable[[str], None] = print,
) -> ProcessedVideo:
    """Run the whole upload pipeline on ``video``; write ``tracks.parquet`` (+ ``annotated.mp4``).

    ``model`` is a loaded detector (``detect.run_detector.load_detector``); ``team_cfg`` is
    ``configs/team.yaml`` and ``tracker_cfg`` a tracker config with the tuned thresholds applied,
    both unscaled. ``max_frames`` stops after that many frames (a demo cap).
    """
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Settings are counted in frames / pixels of a 25 fps, 1920 px video: rescale for this one.
    src_fps, _, (src_w, _) = video_info(video)
    cfg = scale_team_cfg(team_cfg, src_fps, src_w)
    tcfg = scale_tracker_cfg(tracker_cfg, src_fps)
    tcfg.pop("name", None)  # model.track reads a plain Ultralytics tracker yaml
    tracker_yaml = out_dir / "tracker_tuned.yaml"
    tracker_yaml.write_text(yaml.safe_dump(tcfg))

    t0 = time.time()
    tv = track_video(model, video, tracker_yaml, cfg, device=device, max_frames=max_frames)
    t1 = time.time()
    w, h = tv.size
    log(
        f"pass 1: {tv.n_frames} frames at {tv.fps:g} fps, {w}x{h}, {len(tv.tracks)} boxes, "
        f"{tv.tracks['track_id'].nunique()} tracker ids, {t1 - t0:.0f}s"
    )
    pv = cfg["pitch_view"]
    view = pitch_view_frames(tv.tracks, h, pv["min_players"], pv["max_box_height"])
    small = small_boxes(tv.tracks, h, pv["max_box_height"])
    players = tv.tracks[(tv.tracks["cls"] == PLAYER_CLS) & tv.tracks["frame"].isin(view) & small]
    if players.empty:
        raise NoPitchViewError("no wide pitch view with players found: is this football footage?")
    share = len(view) / tv.n_frames
    log(f"pitch-view frames (teams labelled only here): {len(view)} of {tv.n_frames} ({share:.0%})")

    try:
        labelled = label_from_features(
            players,
            tv.feats.loc[players.index.intersection(tv.feats.index)],
            cfg,
            max_colour_ratio=pv["max_colour_ratio"],
        )
    except TooFewColoursError as e:  # players were found, but their crops were empty or all grass
        raise NoPitchViewError(f"cannot tell two teams apart: {e}") from e
    table = build_tracks(tv.tracks, team_for_tracks(tv.tracks, labelled, cfg, view), fps=tv.fps)
    parquet = export_tracks(table, out_dir / "tracks.parquet", fps=tv.fps)
    back = read_tracks(parquet, fps=tv.fps)
    t2 = time.time()
    roles = back["role"].value_counts().to_dict()
    teams = back["team"].value_counts(dropna=False).to_dict()
    log(
        f"pass 2: wrote {parquet.name}: {len(back)} rows, roles {roles}, team {teams}, "
        f"{t2 - t1:.0f}s"
    )

    annotated = None
    if render:
        annotated = render_video(
            video, back, out_dir / "annotated.mp4", tv.fps, max_frames=max_frames
        )
        log(f"render: wrote {annotated.name}, {time.time() - t2:.0f}s")
    return ProcessedVideo(back, parquet, annotated, tv.fps, tv.n_frames, tv.size, share)
