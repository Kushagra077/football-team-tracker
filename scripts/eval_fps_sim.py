"""Does scaling the settings to the video's frame rate help? Simulate lower frame rates.

Drops frames of real SoccerNet val clips (every 2nd frame = 12.5 fps, every 3rd = 8.3 fps), runs the
upload-path pipeline with the settings unscaled and scaled (``ftt.scale``), and scores both against
the ground truth for the kept frames: tracking (HOTA, IDF1, ID switches) and team accuracy on the
GT-matched players, plus label flips. Lower rates only: there is no real 50/60 fps footage to test.

    uv run python scripts/eval_fps_sim.py --clips SNMOT-064 SNMOT-103 SNMOT-158 --steps 1 2 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from ftt.data.detections import detector_name
from ftt.data.sn_tracking import frame_path, load_persons_gt
from ftt.detect.run_detector import load_detector
from ftt.device import default_device
from ftt.eval.mot_io import write_mot
from ftt.eval.team_metrics import best_permutation_accuracy, match_to_gt
from ftt.eval.trackeval_wrap import evaluate
from ftt.export import build_tracks
from ftt.paths import load_paths
from ftt.scale import scale_team_cfg, scale_tracker_cfg
from ftt.team.assign import label_from_features, team_for_tracks
from ftt.team.gate import pitch_view_frames, small_boxes
from ftt.track.config import load_tracker_cfg
from ftt.video import track_video, video_info

ROOT = Path(__file__).parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--clips", nargs="+", default=["SNMOT-064", "SNMOT-103", "SNMOT-158"])
ap.add_argument("--steps", nargs="+", type=int, default=[1, 2, 3], help="keep every Nth frame")
args = ap.parse_args()
CLIPS, STEPS = args.clips, args.steps
paths = load_paths(ROOT)
CFG0 = yaml.safe_load((ROOT / "configs/team.yaml").read_text())
images = (ROOT / paths["detector_images"]).resolve()
meta = ROOT / paths["meta_root"]
OUT = ROOT / paths["reports_dir"] / detector_name(paths["detector_weights"]) / "fps_sim"
OUT.mkdir(parents=True, exist_ok=True)
model = load_detector((ROOT / paths["detector_weights"]).resolve())


def make_video(clip, step):
    out = OUT / f"{clip}_step{step}.mp4"
    if out.exists():
        return out
    n = int(load_persons_gt(meta / "train" / clip)["frame"].max())
    fr = list(range(1, n + 1, step))
    h, w = cv2.imread(str(frame_path(images, "val", clip, 1))).shape[:2]
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 25 / step, (w, h))
    for f in fr:
        vw.write(cv2.imread(str(frame_path(images, "val", clip, f))))
    vw.release()
    return out


def run(video, scale):
    fps, _, (w, h) = video_info(video)
    cfg = scale_team_cfg(CFG0, fps, w) if scale else CFG0
    tcfg = load_tracker_cfg(ROOT / "configs", "botsort_gmc")
    if scale:
        tcfg = scale_tracker_cfg(tcfg, fps)
    tcfg.pop("name", None)
    ty = OUT / f"trk_{scale}_{int(fps * 100)}.yaml"
    ty.write_text(yaml.safe_dump(tcfg))
    tv = track_video(model, video, ty, cfg, device=default_device())
    pv = cfg["pitch_view"]
    view = pitch_view_frames(tv.tracks, h, pv["min_players"], pv["max_box_height"])
    players = tv.tracks[
        (tv.tracks["cls"] == 2)
        & tv.tracks["frame"].isin(view)
        & small_boxes(tv.tracks, h, pv["max_box_height"])
    ]
    lab = label_from_features(
        players,
        tv.feats.loc[players.index.intersection(tv.feats.index)],
        cfg,
        max_colour_ratio=pv["max_colour_ratio"],
    )
    return build_tracks(tv.tracks, team_for_tracks(tv.tracks, lab, cfg, view), fps=tv.fps), fps


rows = []
pooled = {}
for clip in CLIPS:
    gt_all = load_persons_gt(meta / "train" / clip)
    for step in STEPS:
        video = make_video(clip, step)
        for scale in [False] if step == 1 else [False, True]:
            t0 = time.time()
            table, fps = run(video, scale)
            if step == 1:
                table.to_parquet(OUT / f"{clip}_full.parquet")
            keep = np.arange(1, int(gt_all["frame"].max()) + 1, step)
            g = gt_all[gt_all["frame"].isin(keep)].copy()
            g["frame"] = (g["frame"] - 1) // step + 1
            gdir = OUT / "mot" / "gt"
            pdir = OUT / "mot" / f"{clip}_{step}_{scale}"
            gdir.mkdir(parents=True, exist_ok=True)
            tag = f"{clip}_{step}_{scale}"
            one_gt = OUT / "mot" / f"gt_{tag}"
            one_gt.mkdir(exist_ok=True)
            write_mot(g, one_gt / f"{clip}.txt")
            pdir.mkdir(parents=True, exist_ok=True)
            write_mot(table.rename(columns={"frame_id": "frame"}), pdir / f"{clip}.txt")
            hota = evaluate(one_gt, pdir)
            # team accuracy on GT-matched player boxes (frames of this video)
            pl = table[table["role"] == "player"].reset_index(drop=True)
            true = np.full(len(pl), -1)
            gp = g[g["cls"] == "player"]
            for fr, idx in pl.groupby("frame_id").indices.items():
                gg = gp[gp["frame"] == fr].reset_index(drop=True)
                m = match_to_gt(
                    pl.loc[idx, ["x1", "y1", "x2", "y2"]].to_numpy(),
                    gg[["x1", "y1", "x2", "y2"]].to_numpy(),
                    0.5,
                )
                for i, j in zip(idx, m, strict=True):
                    if j >= 0:
                        true[i] = 0 if gg.loc[j, "team"] == "left" else 1
            sc = true >= 0
            has = pl["team"].notna().to_numpy()
            pred = pl["team"].astype("float").to_numpy()
            ev = sc & has
            acc = best_permutation_accuracy(pred[ev].astype(int), true[ev])
            flips = 0
            for _tid, d in pl[has].sort_values("frame_id").groupby("track_id"):
                flips += int((np.diff(d["team"].astype(int).to_numpy()) != 0).sum())
            r = dict(
                clip=clip,
                step=step,
                fps=round(fps, 2),
                scaled=scale,
                secs=round(time.time() - t0),
                HOTA=round(hota["HOTA"], 1),
                IDF1=round(hota["IDF1"], 1),
                IDSW=int(hota["IDSW"]),
                ids=int(pl["track_id"].nunique()),
                team_acc=round(100 * acc, 1),
                labelled=round(100 * ev.sum() / max(sc.sum(), 1), 1),
                flips=flips,
                n=int(ev.sum()),
            )
            rows.append(r)
            print(r, flush=True)
json.dump(rows, open(OUT / "results.json", "w"), indent=1)
