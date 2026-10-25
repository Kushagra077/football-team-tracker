"""One clip, end to end, to ``tracks.parquet`` in A2's schema.

cached A1 boxes -> tracker (BoT-SORT + GMC, tuned shared setting) -> team labels for the
tracked players (torso colour, KMeans, rolling vote) -> ``build_tracks`` -> validated parquet.
Referees get no team; goalkeepers get the keeper rule's team or null when it cannot tell; pitch and
jersey columns are null. The file goes under ``reports/<detector>/exports/`` (gitignored).

    uv run python scripts/export_clip.py --split val --clip SNMOT-064
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from ftt.data.detections import detector_name, load_cached_detections
from ftt.data.sn_tracking import frame_path, load_persons_gt
from ftt.export import build_tracks, export_tracks, read_tracks
from ftt.team.assign import label_players, team_for_tracks
from ftt.track.adapters import UltralyticsTracker
from ftt.track.runner import tracks_for_clip

ROOT = Path(__file__).parents[1]
PLAYER_CLS = 2
GK_CLS = 1
TUNED = {"track_high_thresh": 0.25, "match_thresh": 0.9}  # the shared tuned setting (DD-018)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--tracker", default="botsort_gmc")
    args = ap.parse_args()

    paths = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text())
    cfg = yaml.safe_load((ROOT / "configs" / "team.yaml").read_text())
    images_root = (ROOT / paths["a1_images"]).resolve()
    folder = "test" if args.split == "test" else "train"
    detector = detector_name(paths["a1_weights"])
    n_frames = int(load_persons_gt(ROOT / paths["meta_root"] / folder / args.clip)["frame"].max())

    def loader(f):
        return cv2.imread(str(frame_path(images_root, args.split, args.clip, f)))

    tcfg = {**yaml.safe_load((ROOT / "configs" / f"{args.tracker}.yaml").read_text()), **TUNED}
    tracks = tracks_for_clip(
        ROOT
        / paths["cache_dir"]
        / "tracks"
        / detector
        / f"{args.clip}__{args.tracker}__tuned.parquet",
        UltralyticsTracker(tcfg),
        lambda: load_cached_detections(ROOT / paths["cache_dir"], detector, args.clip),
        n_frames,
        loader,
    )

    labelled = label_players(
        tracks[tracks["cls"] == PLAYER_CLS].reset_index(drop=True), loader, cfg
    )
    table = build_tracks(tracks, team_for_tracks(tracks, labelled, cfg))

    out = ROOT / paths["reports_dir"] / detector / "exports" / f"{args.clip}__tracks.parquet"
    export_tracks(table, out)
    back = read_tracks(out)  # re-validates what is on disk
    print(f"wrote {out}")
    n_tracks, n_frames_out = back["track_id"].nunique(), back["frame_id"].nunique()
    print(f"rows {len(back)}, tracks {n_tracks}, frames {n_frames_out}")
    print("rows by role:", back["role"].value_counts().to_dict())
    print(
        "team (null = referee, keeper, or no usable crop):",
        back["team"].value_counts(dropna=False).to_dict(),
    )
    print(
        "pitch/jersey all null:",
        bool(back[["x_pitch", "y_pitch", "jersey_number"]].isna().all().all()),
    )


if __name__ == "__main__":
    main()
