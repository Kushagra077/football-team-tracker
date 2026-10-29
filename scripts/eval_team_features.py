"""Which colour description separates two teams best? Mean Lab (current) vs alternatives.

Every variant sees the same torso crop and grass mask and goes through the same KMeans k=2 fitted
on the same frames; only the per-box feature differs. Ground-truth player boxes (no detector, no
tracker, no vote), scored every Nth frame with the permutation-invariant accuracy. The point is the
feature, so noise from the rest of the pipeline is left out.

Splits: ``train`` = the 46 clips that are neither val nor test (dev set: the colour clustering
learns nothing from labels, so these are free to experiment on); ``val`` to confirm; ``test`` once.

    uv run python scripts/eval_team_features.py --split train
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import yaml
from sklearn.cluster import KMeans

from ftt.data.detections import detector_name
from ftt.data.sn_tracking import frame_path, list_clips, load_persons_gt
from ftt.eval.team_metrics import best_permutation_accuracy
from ftt.paths import load_paths
from ftt.team.cluster import choose_fit_frames
from ftt.team.crops import grass_mask, torso_crop
from ftt.team.features import HIST_RANGE, histogram_feature, lab_pixels, mean_feature

ROOT = Path(__file__).parents[1]


def _hist(px: np.ndarray, bins: tuple[int, ...], dims: slice) -> np.ndarray:
    """Other bin layouts, for the sensitivity check (``hist_lab`` is the production one)."""
    rng = HIST_RANGE[dims]
    h, _ = np.histogramdd(px[:, dims], bins=bins, range=rng)
    return np.sqrt(h.ravel() / len(px))


VARIANTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean_lab": mean_feature,  # current default
    "mean_ab": lambda px: px.mean(axis=0)[1:],  # drop brightness (shadows)
    "median_lab": lambda px: np.median(px, axis=0),
    "mean_std": lambda px: np.concatenate([px.mean(axis=0), px.std(axis=0)]),
    "hist_lab": histogram_feature,  # candidate
    "hist_ab": lambda px: _hist(px, (8, 8), slice(1, 3)),
    "hist_coarse": lambda px: _hist(px, (3, 5, 5), slice(0, 3)),
    "hist_fine": lambda px: _hist(px, (5, 8, 8), slice(0, 3)),
}


def box_pixels(img: np.ndarray, box: np.ndarray, cfg: dict) -> np.ndarray | None:
    """Lab values (n, 3) of the non-grass torso pixels, or None if there are none."""
    crop = torso_crop(img, box, cfg)
    if crop.size == 0:
        return None
    keep = grass_mask(crop, cfg)
    if not keep.any():
        return None
    return lab_pixels(crop, keep)


def clip_scores(clip: str, split: str, cfg: dict, paths: dict, stride: int) -> dict | None:
    meta_root = ROOT / paths["meta_root"]
    folder = "test" if split == "test" else "train"
    gt = load_persons_gt(meta_root / folder / clip)
    pl = gt[gt["cls"] == "player"]
    fit = set(choose_fit_frames(pl["frame"].to_numpy(), cfg["cluster"]["fit_frames"], 3).tolist())
    have = set(pl["frame"].tolist())
    ev = {f for f in range(1, 751, stride) if f in have}
    images_root = (ROOT / paths["detector_images"]).resolve()
    feats = {n: ([], []) for n in VARIANTS}  # name -> (fit rows, eval rows)
    true = []
    for fr in sorted(fit | ev):
        img = cv2.imread(str(frame_path(images_root, split, clip, fr)))
        for r in pl[pl["frame"] == fr].itertuples():
            px = box_pixels(img, np.array([r.x1, r.y1, r.x2, r.y2]), cfg)
            if px is None:
                continue
            for n, fn in VARIANTS.items():
                if fr in fit:
                    feats[n][0].append(fn(px))
                if fr in ev:
                    feats[n][1].append(fn(px))
            if fr in ev:
                true.append(0 if r.team == "left" else 1)
    true = np.array(true)
    if len(true) < 20:
        return None
    return {
        n: best_permutation_accuracy(
            KMeans(2, n_init=10, random_state=0).fit(np.array(f)).predict(np.array(e)), true
        )
        for n, (f, e) in feats.items()
    } | {"scored": len(true)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--stride", type=int, default=25)
    args = ap.parse_args()
    paths = load_paths(ROOT)
    cfg = yaml.safe_load((ROOT / "configs" / "team.yaml").read_text())

    results = {}
    for clip in list_clips(ROOT / paths["meta_root"], args.split):
        s = clip_scores(clip, args.split, cfg, paths, args.stride)
        if s is None:
            continue
        results[clip] = s
        cells = "  ".join(f"{n} {s[n]:6.1%}" for n in VARIANTS)
        print(clip, f"{s['scored']:5d}", cells, flush=True)

    base = np.array([r["mean_lab"] for r in results.values()])
    head = f"{'variant':<11}{'mean':>7}{'worst':>8}{'<90%':>6}{'better':>8}{'worse':>7}"
    print(f"\n{head}  (vs mean_lab, +-0.5 pt)")
    for n in VARIANTS:
        a = np.array([r[n] for r in results.values()])
        print(
            f"{n:<11}{a.mean():7.1%}{a.min():8.1%}{(a < 0.9).sum():6d}"
            f"{(a > base + 0.005).sum():8d}{(a < base - 0.005).sum():7d}"
        )
    out = ROOT / paths["reports_dir"] / detector_name(paths["detector_weights"])
    (out / f"team_features_{args.split}.json").write_text(
        json.dumps({"args": vars(args), "clips": results}, indent=2)
    )


if __name__ == "__main__":
    main()
