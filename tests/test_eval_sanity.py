"""The evaluator must be trustworthy BEFORE any tracker is blamed."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ftt.data.sn_tracking import list_clips, load_persons_gt
from ftt.eval.mot_io import read_mot, write_mot
from ftt.eval.trackeval_wrap import evaluate


def synthetic_gt(n_tracks=10, n_frames=200, seed=0) -> pd.DataFrame:
    """Players drifting across a 1920x1080 frame, all visible the whole clip."""
    rng = np.random.default_rng(seed)
    rows = []
    for tid in range(1, n_tracks + 1):
        x0, y0 = rng.uniform(100, 1500), rng.uniform(100, 800)
        vx, vy = rng.uniform(-2, 2, 2)
        for f in range(1, n_frames + 1):
            x, y = x0 + vx * f, y0 + vy * f
            rows.append((f, tid, x, y, x + 40, y + 100, 1.0))
    return pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])


def shuffle_ids(df: pd.DataFrame, every: int = 50, seed: int = 1) -> pd.DataFrame:
    """Permute track ids among the boxes at the start of every ``every``-frame block."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    ids = sorted(df["track_id"].unique())
    for block in range(df["frame"].max() // every + 1):
        sel = (out["frame"] - 1) // every == block
        perm = dict(zip(ids, rng.permutation(ids), strict=True))
        out.loc[sel, "track_id"] = out.loc[sel, "track_id"].map(perm)
    return out


def run(tmp_path: Path, gt: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    write_mot(gt, tmp_path / "gt" / "clip.txt")
    write_mot(pred, tmp_path / "pred" / "clip.txt")
    return evaluate(tmp_path / "gt", tmp_path / "pred")


def test_mot_round_trip_is_one_indexed(tmp_path):
    gt = synthetic_gt(2, 5)
    write_mot(gt, tmp_path / "a.txt")
    back = read_mot(tmp_path / "a.txt")
    assert back["frame"].min() == 1
    assert np.allclose(
        back["x"] + back["w"], gt.sort_values(["frame", "track_id"])["x2"], atol=0.01
    )
    with pytest.raises(ValueError):
        write_mot(gt.assign(frame=gt["frame"] - 1), tmp_path / "b.txt")


def test_gt_vs_gt_is_perfect(tmp_path):
    """GT submitted as prediction -> HOTA 100, IDSW 0 (failure usually = frame-indexing bug)."""
    gt = synthetic_gt()
    res = run(tmp_path, gt, gt)
    assert res["HOTA"] == pytest.approx(100, abs=0.1)
    assert res["IDF1"] == pytest.approx(100, abs=0.1)
    assert res["IDSW"] == 0


def test_shuffled_ids_degrade(tmp_path):
    """GT with IDs shuffled every 50 frames -> HOTA drops, IDSW spikes."""
    gt = synthetic_gt()
    res = run(tmp_path, gt, shuffle_ids(gt))
    assert res["HOTA"] < 80
    assert res["AssA"] < res["DetA"]  # detection stays perfect, association is what breaks
    assert res["DetA"] == pytest.approx(100, abs=0.1)
    assert res["IDSW"] > 10


def test_missing_prediction_file_counts_as_misses(tmp_path):
    write_mot(synthetic_gt(3, 20), tmp_path / "gt" / "clip.txt")
    (tmp_path / "pred").mkdir()
    res = evaluate(tmp_path / "gt", tmp_path / "pred")
    assert res["HOTA"] == 0


@pytest.mark.integration
def test_real_clip_gt_vs_gt(tmp_path):
    root = Path(__file__).parents[1] / "data" / "raw_meta"
    if not root.exists():
        pytest.skip("no SoccerNet metadata")
    clip = list_clips(root, "val")[0]
    folder = "train"
    gt = load_persons_gt(root / folder / clip)
    res = run(tmp_path, gt, gt)
    assert res["HOTA"] == pytest.approx(100, abs=0.1)
    assert res["IDSW"] == 0
