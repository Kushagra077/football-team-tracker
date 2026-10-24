"""Team labels for one clip's tracked players: torso colour -> KMeans -> rolling vote."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ftt.team.cluster import choose_fit_frames, fit_team_clusters
from ftt.team.features import box_features
from ftt.team.vote import vote_labels


def label_players(
    tracks: pd.DataFrame, frame_loader: Callable[[int], np.ndarray], cfg: dict
) -> pd.DataFrame:
    """Per-frame cluster label (``raw``) and voted label (``team``) for tracked PLAYER rows.

    ``tracks`` needs frame, track_id, x1, y1, x2, y2 (players only: keepers and referees would
    pull the centroids apart). KMeans fits on the first ``fit_frames`` frames that look like a
    pitch view and never sees a label. Cluster ids are arbitrary per clip. Rows whose crop is
    empty or all grass are dropped. Output is sorted by (frame, track_id).
    """
    feats = box_features(tracks, frame_loader, cfg)
    out = feats.join(tracks["track_id"]).sort_values(["frame", "track_id"]).reset_index(drop=True)
    lab = out[["L", "a", "b"]].to_numpy()
    c = cfg["cluster"]
    fit_frames = choose_fit_frames(
        tracks["frame"].to_numpy(), c["fit_frames"], c["fit_min_players"]
    )
    km = fit_team_clusters(lab[np.isin(out["frame"].to_numpy(), fit_frames)])
    raw = km.predict(lab)
    v = cfg["vote"]
    team = vote_labels(
        out["track_id"].to_numpy(), out["frame"].to_numpy(), raw, v["window"], v["lock_min_agree"]
    )
    return out.assign(raw=raw, team=team)
