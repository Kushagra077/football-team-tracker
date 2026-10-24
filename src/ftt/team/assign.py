"""Team labels for one clip's tracked players: torso colour -> KMeans -> rolling vote."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ftt.team.cluster import assign_goalkeepers, choose_fit_frames, fit_team_clusters
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


def label_goalkeepers(keepers: pd.DataFrame, players: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Team (0/1, or NA when undecided) per goalkeeper TRACK.

    ``keepers``: tracked goalkeeper rows (track_id, x1, x2). ``players``: ``label_players`` output
    (voted ``team``, x1, x2). Each team's position is the median image-x of its players over the
    whole clip, each keeper's the median over his track; the rule itself is ``assign_goalkeepers``.
    """
    g = cfg["goalkeeper"]

    def centre_x(d: pd.DataFrame) -> pd.Series:
        return (d["x1"] + d["x2"]) / 2

    team_x = np.array([centre_x(players[players["team"] == t]).median() for t in (0, 1)])
    keeper_x = centre_x(keepers).groupby(keepers["track_id"]).median()
    if len(keeper_x) == 0 or np.isnan(team_x).any():
        pred = np.full(len(keeper_x), -1)
    else:
        pred = assign_goalkeepers(keeper_x.to_numpy(), team_x, g["min_team_gap"], g["min_margin"])
    team = pd.array(np.where(pred < 0, np.nan, pred), dtype="Int8")
    return pd.DataFrame({"track_id": keeper_x.index.to_numpy(), "team": team})
