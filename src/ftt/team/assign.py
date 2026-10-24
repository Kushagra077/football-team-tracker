"""Team labels for one clip's tracked players: torso colour -> KMeans -> rolling vote."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ftt.team.cluster import (
    assign_goalkeepers,
    choose_fit_frames,
    far_from_both_teams,
    fit_team_clusters,
)
from ftt.team.features import box_features
from ftt.team.vote import vote_labels


def label_players(
    tracks: pd.DataFrame, frame_loader: Callable[[int], np.ndarray], cfg: dict
) -> pd.DataFrame:
    """Per-frame cluster label (``raw``) and voted label (``team``) for tracked PLAYER rows.

    ``tracks`` needs frame, track_id, x1, y1, x2, y2 (players only: keepers and referees would
    pull the centroids apart). Reads the frames itself; see ``label_from_features`` when the
    colours were already computed (streaming a video).
    """
    return label_from_features(tracks, box_features(tracks, frame_loader, cfg), cfg)


def label_from_features(
    tracks: pd.DataFrame,
    feats: pd.DataFrame,
    cfg: dict,
    max_colour_ratio: float | None = None,
) -> pd.DataFrame:
    """Same as ``label_players`` from ``box_features`` output (indexed like ``tracks``).

    KMeans fits on the first ``fit_frames`` frames that look like a pitch view and never sees a
    label. Cluster ids are arbitrary per clip. Rows whose crop is empty or all grass are dropped.
    With ``max_colour_ratio`` set, a colour far from both teams is labelled -1 (no team) before
    the vote, so a referee or keeper the detector called "player" ends up with no team.
    Output is sorted by (frame, track_id).
    """
    out = feats.join(tracks["track_id"]).sort_values(["frame", "track_id"]).reset_index(drop=True)
    lab = out[["L", "a", "b"]].to_numpy()
    c = cfg["cluster"]
    fit_frames = choose_fit_frames(
        tracks["frame"].to_numpy(), c["fit_frames"], c["fit_min_players"]
    )
    km = fit_team_clusters(lab[np.isin(out["frame"].to_numpy(), fit_frames)])
    raw = km.predict(lab)
    if max_colour_ratio is not None:
        raw[far_from_both_teams(lab, km.cluster_centers_, max_colour_ratio)] = -1
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


PLAYER_CLS, KEEPER_CLS = 2, 1  # A1 class ids


def team_for_tracks(
    tracks: pd.DataFrame, labelled: pd.DataFrame, cfg: dict, frames: np.ndarray | None = None
) -> np.ndarray:
    """One team value per ROW of ``tracks`` (all classes): 0/1, or NaN where there is none.

    Players get their voted label (NaN if their crop was unusable), keepers the keeper rule's
    decision per track (NaN if undecided), referees and anything else NaN. With ``frames``
    (e.g. the pitch-view frames) only those frames count: keepers are judged from them alone and
    every other row gets NaN.
    """
    merged = tracks.merge(
        labelled[["frame", "track_id", "team"]], on=["frame", "track_id"], how="left"
    )
    team = merged["team"].astype("float64").to_numpy(copy=True)
    team[team < 0] = np.nan  # -1: colour fits neither team
    team[(merged["cls"] != PLAYER_CLS).to_numpy()] = np.nan
    in_view = (
        np.ones(len(merged), bool) if frames is None else merged["frame"].isin(frames).to_numpy()
    )
    keepers = tracks[
        (tracks["cls"] == KEEPER_CLS)
        & (in_view if frames is None else tracks["frame"].isin(frames))
    ]
    gk = label_goalkeepers(keepers, labelled, cfg).set_index("track_id")["team"].astype("float64")
    is_gk = (merged["cls"] == KEEPER_CLS).to_numpy()
    team[is_gk] = merged.loc[is_gk, "track_id"].map(gk).to_numpy()
    team[~in_view] = np.nan
    return team
