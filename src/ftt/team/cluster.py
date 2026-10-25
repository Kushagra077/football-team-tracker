"""Per-clip KMeans k=2 on PLAYER detections only (first ~50 frames).

Also holds the goalkeeper heuristic.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def choose_fit_frames(box_frames: np.ndarray, fit_n: int, min_players: int) -> np.ndarray:
    """The first ``fit_n`` frames that show at least ``min_players`` player boxes.

    Frames 1..fit_n would do for clips that open on the pitch, but a clip can open on a
    close-up or replay (SNMOT-137: no players until frame 187); colours learned there are junk.
    ``box_frames`` holds one frame number per player box.
    """
    frames, counts = np.unique(box_frames, return_counts=True)
    return frames[counts >= min_players][:fit_n]


class TooFewColoursError(ValueError):
    """Fewer usable shirt colours than teams: KMeans cannot split them."""


def fit_team_clusters(features: np.ndarray, k: int = 2) -> KMeans:
    """Fit KMeans on an (n, d) colour feature matrix (d=3 mean Lab, d=144 Lab histogram).

    n_init restarts + fixed seed keep results repeatable.

    Cluster ids are arbitrary: cluster 0 is not 'team A'. Score with a permutation-invariant metric.
    """
    if len(features) < k:
        raise TooFewColoursError(
            f"only {len(features)} usable shirt colour(s) in the frames used to learn the teams, "
            f"need at least {k}"
        )
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit(features)


def far_from_both_teams(features: np.ndarray, centres: np.ndarray, max_ratio: float) -> np.ndarray:
    """True where a colour fits neither team: a referee or keeper the detector called "player".

    Distance to the nearest of the two centroids, divided by the distance between them, so the
    rule does not depend on how different the two kits are. On val GT (DD-028) players sit at a
    median ratio of 0.24, referees and keepers at about 1.1.
    """
    gap = np.linalg.norm(centres[0] - centres[1])
    nearest = np.linalg.norm(features[:, None, :] - centres[None], axis=2).min(axis=1)
    return nearest > max_ratio * gap


def assign_goalkeepers(
    gk_centres_x: np.ndarray,
    team_centroids_x: np.ndarray,
    min_team_gap: float,
    min_margin: float,
) -> np.ndarray:
    """Keeper goes to the team whose players' mean image-x is nearest. Crude: pitch coords fix it.

    Returns 0/1 per keeper, or -1 (unknown) when the evidence is weak: the two teams' average
    positions are closer than ``min_team_gap`` (everyone bunched up), or the keeper is less than
    ``min_margin`` nearer to one team than to the other (standing between them). Pixels.
    """
    t0, t1 = team_centroids_x
    out = np.full(len(gk_centres_x), -1)
    if abs(t0 - t1) < min_team_gap:
        return out
    d0, d1 = np.abs(gk_centres_x - t0), np.abs(gk_centres_x - t1)
    sure = np.abs(d0 - d1) >= min_margin
    out[sure] = (d1 < d0)[sure].astype(int)
    return out
