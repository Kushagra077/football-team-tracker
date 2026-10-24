"""Per-clip KMeans k=2 on PLAYER detections only (first ~50 frames).

Also holds the goalkeeper heuristic.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def fit_team_clusters(features: np.ndarray, k: int = 2) -> KMeans:
    """Fit KMeans on (n, 3) Lab colours. n_init restarts + fixed seed keep results repeatable.

    Cluster ids are arbitrary: cluster 0 is not 'team A'. Score with a permutation-invariant metric.
    """
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit(features)


def assign_goalkeepers(
    gk_centres_x: np.ndarray,
    team_centroids_x: np.ndarray,
    min_team_gap: float,
    min_margin: float,
) -> np.ndarray:
    """Keeper goes to the team whose players' mean image-x is nearest. Crude; A3 fixes it.

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
