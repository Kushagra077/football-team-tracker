"""Per-clip KMeans k=2 on PLAYER detections only (first ~50 frames), plus the goalkeeper heuristic."""

from __future__ import annotations

import numpy as np


def fit_team_clusters(features: np.ndarray, k: int = 2):
    raise NotImplementedError


def assign_goalkeepers(gk_centres_x: np.ndarray, team_centroids_x: np.ndarray) -> np.ndarray:
    """Keeper goes to the team whose players' mean image-x is nearest. Crude; A3 fixes it."""
    raise NotImplementedError
