"""Which frames are a normal wide pitch view? Team colours are only trusted there.

Close-ups put huge players in the frame: the fixed torso fractions and the colours learned on
distant pitch views no longer fit them (a red shirt came out as the other team's colour). A frame
counts as a pitch view when it shows enough players and the median player is small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PLAYER_CLS = 2


def pitch_view_frames(
    tracks: pd.DataFrame, frame_height: float, min_players: int, max_box_height: float
) -> np.ndarray:
    """Frame numbers with >= ``min_players`` player boxes whose median height is at most
    ``max_box_height`` x ``frame_height``."""
    p = tracks[tracks["cls"] == PLAYER_CLS]
    g = ((p["y2"] - p["y1"]) / frame_height).groupby(p["frame"])
    stats = pd.DataFrame({"n": g.size(), "median_height": g.median()})
    ok = (stats["n"] >= min_players) & (stats["median_height"] <= max_box_height)
    return stats.index[ok].to_numpy()
