"""Does the tactical-metrics project's contract (``ftm.schema.validate``) accept our tracks?

Not yet, and this test pins down exactly why. Its canonical frame needs real pitch coordinates
(``x_pitch``/``y_pitch`` may not be null) plus ``period``, ``is_gk``, ``ball_state`` and a
home/away/ball team, and no row may lack a team. Two checks:

* with our real output, validation fails on the pitch columns and nothing else (the gap is A3's);
* once the pitch columns are filled (placeholder metres here), the same frame passes, so the
  rest of the output (ids, timestamps, teams, goalkeeper flag, dtypes) already fits.

Skipped when ``ftm`` is not installed; CI installs it from a pinned commit of the public repo:
    uv run --with ../football-tactical-metrics python -m pytest tests/test_b1_contract.py
"""

import numpy as np
import pandas as pd
import pytest

ftm_schema = pytest.importorskip("ftm.schema")

from ftt.data.detections import CLASS_IDS  # noqa: E402
from ftt.export import build_tracks, export_tracks, read_tracks  # noqa: E402


def _our_tracks(tmp_path) -> pd.DataFrame:
    """Two frames: a home player, an away player, a goalkeeper and a referee (no team)."""
    cls = [CLASS_IDS["player"], CLASS_IDS["player"], CLASS_IDS["goalkeeper"], CLASS_IDS["referee"]]
    tracked = pd.DataFrame(
        {
            "frame": [1] * 4 + [2] * 4,
            "track_id": [1, 2, 3, 4] * 2,
            "x1": [100.0, 900.0, 1500.0, 500.0] * 2, "y1": [200.0, 300.0, 400.0, 500.0] * 2,
            "x2": [140.0, 940.0, 1540.0, 540.0] * 2, "y2": [300.0, 400.0, 500.0, 600.0] * 2,
            "conf": [0.9] * 8,
            "cls": cls * 2,
        }
    )  # fmt: skip
    team = [0, 1, 0, np.nan] * 2
    return read_tracks(export_tracks(build_tracks(tracked, team=team), tmp_path / "t.parquet"))


def _to_b1(ours: pd.DataFrame, pitch: bool) -> pd.DataFrame:
    """Our table -> B1's canonical frame. Rows without a team (referees, unsure keepers) are
    dropped because B1 allows no null team; with ``pitch`` the pitch columns get placeholders."""
    d = ours[ours["team"].notna()].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "frame_id": d["frame_id"],
            "period": 1,
            "timestamp": d["timestamp"],
            "track_id": d["track_id"].astype(str),
            "team": np.where(d["team"] == 0, "home", "away"),
            "jersey_number": d["jersey_number"],
            "x_pitch": d["x_pitch"],
            "y_pitch": d["y_pitch"],
            "is_ball": False,
            "is_gk": (d["role"] == "goalkeeper").to_numpy(),
            "ball_state": "alive",
        }
    )
    if pitch:  # stand-in metres from the box centre: only the contract's shape is under test
        out["x_pitch"] = ((d["x1"] + d["x2"]) / 2 / 1920 - 0.5) * 90
        out["y_pitch"] = ((d["y1"] + d["y2"]) / 2 / 1080 - 0.5) * 60
    return ftm_schema.coerce(out)


def test_without_pitch_coordinates_only_the_pitch_columns_stand_in_the_way(tmp_path):
    with pytest.raises(ftm_schema.SchemaError, match="x_pitch"):
        ftm_schema.validate(_to_b1(_our_tracks(tmp_path), pitch=False))


def test_with_pitch_coordinates_filled_in_our_output_passes_the_contract(tmp_path):
    frame = ftm_schema.validate(_to_b1(_our_tracks(tmp_path), pitch=True))
    assert len(frame) == 6  # the referee rows have no team and are left out
    assert frame["is_gk"].sum() == 2 and set(frame["team"]) == {"home", "away"}
