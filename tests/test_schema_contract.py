"""Step 7 gate: tracks.parquet is accepted by B1's loader/validator.

Heads-up (design_decision.md DD-003): B1's ``ftm.schema.validate()`` currently rejects null
x_pitch/y_pitch and requires period/is_gk/ball_state. Resolve that before writing this test.
"""

import pytest


@pytest.mark.skip(reason="step 7: needs export.py and the B1-contract decision (DD-003)")
def test_tracks_parquet_passes_b1_contract(): ...


@pytest.mark.skip(reason="step 7")
def test_pitch_and_jersey_columns_are_null_not_zero(): ...
