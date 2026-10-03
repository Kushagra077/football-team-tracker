"""Step 2 gate: the evaluator must be trustworthy BEFORE any tracker is blamed."""

import pytest


@pytest.mark.skip(reason="step 2: needs eval harness")
def test_gt_vs_gt_is_perfect():
    """GT submitted as prediction -> HOTA 100, IDSW 0 (failure usually = frame-indexing bug)."""


@pytest.mark.skip(reason="step 2: needs eval harness")
def test_shuffled_ids_degrade():
    """GT with IDs shuffled every 50 frames -> HOTA drops, IDSW spikes."""
