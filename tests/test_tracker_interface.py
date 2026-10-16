"""Every adapter obeys ``track/base.Tracker`` (input (N,6), output (M,7), reset() works)."""

import pytest


@pytest.mark.skip(reason="step 3: needs adapters")
def test_adapters_obey_interface(): ...
