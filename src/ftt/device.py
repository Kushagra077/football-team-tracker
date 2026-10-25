"""Pick the best available torch device, so nothing is hard-coded to one machine."""

from __future__ import annotations

import torch


def default_device() -> str:
    """``cuda`` if there is an NVIDIA GPU, ``mps`` on Apple silicon, otherwise ``cpu``."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
