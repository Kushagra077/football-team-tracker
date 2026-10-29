"""Read ``configs/paths.yaml``; the detector can be switched without editing the file.

``paths.yaml`` pins one detector (``detector_weights``). To run any script with another model from
the same folder, set ``FTT_DETECTOR`` to its name:

    FTT_DETECTOR=football_detection_v6 uv run python scripts/cache_detections.py --split val

Everything that depends on the detector (cached boxes, reports) is stored under its name, so the
results of different detectors never overwrite each other.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

DETECTOR_ENV = "FTT_DETECTOR"


def load_paths(root: Path) -> dict:
    """``configs/paths.yaml`` as a dict; ``FTT_DETECTOR``, if set, swaps ``detector_weights``."""
    paths = yaml.safe_load((Path(root) / "configs" / "paths.yaml").read_text())
    name = os.environ.get(DETECTOR_ENV)
    if name:
        folder = Path(paths["detector_weights"]).parent
        paths["detector_weights"] = str(folder / f"{Path(name).stem}.pt")
    return paths
