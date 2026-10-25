"""Tracker settings: a tracker's own config plus the shared tuned setting (configs/tuned.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_tuned(configs_dir: Path) -> dict:
    """The shared tuned setting: the keys that override every tracker's own config."""
    return yaml.safe_load((Path(configs_dir) / "tuned.yaml").read_text())


def load_tracker_cfg(configs_dir: Path, name: str, *, tuned: bool = True) -> dict:
    """``configs/<name>.yaml``, with the shared tuned setting applied unless ``tuned=False``."""
    cfg = yaml.safe_load((Path(configs_dir) / f"{name}.yaml").read_text())
    return {**cfg, **load_tuned(configs_dir)} if tuned else cfg
