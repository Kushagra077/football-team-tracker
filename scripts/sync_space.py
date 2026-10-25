"""Copy what the Hugging Face demo needs from this repo into a clone of the Space repo.

The Space runs the same ``ftt`` package, so it gets a copy of ``src/ftt`` plus the two configs the
pipeline reads (``team.yaml`` and the tuned BoT-SORT + GMC tracker config). The Space's own files
(``app/hub_app.py``, ``app/track_tab.py``, requirements, README) live in the Space repo itself.
Nothing is pushed: review the Space folder with ``git status`` and push it yourself.

    uv run python scripts/sync_space.py ~/Desktop/hf-space
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).parents[1]
CONFIGS = ["team.yaml", "botsort_gmc.yaml", "tuned.yaml"]


def sync(space: Path) -> list[str]:
    """Replace ``<space>/ftt`` and the two configs with this repo's current versions."""
    if not (space / ".git").exists():
        raise SystemExit(f"{space} is not a git clone of the Space repo")
    done = []
    dest = space / "ftt"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ROOT / "src" / "ftt", dest, ignore=shutil.ignore_patterns("__pycache__"))
    done.append("ftt/")
    (space / "configs").mkdir(exist_ok=True)
    for name in CONFIGS:
        shutil.copy2(ROOT / "configs" / name, space / "configs" / name)
        done.append(f"configs/{name}")
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("space", type=Path, help="local clone of the Hugging Face Space repo")
    args = ap.parse_args()
    for item in sync(args.space.expanduser().resolve()):
        print("synced", item)


if __name__ == "__main__":
    main()
