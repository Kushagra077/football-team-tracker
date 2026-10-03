"""football-team-tracker (A2): persistent IDs + team labels from per-frame detections.

Two design rules carry the whole package. Do not break them:

1. ``data/detections.py`` returns ground-truth boxes and A1 boxes in ONE
   identical format, and every tracker consumes it through ``track/base.py``.
   The oracle-vs-real comparison is only valid if the only thing that differs
   between the two rows is where the boxes came from.
2. The detector runs ONCE per clip and its output is cached to parquet.
   Tracker ablations then take seconds instead of re-running YOLO on thousands
   of 1080p frames.
"""

__version__ = "0.1.0"
