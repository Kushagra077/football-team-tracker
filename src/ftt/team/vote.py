"""Rolling per-track majority vote (window ~25 frames) with a lock once enough frames agree."""

from __future__ import annotations

from collections import Counter, deque

import numpy as np


def vote_labels(
    track_ids: np.ndarray,
    frames: np.ndarray,
    labels: np.ndarray,
    window: int = 25,
    lock_min_agree: int | None = 15,
) -> np.ndarray:
    """Smooth per-frame team labels into per-track labels. Output is aligned with the input rows.

    Causal: the label at frame t uses only that track's labels from frames (t - window, t].
    Ties keep the previous output, so a 50/50 split does not flicker. Once ``lock_min_agree``
    labels in the window agree, the track's label is frozen for good (no later flips, and also
    no recovery if that first lock was wrong or the tracker later swaps ids). None = never lock.
    """
    if lock_min_agree is not None and lock_min_agree > window:
        raise ValueError(f"lock_min_agree {lock_min_agree} > window {window}: could never lock")
    out = np.empty_like(labels)
    for tid in np.unique(track_ids):
        rows = np.flatnonzero(track_ids == tid)
        rows = rows[np.argsort(frames[rows], kind="stable")]
        recent: deque[tuple[int, int]] = deque()  # (frame, label) inside the window
        prev, locked = None, None
        for i in rows:
            if locked is not None:
                out[i] = locked
                continue
            recent.append((frames[i], labels[i]))
            while recent[0][0] <= frames[i] - window:
                recent.popleft()
            counts = Counter(label for _, label in recent).most_common()
            top, n = counts[0]
            tie = len(counts) > 1 and counts[1][1] == n
            choice = prev if (tie and prev is not None) else top
            if lock_min_agree is not None and n >= lock_min_agree:
                locked = top
                choice = top
            out[i] = prev = choice
    return out
