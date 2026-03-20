"""Temporal train/val/test splitting with purge gaps for LOB data.

Purge gaps prevent data leakage across split boundaries by inserting
unused rows between segments. This is critical for valid walk-forward
evaluation of time-series models.
"""

from __future__ import annotations

import numpy as np


def temporal_split(
    n_rows: int,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    purge_gap: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split row indices temporally into train/val/test with purge gaps.

    Parameters
    ----------
    n_rows:
        Total number of rows in the dataset.
    ratios:
        Fraction of data for (train, val, test). Must sum to ~1.0.
    purge_gap:
        Number of rows to skip between each segment to prevent leakage.

    Returns
    -------
    tuple of three np.ndarray (dtype int64):
        (train_indices, val_indices, test_indices).
        Segments that don't fit return empty arrays.
    """
    _empty = np.array([], dtype=np.int64)

    train_end = int(n_rows * ratios[0])
    if train_end <= 0:
        return _empty.copy(), _empty.copy(), _empty.copy()

    train = np.arange(0, train_end, dtype=np.int64)

    # Validation segment
    val_start = train_end + purge_gap
    if val_start >= n_rows:
        return train, _empty.copy(), _empty.copy()

    val_end = val_start + int(n_rows * ratios[1])
    val_end = min(val_end, n_rows)
    if val_start >= val_end:
        return train, _empty.copy(), _empty.copy()

    val = np.arange(val_start, val_end, dtype=np.int64)

    # Test segment
    test_start = val_end + purge_gap
    if test_start >= n_rows:
        return train, val, _empty.copy()

    test = np.arange(test_start, n_rows, dtype=np.int64)

    return train, val, test
