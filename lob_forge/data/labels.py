"""Mid-price movement labeling with multi-horizon support.

Labels classify future mid-price movement as UP (2), STATIONARY (1), or DOWN (0)
at multiple horizons. Strict causality is guaranteed: the label at row t depends
only on mid_price values at rows t+1 through t+h.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lob_forge.data.schema import MID_PRICE

# Label constants
UP: float = 2.0
STATIONARY: float = 1.0
DOWN: float = 0.0


def compute_labels(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
    threshold: float = 0.00002,
) -> pd.DataFrame:
    """Compute mid-price movement labels at multiple horizons.

    For each horizon *h*, the smoothed future mid-price at row *t* is defined as
    ``mean(mid_price[t+1], mid_price[t+2], ..., mid_price[t+h])``.  The return is
    ``(smoothed_future - mid_price[t]) / mid_price[t]``, which is then classified:

    * ``return >  threshold`` → **UP** (2)
    * ``return < -threshold`` → **DOWN** (0)
    * otherwise              → **STATIONARY** (1)

    The last *h* rows of each ``label_h{h}`` column are ``NaN`` because
    insufficient future data is available.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``mid_price`` column.
    horizons : list[int], optional
        Tick horizons to label.  Defaults to ``[10, 20, 50, 100]``.
    threshold : float, optional
        Classification threshold (default 2 bps = 0.00002).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``label_h{h}`` columns appended.
    """
    if horizons is None:
        horizons = [10, 20, 50, 100]

    mid = df[MID_PRICE].values.astype(np.float64)
    n = len(mid)

    # Cumulative sum trick for O(n) computation of future means.
    # cs[i] = sum(mid[0:i]), so sum(mid[a:b]) = cs[b] - cs[a].
    cs = np.concatenate([[0.0], np.cumsum(mid)])

    result = df.copy()

    for h in horizons:
        # future_mean[t] = mean(mid[t+1], ..., mid[t+h])
        #                 = (cs[t+h+1] - cs[t+1]) / h
        # Valid for t in [0, n-h-1]; last h rows are NaN.
        valid = n - h
        if valid <= 0:
            result[f"label_h{h}"] = np.nan
            continue

        future_sum = cs[h + 1 : n + 1] - cs[1 : n - h + 1]
        future_mean = future_sum / h

        ret = (future_mean - mid[:valid]) / mid[:valid]

        labels = np.full(n, np.nan, dtype=np.float64)
        labels[:valid] = np.where(
            ret > threshold,
            UP,
            np.where(ret < -threshold, DOWN, STATIONARY),
        )

        result[f"label_h{h}"] = labels

    return result
