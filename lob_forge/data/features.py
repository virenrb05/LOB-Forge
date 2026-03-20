"""Derived LOB feature computations for predictor transformer inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from lob_forge.data.schema import (
    ASK_PRICE_COLS,
    ASK_SIZE_COLS,
    BID_PRICE_COLS,
    BID_SIZE_COLS,
    MID_PRICE,
    SPREAD,
    TRADE_PRICE,
    TRADE_SIZE,
)

_EPS: float = 1e-12


def compute_mid_returns(
    df: pd.DataFrame, lookbacks: list[int] | None = None
) -> pd.DataFrame:
    """Compute mid-price percentage returns over multiple lookback windows.

    For each K in *lookbacks*, adds column ``mid_return_K`` defined as
    ``(mid_price - mid_price.shift(K)) / mid_price.shift(K)``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain :pydata:`MID_PRICE` column.
    lookbacks : list[int], optional
        Lookback periods.  Defaults to ``[1, 5, 10, 50]``.

    Returns
    -------
    pd.DataFrame
        DataFrame with one ``mid_return_K`` column per lookback.
    """
    if lookbacks is None:
        lookbacks = [1, 5, 10, 50]

    mid = df[MID_PRICE]
    result = pd.DataFrame(index=df.index)
    for k in lookbacks:
        shifted = mid.shift(k)
        result[f"mid_return_{k}"] = (mid - shifted) / (shifted + _EPS)
    return result


def compute_order_imbalance(df: pd.DataFrame) -> pd.Series:
    """Compute level-1 order imbalance.

    Formula: ``(bid_size_1 - ask_size_1) / (bid_size_1 + ask_size_1)``

    Returns
    -------
    pd.Series
        Values in [-1, 1].
    """
    bid = df[BID_SIZE_COLS[0]]
    ask = df[ASK_SIZE_COLS[0]]
    return (bid - ask) / (bid + ask + _EPS)


def compute_microprice(df: pd.DataFrame) -> pd.Series:
    """Compute microprice (size-weighted mid-price).

    Formula: ``(bid_price_1 * ask_size_1 + ask_price_1 * bid_size_1)
              / (bid_size_1 + ask_size_1)``

    Returns
    -------
    pd.Series
        Microprice series.
    """
    bp = df[BID_PRICE_COLS[0]]
    ap = df[ASK_PRICE_COLS[0]]
    bs = df[BID_SIZE_COLS[0]]
    as_ = df[ASK_SIZE_COLS[0]]
    return (bp * as_ + ap * bs) / (bs + as_ + _EPS)


def compute_depth_imbalance(
    df: pd.DataFrame, levels: int = 10
) -> pd.DataFrame:
    """Compute depth imbalance per book level.

    For each level i=1..levels:
    ``depth_imb_i = (bid_size_i - ask_size_i) / (bid_size_i + ask_size_i)``

    Returns
    -------
    pd.DataFrame
        DataFrame with ``depth_imb_1`` through ``depth_imb_{levels}`` columns.
    """
    result = pd.DataFrame(index=df.index)
    for i in range(levels):
        bid = df[BID_SIZE_COLS[i]]
        ask = df[ASK_SIZE_COLS[i]]
        result[f"depth_imb_{i + 1}"] = (bid - ask) / (bid + ask + _EPS)
    return result


def compute_spread_bps(df: pd.DataFrame) -> pd.Series:
    """Compute spread in basis points.

    Formula: ``spread / mid_price * 10_000``

    Returns
    -------
    pd.Series
        Spread in bps.
    """
    return df[SPREAD] / (df[MID_PRICE] + _EPS) * 10_000


def compute_vpin(df: pd.DataFrame, n_buckets: int = 50) -> pd.Series:
    """Compute Volume-Synchronized Probability of Informed Trading (VPIN).

    Uses bulk volume classification:
    - ``buy_vol = trade_size * CDF(ln(P_t / P_{t-1}) / sigma)``
    - ``sell_vol = trade_size - buy_vol``
    - Aggregates into equal-volume buckets
    - ``VPIN = mean(|buy_vol - sell_vol| / bucket_vol)`` over *n_buckets*

    Rows without trades (NaN trade_size) get forward-filled VPIN values.

    Returns
    -------
    pd.Series
        VPIN values in [0, 1].
    """
    trade_mask = df[TRADE_SIZE].notna() & (df[TRADE_SIZE] > 0)
    trade_idx = df.index[trade_mask]

    if len(trade_idx) < 2:
        return pd.Series(np.nan, index=df.index, name="vpin")

    prices = df.loc[trade_idx, TRADE_PRICE].values
    sizes = df.loc[trade_idx, TRADE_SIZE].values

    # Log returns
    log_ret = np.zeros(len(prices))
    log_ret[1:] = np.log(prices[1:] / (prices[:-1] + _EPS))

    # Rolling std of log returns (window=50, min_periods=2)
    log_ret_series = pd.Series(log_ret)
    sigma = log_ret_series.rolling(window=50, min_periods=2).std().values
    sigma = np.where(sigma < _EPS, _EPS, sigma)

    # Bulk volume classification
    z = log_ret / sigma
    buy_frac = norm.cdf(z)
    buy_vol = sizes * buy_frac
    sell_vol = sizes * (1.0 - buy_frac)

    # Build equal-volume buckets
    total_vol = np.nansum(sizes)
    if total_vol <= 0:
        return pd.Series(np.nan, index=df.index, name="vpin")

    bucket_vol_target = total_vol / n_buckets
    if bucket_vol_target <= 0:
        return pd.Series(np.nan, index=df.index, name="vpin")

    # Accumulate trades into buckets and compute |buy - sell| / bucket_vol
    cum_vol = np.cumsum(sizes)
    bucket_boundaries = np.arange(1, n_buckets + 1) * bucket_vol_target

    # Assign each trade to a bucket
    bucket_ids = np.searchsorted(bucket_boundaries, cum_vol, side="left")
    bucket_ids = np.clip(bucket_ids, 0, n_buckets - 1)

    # Per-bucket aggregation
    bucket_buy = np.zeros(n_buckets)
    bucket_sell = np.zeros(n_buckets)
    bucket_total = np.zeros(n_buckets)

    for b in range(n_buckets):
        mask = bucket_ids == b
        bucket_buy[b] = np.nansum(buy_vol[mask])
        bucket_sell[b] = np.nansum(sell_vol[mask])
        bucket_total[b] = np.nansum(sizes[mask])

    # VPIN per bucket
    with np.errstate(divide="ignore", invalid="ignore"):
        bucket_vpin = np.abs(bucket_buy - bucket_sell) / (bucket_total + _EPS)

    # Rolling mean VPIN over n_buckets (assign to last trade in each bucket)
    # For simplicity: compute a single VPIN per trade using rolling mean of
    # the bucket-level VPIN values up to the current bucket
    vpin_per_trade = np.full(len(trade_idx), np.nan)
    for i in range(len(trade_idx)):
        b = bucket_ids[i]
        # Rolling mean of all completed buckets up to current
        end = b + 1
        start = max(0, end - n_buckets)
        vpin_per_trade[i] = np.mean(bucket_vpin[start:end])

    # Map back to full DataFrame index, forward-fill
    vpin_series = pd.Series(np.nan, index=df.index, name="vpin")
    vpin_series.loc[trade_idx] = vpin_per_trade
    vpin_series = vpin_series.ffill()

    # Clip to [0, 1]
    vpin_series = vpin_series.clip(0.0, 1.0)

    return vpin_series


def compute_all_features(
    df: pd.DataFrame, lookbacks: list[int] | None = None
) -> pd.DataFrame:
    """Compute all 6 derived features and return augmented DataFrame.

    Adds 18 new columns:
    - 4 mid-return columns (mid_return_1, _5, _10, _50)
    - 1 order_imbalance
    - 1 microprice
    - 10 depth_imb columns (depth_imb_1 through depth_imb_10)
    - 1 spread_bps
    - 1 vpin

    Parameters
    ----------
    df : pd.DataFrame
        LOB DataFrame with unified schema columns.
    lookbacks : list[int], optional
        Lookback periods for mid returns.  Defaults to ``[1, 5, 10, 50]``.

    Returns
    -------
    pd.DataFrame
        Original DataFrame augmented with 18 feature columns.
    """
    result = df.copy()

    # Mid returns
    mid_ret = compute_mid_returns(df, lookbacks=lookbacks)
    for col in mid_ret.columns:
        result[col] = mid_ret[col].values

    # Order imbalance
    result["order_imbalance"] = compute_order_imbalance(df).values

    # Microprice
    result["microprice"] = compute_microprice(df).values

    # Depth imbalance
    depth_imb = compute_depth_imbalance(df)
    for col in depth_imb.columns:
        result[col] = depth_imb[col].values

    # Spread bps
    result["spread_bps"] = compute_spread_bps(df).values

    # VPIN
    result["vpin"] = compute_vpin(df).values

    return result
