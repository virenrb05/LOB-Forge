"""Statistical tests for validating stylized facts of generated LOB data.

Each function follows a consistent interface:
    def test_name(real, synthetic, **kwargs) -> dict[str, float | bool]

The returned dict always contains at minimum a ``passed`` key indicating
whether the synthetic data reproduces the tested stylized fact.

LOB book arrays are expected in the 40-column layout:
    ask_price_1..10 (cols 0-9), ask_size_1..10 (cols 10-19),
    bid_price_1..10 (cols 20-29), bid_size_1..10 (cols 30-39).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "return_distribution_test",
    "volatility_clustering_test",
    "bid_ask_bounce_test",
    "spread_cdf_test",
    "book_shape_test",
    "market_impact_test",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _log_returns(mid: np.ndarray) -> np.ndarray:
    """Compute log-returns from a mid-price series."""
    mid = np.asarray(mid, dtype=np.float64)
    return np.diff(np.log(mid))


def _lag1_autocorrelation(x: np.ndarray) -> float:
    """Compute lag-1 autocorrelation of a 1-D array."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 2:
        return 0.0
    c0 = np.dot(x, x)
    if c0 == 0:
        return 0.0
    c1 = np.dot(x[:-1], x[1:])
    return float(c1 / c0)


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation at lags 1..max_lag (normalised by lag-0)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    c0 = np.dot(x, x)
    if c0 == 0:
        return np.zeros(max_lag)
    result = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        if lag >= n:
            result[lag - 1] = 0.0
        else:
            result[lag - 1] = np.dot(x[: n - lag], x[lag:]) / c0
    return result


# ---------------------------------------------------------------------------
# Test 1: Return distribution
# ---------------------------------------------------------------------------


def return_distribution_test(
    real_mid: np.ndarray,
    synthetic_mid: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """Two-sample KS test on log-return distributions.

    Parameters
    ----------
    real_mid : array-like
        Real mid-price series (1-D).
    synthetic_mid : array-like
        Synthetic mid-price series (1-D).
    alpha : float
        Significance level for the KS test.

    Returns
    -------
    dict
        statistic, p_value, passed, real_kurtosis, synthetic_kurtosis.
    """
    real_ret = _log_returns(real_mid)
    synth_ret = _log_returns(synthetic_mid)

    ks_stat, p_value = stats.ks_2samp(real_ret, synth_ret)
    real_kurt = float(stats.kurtosis(real_ret, fisher=True))
    synth_kurt = float(stats.kurtosis(synth_ret, fisher=True))

    return {
        "statistic": float(ks_stat),
        "p_value": float(p_value),
        "passed": bool(p_value > alpha),
        "real_kurtosis": real_kurt,
        "synthetic_kurtosis": synth_kurt,
    }


# ---------------------------------------------------------------------------
# Test 2: Volatility clustering
# ---------------------------------------------------------------------------


def volatility_clustering_test(
    real_mid: np.ndarray,
    synthetic_mid: np.ndarray,
    *,
    max_lag: int = 20,
) -> dict[str, float | bool]:
    """Test for volatility clustering via autocorrelation of absolute returns.

    Parameters
    ----------
    real_mid : array-like
        Real mid-price series (1-D).
    synthetic_mid : array-like
        Synthetic mid-price series (1-D).
    max_lag : int
        Maximum lag for autocorrelation computation.

    Returns
    -------
    dict
        mean_acf_real, mean_acf_synthetic, acf_ratio, passed.
    """
    real_abs = np.abs(_log_returns(real_mid))
    synth_abs = np.abs(_log_returns(synthetic_mid))

    acf_real = _acf(real_abs, max_lag)
    acf_synth = _acf(synth_abs, max_lag)

    mean_real = float(np.mean(acf_real))
    mean_synth = float(np.mean(acf_synth))

    # Avoid division by zero when real mean ACF is zero
    if mean_real == 0:
        ratio = 0.0
    else:
        ratio = mean_synth / mean_real

    passed = mean_synth > 0 and 0.3 <= ratio <= 3.0

    return {
        "mean_acf_real": mean_real,
        "mean_acf_synthetic": mean_synth,
        "acf_ratio": float(ratio),
        "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# Test 3: Bid-ask bounce
# ---------------------------------------------------------------------------


def bid_ask_bounce_test(
    real_returns: np.ndarray,
    synthetic_returns: np.ndarray,
) -> dict[str, float | bool]:
    """Test for negative lag-1 autocorrelation (bid-ask bounce).

    Parameters
    ----------
    real_returns : array-like
        Real trade-signed return series (1-D).
    synthetic_returns : array-like
        Synthetic trade-signed return series (1-D).

    Returns
    -------
    dict
        real_lag1_acf, synthetic_lag1_acf, passed.
    """
    real_acf = _lag1_autocorrelation(np.asarray(real_returns, dtype=np.float64))
    synth_acf = _lag1_autocorrelation(np.asarray(synthetic_returns, dtype=np.float64))

    return {
        "real_lag1_acf": float(real_acf),
        "synthetic_lag1_acf": float(synth_acf),
        "passed": bool(real_acf < 0 and synth_acf < 0),
    }


# ---------------------------------------------------------------------------
# Test 4: Spread CDF
# ---------------------------------------------------------------------------


def spread_cdf_test(
    real_book: np.ndarray,
    synthetic_book: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """KS test on bid-ask spread distributions.

    Parameters
    ----------
    real_book : ndarray, shape (N, 40)
        Real LOB book data.  Column 0 = ask_price_1, column 20 = bid_price_1.
    synthetic_book : ndarray, shape (M, 40)
        Synthetic LOB book data (same column layout).
    alpha : float
        Significance level.

    Returns
    -------
    dict
        statistic, p_value, passed, real_mean_spread, synthetic_mean_spread.
    """
    real_book = np.asarray(real_book, dtype=np.float64)
    synthetic_book = np.asarray(synthetic_book, dtype=np.float64)

    real_spread = real_book[:, 0] - real_book[:, 20]
    synth_spread = synthetic_book[:, 0] - synthetic_book[:, 20]

    ks_stat, p_value = stats.ks_2samp(real_spread, synth_spread)

    return {
        "statistic": float(ks_stat),
        "p_value": float(p_value),
        "passed": bool(p_value > alpha),
        "real_mean_spread": float(np.mean(real_spread)),
        "synthetic_mean_spread": float(np.mean(synth_spread)),
    }


# ---------------------------------------------------------------------------
# Test 5: Book shape
# ---------------------------------------------------------------------------


def book_shape_test(
    real_book: np.ndarray,
    synthetic_book: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, float | bool | list[float]]:
    """Compare depth profiles across 10 price levels.

    Ask sizes occupy columns 10-19 and bid sizes columns 30-39.
    The depth profile averages ask_size + bid_size at each level.

    Parameters
    ----------
    real_book : ndarray, shape (N, 40)
        Real LOB book data.
    synthetic_book : ndarray, shape (M, 40)
        Synthetic LOB book data.
    alpha : float
        Significance level for per-level KS tests.

    Returns
    -------
    dict
        mean_ks_statistic, min_p_value, passed, real_shape, synthetic_shape.
    """
    real_book = np.asarray(real_book, dtype=np.float64)
    synthetic_book = np.asarray(synthetic_book, dtype=np.float64)

    n_levels = 10
    ks_stats = np.empty(n_levels)
    p_values = np.empty(n_levels)

    # Average size = (ask_size + bid_size) / 2 at each level
    real_shape = np.empty(n_levels)
    synth_shape = np.empty(n_levels)

    for i in range(n_levels):
        ask_size_col = 10 + i  # ask_size columns 10-19
        bid_size_col = 30 + i  # bid_size columns 30-39

        real_depth = real_book[:, ask_size_col] + real_book[:, bid_size_col]
        synth_depth = synthetic_book[:, ask_size_col] + synthetic_book[:, bid_size_col]

        real_shape[i] = float(np.mean(real_depth))
        synth_shape[i] = float(np.mean(synth_depth))

        ks_stats[i], p_values[i] = stats.ks_2samp(real_depth, synth_depth)

    return {
        "mean_ks_statistic": float(np.mean(ks_stats)),
        "min_p_value": float(np.min(p_values)),
        "passed": bool(np.min(p_values) > alpha),
        "real_shape": real_shape.tolist(),
        "synthetic_shape": synth_shape.tolist(),
    }


# ---------------------------------------------------------------------------
# Test 6: Market impact
# ---------------------------------------------------------------------------


def market_impact_test(
    real_book: np.ndarray,
    real_mid: np.ndarray,
    synthetic_book: np.ndarray,
    synthetic_mid: np.ndarray,
    *,
    n_bins: int = 20,
) -> dict[str, float | bool]:
    """Test concavity of price impact vs volume (square-root law).

    Volume proxy: total absolute bid-size change across levels between
    consecutive snapshots.  Impact: absolute mid-price change.

    A log-log regression slope (beta) < 1 indicates concavity.

    Parameters
    ----------
    real_book : ndarray, shape (N, 40)
        Real LOB book data.
    real_mid : array-like, shape (N,)
        Real mid-price series.
    synthetic_book : ndarray, shape (M, 40)
        Synthetic LOB book data.
    synthetic_mid : array-like, shape (M,)
        Synthetic mid-price series.
    n_bins : int
        Number of volume bins for aggregation.

    Returns
    -------
    dict
        real_beta, synthetic_beta, passed.
    """

    def _estimate_beta(book: np.ndarray, mid: np.ndarray) -> float:
        book = np.asarray(book, dtype=np.float64)
        mid = np.asarray(mid, dtype=np.float64)

        # Volume proxy: sum of absolute bid-size changes across 10 levels
        bid_sizes = book[:, 30:40]  # bid_size columns 30-39
        vol = np.sum(np.abs(np.diff(bid_sizes, axis=0)), axis=1)
        impact = np.abs(np.diff(mid))

        # Filter out zeros (log undefined)
        mask = (vol > 0) & (impact > 0)
        vol = vol[mask]
        impact = impact[mask]

        if len(vol) < n_bins:
            return float("nan")

        # Bin by volume and average impact per bin
        bin_edges = np.percentile(vol, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-12  # ensure last point is included
        bin_idx = np.digitize(vol, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        bin_vol = np.empty(n_bins)
        bin_impact = np.empty(n_bins)
        valid_bins = 0
        for b in range(n_bins):
            sel = bin_idx == b
            if np.any(sel):
                bin_vol[valid_bins] = np.mean(vol[sel])
                bin_impact[valid_bins] = np.mean(impact[sel])
                valid_bins += 1

        bin_vol = bin_vol[:valid_bins]
        bin_impact = bin_impact[:valid_bins]

        # Filter positive for log
        pos = (bin_vol > 0) & (bin_impact > 0)
        bin_vol = bin_vol[pos]
        bin_impact = bin_impact[pos]

        if len(bin_vol) < 2:
            return float("nan")

        # Log-log OLS: log(impact) = beta * log(vol) + intercept
        log_vol = np.log(bin_vol)
        log_impact = np.log(bin_impact)
        slope, _intercept, _r, _p, _se = stats.linregress(log_vol, log_impact)
        return float(slope)

    real_beta = _estimate_beta(real_book, real_mid)
    synth_beta = _estimate_beta(synthetic_book, synthetic_mid)

    # Pass if both betas are positive but < 1 (concave)
    passed = (
        not np.isnan(real_beta)
        and not np.isnan(synth_beta)
        and 0 < real_beta < 1.0
        and 0 < synth_beta < 1.0
    )

    return {
        "real_beta": real_beta,
        "synthetic_beta": synth_beta,
        "passed": bool(passed),
    }
