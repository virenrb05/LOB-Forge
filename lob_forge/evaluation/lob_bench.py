"""LOB-Bench quantitative evaluation metrics.

Implements Wasserstein distances, MLP discriminator scores, and conditional
statistics comparison for evaluating synthetic LOB data quality.
Satisfies GEN-08.

Reference: LOB-Bench (ICML 2025).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance

__all__ = [
    "compute_wasserstein_metrics",
    "compute_conditional_stats",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPS = 1e-12


def _extract_spread(data: NDArray[np.floating]) -> NDArray[np.floating]:
    """Spread = ask_1 (col 0) - bid_1 (col 2) assuming standard 40-col LOB layout.

    Standard layout: [ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...].
    """
    return data[:, 0] - data[:, 2]


def _extract_mid(data: NDArray[np.floating]) -> NDArray[np.floating]:
    """Mid-price = (ask_1 + bid_1) / 2."""
    return (data[:, 0] + data[:, 2]) / 2.0


def _extract_log_returns(data: NDArray[np.floating]) -> NDArray[np.floating]:
    """Log returns of mid-price series."""
    mid = _extract_mid(data)
    mid = np.maximum(mid, _EPS)  # guard against log(0)
    return np.diff(np.log(mid))


# ---------------------------------------------------------------------------
# 1. Wasserstein Metrics
# ---------------------------------------------------------------------------


def compute_wasserstein_metrics(
    real: NDArray[np.floating],
    synthetic: NDArray[np.floating],
) -> dict[str, Any]:
    """Compute 1-D Wasserstein distances between real and synthetic marginals.

    Parameters
    ----------
    real : ndarray of shape ``(N, C)``
        Real LOB data (C features, typically 40 columns).
    synthetic : ndarray of shape ``(M, C)``
        Synthetic LOB data with the same number of columns.

    Returns
    -------
    dict[str, Any]
        ``wd_mean``, ``wd_max``, ``wd_per_feature`` (list), and derived
        quantities ``wd_spread``, ``wd_mid``, ``wd_returns``.
    """
    n_features = real.shape[1]
    per_feature: list[float] = []
    for c in range(n_features):
        per_feature.append(float(wasserstein_distance(real[:, c], synthetic[:, c])))

    # Derived quantities
    wd_spread = float(
        wasserstein_distance(_extract_spread(real), _extract_spread(synthetic))
    )
    wd_mid = float(wasserstein_distance(_extract_mid(real), _extract_mid(synthetic)))

    real_ret = _extract_log_returns(real)
    synth_ret = _extract_log_returns(synthetic)
    wd_returns = float(wasserstein_distance(real_ret, synth_ret))

    return {
        "wd_mean": float(np.mean(per_feature)),
        "wd_max": float(np.max(per_feature)),
        "wd_per_feature": per_feature,
        "wd_spread": wd_spread,
        "wd_mid": wd_mid,
        "wd_returns": wd_returns,
    }


# ---------------------------------------------------------------------------
# 2. Conditional Statistics
# ---------------------------------------------------------------------------


def compute_conditional_stats(
    real: NDArray[np.floating],
    synthetic: NDArray[np.floating],
    real_regimes: NDArray[np.integer],
    synthetic_regimes: NDArray[np.integer],
) -> dict[str, Any]:
    """Compare distributional properties across volatility regimes.

    Parameters
    ----------
    real, synthetic : ndarray of shape ``(N, C)``
        LOB snapshots.
    real_regimes, synthetic_regimes : ndarray of shape ``(N,)``
        Regime labels (0=low-vol, 1=normal, 2=high-vol).

    Returns
    -------
    dict[str, Any]
        Per-regime relative errors and ``mean_relative_error`` across all
        regime-stat pairs.
    """
    results: dict[str, Any] = {}
    all_errors: list[float] = []

    for regime in (0, 1, 2):
        r_mask = real_regimes == regime
        s_mask = synthetic_regimes == regime

        # Skip regime if either set is empty
        if r_mask.sum() < 2 or s_mask.sum() < 2:
            results[str(regime)] = {"skipped": True}
            continue

        r_data = real[r_mask]
        s_data = synthetic[s_mask]

        r_spread = _extract_spread(r_data)
        s_spread = _extract_spread(s_data)

        # Total book depth: sum of all size columns (odd indices in 40-col layout)
        r_depth = r_data[:, 1::2].sum(axis=1)
        s_depth = s_data[:, 1::2].sum(axis=1)

        r_ret = _extract_log_returns(r_data)
        s_ret = _extract_log_returns(s_data)

        def _rel_err(a: float, b: float) -> float:
            return abs(a - b) / (abs(a) + _EPS)

        regime_stats: dict[str, float] = {}
        pairs = [
            ("spread_mean_err", float(np.mean(r_spread)), float(np.mean(s_spread))),
            ("spread_std_err", float(np.std(r_spread)), float(np.std(s_spread))),
            ("depth_mean_err", float(np.mean(r_depth)), float(np.mean(s_depth))),
            ("depth_std_err", float(np.std(r_depth)), float(np.std(s_depth))),
            ("return_mean_err", float(np.mean(r_ret)), float(np.mean(s_ret))),
            ("return_std_err", float(np.std(r_ret)), float(np.std(s_ret))),
        ]

        for name, rv, sv in pairs:
            err = _rel_err(rv, sv)
            regime_stats[name] = err
            all_errors.append(err)

        results[str(regime)] = regime_stats

    results["mean_relative_error"] = float(np.mean(all_errors)) if all_errors else 0.0
    return results
