"""Regime-conditioned generation validation.

Validates that conditioning on different volatility regimes produces
statistically distinct LOB distributions, satisfying GEN-07.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats

__all__ = [
    "compare_regime_distributions",
    "compute_regime_divergence",
    "validate_regime_conditioning",
]


def _mid_prices(data: np.ndarray) -> np.ndarray:
    """Compute mid-prices from LOB snapshot columns 0 (ask_1) and 20 (bid_1)."""
    return (data[:, 0] + data[:, 20]) / 2.0


def _returns(data: np.ndarray) -> np.ndarray:
    """Compute simple returns from mid-prices."""
    mid = _mid_prices(data)
    return np.diff(mid) / (mid[:-1] + 1e-12)


def _spreads(data: np.ndarray) -> np.ndarray:
    """Compute bid-ask spreads from columns 0 (ask_1) and 20 (bid_1)."""
    return data[:, 0] - data[:, 20]


def compare_regime_distributions(
    data_by_regime: dict[int, np.ndarray],
) -> dict[str, dict]:
    """Compare LOB distributions across volatility regimes.

    For each pair of regimes, computes KS tests on mid-price returns
    and spreads, plus the ratio of return standard deviations.

    Parameters
    ----------
    data_by_regime : dict[int, np.ndarray]
        Mapping from regime label (0, 1, 2) to LOB data arrays of shape (N, C)
        where C >= 21 (at minimum columns 0 and 20 for ask_1 and bid_1).

    Returns
    -------
    dict[str, dict]
        Keyed by regime pair string (e.g. ``"0_1"``) with values containing
        ``ks_returns_stat``, ``ks_returns_p``, ``ks_spread_stat``,
        ``ks_spread_p``, and ``vol_ratio``.
    """
    regimes = sorted(data_by_regime.keys())
    results: dict[str, dict] = {}

    for r_a, r_b in combinations(regimes, 2):
        ret_a = _returns(data_by_regime[r_a])
        ret_b = _returns(data_by_regime[r_b])
        spr_a = _spreads(data_by_regime[r_a])
        spr_b = _spreads(data_by_regime[r_b])

        ks_ret = stats.ks_2samp(ret_a, ret_b)
        ks_spr = stats.ks_2samp(spr_a, spr_b)

        vol_a = float(np.std(ret_a))
        vol_b = float(np.std(ret_b))
        vol_ratio = vol_b / (vol_a + 1e-12)

        results[f"{r_a}_{r_b}"] = {
            "ks_returns_stat": float(ks_ret.statistic),
            "ks_returns_p": float(ks_ret.pvalue),
            "ks_spread_stat": float(ks_spr.statistic),
            "ks_spread_p": float(ks_spr.pvalue),
            "vol_ratio": vol_ratio,
        }

    return results


def compute_regime_divergence(
    data_by_regime: dict[int, np.ndarray],
    n_bins: int = 50,
) -> dict[str, Any]:
    """Compute discrete KL divergence between regime return distributions.

    Parameters
    ----------
    data_by_regime : dict[int, np.ndarray]
        Mapping from regime label to LOB data arrays.
    n_bins : int
        Number of histogram bins for discretising return distributions.

    Returns
    -------
    dict[str, Any]
        Keys ``kl_01``, ``kl_02``, ``kl_12`` (pairwise KL divergences),
        ``mean_kl`` (average), and ``regime_separability`` (True if
        ``mean_kl > 0.1``).
    """
    regimes = sorted(data_by_regime.keys())
    returns_by_regime: dict[int, np.ndarray] = {}
    for r in regimes:
        returns_by_regime[r] = _returns(data_by_regime[r])

    # Determine shared bin edges across all regimes
    all_returns = np.concatenate(list(returns_by_regime.values()))
    bin_edges = np.linspace(
        float(np.min(all_returns)),
        float(np.max(all_returns)),
        n_bins + 1,
    )

    eps = 1e-10
    histograms: dict[int, np.ndarray] = {}
    for r in regimes:
        counts, _ = np.histogram(returns_by_regime[r], bins=bin_edges)
        prob = counts.astype(np.float64) / counts.sum() + eps
        prob = prob / prob.sum()  # re-normalise after epsilon
        histograms[r] = prob

    results: dict[str, Any] = {}
    kl_values: list[float] = []
    for r_a, r_b in combinations(regimes, 2):
        kl = float(stats.entropy(histograms[r_a], histograms[r_b]))
        key = f"kl_{r_a}{r_b}"
        results[key] = kl
        kl_values.append(kl)

    mean_kl = float(np.mean(kl_values))
    results["mean_kl"] = mean_kl
    results["regime_separability"] = mean_kl > 0.1

    return results


def validate_regime_conditioning(
    real_by_regime: dict[int, np.ndarray],
    synthetic_by_regime: dict[int, np.ndarray],
) -> dict[str, Any]:
    """Orchestrate regime-conditioned generation validation.

    Checks three properties:

    1. **Distinctness** -- synthetic regimes are statistically distinct from
       each other.
    2. **Fidelity** -- each synthetic regime matches its real counterpart
       (KS test on returns).
    3. **Ordering** -- volatility ordering is preserved:
       ``vol(regime=2) > vol(regime=1) > vol(regime=0)``.

    Parameters
    ----------
    real_by_regime : dict[int, np.ndarray]
        Real LOB data grouped by regime.
    synthetic_by_regime : dict[int, np.ndarray]
        Synthetic LOB data grouped by regime.

    Returns
    -------
    dict[str, Any]
        ``regime_distinct`` (bool), ``regime_matched`` (bool),
        ``ordering_preserved`` (bool), ``all_passed`` (bool), and
        ``details`` (nested dict).
    """
    details: dict[str, Any] = {}

    # 1. Synthetic regimes distinct from each other
    syn_comparison = compare_regime_distributions(synthetic_by_regime)
    details["synthetic_comparison"] = syn_comparison
    # Distinct if at least one KS p-value < 0.05 for each pair
    regime_distinct = all(v["ks_returns_p"] < 0.05 for v in syn_comparison.values())

    # 2. Each synthetic regime matches its real counterpart
    match_results: dict[str, dict] = {}
    regimes = sorted(set(real_by_regime.keys()) & set(synthetic_by_regime.keys()))
    for r in regimes:
        ret_real = _returns(real_by_regime[r])
        ret_syn = _returns(synthetic_by_regime[r])
        ks = stats.ks_2samp(ret_real, ret_syn)
        match_results[str(r)] = {
            "ks_stat": float(ks.statistic),
            "ks_p": float(ks.pvalue),
        }
    details["regime_match"] = match_results
    # Matched if KS p-value > 0.05 (cannot reject same distribution)
    regime_matched = all(v["ks_p"] > 0.05 for v in match_results.values())

    # 3. Volatility ordering: vol(2) > vol(1) > vol(0)
    syn_vols: dict[int, float] = {}
    for r in sorted(synthetic_by_regime.keys()):
        syn_vols[r] = float(np.std(_returns(synthetic_by_regime[r])))
    details["synthetic_vols"] = syn_vols

    ordering_preserved = True
    sorted_regimes = sorted(syn_vols.keys())
    for i in range(len(sorted_regimes) - 1):
        if syn_vols[sorted_regimes[i + 1]] <= syn_vols[sorted_regimes[i]]:
            ordering_preserved = False
            break

    all_passed = regime_distinct and regime_matched and ordering_preserved

    return {
        "regime_distinct": regime_distinct,
        "regime_matched": regime_matched,
        "ordering_preserved": ordering_preserved,
        "all_passed": all_passed,
        "details": details,
    }
