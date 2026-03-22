"""Unit tests for regime-conditioned generation validation."""

from __future__ import annotations

import numpy as np
import pytest

from lob_forge.evaluation.regime_validation import (
    compare_regime_distributions,
    compute_regime_divergence,
    validate_regime_conditioning,
)


def _make_lob_data(
    n: int,
    return_std: float,
    spread_mean: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic LOB-like data with controlled return volatility.

    Constructs mid-prices via cumulative returns with specified std,
    then places ask/bid around mid with given spread.

    Returns array of shape (n, 40).
    """
    data = rng.uniform(90, 110, size=(n, 40))
    # Generate returns with controlled volatility, then build prices
    returns = rng.normal(0, return_std, size=n)
    mid = 100.0 * np.exp(np.cumsum(returns))
    spread = np.abs(rng.normal(spread_mean, spread_mean * 0.1, size=n))
    data[:, 0] = mid + spread / 2  # ask_1
    data[:, 20] = mid - spread / 2  # bid_1
    return data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def distinct_regimes() -> dict[int, np.ndarray]:
    """Three clearly distinct regimes: increasing volatility."""
    rng = np.random.default_rng(42)
    return {
        0: _make_lob_data(200, return_std=0.001, spread_mean=0.05, rng=rng),
        1: _make_lob_data(200, return_std=0.010, spread_mean=0.15, rng=rng),
        2: _make_lob_data(200, return_std=0.100, spread_mean=0.50, rng=rng),
    }


@pytest.fixture()
def identical_regimes() -> dict[int, np.ndarray]:
    """All three regimes drawn from the same distribution (same seed reset)."""
    data = {}
    for r in range(3):
        rng = np.random.default_rng(42)
        data[r] = _make_lob_data(200, return_std=0.005, spread_mean=0.10, rng=rng)
    return data


# ---------------------------------------------------------------------------
# TestCompareRegimeDistributions
# ---------------------------------------------------------------------------


class TestCompareRegimeDistributions:
    """Tests for compare_regime_distributions."""

    def test_distinct_regimes(self, distinct_regimes: dict[int, np.ndarray]) -> None:
        result = compare_regime_distributions(distinct_regimes)
        for pair_key, pair_val in result.items():
            assert pair_val["ks_returns_p"] < 0.05, (
                f"Pair {pair_key}: expected distinct returns (p < 0.05), "
                f"got p={pair_val['ks_returns_p']:.4f}"
            )

    def test_identical_regimes(self, identical_regimes: dict[int, np.ndarray]) -> None:
        result = compare_regime_distributions(identical_regimes)
        for pair_key, pair_val in result.items():
            assert pair_val["ks_returns_p"] > 0.05, (
                f"Pair {pair_key}: expected similar returns (p > 0.05), "
                f"got p={pair_val['ks_returns_p']:.4f}"
            )

    def test_vol_ordering(self, distinct_regimes: dict[int, np.ndarray]) -> None:
        result = compare_regime_distributions(distinct_regimes)
        assert result["0_1"]["vol_ratio"] > 1.0
        assert result["1_2"]["vol_ratio"] > 1.0
        assert result["0_2"]["vol_ratio"] > 1.0


# ---------------------------------------------------------------------------
# TestComputeRegimeDivergence
# ---------------------------------------------------------------------------


class TestComputeRegimeDivergence:
    """Tests for compute_regime_divergence."""

    def test_distinct_kl(self, distinct_regimes: dict[int, np.ndarray]) -> None:
        result = compute_regime_divergence(distinct_regimes)
        assert result["kl_01"] > 0
        assert result["kl_02"] > 0
        assert result["kl_12"] > 0

    def test_similar_kl(self, identical_regimes: dict[int, np.ndarray]) -> None:
        result = compute_regime_divergence(identical_regimes)
        assert (
            result["mean_kl"] < 0.1
        ), f"Expected mean_kl near 0, got {result['mean_kl']:.4f}"

    def test_separability(self, distinct_regimes: dict[int, np.ndarray]) -> None:
        result = compute_regime_divergence(distinct_regimes)
        assert result["regime_separability"] is True
        assert result["mean_kl"] > 0.1


# ---------------------------------------------------------------------------
# TestValidateRegimeConditioning
# ---------------------------------------------------------------------------


class TestValidateRegimeConditioning:
    """Tests for validate_regime_conditioning."""

    def test_well_conditioned(self, distinct_regimes: dict[int, np.ndarray]) -> None:
        """Synthetic matches real regime structure -> all_passed=True."""
        rng = np.random.default_rng(99)
        synthetic = {
            0: _make_lob_data(200, return_std=0.001, spread_mean=0.05, rng=rng),
            1: _make_lob_data(200, return_std=0.010, spread_mean=0.15, rng=rng),
            2: _make_lob_data(200, return_std=0.100, spread_mean=0.50, rng=rng),
        }
        result = validate_regime_conditioning(distinct_regimes, synthetic)
        assert result["regime_distinct"] is True
        assert result["ordering_preserved"] is True
        assert "details" in result

    def test_collapsed_regimes(self, distinct_regimes: dict[int, np.ndarray]) -> None:
        """All synthetic regimes identical -> regime_distinct=False."""
        collapsed = {}
        for r in range(3):
            rng = np.random.default_rng(99)
            collapsed[r] = _make_lob_data(
                200, return_std=0.005, spread_mean=0.10, rng=rng
            )
        result = validate_regime_conditioning(distinct_regimes, collapsed)
        assert result["regime_distinct"] is False
        assert result["all_passed"] is False
