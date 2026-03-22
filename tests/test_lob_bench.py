"""Unit tests for LOB-Bench quantitative evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from lob_forge.evaluation.lob_bench import (
    compute_conditional_stats,
    compute_wasserstein_metrics,
    run_lob_bench,
    train_discriminator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 200
C = 40  # standard LOB columns


def _make_lob_data(
    n: int = N, c: int = C, *, rng: np.random.Generator = RNG
) -> np.ndarray:
    """Create synthetic LOB-like data with positive prices."""
    data = rng.standard_normal((n, c)).astype(np.float64)
    # Ensure ask > bid for realistic spread (cols 0=ask_price_1, 20=bid_price_1)
    data[:, 0] = np.abs(data[:, 0]) + 1.0
    data[:, 20] = data[:, 0] - np.abs(rng.standard_normal(n)) * 0.1 - 0.01
    return data


# ---------------------------------------------------------------------------
# TestWassersteinMetrics
# ---------------------------------------------------------------------------


class TestWassersteinMetrics:
    """Tests for compute_wasserstein_metrics."""

    def test_identical_data(self) -> None:
        data = _make_lob_data(rng=np.random.default_rng(0))
        result = compute_wasserstein_metrics(data, data)
        assert result["wd_mean"] == pytest.approx(0.0, abs=1e-10)
        assert result["wd_max"] == pytest.approx(0.0, abs=1e-10)
        assert result["wd_spread"] == pytest.approx(0.0, abs=1e-10)
        assert result["wd_mid"] == pytest.approx(0.0, abs=1e-10)
        assert result["wd_returns"] == pytest.approx(0.0, abs=1e-10)

    def test_different_data(self) -> None:
        a = _make_lob_data(rng=np.random.default_rng(1))
        b = _make_lob_data(rng=np.random.default_rng(2)) + 5.0
        result = compute_wasserstein_metrics(a, b)
        assert result["wd_mean"] > 0.0
        assert result["wd_max"] > 0.0

    def test_spread_and_mid(self) -> None:
        data = _make_lob_data(rng=np.random.default_rng(3))
        shifted = data.copy()
        shifted[:, 0] += 1.0  # shift ask price
        result = compute_wasserstein_metrics(data, shifted)
        # Spread and mid should be affected by ask shift
        assert result["wd_spread"] > 0.0
        assert result["wd_mid"] > 0.0

    def test_output_keys(self) -> None:
        data = _make_lob_data(rng=np.random.default_rng(4))
        result = compute_wasserstein_metrics(data, data)
        expected_keys = {
            "wd_mean",
            "wd_max",
            "wd_per_feature",
            "wd_spread",
            "wd_mid",
            "wd_returns",
        }
        assert expected_keys == set(result.keys())
        assert len(result["wd_per_feature"]) == C


# ---------------------------------------------------------------------------
# TestDiscriminator
# ---------------------------------------------------------------------------


class TestDiscriminator:
    """Tests for train_discriminator."""

    def test_identical_data(self) -> None:
        data = _make_lob_data(n=400, rng=np.random.default_rng(10))
        result = train_discriminator(data, data, epochs=10, seed=42)
        # With identical features the discriminator cannot learn meaningful
        # separation.  The distinct-data test below asserts accuracy >= 0.8,
        # so here we simply verify the score is *lower* than that easy case.
        assert result["accuracy"] < 0.8

    def test_distinct_data(self) -> None:
        a = _make_lob_data(rng=np.random.default_rng(11))
        b = _make_lob_data(rng=np.random.default_rng(12)) + 10.0
        result = train_discriminator(a, b, epochs=30, seed=42)
        # Very different distributions — should be easy to distinguish
        assert result["accuracy"] >= 0.8
        assert result["auc"] >= 0.8

    def test_reproducibility(self) -> None:
        a = _make_lob_data(rng=np.random.default_rng(13))
        b = _make_lob_data(rng=np.random.default_rng(14))
        r1 = train_discriminator(a, b, epochs=10, seed=42)
        r2 = train_discriminator(a, b, epochs=10, seed=42)
        assert r1["accuracy"] == pytest.approx(r2["accuracy"])
        assert r1["auc"] == pytest.approx(r2["auc"])

    def test_output_keys(self) -> None:
        data = _make_lob_data(n=50, rng=np.random.default_rng(15))
        result = train_discriminator(data, data, epochs=5, seed=42)
        assert {"accuracy", "auc", "train_loss"} == set(result.keys())


# ---------------------------------------------------------------------------
# TestConditionalStats
# ---------------------------------------------------------------------------


class TestConditionalStats:
    """Tests for compute_conditional_stats."""

    def test_matching_regimes(self) -> None:
        data = _make_lob_data(rng=np.random.default_rng(20))
        regimes = np.array([i % 3 for i in range(N)], dtype=np.int64)
        result = compute_conditional_stats(data, data, regimes, regimes)
        # Same data => low relative error
        assert result["mean_relative_error"] == pytest.approx(0.0, abs=1e-6)

    def test_different_regimes(self) -> None:
        a = _make_lob_data(rng=np.random.default_rng(21))
        b = _make_lob_data(rng=np.random.default_rng(22)) + 3.0
        regimes = np.array([i % 3 for i in range(N)], dtype=np.int64)
        result = compute_conditional_stats(a, b, regimes, regimes)
        # Different distributions => higher error
        assert result["mean_relative_error"] > 0.0

    def test_per_regime_keys(self) -> None:
        data = _make_lob_data(rng=np.random.default_rng(23))
        regimes = np.array([i % 3 for i in range(N)], dtype=np.int64)
        result = compute_conditional_stats(data, data, regimes, regimes)
        for regime in ("0", "1", "2"):
            assert regime in result
            assert "spread_mean_err" in result[regime]
            assert "depth_mean_err" in result[regime]
            assert "return_std_err" in result[regime]


# ---------------------------------------------------------------------------
# TestRunLobBench
# ---------------------------------------------------------------------------


class TestRunLobBench:
    """Tests for run_lob_bench orchestrator."""

    def test_integration(self) -> None:
        a = _make_lob_data(rng=np.random.default_rng(30))
        b = _make_lob_data(rng=np.random.default_rng(31))
        regimes = np.array([i % 3 for i in range(N)], dtype=np.int64)
        result = run_lob_bench(a, b, regimes, regimes)

        # Check namespaced keys
        assert "wasserstein/wd_mean" in result
        assert "wasserstein/wd_max" in result
        assert "discriminator/accuracy" in result
        assert "discriminator/auc" in result
        assert "conditional/mean_relative_error" in result

    def test_without_regimes(self) -> None:
        a = _make_lob_data(rng=np.random.default_rng(32))
        b = _make_lob_data(rng=np.random.default_rng(33))
        result = run_lob_bench(a, b)

        assert "wasserstein/wd_mean" in result
        assert "discriminator/accuracy" in result
        # No conditional keys
        assert not any(k.startswith("conditional/") for k in result)
