"""Unit tests for evaluation metrics: IS, IS Sharpe, slippage-vs-TWAP.

Verifies mathematical correctness of:
- compute_implementation_shortfall
- compute_is_sharpe
- compute_slippage_vs_twap
"""

from __future__ import annotations

import pytest

from lob_forge.evaluation.metrics import (
    compute_implementation_shortfall,
    compute_is_sharpe,
    compute_slippage_vs_twap,
)
from lob_forge.executor.baselines import ExecutionResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(episode_cost: float) -> ExecutionResult:
    """Create a minimal ExecutionResult with known episode_cost."""
    return ExecutionResult(
        episode_cost=episode_cost,
        implementation_shortfall=episode_cost,
        remaining_inventory=0,
        n_steps=10,
        actions=[],
    )


# ---------------------------------------------------------------------------
# compute_implementation_shortfall
# ---------------------------------------------------------------------------


class TestComputeImplementationShortfall:
    def test_known_values(self) -> None:
        """IS mean/std/sharpe are mathematically correct for [10, 20, 30]."""
        results = [_make_result(c) for c in [10.0, 20.0, 30.0]]
        metrics = compute_implementation_shortfall(results)

        assert metrics["is_mean"] == pytest.approx(20.0)
        # std (population, ddof=0): sqrt(200/3) ≈ 8.1650
        assert metrics["is_std"] == pytest.approx(8.16496580928, rel=1e-4)
        # sharpe: 20 / 8.1650 ≈ 2.449
        assert metrics["is_sharpe"] == pytest.approx(2.449, rel=1e-3)

    def test_output_keys(self) -> None:
        """Dict contains is_mean, is_std, is_sharpe (and optionally slippage_vs_twap)."""
        results = [_make_result(5.0), _make_result(10.0)]
        metrics = compute_implementation_shortfall(results)
        assert "is_mean" in metrics
        assert "is_std" in metrics
        assert "is_sharpe" in metrics

    def test_single_episode(self) -> None:
        """Single episode: is_mean equals episode_cost, is_sharpe is 0."""
        results = [_make_result(7.5)]
        metrics = compute_implementation_shortfall(results)
        assert metrics["is_mean"] == pytest.approx(7.5)
        assert metrics["is_sharpe"] == pytest.approx(0.0)

    def test_is_mean_float(self) -> None:
        """is_mean returns a Python float."""
        results = [_make_result(1.0), _make_result(3.0)]
        metrics = compute_implementation_shortfall(results)
        assert isinstance(metrics["is_mean"], float)

    def test_is_std_non_negative(self) -> None:
        """is_std is always non-negative."""
        results = [_make_result(c) for c in [5.0, 10.0, 15.0]]
        metrics = compute_implementation_shortfall(results)
        assert metrics["is_std"] >= 0.0


# ---------------------------------------------------------------------------
# compute_is_sharpe
# ---------------------------------------------------------------------------


class TestComputeIsSharpe:
    def test_zero_variance(self) -> None:
        """All identical costs → IS Sharpe == 0.0 (no ZeroDivisionError)."""
        results = [_make_result(5.0) for _ in range(5)]
        assert compute_is_sharpe(results) == pytest.approx(0.0)

    def test_known_values(self) -> None:
        """IS Sharpe for [10, 20, 30] ≈ 2.449."""
        results = [_make_result(c) for c in [10.0, 20.0, 30.0]]
        sharpe = compute_is_sharpe(results)
        assert sharpe == pytest.approx(2.449, rel=1e-3)

    def test_returns_float(self) -> None:
        """Return type is float."""
        results = [_make_result(1.0), _make_result(2.0)]
        assert isinstance(compute_is_sharpe(results), float)

    def test_positive_when_positive_mean(self) -> None:
        """Positive costs → positive Sharpe."""
        results = [_make_result(c) for c in [1.0, 2.0, 3.0]]
        assert compute_is_sharpe(results) > 0.0

    def test_consistent_with_is_dict(self) -> None:
        """compute_is_sharpe matches is_sharpe inside compute_implementation_shortfall."""
        results = [_make_result(c) for c in [2.0, 4.0, 6.0, 8.0]]
        sharpe_standalone = compute_is_sharpe(results)
        metrics = compute_implementation_shortfall(results)
        assert sharpe_standalone == pytest.approx(metrics["is_sharpe"])


# ---------------------------------------------------------------------------
# compute_slippage_vs_twap
# ---------------------------------------------------------------------------


class TestComputeSlippageVsTwap:
    def test_agent_better_than_twap(self) -> None:
        """agent_mean_cost=8, twap_mean_cost=10 → slippage == -0.2."""
        agent = [_make_result(8.0)]
        twap = [_make_result(10.0)]
        slippage = compute_slippage_vs_twap(agent, twap)
        assert slippage == pytest.approx(-0.2)

    def test_agent_worse_than_twap(self) -> None:
        """agent_mean_cost=12, twap_mean_cost=10 → slippage == 0.2."""
        agent = [_make_result(12.0)]
        twap = [_make_result(10.0)]
        slippage = compute_slippage_vs_twap(agent, twap)
        assert slippage == pytest.approx(0.2)

    def test_equal_cost(self) -> None:
        """Same cost → slippage == 0.0."""
        results = [_make_result(10.0)]
        slippage = compute_slippage_vs_twap(results, results)
        assert slippage == pytest.approx(0.0)

    def test_multiple_episodes_mean(self) -> None:
        """Slippage uses mean across episodes."""
        agent = [_make_result(8.0), _make_result(12.0)]  # mean = 10
        twap = [_make_result(10.0), _make_result(10.0)]  # mean = 10
        slippage = compute_slippage_vs_twap(agent, twap)
        assert slippage == pytest.approx(0.0)

    def test_negative_is_better(self) -> None:
        """Negative slippage means agent outperforms TWAP."""
        agent = [_make_result(c) for c in [6.0, 7.0, 8.0]]  # mean = 7
        twap = [_make_result(10.0)]
        slippage = compute_slippage_vs_twap(agent, twap)
        assert slippage < 0.0

    def test_returns_float(self) -> None:
        """Return type is float."""
        agent = [_make_result(5.0)]
        twap = [_make_result(10.0)]
        assert isinstance(compute_slippage_vs_twap(agent, twap), float)
