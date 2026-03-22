"""Tests for execution baselines: TWAP, VWAP, AlmgrenChriss, Random."""

from __future__ import annotations

import numpy as np
import pytest

from lob_forge.executor.baselines import (
    AlmgrenChrissBaseline,
    ExecutionResult,
    RandomBaseline,
    TWAPBaseline,
    VWAPBaseline,
)
from lob_forge.executor.environment import LOBExecutionEnv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N_ROWS = 2000
_SEQ_LEN = 50
_HORIZON = 200
_INVENTORY = 1000.0


@pytest.fixture(scope="module")
def lob_data() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((_N_ROWS, 40)).astype(np.float32)


@pytest.fixture()
def env(lob_data: np.ndarray) -> LOBExecutionEnv:
    return LOBExecutionEnv(
        lob_data=lob_data,
        seq_len=_SEQ_LEN,
        horizon=_HORIZON,
        inventory=_INVENTORY,
    )


# ---------------------------------------------------------------------------
# ExecutionResult typing
# ---------------------------------------------------------------------------


class TestExecutionResult:
    def test_fields_types(self, env: LOBExecutionEnv) -> None:
        result = TWAPBaseline().run_episode(env, seed=0)
        assert isinstance(result.episode_cost, float)
        assert isinstance(result.implementation_shortfall, float)
        assert isinstance(result.remaining_inventory, float)
        assert isinstance(result.n_steps, int)
        assert isinstance(result.actions, list)

    def test_n_steps_positive(self, env: LOBExecutionEnv) -> None:
        result = TWAPBaseline().run_episode(env, seed=0)
        assert result.n_steps > 0

    def test_remaining_non_negative(self, env: LOBExecutionEnv) -> None:
        result = TWAPBaseline().run_episode(env, seed=0)
        assert result.remaining_inventory >= 0.0

    def test_actions_length_matches_n_steps(self, env: LOBExecutionEnv) -> None:
        result = RandomBaseline().run_episode(env, seed=0)
        assert len(result.actions) == result.n_steps


# ---------------------------------------------------------------------------
# All 4 baselines return ExecutionResult
# ---------------------------------------------------------------------------


class TestAllBaselinesReturnResult:
    @pytest.mark.parametrize(
        "baseline_cls",
        [TWAPBaseline, VWAPBaseline, AlmgrenChrissBaseline, RandomBaseline],
    )
    def test_returns_execution_result(
        self, baseline_cls: type, env: LOBExecutionEnv
    ) -> None:
        baseline = baseline_cls()
        result = baseline.run_episode(env, seed=0)
        assert isinstance(result, ExecutionResult)

    @pytest.mark.parametrize(
        "baseline_cls",
        [TWAPBaseline, VWAPBaseline, AlmgrenChrissBaseline, RandomBaseline],
    )
    def test_n_steps_bounded_by_horizon(
        self, baseline_cls: type, env: LOBExecutionEnv
    ) -> None:
        baseline = baseline_cls()
        result = baseline.run_episode(env, seed=0)
        assert 1 <= result.n_steps <= _HORIZON


# ---------------------------------------------------------------------------
# TWAP always makes progress
# ---------------------------------------------------------------------------


class TestTWAPBaseline:
    def test_executes_inventory(self, env: LOBExecutionEnv) -> None:
        result = TWAPBaseline().run_episode(env, seed=0)
        # TWAP always sends MARKET_SMALL; inventory must decrease
        assert result.remaining_inventory < _INVENTORY

    def test_all_actions_are_market_small_or_wait(self, env: LOBExecutionEnv) -> None:
        result = TWAPBaseline().run_episode(env, seed=0)
        for action in result.actions:
            assert action in (0, 1), f"Unexpected action: {action}"


# ---------------------------------------------------------------------------
# VWAP baseline
# ---------------------------------------------------------------------------


class TestVWAPBaseline:
    def test_executes_inventory(self, env: LOBExecutionEnv) -> None:
        result = VWAPBaseline().run_episode(env, seed=0)
        assert result.remaining_inventory < _INVENTORY

    def test_actions_are_valid(self, env: LOBExecutionEnv) -> None:
        result = VWAPBaseline().run_episode(env, seed=0)
        for action in result.actions:
            assert action in (0, 1)


# ---------------------------------------------------------------------------
# RandomBaseline actions are in [0, 6]
# ---------------------------------------------------------------------------


class TestRandomBaseline:
    def test_actions_in_valid_range(self, env: LOBExecutionEnv) -> None:
        result = RandomBaseline().run_episode(env, seed=0)
        for action in result.actions:
            assert 0 <= action <= 6, f"Action {action} out of range"

    def test_all_action_types_possible(self, env: LOBExecutionEnv) -> None:
        """Over many steps, random should sample multiple distinct actions."""
        result = RandomBaseline().run_episode(env, seed=42)
        unique_actions = set(result.actions)
        assert len(unique_actions) > 1, "Random should produce varied actions"


# ---------------------------------------------------------------------------
# AlmgrenChriss baseline
# ---------------------------------------------------------------------------


class TestAlmgrenChrissBaseline:
    def test_executes_inventory(self, env: LOBExecutionEnv) -> None:
        result = AlmgrenChrissBaseline().run_episode(env, seed=0)
        assert result.remaining_inventory < _INVENTORY

    def test_trajectory_length_equals_horizon(self, env: LOBExecutionEnv) -> None:
        ac = AlmgrenChrissBaseline()
        ac.reset_episode(env.inventory, env.horizon)
        assert len(ac._deltas) == env.horizon

    def test_trajectory_sums_to_inventory(self, env: LOBExecutionEnv) -> None:
        ac = AlmgrenChrissBaseline()
        ac.reset_episode(env.inventory, env.horizon)
        assert abs(ac._deltas.sum() - env.inventory) < 1e-4

    def test_actions_are_market_med_or_wait(self, env: LOBExecutionEnv) -> None:
        result = AlmgrenChrissBaseline().run_episode(env, seed=0)
        for action in result.actions:
            assert action in (0, 2), f"Unexpected action: {action}"


# ---------------------------------------------------------------------------
# Reproducibility: same seed -> same result
# ---------------------------------------------------------------------------


class TestReproducibility:
    @pytest.mark.parametrize(
        "baseline_cls",
        [TWAPBaseline, VWAPBaseline, AlmgrenChrissBaseline],
    )
    def test_same_seed_same_cost(
        self, baseline_cls: type, lob_data: np.ndarray
    ) -> None:
        env1 = LOBExecutionEnv(
            lob_data=lob_data, seq_len=_SEQ_LEN, horizon=_HORIZON, inventory=_INVENTORY
        )
        env2 = LOBExecutionEnv(
            lob_data=lob_data, seq_len=_SEQ_LEN, horizon=_HORIZON, inventory=_INVENTORY
        )
        r1 = baseline_cls().run_episode(env1, seed=7)
        r2 = baseline_cls().run_episode(env2, seed=7)
        assert r1.episode_cost == pytest.approx(r2.episode_cost, rel=1e-6)
        assert r1.n_steps == r2.n_steps
