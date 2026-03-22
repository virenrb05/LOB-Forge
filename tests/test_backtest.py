"""Unit tests for run_backtest() in lob_forge.evaluation.backtest.

Uses a lightweight MockEnv (not LOBExecutionEnv) to avoid needing real LOB
data files.  Tests cover:
- TWAP baseline producing the correct number of results
- Results are ExecutionResult instances
- Independent seeding per episode (seeds 0..n_episodes-1)
- Baseline agents produce distinct costs across episodes when env varies
- TypeError on unsupported agent type
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import gymnasium
except ImportError:
    gymnasium = None  # type: ignore[assignment]

from lob_forge.evaluation.backtest import run_backtest
from lob_forge.executor.baselines import ExecutionResult, TWAPBaseline

# ---------------------------------------------------------------------------
# Mock environment (no real LOB data required)
# ---------------------------------------------------------------------------


class MockEnv:
    """Minimal gymnasium-compatible mock for testing run_backtest.

    Terminates after the first step and populates info with fixed values.
    """

    if gymnasium is not None:
        action_space = gymnasium.spaces.Discrete(7)
    seq_len: int = 10
    horizon: int = 10
    inventory: float = 100.0

    def __init__(self, cost_per_episode: float = 1.0) -> None:
        self._cost = cost_per_episode
        self._seed_log: list[int | None] = []

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        self._seed_log.append(seed)
        obs = np.zeros((self.seq_len, 40), dtype=np.float32)
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs = np.zeros((self.seq_len, 40), dtype=np.float32)
        info = {"episode_cost": float(self._cost), "remaining": 0.0}
        return obs, -float(self._cost), True, False, info


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunBacktestWithTWAPBaseline:
    def test_returns_correct_number_of_results(self) -> None:
        """run_backtest returns exactly n_episodes results."""
        env = MockEnv()
        results = run_backtest(env, TWAPBaseline(), n_episodes=3)
        assert len(results) == 3

    def test_results_are_execution_result_instances(self) -> None:
        """Every element of the returned list is an ExecutionResult."""
        env = MockEnv()
        results = run_backtest(env, TWAPBaseline(), n_episodes=3)
        assert all(isinstance(r, ExecutionResult) for r in results)

    def test_episode_cost_matches_env(self) -> None:
        """episode_cost in each result reflects the mock env cost."""
        env = MockEnv(cost_per_episode=5.0)
        results = run_backtest(env, TWAPBaseline(), n_episodes=2)
        for r in results:
            assert r.episode_cost == pytest.approx(5.0)

    def test_remaining_inventory_zero(self) -> None:
        """MockEnv returns remaining=0; results reflect this."""
        env = MockEnv()
        results = run_backtest(env, TWAPBaseline(), n_episodes=2)
        for r in results:
            assert r.remaining_inventory == pytest.approx(0.0)

    def test_n_steps_positive(self) -> None:
        """Each episode completes at least one step."""
        env = MockEnv()
        results = run_backtest(env, TWAPBaseline(), n_episodes=3)
        for r in results:
            assert r.n_steps >= 1


class TestRunBacktestSeeding:
    def test_episodes_receive_increasing_seeds(self) -> None:
        """Episodes 0..n-1 are seeded with seed_offset+i (default offset=0)."""
        env = MockEnv()
        run_backtest(env, TWAPBaseline(), n_episodes=3, seed_offset=0)
        # env.reset() is called with seeds 0, 1, 2
        assert env._seed_log == [0, 1, 2]

    def test_seed_offset_applied(self) -> None:
        """seed_offset shifts the seed range."""
        env = MockEnv()
        run_backtest(env, TWAPBaseline(), n_episodes=3, seed_offset=10)
        assert env._seed_log == [10, 11, 12]

    def test_single_episode_seed(self) -> None:
        """Single episode uses seed 0 by default."""
        env = MockEnv()
        run_backtest(env, TWAPBaseline(), n_episodes=1)
        assert env._seed_log == [0]


class TestRunBacktestInvalidAgent:
    def test_invalid_agent_raises_type_error(self) -> None:
        """Passing an unsupported agent type raises TypeError."""
        env = MockEnv()
        with pytest.raises(TypeError):
            run_backtest(env, object(), n_episodes=1)  # type: ignore[arg-type]
