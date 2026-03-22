"""Execution baselines: TWAP, VWAP, Almgren-Chriss, and Random.

Each baseline implements a ``run_episode(env, seed)`` method that runs a full
episode in a :class:`~lob_forge.executor.environment.LOBExecutionEnv` and
returns an :class:`ExecutionResult` dataclass.

These baselines provide the performance floor that the DQN agent must beat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lob_forge.executor.environment import LOBExecutionEnv


@dataclass
class ExecutionResult:
    """Aggregated metrics from a single execution episode.

    Attributes
    ----------
    episode_cost : float
        Total transaction cost paid (from CostModel).
    implementation_shortfall : float
        ``(exec_vwap - arrival_price) * inventory``. Positive means executed at
        prices above the arrival mid-price (worse); negative means better.
    remaining_inventory : float
        Shares/units not executed by episode end. 0.0 = fully liquidated.
    n_steps : int
        Number of environment steps taken before termination/truncation.
    actions : list[int]
        Sequence of actions taken during the episode.
    """

    episode_cost: float
    implementation_shortfall: float
    remaining_inventory: float
    n_steps: int
    actions: list[int] = field(default_factory=list)


class BaselineAgent:
    """Abstract base class for all execution baselines."""

    def select_action(self, obs: np.ndarray, remaining: float, step: int) -> int:
        """Select an action given the current observation and state.

        Parameters
        ----------
        obs : np.ndarray
            Current observation, shape ``(seq_len, 40)``.
        remaining : float
            Remaining inventory to execute.
        step : int
            Current step index within the episode (0-indexed).

        Returns
        -------
        int
            Action in ``[0, 6]``.
        """
        raise NotImplementedError

    def run_episode(self, env: LOBExecutionEnv, seed: int = 0) -> ExecutionResult:
        """Run a full episode in *env* and return aggregated metrics.

        Parameters
        ----------
        env : LOBExecutionEnv
            A gymnasium environment following the LOBExecutionEnv interface.
        seed : int
            Seed passed to ``env.reset()`` for reproducibility.

        Returns
        -------
        ExecutionResult
        """
        obs, info = env.reset(seed=seed)

        # Arrival price: mid from last row of initial observation
        # Grouped layout: col 0 = ask_1, col 20 = bid_1
        arrival_price: float = (float(obs[-1, 0]) + float(obs[-1, 20])) / 2.0

        actions: list[int] = []
        step: int = 0

        while True:
            remaining = (
                info.get("remaining", env.inventory) if step > 0 else env.inventory
            )
            action = self.select_action(obs, remaining, step)
            obs, _reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            step += 1
            if terminated or truncated:
                break

        executed = env.inventory - info["remaining"]
        exec_vwap_approx = arrival_price + (info["episode_cost"] / max(executed, 1e-9))
        is_shortfall = (exec_vwap_approx - arrival_price) * env.inventory

        return ExecutionResult(
            episode_cost=info["episode_cost"],
            implementation_shortfall=is_shortfall,
            remaining_inventory=info["remaining"],
            n_steps=step,
            actions=actions,
        )


class TWAPBaseline(BaselineAgent):
    """Time-Weighted Average Price (TWAP) baseline.

    Executes using MARKET_SMALL (action 1, 1 % of remaining) at every step
    until inventory is exhausted.  If remaining is already zero, it waits.
    This is the simplest uniform-pacing approximation of TWAP.
    """

    def select_action(self, obs: np.ndarray, remaining: float, step: int) -> int:
        if remaining <= 0.0:
            return 0  # WAIT — nothing left
        return 1  # MARKET_SMALL


class VWAPBaseline(BaselineAgent):
    """Volume-Weighted Average Price (VWAP) baseline.

    Approximates intraday volume variation with a sinusoidal profile plus
    small random noise.  The volume schedule is precomputed in ``__init__``
    so that episodes are reproducible given a fixed horizon.

    Parameters
    ----------
    horizon : int
        Episode horizon (number of steps).  Defaults to 500 to match the
        environment default; will be overridden lazily if a different horizon
        is detected on first use.
    seed : int
        Seed for the volume-proxy RNG (default 42 for reproducibility).
    """

    def __init__(self, horizon: int = 500, seed: int = 42) -> None:
        self._horizon = horizon
        self._seed = seed
        self._target_fractions: np.ndarray = self._compute_schedule(horizon, seed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_schedule(horizon: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        weights = np.array(
            [
                max(0.01, math.sin(math.pi * t / horizon) + 0.1 * rng.random())
                for t in range(horizon)
            ]
        )
        return weights / weights.sum()

    def _ensure_horizon(self, env: LOBExecutionEnv) -> None:
        """Lazily recompute schedule if the env horizon differs."""
        if env.horizon != self._horizon:
            self._horizon = env.horizon
            self._target_fractions = self._compute_schedule(self._horizon, self._seed)

    # ------------------------------------------------------------------
    # BaselineAgent interface
    # ------------------------------------------------------------------

    def run_episode(self, env: LOBExecutionEnv, seed: int = 0) -> ExecutionResult:
        self._ensure_horizon(env)
        return super().run_episode(env, seed=seed)

    def select_action(self, obs: np.ndarray, remaining: float, step: int) -> int:
        if remaining <= 0.0:
            return 0  # WAIT

        if step >= len(self._target_fractions):
            return 0  # beyond schedule — wait

        # Execute if this step has non-zero target volume allocation
        if self._target_fractions[step] > 0.0:
            return 1  # MARKET_SMALL
        return 0  # WAIT


class AlmgrenChrissBaseline(BaselineAgent):
    """Almgren-Chriss optimal liquidation baseline.

    Computes a deterministic liquidation trajectory from the closed-form
    Almgren-Chriss model and maps trajectory deltas to discrete actions.

    Parameters
    ----------
    eta : float
        Temporary market impact coefficient.
    sigma : float
        Volatility estimate (annualised fraction).
    lam : float
        Risk-aversion coefficient.
    """

    def __init__(
        self,
        eta: float = 0.1,
        sigma: float = 0.3,
        lam: float = 1e-5,
    ) -> None:
        self.eta = eta
        self.sigma = sigma
        self.lam = lam

        # Episode-level trajectory, set by reset_episode()
        self._deltas: np.ndarray = np.array([])

    # ------------------------------------------------------------------
    # Trajectory helpers
    # ------------------------------------------------------------------

    def reset_episode(self, inventory: float, horizon: int) -> None:
        """Precompute the Almgren-Chriss liquidation schedule.

        ``n_j`` gives the remaining inventory at step *j*; ``delta_j`` is the
        trade size executed at step *j*.

        kappa = sqrt(lam * sigma^2 / eta)
        n_j   = inventory * sinh(kappa * (horizon - j)) / sinh(kappa * horizon)
        delta_j = n_j - n_{j+1}
        """
        kappa = math.sqrt(self.lam * self.sigma**2 / self.eta)
        sinh_total = math.sinh(kappa * horizon)

        # Protect against degenerate case (kappa ≈ 0 → uniform schedule)
        if sinh_total < 1e-12:
            deltas = np.full(horizon, inventory / horizon)
        else:
            n = np.array(
                [
                    inventory * math.sinh(kappa * (horizon - j)) / sinh_total
                    for j in range(horizon + 1)
                ]
            )
            deltas = n[:-1] - n[1:]

        self._deltas = deltas

    # ------------------------------------------------------------------
    # BaselineAgent interface
    # ------------------------------------------------------------------

    def run_episode(self, env: LOBExecutionEnv, seed: int = 0) -> ExecutionResult:
        self.reset_episode(env.inventory, env.horizon)
        return super().run_episode(env, seed=seed)

    def select_action(self, obs: np.ndarray, remaining: float, step: int) -> int:
        if remaining <= 0.0:
            return 0  # WAIT

        if step >= len(self._deltas):
            return 0  # beyond schedule

        delta = self._deltas[step]
        # Execute if this step's Almgren-Chriss trade is meaningful.
        # Threshold: half the uniform TWAP rate (inventory / horizon / 2).
        # This allows the AC schedule to deviate from uniform while still
        # capturing most of the trajectory.
        inventory_total = self._deltas.sum()
        horizon = len(self._deltas)
        threshold = 0.5 * inventory_total / max(horizon, 1)
        if delta > threshold:
            return 2  # MARKET_MED
        return 0  # WAIT


class RandomBaseline(BaselineAgent):
    """Uniform-random baseline.

    Samples an action uniformly from the environment's discrete action space
    ``[0, 6]`` at every step.  Relies on gymnasium's seeded ``action_space``
    RNG for reproducibility.
    """

    def __init__(self) -> None:
        self._env: LOBExecutionEnv | None = None

    def run_episode(self, env: LOBExecutionEnv, seed: int = 0) -> ExecutionResult:
        self._env = env
        return super().run_episode(env, seed=seed)

    def select_action(self, obs: np.ndarray, remaining: float, step: int) -> int:
        if self._env is None:
            raise RuntimeError("Call run_episode() before select_action().")
        return int(self._env.action_space.sample())


__all__ = [
    "ExecutionResult",
    "BaselineAgent",
    "TWAPBaseline",
    "VWAPBaseline",
    "AlmgrenChrissBaseline",
    "RandomBaseline",
]
