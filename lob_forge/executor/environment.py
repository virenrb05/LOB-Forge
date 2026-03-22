"""LOBExecutionEnv: Gymnasium environment for optimal trade execution in real LOB data.

Action space (7 discrete actions):
  0: WAIT           — no execution, zero cost
  1: MARKET_SMALL   — market order, order_sizes[0] * remaining_inventory
  2: MARKET_MED     — market order, order_sizes[1] * remaining_inventory
  3: MARKET_LARGE   — market order, order_sizes[2] * remaining_inventory
  4: LIMIT_AGGRESSIVE — limit order, 50% fill prob, offset limit_offsets_bps[0] bps inside ask
  5: LIMIT_MID        — limit order, 30% fill prob, offset limit_offsets_bps[1] bps inside ask
  6: LIMIT_PASSIVE    — limit order, 10% fill prob, offset limit_offsets_bps[2] bps inside ask

Observation space:
  Box(low=-inf, high=inf, shape=(seq_len, 40), dtype=float32) — sliding window of LOB snapshots
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from lob_forge.executor.cost_model import CostModel

# Column indices in grouped LOB layout (40 columns):
#   ask_price cols: 0-9
#   ask_size  cols: 10-19
#   bid_price cols: 20-29
#   bid_size  cols: 30-39
_ASK1_COL: int = 0
_BID1_COL: int = 20

# Constant proxy ADV (Phase 9+ calibration concern)
_AVG_DAILY_VOLUME: float = 1_000_000.0

# Fill probabilities for limit actions 4-6
_LIMIT_FILL_PROBS: tuple[float, float, float] = (0.50, 0.30, 0.10)


class LOBExecutionEnv(gymnasium.Env):
    """Gymnasium environment for executing a liquidation task on real LOB data.

    Parameters
    ----------
    lob_data : np.ndarray
        Shape (N, 40), the full LOB sequence (40 book columns only, z-score normalised).
    seq_len : int
        Number of LOB snapshots in each observation window.
    inventory : float
        Total shares/units to execute per episode (liquidation task).
    horizon : int
        Max steps before episode truncates.
    order_sizes : tuple[float, float, float]
        Market order sizes as fraction of remaining inventory for actions 1-3.
    limit_offsets_bps : tuple[float, float, float]
        Limit order price offsets in bps inside the ask for actions 4-6.
    cost_model : CostModel | None
        Cost model instance; defaults to CostModel() if None.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    ACTION_NAMES: list[str] = [
        "WAIT",
        "MARKET_SMALL",
        "MARKET_MED",
        "MARKET_LARGE",
        "LIMIT_AGGRESSIVE",
        "LIMIT_MID",
        "LIMIT_PASSIVE",
    ]

    def __init__(
        self,
        lob_data: np.ndarray,
        seq_len: int = 100,
        inventory: float = 1000.0,
        horizon: int = 500,
        order_sizes: tuple[float, float, float] = (0.01, 0.05, 0.20),
        limit_offsets_bps: tuple[float, float, float] = (1, 5, 20),
        cost_model: CostModel | None = None,
    ) -> None:
        super().__init__()

        if lob_data.ndim != 2 or lob_data.shape[1] != 40:
            raise ValueError(f"lob_data must have shape (N, 40), got {lob_data.shape}")

        self.lob_data: np.ndarray = lob_data.astype(np.float32)
        self.seq_len: int = seq_len
        self.inventory: float = float(inventory)
        self.horizon: int = horizon
        self.order_sizes: tuple[float, float, float] = order_sizes
        self.limit_offsets_bps: tuple[float, float, float] = limit_offsets_bps
        self.cost_model: CostModel = (
            cost_model if cost_model is not None else CostModel()
        )

        # Gymnasium spaces
        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(seq_len, 40),
            dtype=np.float32,
        )
        self.action_space: spaces.Discrete = spaces.Discrete(7)

        # Episode state (initialised in reset)
        self._step: int = 0
        self._start: int = 0
        self._remaining: float = self.inventory
        self._executed_vwap: float = 0.0
        self._episode_cost: float = 0.0
        self._total_executed: float = 0.0  # track for VWAP numerator

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment for a new episode.

        Returns
        -------
        obs : np.ndarray
            Shape (seq_len, 40).
        info : dict
        """
        super().reset(seed=seed)

        n = len(self.lob_data)
        min_start = self.seq_len
        max_start = n - self.horizon - 1

        if max_start < min_start:
            raise ValueError(
                f"lob_data too short ({n} rows) for seq_len={self.seq_len} "
                f"and horizon={self.horizon}. Need at least "
                f"{self.seq_len + self.horizon + 1} rows."
            )

        self._start = int(self.np_random.integers(min_start, max_start + 1))
        self._step = self._start

        self._remaining = self.inventory
        self._executed_vwap = 0.0
        self._episode_cost = 0.0
        self._total_executed = 0.0

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Parameters
        ----------
        action : int
            Integer in [0, 6].

        Returns
        -------
        obs : np.ndarray  shape (seq_len, 40)
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        raw_lob = self.lob_data[self._step]

        ask_price: float = float(raw_lob[_ASK1_COL])
        bid_price: float = float(raw_lob[_BID1_COL])
        mid_price: float = (ask_price + bid_price) / 2.0
        spread: float = max(ask_price - bid_price, 0.0)
        avg_daily_volume: float = _AVG_DAILY_VOLUME

        exec_size: float = 0.0
        exec_price: float = 0.0
        cost: float = 0.0

        if action == 0:
            # WAIT: no execution
            pass

        elif 1 <= action <= 3:
            # Market orders
            size_frac = self.order_sizes[action - 1]
            exec_size = size_frac * self._remaining
            exec_price = ask_price
            cost = self.cost_model.compute(
                exec_price=exec_price,
                exec_size=exec_size,
                mid_price=mid_price,
                spread=spread,
                avg_daily_volume=avg_daily_volume,
            )

        elif 4 <= action <= 6:
            # Limit orders
            limit_idx = action - 4
            fill_prob = _LIMIT_FILL_PROBS[limit_idx]
            offset_bps = self.limit_offsets_bps[limit_idx]
            filled = bool(self.np_random.random() < fill_prob)

            if filled:
                exec_price = ask_price - offset_bps * 1e-4 * abs(ask_price)
                # Use the same size fraction as the corresponding market action
                size_frac = self.order_sizes[limit_idx]
                exec_size = size_frac * self._remaining
                cost = self.cost_model.compute(
                    exec_price=exec_price,
                    exec_size=exec_size,
                    mid_price=mid_price,
                    spread=spread,
                    avg_daily_volume=avg_daily_volume,
                )

        # Update state
        if exec_size > 0.0:
            # Update running VWAP
            new_total = self._total_executed + exec_size
            if new_total > 0.0:
                self._executed_vwap = (
                    self._executed_vwap * self._total_executed + exec_price * exec_size
                ) / new_total
            self._total_executed = new_total
            self._remaining = max(self._remaining - exec_size, 0.0)
            self._episode_cost += cost

        reward: float = -cost / self.inventory

        self._step += 1

        terminated: bool = self._remaining <= 1e-6
        truncated: bool = self._step >= self._start + self.horizon

        info: dict[str, Any] = {
            "remaining": self._remaining,
            "step": self._step,
            "episode_cost": self._episode_cost,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        """Render is not implemented for this environment."""
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Return sliding-window observation of shape (seq_len, 40).

        Pads with zeros at episode start when there is insufficient history.
        """
        start_idx = self._step - self.seq_len
        if start_idx >= 0:
            return self.lob_data[start_idx : self._step].copy()

        # Pad with zeros for the missing history
        available = self.lob_data[: self._step]
        pad_rows = self.seq_len - len(available)
        padding = np.zeros((pad_rows, 40), dtype=np.float32)
        return np.concatenate([padding, available], axis=0)
