"""CostModel: realistic execution cost calculation for LOB environments.

Three-component model:
  spread_cost  = 0.5 * spread * exec_size
  fee_cost     = fee_bps * 1e-4 * exec_price * exec_size
  impact_cost  = impact_eta * mid_price * sqrt(exec_size / avg_daily_volume) * exec_size
  total        = spread_cost + fee_cost + impact_cost  (always >= 0)
"""

import math
from dataclasses import dataclass


@dataclass
class CostModel:
    """Compute total execution cost for a single order.

    Parameters
    ----------
    fee_bps : float
        Exchange fee in basis points (1 bps = 0.01%). Default 2.0 bps.
    impact_eta : float
        Market impact coefficient (Kyle lambda variant). Default 0.1.
    """

    fee_bps: float = 2.0
    impact_eta: float = 0.1

    def compute(
        self,
        exec_price: float,
        exec_size: float,
        mid_price: float,
        spread: float,
        avg_daily_volume: float,
    ) -> float:
        """Compute total execution cost for a single order.

        Parameters
        ----------
        exec_price : float
            Price at which the order was executed.
        exec_size : float
            Number of units executed. Must be >= 0.
        mid_price : float
            Mid-price of the LOB at time of execution.
        spread : float
            Best bid-ask spread at time of execution.
        avg_daily_volume : float
            Average daily volume used to compute participation rate.
            Must be > 0.

        Returns
        -------
        float
            Total execution cost >= 0.

        Raises
        ------
        ValueError
            If exec_size < 0.
        """
        if exec_size < 0:
            raise ValueError(f"exec_size must be >= 0, got {exec_size}")

        if exec_size == 0.0:
            return 0.0

        # Spread cost: half-spread model
        spread_cost: float = 0.5 * spread * exec_size

        # Exchange fee: fixed bps fraction of notional
        fee_cost: float = self.fee_bps * 1e-4 * exec_price * exec_size

        # Market impact: square-root participation rate model
        participation_rate: float = exec_size / avg_daily_volume
        impact_cost: float = (
            self.impact_eta * mid_price * math.sqrt(participation_rate) * exec_size
        )

        total: float = spread_cost + fee_cost + impact_cost
        return total
