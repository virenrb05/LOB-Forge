"""Execution performance metrics for LOB trading agents.

Provides IS (Implementation Shortfall), IS Sharpe, and slippage-vs-TWAP
computed over lists of :class:`~lob_forge.executor.baselines.ExecutionResult`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lob_forge.executor.baselines import ExecutionResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_implementation_shortfall(results: list[ExecutionResult]) -> dict:
    """Compute IS statistics over a list of execution episodes.

    Parameters
    ----------
    results : list[ExecutionResult]
        Episodes from a single agent (baseline or DQN).

    Returns
    -------
    dict
        Keys:

        - ``"is_mean"`` – mean episode cost across episodes.
        - ``"is_std"``  – standard deviation of episode cost.
        - ``"is_sharpe"`` – IS Sharpe ratio (mean / std); 0.0 when std is 0.
        - ``"slippage_vs_twap"`` – set to ``float("nan")``.  Populate via
          :func:`compute_slippage_vs_twap` after computing TWAP results.
    """
    if not results:
        return {
            "is_mean": float("nan"),
            "is_std": float("nan"),
            "is_sharpe": 0.0,
            "slippage_vs_twap": float("nan"),
        }

    costs = np.array([r.episode_cost for r in results], dtype=float)
    is_mean = float(np.mean(costs))
    is_std = float(np.std(costs, ddof=0))
    is_sharpe = is_mean / is_std if is_std > 0.0 else 0.0

    return {
        "is_mean": is_mean,
        "is_std": is_std,
        "is_sharpe": is_sharpe,
        "slippage_vs_twap": float("nan"),
    }


def compute_is_sharpe(results: list[ExecutionResult]) -> float:
    """Return the IS Sharpe ratio for *results*.

    IS Sharpe = mean(episode_cost) / std(episode_cost) across episodes.
    Returns ``0.0`` when the standard deviation is zero (or results is empty).

    Parameters
    ----------
    results : list[ExecutionResult]
        Episodes from a single agent.

    Returns
    -------
    float
        IS Sharpe ratio.
    """
    if not results:
        return 0.0

    costs = np.array([r.episode_cost for r in results], dtype=float)
    mean_cost = float(np.mean(costs))
    std_cost = float(np.std(costs, ddof=0))

    if std_cost == 0.0 or math.isnan(std_cost):
        return 0.0

    return mean_cost / std_cost


def compute_slippage_vs_twap(
    agent_results: list[ExecutionResult],
    twap_results: list[ExecutionResult],
) -> float:
    """Return slippage of *agent_results* relative to *twap_results*.

    slippage = (agent_mean_cost - twap_mean_cost) / twap_mean_cost

    A **negative** value means the agent executed at lower cost than TWAP
    (better).  Returns ``float("nan")`` when *twap_results* is empty or the
    TWAP mean cost is zero.

    Parameters
    ----------
    agent_results : list[ExecutionResult]
        Episodes from the agent being evaluated.
    twap_results : list[ExecutionResult]
        Episodes from the TWAP baseline.

    Returns
    -------
    float
        Relative slippage vs TWAP.
    """
    if not agent_results or not twap_results:
        return float("nan")

    agent_mean = float(np.mean([r.episode_cost for r in agent_results]))
    twap_mean = float(np.mean([r.episode_cost for r in twap_results]))

    if twap_mean == 0.0 or math.isnan(twap_mean):
        return float("nan")

    return (agent_mean - twap_mean) / twap_mean


__all__ = [
    "compute_implementation_shortfall",
    "compute_is_sharpe",
    "compute_slippage_vs_twap",
]
