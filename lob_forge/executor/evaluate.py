"""Agent evaluation and baseline comparison for LOB execution.

Provides two top-level functions:

- ``evaluate_agent(checkpoint_path, env, n_episodes, device)``
  Loads a trained DuelingDQN checkpoint and runs greedy evaluation.

- ``compare_to_baselines(checkpoint_path, env, n_episodes, device)``
  Benchmarks the DQN agent against all 4 execution baselines (TWAP, VWAP,
  Almgren-Chriss, Random) and returns a comparison dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from lob_forge.executor.agent import DuelingDQN
from lob_forge.executor.baselines import (
    AlmgrenChrissBaseline,
    ExecutionResult,
    RandomBaseline,
    TWAPBaseline,
    VWAPBaseline,
)

if TYPE_CHECKING:
    from lob_forge.executor.environment import LOBExecutionEnv


# ---------------------------------------------------------------------------
# evaluate_agent
# ---------------------------------------------------------------------------


def evaluate_agent(
    checkpoint_path: str | Path,
    env: LOBExecutionEnv,
    n_episodes: int = 10,
    device: str = "cpu",
) -> list[ExecutionResult]:
    """Load a trained checkpoint and run the DQN agent for *n_episodes*.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to a checkpoint file saved by ``train_agent()``.  Must contain an
        ``"online_net"`` key with DuelingDQN state dict.
    env : LOBExecutionEnv
        Environment to evaluate in.  Episodes are seeded 0 through
        ``n_episodes - 1`` for reproducibility.
    n_episodes : int
        Number of episodes to run (default 10).
    device : str
        Torch device string, e.g. ``"cpu"`` or ``"cuda"`` (default ``"cpu"``).

    Returns
    -------
    list[ExecutionResult]
        One :class:`~lob_forge.executor.baselines.ExecutionResult` per episode.
    """
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)

    obs_shape = (env.seq_len, 40)
    net = DuelingDQN(obs_shape=obs_shape, n_actions=7)
    net.load_state_dict(ckpt["online_net"])
    net.to(device)
    net.eval()

    results: list[ExecutionResult] = []

    for episode_idx in range(n_episodes):
        obs, info = env.reset(seed=episode_idx)

        # Arrival price from the last LOB snapshot in the initial observation.
        # Grouped layout: col 0 = ask_1, col 20 = bid_1.
        arrival_price: float = (float(obs[-1, 0]) + float(obs[-1, 20])) / 2.0

        actions: list[int] = []
        step: int = 0

        while True:
            obs_tensor = (
                torch.from_numpy(obs).unsqueeze(0).float().to(device)
            )  # (1, seq_len, 40)
            with torch.no_grad():
                q_values = net(obs_tensor)  # (1, n_actions)
            action = int(q_values.argmax(dim=1).item())

            obs, _reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            step += 1

            if terminated or truncated:
                break

        executed = env.inventory - info["remaining"]
        exec_vwap_approx = arrival_price + (
            info["episode_cost"] / max(executed, 1e-9)
        )
        is_shortfall = (exec_vwap_approx - arrival_price) * env.inventory

        results.append(
            ExecutionResult(
                episode_cost=info["episode_cost"],
                implementation_shortfall=is_shortfall,
                remaining_inventory=info["remaining"],
                n_steps=step,
                actions=actions,
            )
        )

    return results


# ---------------------------------------------------------------------------
# compare_to_baselines
# ---------------------------------------------------------------------------


def _run_baseline_episodes(
    baseline,
    env: LOBExecutionEnv,
    n_episodes: int,
) -> list[ExecutionResult]:
    """Run *baseline* for *n_episodes* and return the results."""
    return [baseline.run_episode(env, seed=i) for i in range(n_episodes)]


def _mean_metrics(
    results: list[ExecutionResult],
) -> tuple[float, float]:
    """Return (mean_cost, mean_implementation_shortfall) for *results*."""
    mean_cost = float(np.mean([r.episode_cost for r in results]))
    mean_is = float(np.mean([r.implementation_shortfall for r in results]))
    return mean_cost, mean_is


def compare_to_baselines(
    checkpoint_path: str | Path,
    env: LOBExecutionEnv,
    n_episodes: int = 10,
    device: str = "cpu",
) -> dict:
    """Benchmark the DQN agent against all 4 execution baselines.

    Runs the DQN agent and each baseline (TWAP, VWAP, Almgren-Chriss, Random)
    for *n_episodes* episodes on seeds 0 through ``n_episodes - 1``.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the DQN checkpoint file.
    env : LOBExecutionEnv
        Evaluation environment.
    n_episodes : int
        Number of episodes per agent (default 10).
    device : str
        Torch device for the DQN inference (default ``"cpu"``).

    Returns
    -------
    dict
        Keys: ``"dqn"``, ``"twap"``, ``"vwap"``, ``"almgren_chriss"``,
        ``"random"``, ``"dqn_beats_twap"``.

        Each agent entry contains::

            {
                "mean_cost": float,
                "mean_is": float,
                "results": list[ExecutionResult],
            }

        ``"dqn_beats_twap"`` is ``True`` when the DQN mean cost is strictly
        less than the TWAP mean cost.
    """
    # --- DQN ---
    dqn_results = evaluate_agent(checkpoint_path, env, n_episodes, device)
    dqn_mean_cost, dqn_mean_is = _mean_metrics(dqn_results)

    # --- Baselines ---
    twap_results = _run_baseline_episodes(TWAPBaseline(), env, n_episodes)
    twap_mean_cost, twap_mean_is = _mean_metrics(twap_results)

    vwap_results = _run_baseline_episodes(
        VWAPBaseline(horizon=env.horizon), env, n_episodes
    )
    vwap_mean_cost, vwap_mean_is = _mean_metrics(vwap_results)

    ac_results = _run_baseline_episodes(AlmgrenChrissBaseline(), env, n_episodes)
    ac_mean_cost, ac_mean_is = _mean_metrics(ac_results)

    random_results = _run_baseline_episodes(RandomBaseline(), env, n_episodes)
    random_mean_cost, random_mean_is = _mean_metrics(random_results)

    dqn_beats_twap: bool = dqn_mean_cost < twap_mean_cost

    # --- Print summary table ---
    _print_comparison_table(
        dqn=(dqn_mean_cost, dqn_mean_is),
        twap=(twap_mean_cost, twap_mean_is),
        vwap=(vwap_mean_cost, vwap_mean_is),
        almgren_chriss=(ac_mean_cost, ac_mean_is),
        random=(random_mean_cost, random_mean_is),
        dqn_beats_twap=dqn_beats_twap,
    )

    return {
        "dqn": {
            "mean_cost": dqn_mean_cost,
            "mean_is": dqn_mean_is,
            "results": dqn_results,
        },
        "twap": {
            "mean_cost": twap_mean_cost,
            "mean_is": twap_mean_is,
            "results": twap_results,
        },
        "vwap": {
            "mean_cost": vwap_mean_cost,
            "mean_is": vwap_mean_is,
            "results": vwap_results,
        },
        "almgren_chriss": {
            "mean_cost": ac_mean_cost,
            "mean_is": ac_mean_is,
            "results": ac_results,
        },
        "random": {
            "mean_cost": random_mean_cost,
            "mean_is": random_mean_is,
            "results": random_results,
        },
        "dqn_beats_twap": dqn_beats_twap,
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def _print_comparison_table(
    *,
    dqn: tuple[float, float],
    twap: tuple[float, float],
    vwap: tuple[float, float],
    almgren_chriss: tuple[float, float],
    random: tuple[float, float],
    dqn_beats_twap: bool,
) -> None:
    """Print a formatted comparison table to stdout."""
    header = f"{'Agent':<18}| {'Mean Cost':>9} | {'Mean IS':>8} | {'Beats TWAP'}"
    sep = "-" * 18 + "|-" + "-" * 10 + "|-" + "-" * 9 + "|-" + "-" * 10
    rows = [
        ("DQN", dqn[0], dqn[1], f"{'YES' if dqn_beats_twap else 'NO'}"),
        ("TWAP", twap[0], twap[1], "baseline"),
        ("VWAP", vwap[0], vwap[1], "-"),
        ("Almgren-Chriss", almgren_chriss[0], almgren_chriss[1], "-"),
        ("Random", random[0], random[1], "-"),
    ]
    print(header)
    print(sep)
    for name, cost, is_val, beats in rows:
        print(f"{name:<18}| {cost:>9.4f} | {is_val:>8.4f} | {beats}")


__all__ = ["evaluate_agent", "compare_to_baselines"]
