"""Backtesting engine for evaluating execution strategies on historical LOB data.

Provides :func:`run_backtest` which runs either a DQN checkpoint or a
:class:`~lob_forge.executor.baselines.BaselineAgent` for *n_episodes* and
returns a list of :class:`~lob_forge.executor.baselines.ExecutionResult`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

from lob_forge.executor.baselines import BaselineAgent, ExecutionResult

if TYPE_CHECKING:
    from lob_forge.executor.environment import LOBExecutionEnv


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_backtest(
    env: LOBExecutionEnv,
    agent: Union[BaselineAgent, str, Path],
    n_episodes: int = 10,
    seed_offset: int = 0,
) -> list[ExecutionResult]:
    """Run *agent* for *n_episodes* in *env* and return the results.

    Accepts two agent types:

    - **DQN checkpoint** – when *agent* is a ``str`` or :class:`~pathlib.Path`,
      it is treated as a checkpoint file path and loaded via
      :func:`~lob_forge.executor.evaluate.evaluate_agent`.  ``torch`` is
      imported lazily inside the DQN branch to keep the module torch-free at
      import time.
    - **Baseline agent** – when *agent* is a
      :class:`~lob_forge.executor.baselines.BaselineAgent` instance, its
      :meth:`run_episode` method is called directly.

    Parameters
    ----------
    env : LOBExecutionEnv
        A gymnasium environment following the :class:`LOBExecutionEnv` interface.
    agent : BaselineAgent | str | Path
        Either a baseline agent instance or the path to a DQN checkpoint.
    n_episodes : int
        Number of episodes to run (default 10).
    seed_offset : int
        Starting seed; episode *i* uses seed ``seed_offset + i`` (default 0).

    Returns
    -------
    list[ExecutionResult]
        One :class:`~lob_forge.executor.baselines.ExecutionResult` per episode.
    """
    if isinstance(agent, (str, Path)):
        # Lazy import to avoid pulling torch into module namespace.
        from lob_forge.executor.evaluate import evaluate_agent  # noqa: PLC0415

        # evaluate_agent seeds episodes 0..n_episodes-1 internally.
        # When seed_offset != 0 we re-implement the loop here to honour it.
        if seed_offset == 0:
            return evaluate_agent(
                checkpoint_path=agent,
                env=env,
                n_episodes=n_episodes,
                device="cpu",
            )
        else:
            # evaluate_agent does not support a seed_offset; load manually.
            import torch  # noqa: PLC0415

            from lob_forge.executor.agent import DuelingDQN  # noqa: PLC0415

            checkpoint_path = Path(agent)
            ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
            obs_shape = (env.seq_len, 40)
            net = DuelingDQN(obs_shape=obs_shape, n_actions=7)
            net.load_state_dict(ckpt["online_net"])
            net.eval()

            results: list[ExecutionResult] = []
            for i in range(n_episodes):
                seed = seed_offset + i
                obs, info = env.reset(seed=seed)
                actions: list[int] = []
                step = 0
                arrival_price = (float(obs[-1, 0]) + float(obs[-1, 20])) / 2.0

                while True:
                    obs_t = torch.from_numpy(obs).unsqueeze(0).float()
                    with torch.no_grad():
                        action = int(net(obs_t).argmax(dim=1).item())
                    obs, _r, terminated, truncated, info = env.step(action)
                    actions.append(action)
                    step += 1
                    if terminated or truncated:
                        break

                executed = env.inventory - info["remaining"]
                exec_vwap = arrival_price + (
                    info["episode_cost"] / max(executed, 1e-9)
                )
                is_shortfall = (exec_vwap - arrival_price) * env.inventory
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

    # Baseline agent branch
    if not isinstance(agent, BaselineAgent):
        raise TypeError(
            f"agent must be a BaselineAgent instance or a checkpoint path (str/Path), "
            f"got {type(agent).__name__}"
        )

    return [
        agent.run_episode(env, seed=seed_offset + i) for i in range(n_episodes)
    ]


__all__ = ["run_backtest"]
