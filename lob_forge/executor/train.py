"""Training loop for the DQN execution agent via 3-stage curriculum learning.

Implements train_agent(cfg) which trains a DuelingDQN on LOBExecutionEnv using
Double-DQN updates and prioritized experience replay over three curriculum stages:
  1. low_vol   — low-volatility regime (regime=0)
  2. mixed     — mixed regimes (regime=1)
  3. adversarial — adversarial high-vol regime (regime=2)

Each stage loads from the previous stage's checkpoint. Saves a checkpoint after
every stage completes. Returns the path to the final stage checkpoint.

Logging: stdout via print(). No wandb — Phase 10 concern.
"""

from __future__ import annotations

import math
import random
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from lob_forge.executor.agent import DuelingDQN, PrioritizedReplayBuffer
from lob_forge.executor.cost_model import CostModel
from lob_forge.executor.environment import LOBExecutionEnv

# ---------------------------------------------------------------------------
# Curriculum stage definitions
# ---------------------------------------------------------------------------

STAGE_CONFIG: dict[str, dict[str, Any]] = {
    "low_vol": {"regime": 0, "steps": 50_000, "mode": "real"},
    "mixed": {"regime": 1, "steps": 75_000, "mode": "real"},
    "adversarial": {"regime": 2, "steps": 50_000, "mode": "real"},
}

# ---------------------------------------------------------------------------
# Device selection helper
# ---------------------------------------------------------------------------


def _select_device() -> torch.device:
    """Select the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_agent(cfg: Any) -> Path:
    """Train DQN agent via 3-stage curriculum. Returns path to final checkpoint.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Must have a top-level ``executor`` key with all training hyperparameters.

    Returns
    -------
    Path
        Absolute path to the final stage checkpoint file.
    """
    ex = cfg.executor

    # -- Hyperparameters -------------------------------------------------------
    seq_len: int = ex.seq_len
    inventory: float = float(ex.inventory)
    horizon: int = ex.horizon
    order_sizes: list[float] = list(ex.order_sizes)
    limit_offsets_bps: list[float] = list(ex.limit_offsets_bps)
    n_actions: int = int(ex.n_actions)
    gamma: float = float(ex.gamma)
    lr: float = float(ex.lr)
    buffer_size: int = int(ex.buffer_size)
    batch_size: int = int(ex.batch_size)
    epsilon_start: float = float(ex.epsilon_start)
    epsilon_end: float = float(ex.epsilon_end)
    epsilon_decay: float = float(ex.epsilon_decay)
    target_update_freq: int = int(ex.target_update_freq)
    curriculum_stages: list[str] = list(ex.curriculum_stages)

    # Optional cost model config
    cost_model_cfg = ex.get("cost_model", None)
    if cost_model_cfg is not None:
        cost_model = CostModel(
            fee_bps=float(cost_model_cfg.fee_bps),
            impact_eta=float(cost_model_cfg.impact_eta),
        )
    else:
        cost_model = CostModel()

    # -- Data ------------------------------------------------------------------
    data_path = ex.get("data_path", None)
    if data_path is None:
        print(
            "[train_agent] WARNING: data_path is None — using dummy random LOB data "
            "(shape 10000×40). For real training, set executor.data_path to a Parquet file."
        )
        lob_data = np.random.randn(10_000, 40).astype(np.float32)
    else:
        import pandas as pd

        lob_data = pd.read_parquet(data_path).values.astype(np.float32)
        if lob_data.shape[1] != 40:
            raise ValueError(
                f"Loaded parquet has {lob_data.shape[1]} columns; expected 40."
            )
        print(f"[train_agent] Loaded LOB data: {lob_data.shape} from {data_path}")

    # -- Device ----------------------------------------------------------------
    device = _select_device()
    print(f"[train_agent] Using device: {device}")

    # -- Checkpoint directory --------------------------------------------------
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -- Track the path of the most recently saved checkpoint ------------------
    last_ckpt_path: Path | None = None

    # -- Per-stage training ----------------------------------------------------
    epsilon: float = epsilon_start
    total_steps: int = 0
    prev_ckpt_path: Path | None = None

    for stage_name in curriculum_stages:
        stage_cfg = STAGE_CONFIG[stage_name]
        stage_steps: int = stage_cfg["steps"]
        mode: str = stage_cfg["mode"]
        print(
            f"\n[train_agent] ===== Stage: {stage_name} | "
            f"regime={stage_cfg['regime']} | steps={stage_steps} | mode={mode} ====="
        )

        # -- Environment -------------------------------------------------------
        env = LOBExecutionEnv(
            lob_data=lob_data,
            seq_len=seq_len,
            inventory=inventory,
            horizon=horizon,
            order_sizes=tuple(order_sizes),  # type: ignore[arg-type]
            limit_offsets_bps=tuple(limit_offsets_bps),  # type: ignore[arg-type]
            cost_model=cost_model,
            mode=mode,
        )

        # -- Networks ----------------------------------------------------------
        obs_shape = (seq_len, 40)
        online_net = DuelingDQN(obs_shape=obs_shape, n_actions=n_actions).to(device)
        target_net = DuelingDQN(obs_shape=obs_shape, n_actions=n_actions).to(device)

        # Load from previous stage checkpoint if available
        if prev_ckpt_path is not None and prev_ckpt_path.exists():
            ckpt = torch.load(prev_ckpt_path, weights_only=False, map_location=device)
            online_net.load_state_dict(ckpt["online_net"])
            target_net.load_state_dict(ckpt["target_net"])
            epsilon = float(ckpt.get("epsilon", epsilon))
            print(
                f"[train_agent] Loaded checkpoint from {prev_ckpt_path} "
                f"(epsilon={epsilon:.4f})"
            )
        else:
            # Target net starts as a copy of online net
            target_net.load_state_dict(online_net.state_dict())

        target_net.eval()

        optimizer = torch.optim.Adam(online_net.parameters(), lr=lr)

        # Load optimizer state from prev checkpoint if available
        if prev_ckpt_path is not None and prev_ckpt_path.exists():
            ckpt = torch.load(prev_ckpt_path, weights_only=False, map_location=device)
            optimizer.load_state_dict(ckpt["optimizer"])

        replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)

        # -- Training loop -----------------------------------------------------
        stage_step: int = 0
        episode_rewards: deque[float] = deque(maxlen=10)
        loss_history: deque[float] = deque(maxlen=100)

        obs, _ = env.reset()
        episode_reward: float = 0.0

        while stage_step < stage_steps:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = online_net(obs_t)
                action = int(q_values.argmax(dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            episode_reward += reward
            stage_step += 1
            total_steps += 1

            # Update if buffer has enough samples
            if len(replay_buffer) >= batch_size:
                (
                    obs_b,
                    actions_b,
                    rewards_b,
                    next_obs_b,
                    dones_b,
                    weights_b,
                    indices,
                ) = replay_buffer.sample(batch_size)

                obs_b = obs_b.to(device)
                actions_b = actions_b.to(device)
                rewards_b = rewards_b.to(device)
                next_obs_b = next_obs_b.to(device)
                dones_b = dones_b.to(device)
                weights_b = weights_b.to(device)

                # Double-DQN update:
                #   online net selects the action, target net evaluates Q-value
                with torch.no_grad():
                    next_actions = online_net(next_obs_b).argmax(dim=1)
                    next_q = (
                        target_net(next_obs_b)
                        .gather(1, next_actions.unsqueeze(1))
                        .squeeze(1)
                    )
                    targets = rewards_b + gamma * next_q * (1.0 - dones_b)

                current_q = (
                    online_net(obs_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
                )
                td_errors = (targets - current_q).detach().abs()
                loss = (
                    weights_b * F.mse_loss(current_q, targets, reduction="none")
                ).mean()

                # NaN guard
                if math.isnan(loss.item()):
                    print(
                        f"[train_agent] NaN loss detected at step {total_steps}, "
                        "stopping early."
                    )
                    break

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_history.append(loss.item())
                replay_buffer.update_priorities(indices, td_errors.cpu().numpy())

                # Hard-copy target net
                if stage_step % target_update_freq == 0:
                    target_net.load_state_dict(online_net.state_dict())

            # Epsilon decay
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # Periodic logging (every 1000 stage steps)
            if stage_step % 1000 == 0:
                mean_loss = (
                    float(np.mean(list(loss_history))) if loss_history else float("nan")
                )
                mean_ep_reward = (
                    float(np.mean(list(episode_rewards)))
                    if episode_rewards
                    else float("nan")
                )
                print(
                    f"[{stage_name}] step={stage_step}/{stage_steps} | "
                    f"total={total_steps} | "
                    f"eps={epsilon:.4f} | "
                    f"loss={mean_loss:.6f} | "
                    f"ep_reward={mean_ep_reward:.4f}"
                )

            # Episode end
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                obs, _ = env.reset()

        # -- Save stage checkpoint ---------------------------------------------
        ckpt_path = ckpt_dir / f"executor_{stage_name}.pt"
        checkpoint = {
            "stage": stage_name,
            "online_net": online_net.state_dict(),
            "target_net": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epsilon": epsilon,
            "step": total_steps,
        }
        torch.save(checkpoint, ckpt_path)
        print(f"[train_agent] Checkpoint saved: {ckpt_path}")

        prev_ckpt_path = ckpt_path
        last_ckpt_path = ckpt_path

    if last_ckpt_path is None:
        raise RuntimeError(
            "No curriculum stages were executed — curriculum_stages is empty."
        )

    return last_ckpt_path
