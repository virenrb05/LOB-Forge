"""DuelingDQN network and PrioritizedReplayBuffer for EXEC-03.

DuelingDQN implements the Wang et al. (2016) dueling architecture:
    Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

PrioritizedReplayBuffer implements prioritized experience replay (Schaul et al. 2016)
using a simple deque for storage and numpy for priority-weighted sampling.
No external RL libraries — implemented from scratch.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn  # noqa: F401 (re-exported pattern)

# ---------------------------------------------------------------------------
# DuelingDQN
# ---------------------------------------------------------------------------


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network with a shared trunk and separate value/advantage heads.

    Parameters
    ----------
    obs_shape : tuple[int, int]
        (seq_len, n_features) — e.g. (100, 40).
    n_actions : int
        Number of discrete actions (7 for LOB execution env).
    hidden_dim : int
        Width of the shared trunk hidden layers (default 256).
    """

    def __init__(
        self,
        obs_shape: tuple[int, int],
        n_actions: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        seq_len, n_features = obs_shape
        input_dim = seq_len * n_features

        # Shared trunk: two hidden layers with ReLU
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream: V(s) — scalar
        self.value_stream = nn.Linear(hidden_dim, 1)

        # Advantage stream: A(s, a) — one output per action
        self.advantage_stream = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions.

        Parameters
        ----------
        obs : torch.Tensor
            Shape (B, seq_len, n_features) or (B, seq_len * n_features).

        Returns
        -------
        torch.Tensor
            Q-values of shape (B, n_actions).
        """
        if obs.dim() == 3:
            # Flatten (B, seq_len, n_features) → (B, seq_len * n_features)
            obs = obs.flatten(start_dim=1)

        shared = self.trunk(obs)

        value = self.value_stream(shared)  # (B, 1)
        advantage = self.advantage_stream(shared)  # (B, n_actions)

        # Dueling formula: Q = V + A - mean(A)  [Wang et al. 2016]
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer
# ---------------------------------------------------------------------------


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer (Schaul et al. 2016).

    Stores (obs, action, reward, next_obs, done) transitions.
    Sampling probability P(i) ∝ priority_i^alpha.
    Importance-sampling (IS) weights correct for the non-uniform sampling bias.
    beta anneals linearly from beta_start to beta_end over beta_steps calls to sample().

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    alpha : float
        Priority exponent (0 = uniform, 1 = full priority).
    beta_start : float
        Initial IS weight exponent.
    beta_end : float
        Final IS weight exponent (fully corrects for non-uniform sampling).
    beta_steps : int
        Number of sample() calls over which beta anneals linearly.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100_000,
    ) -> None:
        self._capacity = capacity
        self._alpha = alpha
        self._beta = beta_start
        self._beta_start = beta_start
        self._beta_end = beta_end
        self._beta_steps = beta_steps
        self._beta_increment = (beta_end - beta_start) / beta_steps

        # Storage: deque evicts oldest when at capacity
        self._storage: deque[tuple] = deque(maxlen=capacity)

        # Priority for each slot (parallel list, reindexed on access)
        self._priorities: deque[float] = deque(maxlen=capacity)

        self._max_priority: float = 1.0

    def __len__(self) -> int:
        return len(self._storage)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition with maximum current priority.

        Parameters
        ----------
        obs : np.ndarray   shape (seq_len, 40)
        action : int
        reward : float
        next_obs : np.ndarray  shape (seq_len, 40)
        done : bool
        """
        self._storage.append((obs, action, reward, next_obs, done))
        self._priorities.append(self._max_priority)

    def sample(self, batch_size: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[int],
    ]:
        """Sample a batch of transitions with priority-weighted probabilities.

        Parameters
        ----------
        batch_size : int

        Returns
        -------
        obs : FloatTensor (batch, seq_len, 40)
        actions : LongTensor (batch,)
        rewards : FloatTensor (batch,)
        next_obs : FloatTensor (batch, seq_len, 40)
        dones : FloatTensor (batch,)
        weights : FloatTensor (batch,) — IS correction weights, clamped [1e-8, 1]
        indices : list[int] — for update_priorities()
        """
        n = len(self._storage)
        if n < batch_size:
            raise ValueError(
                f"Buffer has {n} transitions but batch_size={batch_size} requested. "
                "Push more transitions before sampling."
            )

        # Build priority array and compute sampling probabilities
        priorities = np.array(list(self._priorities), dtype=np.float64)
        probs = priorities**self._alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)

        # IS weights: w_i = (N * P(i))^(-beta) / max_w
        weights = (n * probs[indices]) ** (-self._beta)
        weights /= weights.max()  # normalise by max weight
        weights = np.clip(weights, 1e-8, 1.0).astype(np.float32)

        # Unpack transitions
        batch = [self._storage[i] for i in indices]
        obs_list, actions_list, rewards_list, next_obs_list, dones_list = zip(
            *batch, strict=True
        )

        obs_t = torch.tensor(np.stack(obs_list), dtype=torch.float32)
        next_obs_t = torch.tensor(np.stack(next_obs_list), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions_list), dtype=torch.long)
        rewards_t = torch.tensor(
            np.array(rewards_list, dtype=np.float32), dtype=torch.float32
        )
        dones_t = torch.tensor(
            np.array(dones_list, dtype=np.float32), dtype=torch.float32
        )
        weights_t = torch.tensor(weights, dtype=torch.float32)

        # Anneal beta
        self._beta = min(self._beta_end, self._beta + self._beta_increment)

        return (
            obs_t,
            actions_t,
            rewards_t,
            next_obs_t,
            dones_t,
            weights_t,
            indices.tolist(),
        )

    def update_priorities(self, indices: list[int], td_errors: np.ndarray) -> None:
        """Update stored priorities for sampled transitions.

        Parameters
        ----------
        indices : list[int]
            Indices returned by sample().
        td_errors : np.ndarray
            TD errors for each sampled transition.
            Priority stored as |td_error| + 1e-6.
        """
        new_priorities = np.abs(td_errors) + 1e-6
        priorities_list = list(self._priorities)
        for idx, priority in zip(indices, new_priorities, strict=True):
            priorities_list[idx] = float(priority)
        self._priorities = deque(priorities_list, maxlen=self._capacity)
        self._max_priority = max(self._max_priority, float(new_priorities.max()))
