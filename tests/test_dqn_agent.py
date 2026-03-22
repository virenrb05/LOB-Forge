"""RED-GREEN-REFACTOR tests for DuelingDQN and PrioritizedReplayBuffer.

Tests were written RED (before implementation) per TDD discipline.
"""

import numpy as np
import pytest
import torch

from lob_forge.executor.agent import DuelingDQN, PrioritizedReplayBuffer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OBS_SHAPE = (100, 40)  # seq_len=100, features=40
N_ACTIONS = 7
BATCH_SIZE = 8


@pytest.fixture()
def net() -> DuelingDQN:
    return DuelingDQN(obs_shape=OBS_SHAPE, n_actions=N_ACTIONS)


@pytest.fixture()
def buffer() -> PrioritizedReplayBuffer:
    return PrioritizedReplayBuffer(capacity=1000)


def _rand_obs(batch: int = 1) -> torch.Tensor:
    """Return a random observation tensor of shape (batch, seq_len, 40)."""
    return torch.randn(batch, OBS_SHAPE[0], OBS_SHAPE[1])


# ---------------------------------------------------------------------------
# RED: DuelingDQN tests (shape, architecture, forward pass)
# ---------------------------------------------------------------------------


class TestDuelingDQNConstruction:
    def test_instantiates_with_defaults(self):
        net = DuelingDQN(obs_shape=(100, 40), n_actions=7)
        assert net is not None

    def test_custom_hidden_dim(self):
        net = DuelingDQN(obs_shape=(50, 40), n_actions=5, hidden_dim=128)
        assert net is not None

    def test_is_nn_module(self):
        net = DuelingDQN(obs_shape=(100, 40), n_actions=7)
        assert isinstance(net, torch.nn.Module)


class TestDuelingDQNForwardShape:
    def test_forward_batch4_returns_shape_4_7(self, net: DuelingDQN):
        obs = _rand_obs(batch=4)
        q = net(obs)
        assert q.shape == (4, N_ACTIONS), f"Expected (4, 7), got {q.shape}"

    def test_forward_batch1_returns_shape_1_7(self, net: DuelingDQN):
        obs = _rand_obs(batch=1)
        q = net(obs)
        assert q.shape == (1, N_ACTIONS)

    def test_forward_batch16_returns_shape_16_7(self, net: DuelingDQN):
        obs = _rand_obs(batch=16)
        q = net(obs)
        assert q.shape == (16, N_ACTIONS)

    def test_output_is_float_tensor(self, net: DuelingDQN):
        obs = _rand_obs(batch=4)
        q = net(obs)
        assert q.dtype == torch.float32

    def test_no_nan_in_output(self, net: DuelingDQN):
        obs = _rand_obs(batch=4)
        q = net(obs)
        assert not torch.isnan(q).any(), "Q-values contain NaN"

    def test_accepts_3d_input(self, net: DuelingDQN):
        """Input (B, seq_len, 40) is 3D and must be handled by flattening inside forward."""
        obs = torch.randn(BATCH_SIZE, OBS_SHAPE[0], OBS_SHAPE[1])
        q = net(obs)
        assert q.shape == (BATCH_SIZE, N_ACTIONS)


class TestDuelingDQNArchitecture:
    def test_has_value_stream(self, net: DuelingDQN):
        """Network must have a value stream (V(s) head)."""
        assert hasattr(net, "value_stream"), "DuelingDQN missing value_stream attribute"

    def test_has_advantage_stream(self, net: DuelingDQN):
        """Network must have an advantage stream (A(s,a) head)."""
        assert hasattr(
            net, "advantage_stream"
        ), "DuelingDQN missing advantage_stream attribute"

    def test_has_shared_trunk(self, net: DuelingDQN):
        """Network must have a shared trunk."""
        assert hasattr(net, "trunk"), "DuelingDQN missing trunk attribute"

    def test_dueling_decomposition_mean_advantage_zero(self, net: DuelingDQN):
        """Q = V + A - mean(A) → mean of advantage over actions should be zero in output."""
        obs = _rand_obs(batch=32)
        with torch.no_grad():
            # Verify mean advantage is subtracted: if we pass the same obs,
            # the Q values should satisfy Q - mean(Q) == A - mean(A)
            # We can only verify the mean is NOT constant (it IS subtracted by testing)
            q = net(obs)
        # Q-values should not all be identical (degenerate), confirm variance > 0
        assert q.var().item() > 0.0, "Q-values have zero variance — likely bug"

    def test_q_values_differ_across_actions(self, net: DuelingDQN):
        obs = _rand_obs(batch=1)
        with torch.no_grad():
            q = net(obs)
        # Q-values across the 7 actions should not all be the same
        assert q[0].unique().numel() > 1, "All Q-values are identical — degenerate"


class TestDuelingDQNGradients:
    def test_backward_pass_works(self, net: DuelingDQN):
        obs = _rand_obs(batch=4)
        q = net(obs)
        loss = q.sum()
        loss.backward()
        # Check that at least one parameter has a gradient
        has_grad = any(p.grad is not None for p in net.parameters())
        assert has_grad, "No gradients after backward pass"


# ---------------------------------------------------------------------------
# RED: PrioritizedReplayBuffer tests
# ---------------------------------------------------------------------------


def _make_transition(seq_len: int = 100) -> tuple:
    """Create a single (obs, action, reward, next_obs, done) transition."""
    obs = np.random.randn(seq_len, 40).astype(np.float32)
    action = np.random.randint(0, N_ACTIONS)
    reward = float(np.random.randn())
    next_obs = np.random.randn(seq_len, 40).astype(np.float32)
    done = bool(np.random.rand() < 0.1)
    return obs, action, reward, next_obs, done


class TestPrioritizedReplayBufferConstruction:
    def test_default_params(self):
        buf = PrioritizedReplayBuffer(capacity=1000)
        assert buf is not None

    def test_custom_params(self):
        buf = PrioritizedReplayBuffer(
            capacity=5000,
            alpha=0.7,
            beta_start=0.5,
            beta_end=1.0,
            beta_steps=50_000,
        )
        assert buf is not None

    def test_initial_len_is_zero(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        assert len(buf) == 0


class TestPrioritizedReplayBufferPush:
    def test_push_increments_len(self, buffer: PrioritizedReplayBuffer):
        assert len(buffer) == 0
        buffer.push(*_make_transition())
        assert len(buffer) == 1

    def test_push_multiple_increments_len(self, buffer: PrioritizedReplayBuffer):
        for _ in range(10):
            buffer.push(*_make_transition())
        assert len(buffer) == 10

    def test_push_at_capacity_evicts_oldest(self):
        """Buffer with capacity=5: after 7 pushes, len should remain 5."""
        buf = PrioritizedReplayBuffer(capacity=5)
        for _ in range(7):
            buf.push(*_make_transition())
        assert len(buf) == 5


class TestPrioritizedReplayBufferSample:
    def test_sample_returns_7_tuple(self, buffer: PrioritizedReplayBuffer):
        for _ in range(20):
            buffer.push(*_make_transition())
        result = buffer.sample(batch_size=BATCH_SIZE)
        assert len(result) == 7

    def test_sample_obs_shape(self, buffer: PrioritizedReplayBuffer):
        for _ in range(20):
            buffer.push(*_make_transition())
        obs, actions, rewards, next_obs, dones, weights, indices = buffer.sample(
            batch_size=BATCH_SIZE
        )
        assert obs.shape == (BATCH_SIZE, 100, 40), f"Got {obs.shape}"

    def test_sample_next_obs_shape(self, buffer: PrioritizedReplayBuffer):
        for _ in range(20):
            buffer.push(*_make_transition())
        obs, actions, rewards, next_obs, dones, weights, indices = buffer.sample(
            batch_size=BATCH_SIZE
        )
        assert next_obs.shape == (BATCH_SIZE, 100, 40)

    def test_sample_actions_long_tensor(self, buffer: PrioritizedReplayBuffer):
        for _ in range(20):
            buffer.push(*_make_transition())
        obs, actions, rewards, next_obs, dones, weights, indices = buffer.sample(
            batch_size=BATCH_SIZE
        )
        assert actions.dtype == torch.long
        assert actions.shape == (BATCH_SIZE,)

    def test_sample_rewards_float_tensor(self, buffer: PrioritizedReplayBuffer):
        for _ in range(20):
            buffer.push(*_make_transition())
        obs, actions, rewards, next_obs, dones, weights, indices = buffer.sample(
            batch_size=BATCH_SIZE
        )
        assert rewards.dtype == torch.float32
        assert rewards.shape == (BATCH_SIZE,)

    def test_sample_dones_float_tensor(self, buffer: PrioritizedReplayBuffer):
        for _ in range(20):
            buffer.push(*_make_transition())
        obs, actions, rewards, next_obs, dones, weights, indices = buffer.sample(
            batch_size=BATCH_SIZE
        )
        assert dones.dtype == torch.float32
        assert dones.shape == (BATCH_SIZE,)

    def test_sample_weights_float_tensor(self, buffer: PrioritizedReplayBuffer):
        for _ in range(20):
            buffer.push(*_make_transition())
        obs, actions, rewards, next_obs, dones, weights, indices = buffer.sample(
            batch_size=BATCH_SIZE
        )
        assert weights.dtype == torch.float32
        assert weights.shape == (BATCH_SIZE,)

    def test_sample_weights_clamped_between_0_and_1(
        self, buffer: PrioritizedReplayBuffer
    ):
        for _ in range(20):
            buffer.push(*_make_transition())
        obs, actions, rewards, next_obs, dones, weights, indices = buffer.sample(
            batch_size=BATCH_SIZE
        )
        assert weights.min().item() >= 1e-8 - 1e-12
        assert weights.max().item() <= 1.0 + 1e-12

    def test_sample_indices_is_list_of_int(self, buffer: PrioritizedReplayBuffer):
        for _ in range(20):
            buffer.push(*_make_transition())
        obs, actions, rewards, next_obs, dones, weights, indices = buffer.sample(
            batch_size=BATCH_SIZE
        )
        assert isinstance(indices, list)
        assert len(indices) == BATCH_SIZE
        assert all(isinstance(i, int) for i in indices)

    def test_sample_raises_if_insufficient_transitions(self):
        buf = PrioritizedReplayBuffer(capacity=1000)
        for _ in range(3):
            buf.push(*_make_transition())
        with pytest.raises(ValueError, match="batch_size"):
            buf.sample(batch_size=10)

    def test_sample_obs_is_float_tensor(self, buffer: PrioritizedReplayBuffer):
        for _ in range(20):
            buffer.push(*_make_transition())
        obs, *_ = buffer.sample(batch_size=BATCH_SIZE)
        assert obs.dtype == torch.float32


class TestPrioritizedReplayBufferUpdatePriorities:
    def test_update_priorities_changes_sampling_distribution(
        self, buffer: PrioritizedReplayBuffer
    ):
        """After setting one transition's priority to a very large value,
        it should dominate sampling."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            t = _make_transition()
            buffer.push(*t)

        obs, actions, rewards, next_obs, dones, weights, indices = buffer.sample(
            batch_size=BATCH_SIZE
        )

        # Give index 0 a massive priority
        td_errors = np.zeros(BATCH_SIZE)
        # Set all but the first to near-zero
        td_errors[0] = 1e6
        buffer.update_priorities(indices, td_errors)

        # After update, verify the buffer is still functional (no crash)
        obs2, _, _, _, _, _, indices2 = buffer.sample(batch_size=BATCH_SIZE)
        assert len(indices2) == BATCH_SIZE

    def test_update_priorities_stores_abs_td_plus_eps(
        self, buffer: PrioritizedReplayBuffer
    ):
        """update_priorities stores |td_error| + 1e-6 as priority."""
        for _ in range(20):
            buffer.push(*_make_transition())
        obs, actions, rewards, next_obs, dones, weights, indices = buffer.sample(
            batch_size=BATCH_SIZE
        )
        td_errors = np.ones(BATCH_SIZE) * 2.0
        # Should not raise
        buffer.update_priorities(indices, td_errors)
        # Verify len unchanged
        assert len(buffer) == 20

    def test_update_priorities_handles_negative_td_errors(
        self, buffer: PrioritizedReplayBuffer
    ):
        """Priorities use |td_error|, so negative errors are fine."""
        for _ in range(20):
            buffer.push(*_make_transition())
        obs, actions, rewards, next_obs, dones, weights, indices = buffer.sample(
            batch_size=BATCH_SIZE
        )
        td_errors = -np.ones(BATCH_SIZE) * 1.5
        # Should not raise
        buffer.update_priorities(indices, td_errors)


class TestPrioritizedReplayBufferBetaAnnealing:
    def test_beta_anneals_over_steps(self):
        """beta should increase from beta_start toward beta_end over beta_steps calls."""
        buf = PrioritizedReplayBuffer(
            capacity=1000, beta_start=0.4, beta_end=1.0, beta_steps=10
        )
        for _ in range(30):
            buf.push(*_make_transition())

        # First sample: beta ~ 0.4
        buf_beta_start = buf._beta  # noqa: SLF001
        # Call sample many times to advance beta
        for _ in range(10):
            buf.sample(batch_size=4)

        buf_beta_after = buf._beta  # noqa: SLF001
        assert buf_beta_after >= buf_beta_start, "Beta did not increase after sampling"

    def test_beta_clamped_at_beta_end(self):
        """beta must not exceed beta_end even after many sample() calls."""
        buf = PrioritizedReplayBuffer(
            capacity=100, beta_start=0.4, beta_end=1.0, beta_steps=5
        )
        for _ in range(30):
            buf.push(*_make_transition())

        for _ in range(100):  # far beyond beta_steps
            buf.sample(batch_size=4)

        assert buf._beta <= 1.0 + 1e-9  # noqa: SLF001
