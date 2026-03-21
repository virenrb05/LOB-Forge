"""Comprehensive tests for the generator module.

Covers noise schedule, EMA, conditioning, blocks, U-Net, and DiffusionModel.
Uses small dimensions for fast execution (d_model=32, T=16, B=2).
"""

from __future__ import annotations

import pytest
import torch

from lob_forge.generator.blocks import AdaptiveLayerNorm, ResBlock1D
from lob_forge.generator.conditioning import (
    ConditioningModule,
    SinusoidalTimestepEmbedding,
)
from lob_forge.generator.ema import ExponentialMovingAverage
from lob_forge.generator.model import DiffusionModel
from lob_forge.generator.noise_schedule import CosineNoiseSchedule
from lob_forge.generator.unet import UNet1D

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

D_MODEL = 32
IN_CHANNELS = 8
NUM_TIMESTEPS = 20
B = 2
T = 16


# ---------------------------------------------------------------------------
# TestCosineNoiseSchedule
# ---------------------------------------------------------------------------


class TestCosineNoiseSchedule:
    def test_alpha_bar_monotonic(self):
        schedule = CosineNoiseSchedule(num_timesteps=NUM_TIMESTEPS)
        diffs = schedule.alphas_cumprod[1:] - schedule.alphas_cumprod[:-1]
        assert (diffs < 0).all(), "alphas_cumprod should be strictly decreasing"

    def test_alpha_bar_bounds(self):
        schedule = CosineNoiseSchedule(num_timesteps=NUM_TIMESTEPS)
        assert schedule.alphas_cumprod[0].item() > 0.9, "alpha_bar_0 should be near 1"
        assert schedule.alphas_cumprod[-1].item() < 0.1, "alpha_bar_T should be near 0"

    def test_betas_positive(self):
        schedule = CosineNoiseSchedule(num_timesteps=NUM_TIMESTEPS)
        assert (schedule.betas > 0).all(), "All betas must be > 0"

    def test_betas_bounded(self):
        schedule = CosineNoiseSchedule(num_timesteps=NUM_TIMESTEPS)
        assert (schedule.betas < 1.0).all(), "All betas must be < 1"

    def test_q_sample_shape(self):
        schedule = CosineNoiseSchedule(num_timesteps=NUM_TIMESTEPS)
        x_0 = torch.randn(B, IN_CHANNELS, T)
        t = torch.randint(0, NUM_TIMESTEPS, (B,))
        x_t = schedule.q_sample(x_0, t)
        assert x_t.shape == x_0.shape

    def test_q_sample_t0_close_to_clean(self):
        schedule = CosineNoiseSchedule(num_timesteps=NUM_TIMESTEPS)
        x_0 = torch.randn(B, IN_CHANNELS, T)
        t = torch.zeros(B, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = schedule.q_sample(x_0, t, noise)
        # At t=0, sqrt_alpha_bar ≈ 1, so x_t ≈ x_0 (generous tolerance
        # because even t=0 adds a small noise fraction)
        assert torch.allclose(x_t, x_0, atol=0.5)

    def test_q_sample_tmax_close_to_noise(self):
        schedule = CosineNoiseSchedule(num_timesteps=NUM_TIMESTEPS)
        x_0 = torch.zeros(B, IN_CHANNELS, T)
        t = torch.full((B,), NUM_TIMESTEPS - 1, dtype=torch.long)
        noise = torch.ones(B, IN_CHANNELS, T)
        _x_t = schedule.q_sample(x_0, t, noise)  # noqa: F841
        # At t=T-1, sqrt_one_minus_alpha_bar ≈ 1, so x_t ≈ noise
        sqrt_omc = schedule.sqrt_one_minus_alphas_cumprod[-1].item()
        assert sqrt_omc > 0.9, "At tmax noise should dominate"


# ---------------------------------------------------------------------------
# TestEMA
# ---------------------------------------------------------------------------


class TestEMA:
    def _make_model(self):
        return torch.nn.Linear(4, 4)

    def test_initial_shadow_equals_params(self):
        model = self._make_model()
        ema = ExponentialMovingAverage(model, decay=0.99)
        for name, p in model.named_parameters():
            assert torch.equal(
                ema.shadow_params[name], p.data
            ), "Shadow should start equal to params"

    def test_update_moves_shadow(self):
        model = self._make_model()
        ema = ExponentialMovingAverage(model, decay=0.9)
        old_shadow = {k: v.clone() for k, v in ema.shadow_params.items()}

        # Change params
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update(model)

        for name in ema.shadow_params:
            # Shadow should differ from both old shadow and new params
            assert not torch.equal(ema.shadow_params[name], old_shadow[name])
            assert not torch.equal(
                ema.shadow_params[name], dict(model.named_parameters())[name].data
            )

    def test_apply_and_restore(self):
        model = self._make_model()
        ema = ExponentialMovingAverage(model, decay=0.9)

        # Change model
        with torch.no_grad():
            for p in model.parameters():
                p.add_(10.0)
        ema.update(model)

        orig_params = {n: p.data.clone() for n, p in model.named_parameters()}

        ema.apply_shadow(model)
        # Params should now be shadow (different from orig)
        for name, p in model.named_parameters():
            assert not torch.equal(p.data, orig_params[name])

        ema.restore(model)
        # Should be back to original
        for name, p in model.named_parameters():
            assert torch.equal(p.data, orig_params[name])

    def test_state_dict_roundtrip(self):
        model = self._make_model()
        ema = ExponentialMovingAverage(model, decay=0.99)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(5.0)
        ema.update(model)

        state = ema.state_dict()
        ema2 = ExponentialMovingAverage(model, decay=0.99)
        ema2.load_state_dict(state)

        for key in ema.shadow_params:
            assert torch.equal(ema.shadow_params[key], ema2.shadow_params[key])


# ---------------------------------------------------------------------------
# TestConditioning
# ---------------------------------------------------------------------------


class TestConditioning:
    def test_timestep_embedding_shape(self):
        emb = SinusoidalTimestepEmbedding(D_MODEL)
        t = torch.randint(0, NUM_TIMESTEPS, (B,))
        out = emb(t)
        assert out.shape == (B, D_MODEL)

    def test_timestep_embedding_unique(self):
        emb = SinusoidalTimestepEmbedding(D_MODEL)
        t1 = torch.tensor([0])
        t2 = torch.tensor([10])
        out1 = emb(t1)
        out2 = emb(t2)
        assert not torch.allclose(
            out1, out2
        ), "Different timesteps must give different embeddings"

    def test_conditioning_module_shape(self):
        cond_mod = ConditioningModule(d_model=D_MODEL, n_regimes=3)
        t = torch.randint(0, NUM_TIMESTEPS, (B,))
        regime = torch.randint(0, 3, (B,))
        tod = torch.rand(B)
        out = cond_mod(t, regime, tod)
        assert out.shape == (B, D_MODEL)

    def test_conditioning_without_tod(self):
        cond_mod = ConditioningModule(d_model=D_MODEL, n_regimes=3)
        t = torch.randint(0, NUM_TIMESTEPS, (B,))
        regime = torch.randint(0, 3, (B,))
        out = cond_mod(t, regime, time_of_day=None)
        assert out.shape == (B, D_MODEL)


# ---------------------------------------------------------------------------
# TestBlocks
# ---------------------------------------------------------------------------


class TestBlocks:
    def test_adaln_shape(self):
        adaln = AdaptiveLayerNorm(channels=D_MODEL, cond_dim=D_MODEL)
        x = torch.randn(B, D_MODEL, T)
        cond = torch.randn(B, D_MODEL)
        out = adaln(x, cond)
        assert out.shape == (B, D_MODEL, T)

    def test_resblock_same_channels(self):
        block = ResBlock1D(channels=D_MODEL, cond_dim=D_MODEL, dropout=0.0)
        x = torch.randn(B, D_MODEL, T)
        cond = torch.randn(B, D_MODEL)
        out = block(x, cond)
        assert out.shape == (B, D_MODEL, T)

    def test_resblock_channel_change(self):
        block = ResBlock1D(
            channels=D_MODEL, cond_dim=D_MODEL, dropout=0.0, out_channels=D_MODEL * 2
        )
        x = torch.randn(B, D_MODEL, T)
        cond = torch.randn(B, D_MODEL)
        out = block(x, cond)
        assert out.shape == (B, D_MODEL * 2, T)


# ---------------------------------------------------------------------------
# TestUNet1D
# ---------------------------------------------------------------------------


class TestUNet1D:
    def _make_unet(self):
        return UNet1D(
            in_channels=IN_CHANNELS,
            d_model=D_MODEL,
            channel_mults=(1, 2),
            n_res_blocks=1,
            cond_dim=D_MODEL,
            dropout=0.0,
            attention_levels=(1,),
            n_heads=2,
        )

    def test_output_shape_matches_input(self):
        unet = self._make_unet()
        x = torch.randn(B, IN_CHANNELS, T)
        cond = torch.randn(B, D_MODEL)
        out = unet(x, cond)
        assert out.shape == (B, IN_CHANNELS, T)

    def test_different_sequence_lengths(self):
        unet = self._make_unet()
        cond = torch.randn(B, D_MODEL)
        for seq_len in [16, 32, 64]:
            x = torch.randn(B, IN_CHANNELS, seq_len)
            out = unet(x, cond)
            assert out.shape == (B, IN_CHANNELS, seq_len)

    def test_conditioning_affects_output(self):
        unet = self._make_unet()
        x = torch.randn(B, IN_CHANNELS, T)
        cond1 = torch.randn(B, D_MODEL)
        cond2 = torch.randn(B, D_MODEL)
        out1 = unet(x, cond1)
        out2 = unet(x, cond2)
        assert not torch.allclose(
            out1, out2
        ), "Different cond must give different output"


# ---------------------------------------------------------------------------
# TestDiffusionModel
# ---------------------------------------------------------------------------


class TestDiffusionModel:
    def _make_model(self):
        return DiffusionModel(
            in_channels=IN_CHANNELS,
            d_model=D_MODEL,
            channel_mults=(1, 2),
            n_res_blocks=1,
            num_timesteps=NUM_TIMESTEPS,
            ddim_steps=5,
            n_regimes=3,
            dropout=0.0,
            attention_levels=(1,),
            n_heads=2,
        )

    def test_training_loss_scalar(self):
        model = self._make_model()
        x_0 = torch.randn(B, T, IN_CHANNELS)
        regime = torch.randint(0, 3, (B,))
        loss = model.training_loss(x_0, regime)
        assert loss.ndim == 0, "Loss must be scalar"
        assert loss.item() > 0, "Loss must be positive"

    def test_training_loss_gradients(self):
        model = self._make_model()
        x_0 = torch.randn(B, T, IN_CHANNELS)
        regime = torch.randint(0, 3, (B,))
        # Pass time_of_day so all conditioning params get gradients
        tod = torch.rand(B)
        loss = model.training_loss(x_0, regime, time_of_day=tod)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"Missing grad for {name}"

    def test_ddpm_sample_shape(self):
        model = self._make_model()
        model.eval()
        regime = torch.zeros(B, dtype=torch.long)
        sample = model.ddpm_sample(n_samples=B, seq_len=T, regime=regime)
        assert sample.shape == (B, T, IN_CHANNELS)

    def test_ddim_sample_shape(self):
        model = self._make_model()
        model.eval()
        regime = torch.zeros(B, dtype=torch.long)
        sample = model.ddim_sample(n_samples=B, seq_len=T, regime=regime, ddim_steps=5)
        assert sample.shape == (B, T, IN_CHANNELS)

    def test_ddim_deterministic(self):
        model = self._make_model()
        model.eval()
        regime = torch.zeros(B, dtype=torch.long)
        torch.manual_seed(42)
        s1 = model.ddim_sample(
            n_samples=B, seq_len=T, regime=regime, ddim_steps=5, eta=0.0
        )
        torch.manual_seed(42)
        s2 = model.ddim_sample(
            n_samples=B, seq_len=T, regime=regime, ddim_steps=5, eta=0.0
        )
        assert torch.allclose(s1, s2), "eta=0 with same seed must be deterministic"

    def test_generate_routes(self):
        model = self._make_model()
        model.eval()
        regime = torch.zeros(B, dtype=torch.long)

        ddim_out = model.generate(
            n_samples=B, seq_len=T, regime=regime, method="ddim", ddim_steps=5
        )
        assert ddim_out.shape == (B, T, IN_CHANNELS)

        ddpm_out = model.generate(n_samples=B, seq_len=T, regime=regime, method="ddpm")
        assert ddpm_out.shape == (B, T, IN_CHANNELS)

        with pytest.raises(ValueError, match="Unknown sampling method"):
            model.generate(n_samples=B, seq_len=T, regime=regime, method="bad")
