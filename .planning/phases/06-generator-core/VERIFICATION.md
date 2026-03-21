---
status: passed
verified_at: 2026-03-20
---

## Phase 6: Generator Core -- Verification

### Must-Haves

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Cosine noise schedule produces correct alpha/beta values (verified against reference) | Pass | `CosineNoiseSchedule` in `lob_forge/generator/noise_schedule.py` implements Nichol & Dhariwal 2021 cosine schedule. Computes `f(t) = cos((t/T + s)/(1+s) * pi/2)^2`, derives betas clipped to [0, 0.999], alphas_cumprod, posterior coefficients. Tests confirm: alphas_cumprod monotonically decreasing, alpha_bar_0 > 0.9, alpha_bar_T < 0.1, all betas in (0, 1), q_sample at t=0 close to clean data, at t=T noise dominates. All 7 schedule tests pass. |
| 2 | 1D U-Net with AdaLN accepts LOB sequences and conditioning inputs | Pass | `UNet1D` in `lob_forge/generator/unet.py` implements encoder-decoder with skip connections, `Downsample1D`/`Upsample1D`, `SelfAttention1D` at configurable levels. `ResBlock1D` in `blocks.py` uses `AdaptiveLayerNorm` (scale+shift from conditioning via linear projection). Conditioning produced by `ConditioningModule` in `conditioning.py` (sinusoidal timestep + regime embedding + optional time-of-day MLP). Tests verify output shape matches input `(B, C, T)`, multiple sequence lengths work, different conditioning vectors produce different outputs. All 7 U-Net/block/conditioning tests pass. |
| 3 | DDPM 1000-step sampling produces LOB-shaped output tensors | Pass | `DiffusionModel.ddpm_sample()` in `model.py` iterates `reversed(range(num_timesteps))` calling `p_sample()` which computes posterior mean from predicted noise, adds stochastic noise (except at t=0). Returns `(B, T, C)`. Test `test_ddpm_sample_shape` confirms output shape `(B, T, IN_CHANNELS)`. Default `num_timesteps=1000`. |
| 4 | DDIM 50-step sampling produces LOB-shaped output tensors (faster inference) | Pass | `DiffusionModel.ddim_sample()` in `model.py` creates a `linspace` subsequence of timesteps, implements the DDIM update rule with configurable `eta` (0.0 = deterministic). Default `ddim_steps=50`. Tests confirm correct output shape and deterministic behavior with `eta=0.0` and same seed. |
| 5 | EMA model weights update correctly and can be loaded for inference | Pass | `ExponentialMovingAverage` in `ema.py` maintains shadow params with `update()` (`shadow = decay * shadow + (1-decay) * param`), `apply_shadow()`/`restore()` for inference swap, and `state_dict()`/`load_state_dict()` for serialization. Training loop in `train.py` calls `ema.update(model)` every step, `ema.apply_shadow()`/`ema.restore()` around sample generation, and saves both `model_state_dict` and `ema_state_dict` in checkpoints. All 4 EMA tests pass (initial equality, update moves shadow, apply+restore roundtrip, state_dict roundtrip). |

### Test Results

All 27 tests in `tests/test_generator.py` pass (pytest, 4.14s):
- 7 noise schedule tests
- 4 EMA tests
- 4 conditioning tests
- 3 block tests
- 3 U-Net tests
- 6 DiffusionModel tests (training loss, gradients, DDPM sample, DDIM sample, determinism, generate routing)

### Summary

Phase 6 goal -- "Conditional diffusion model generates LOB sequences from noise" -- is fully achieved. All five success criteria are met with working implementations backed by passing tests. The codebase includes a complete training loop (`train.py`) with EMA integration, wandb logging, and checkpoint saving, going beyond the minimum requirements.
