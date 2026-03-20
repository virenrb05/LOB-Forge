"""Conditional diffusion model combining noise schedule, U-Net, and conditioning.

Provides :class:`DiffusionModel`, the top-level generative model that composes
:class:`~lob_forge.generator.noise_schedule.CosineNoiseSchedule`,
:class:`~lob_forge.generator.unet.UNet1D`, and
:class:`~lob_forge.generator.conditioning.ConditioningModule` into a trainable
model with DDPM and DDIM sampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lob_forge.generator.conditioning import ConditioningModule
from lob_forge.generator.noise_schedule import CosineNoiseSchedule
from lob_forge.generator.unet import UNet1D


class DiffusionModel(nn.Module):
    """Conditional diffusion model for LOB sequence generation.

    Wraps the noise schedule, U-Net denoiser, and conditioning module into a
    single interface supporting training loss computation and both DDPM
    (1000-step) and DDIM (50-step) sampling.

    Parameters
    ----------
    in_channels : int
        Number of LOB feature channels (default 40).
    d_model : int
        Base channel width / conditioning dimension (default 128).
    channel_mults : tuple[int, ...]
        Channel multipliers per U-Net resolution level.
    n_res_blocks : int
        Residual blocks per resolution level (default 2).
    num_timesteps : int
        Total diffusion timesteps *T* (default 1000).
    ddim_steps : int
        Default number of DDIM sampling steps (default 50).
    n_regimes : int
        Number of volatility regime classes (default 3).
    dropout : float
        Dropout probability (default 0.1).
    attention_levels : tuple[int, ...]
        U-Net levels that receive self-attention (default ``(2, 3)``).
    n_heads : int
        Number of attention heads (default 4).
    """

    def __init__(
        self,
        in_channels: int = 40,
        d_model: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4, 4),
        n_res_blocks: int = 2,
        num_timesteps: int = 1000,
        ddim_steps: int = 50,
        n_regimes: int = 3,
        dropout: float = 0.1,
        attention_levels: tuple[int, ...] = (2, 3),
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_timesteps = num_timesteps
        self.ddim_steps = ddim_steps

        self.schedule = CosineNoiseSchedule(num_timesteps)
        self.conditioning = ConditioningModule(d_model, n_regimes)
        self.unet = UNet1D(
            in_channels=in_channels,
            d_model=d_model,
            channel_mults=channel_mults,
            n_res_blocks=n_res_blocks,
            cond_dim=d_model,
            dropout=dropout,
            attention_levels=attention_levels,
            n_heads=n_heads,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_loss(
        self,
        x_0: Tensor,
        regime: Tensor,
        time_of_day: Tensor | None = None,
    ) -> Tensor:
        """Compute simple MSE diffusion training loss.

        Parameters
        ----------
        x_0 : Tensor
            Clean LOB sequences ``(B, T, C)``.
        regime : Tensor
            Volatility regime indices ``(B,)``.
        time_of_day : Tensor or None
            Normalized time-of-day ``(B,)`` in ``[0, 1]``.

        Returns
        -------
        Tensor
            Scalar MSE loss between predicted and actual noise.
        """
        batch_size = x_0.shape[0]

        # Permute (B, T, C) -> (B, C, T) for U-Net
        x_0_perm = x_0.permute(0, 2, 1)

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device)

        # Sample noise
        noise = torch.randn_like(x_0_perm)

        # Forward diffusion
        x_t = self.schedule.q_sample(x_0_perm, t, noise)

        # Get conditioning
        cond = self.conditioning(t, regime, time_of_day)

        # Predict noise
        noise_pred = self.unet(x_t, cond)

        return F.mse_loss(noise_pred, noise)

    # ------------------------------------------------------------------
    # DDPM sampling
    # ------------------------------------------------------------------

    def p_sample(self, x_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        """Single DDPM reverse step: predict x_{t-1} from x_t.

        Parameters
        ----------
        x_t : Tensor
            Current noisy state ``(B, C, T)``.
        t : Tensor
            Current timesteps ``(B,)``.
        cond : Tensor
            Conditioning vector ``(B, d_model)``.

        Returns
        -------
        Tensor
            Denoised state ``(B, C, T)`` at timestep ``t - 1``.
        """
        noise_pred = self.unet(x_t, cond)

        # Extract schedule coefficients
        sqrt_alpha_cumprod = self.schedule._extract(
            self.schedule.sqrt_alphas_cumprod, t, x_t.shape
        )
        sqrt_one_minus_alpha_cumprod = self.schedule._extract(
            self.schedule.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        posterior_mean_coef1 = self.schedule._extract(
            self.schedule.posterior_mean_coef1, t, x_t.shape
        )
        posterior_mean_coef2 = self.schedule._extract(
            self.schedule.posterior_mean_coef2, t, x_t.shape
        )
        posterior_variance = self.schedule._extract(
            self.schedule.posterior_variance, t, x_t.shape
        )

        # Predict x_0
        x_0_pred = (
            x_t - sqrt_one_minus_alpha_cumprod * noise_pred
        ) / sqrt_alpha_cumprod

        # Posterior mean
        mu = posterior_mean_coef1 * x_0_pred + posterior_mean_coef2 * x_t

        # Add noise (except at t=0)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().reshape(-1, *([1] * (x_t.ndim - 1)))
        sigma = posterior_variance.sqrt()

        return mu + nonzero_mask * sigma * noise

    @torch.no_grad()
    def ddpm_sample(
        self,
        n_samples: int,
        seq_len: int,
        regime: Tensor,
        time_of_day: Tensor | None = None,
    ) -> Tensor:
        """Generate LOB sequences via full DDPM reverse process.

        Parameters
        ----------
        n_samples : int
            Number of sequences to generate.
        seq_len : int
            Temporal length of each sequence.
        regime : Tensor
            Volatility regime indices ``(n_samples,)``.
        time_of_day : Tensor or None
            Normalized time-of-day ``(n_samples,)`` in ``[0, 1]``.

        Returns
        -------
        Tensor
            Generated LOB sequences ``(B, T, C)``.
        """
        device = regime.device
        x_t = torch.randn(n_samples, self.in_channels, seq_len, device=device)

        for step in reversed(range(self.num_timesteps)):
            t_batch = torch.full((n_samples,), step, device=device, dtype=torch.long)
            cond = self.conditioning(t_batch, regime, time_of_day)
            x_t = self.p_sample(x_t, t_batch, cond)

        # Permute (B, C, T) -> (B, T, C)
        return x_t.permute(0, 2, 1)

    # ------------------------------------------------------------------
    # DDIM sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        n_samples: int,
        seq_len: int,
        regime: Tensor,
        time_of_day: Tensor | None = None,
        ddim_steps: int | None = None,
        eta: float = 0.0,
    ) -> Tensor:
        """Generate LOB sequences via DDIM accelerated sampling.

        Parameters
        ----------
        n_samples : int
            Number of sequences to generate.
        seq_len : int
            Temporal length of each sequence.
        regime : Tensor
            Volatility regime indices ``(n_samples,)``.
        time_of_day : Tensor or None
            Normalized time-of-day ``(n_samples,)`` in ``[0, 1]``.
        ddim_steps : int or None
            Number of DDIM steps; uses ``self.ddim_steps`` if *None*.
        eta : float
            Stochasticity parameter. ``0.0`` = deterministic DDIM,
            ``1.0`` = equivalent to DDPM noise level.

        Returns
        -------
        Tensor
            Generated LOB sequences ``(B, T, C)``.
        """
        if ddim_steps is None:
            ddim_steps = self.ddim_steps

        device = regime.device

        # Create subsequence of timesteps
        timesteps = torch.linspace(
            0, self.num_timesteps - 1, ddim_steps, device=device
        ).long()

        x_t = torch.randn(n_samples, self.in_channels, seq_len, device=device)

        # Iterate through reversed timestep pairs
        for i in reversed(range(len(timesteps))):
            t = timesteps[i]
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

            cond = self.conditioning(t_batch, regime, time_of_day)
            noise_pred = self.unet(x_t, cond)

            # Current and previous alpha_cumprod
            alpha_cumprod_t = self.schedule._extract(
                self.schedule.alphas_cumprod, t_batch, x_t.shape
            )
            sqrt_alpha_cumprod_t = self.schedule._extract(
                self.schedule.sqrt_alphas_cumprod, t_batch, x_t.shape
            )
            sqrt_one_minus_alpha_cumprod_t = self.schedule._extract(
                self.schedule.sqrt_one_minus_alphas_cumprod, t_batch, x_t.shape
            )

            # Previous timestep alpha_cumprod
            if i > 0:
                t_prev = timesteps[i - 1]
                t_prev_batch = torch.full(
                    (n_samples,), t_prev, device=device, dtype=torch.long
                )
                alpha_cumprod_t_prev = self.schedule._extract(
                    self.schedule.alphas_cumprod, t_prev_batch, x_t.shape
                )
            else:
                alpha_cumprod_t_prev = torch.ones_like(alpha_cumprod_t)

            # Predict x_0
            x_0_pred = (
                x_t - sqrt_one_minus_alpha_cumprod_t * noise_pred
            ) / sqrt_alpha_cumprod_t

            # Compute sigma for stochasticity
            sigma = eta * torch.sqrt(
                (1.0 - alpha_cumprod_t_prev)
                / (1.0 - alpha_cumprod_t)
                * (1.0 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )

            # Direction pointing to x_t
            dir_xt = (
                torch.sqrt((1.0 - alpha_cumprod_t_prev - sigma**2).clamp(min=0))
                * noise_pred
            )

            # Combine
            x_t = torch.sqrt(alpha_cumprod_t_prev) * x_0_pred + dir_xt

            # Add noise if eta > 0
            if eta > 0 and i > 0:
                x_t = x_t + sigma * torch.randn_like(x_t)

        # Permute (B, C, T) -> (B, T, C)
        return x_t.permute(0, 2, 1)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def generate(
        self,
        n_samples: int,
        seq_len: int,
        regime: Tensor,
        time_of_day: Tensor | None = None,
        method: str = "ddim",
        **kwargs: object,
    ) -> Tensor:
        """Generate LOB sequences using the specified sampling method.

        Parameters
        ----------
        n_samples : int
            Number of sequences to generate.
        seq_len : int
            Temporal length of each sequence.
        regime : Tensor
            Volatility regime indices ``(n_samples,)``.
        time_of_day : Tensor or None
            Normalized time-of-day ``(n_samples,)`` in ``[0, 1]``.
        method : str
            Sampling method: ``"ddpm"`` or ``"ddim"`` (default).
        **kwargs
            Additional keyword arguments forwarded to the sampler.

        Returns
        -------
        Tensor
            Generated LOB sequences ``(B, T, C)``.

        Raises
        ------
        ValueError
            If *method* is not ``"ddpm"`` or ``"ddim"``.
        """
        if method == "ddim":
            return self.ddim_sample(n_samples, seq_len, regime, time_of_day, **kwargs)
        elif method == "ddpm":
            return self.ddpm_sample(n_samples, seq_len, regime, time_of_day)
        else:
            raise ValueError(
                f"Unknown sampling method: {method!r}. Use 'ddpm' or 'ddim'."
            )
