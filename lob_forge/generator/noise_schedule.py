"""Noise scheduling strategies for the diffusion forward and reverse processes.

Implements the cosine noise schedule from Nichol & Dhariwal 2021.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class CosineNoiseSchedule(nn.Module):
    """Cosine noise schedule for diffusion models (Nichol & Dhariwal 2021).

    Computes and stores all schedule coefficients as registered buffers
    so they automatically transfer with ``.to(device)``.

    Parameters
    ----------
    num_timesteps : int
        Total diffusion timesteps *T*.
    s : float
        Small offset preventing beta from being too small near *t = 0*.
    """

    def __init__(self, num_timesteps: int = 1000, s: float = 0.008) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps

        # --- cosine schedule: f(t) = cos((t/T + s) / (1 + s) * pi/2)^2 ---
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        f = torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]

        # Clip betas to [0, 0.999]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(max=0.999)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float64), alphas_cumprod[:-1]]
        )

        # Posterior coefficients for q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
        posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # Register all as float32 buffers
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod).float(),
        )
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1.float())
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2.float())
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer(
            "posterior_log_variance_clipped", posterior_log_variance_clipped.float()
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(a: Tensor, t: Tensor, x_shape: tuple) -> Tensor:
        """Index into schedule tensor *a* at timestep positions *t*.

        Returns tensor reshaped to ``(B, 1, 1, ...)`` for broadcasting
        with a tensor of shape *x_shape*.
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    # ------------------------------------------------------------------
    # Forward diffusion
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x_0: Tensor,
        t: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Forward diffusion: sample *x_t* from *q(x_t | x_0)*.

        .. math::
            x_t = \\sqrt{\\bar\\alpha_t}\\, x_0
                  + \\sqrt{1 - \\bar\\alpha_t}\\, \\varepsilon

        Parameters
        ----------
        x_0 : Tensor
            Clean data, arbitrary shape with leading batch dimension.
        t : Tensor
            Integer timesteps of shape ``(B,)``.
        noise : Tensor or None
            Pre-sampled noise; generated from standard normal if *None*.

        Returns
        -------
        Tensor
            Noised data *x_t* with the same shape as *x_0*.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
