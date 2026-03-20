"""Conditioning embeddings for the diffusion U-Net.

Provides sinusoidal timestep embedding, regime embedding, and time-of-day
embedding, combined into a single conditioning vector via ConditioningModule.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timesteps.

    Maps integer timesteps to dense float embeddings using fixed sinusoidal
    frequencies followed by a learned 2-layer MLP.

    Parameters
    ----------
    dim : int
        Output embedding dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Embed integer timesteps.

        Parameters
        ----------
        t : Tensor
            Integer timesteps of shape ``(B,)``.

        Returns
        -------
        Tensor
            Embeddings of shape ``(B, dim)``.
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ConditioningModule(nn.Module):
    """Combine timestep, regime, and time-of-day into a conditioning vector.

    Parameters
    ----------
    d_model : int
        Conditioning vector dimension (default 128).
    n_regimes : int
        Number of volatility regime classes (default 3).
    time_of_day_dim : int
        Hidden dimension for the time-of-day MLP (default 16).
    """

    def __init__(
        self,
        d_model: int = 128,
        n_regimes: int = 3,
        time_of_day_dim: int = 16,
    ) -> None:
        super().__init__()
        self.timestep_emb = SinusoidalTimestepEmbedding(d_model)
        self.regime_emb = nn.Embedding(n_regimes, d_model)
        self.tod_mlp = nn.Sequential(
            nn.Linear(1, time_of_day_dim),
            nn.GELU(),
            nn.Linear(time_of_day_dim, d_model),
        )

    def forward(
        self,
        t: Tensor,
        regime: Tensor,
        time_of_day: Tensor | None = None,
    ) -> Tensor:
        """Produce combined conditioning vector.

        Parameters
        ----------
        t : Tensor
            Diffusion timesteps ``(B,)``.
        regime : Tensor
            Volatility regime indices ``(B,)``.
        time_of_day : Tensor or None
            Normalized time-of-day in ``[0, 1]`` with shape ``(B,)``.
            Omitted when not available at inference.

        Returns
        -------
        Tensor
            Conditioning vector ``(B, d_model)``.
        """
        cond = self.timestep_emb(t) + self.regime_emb(regime)
        if time_of_day is not None:
            cond = cond + self.tod_mlp(time_of_day.unsqueeze(-1))
        return cond
