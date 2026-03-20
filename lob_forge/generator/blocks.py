"""Core building blocks for the 1-D diffusion U-Net.

Provides Adaptive Layer Normalization (AdaLN) and a conditioning-aware
residual block (ResBlock1D) used throughout the U-Net encoder/decoder.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


def _init_conv(conv: nn.Conv1d) -> None:
    """Xavier-uniform weights, zero biases."""
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization (AdaLN) from DiT.

    Applies GroupNorm then modulates with conditioning-derived scale and shift:
    ``out = norm(x) * (1 + scale) + shift``.

    Parameters
    ----------
    channels : int
        Number of input channels.
    cond_dim : int
        Dimension of the conditioning vector.
    """

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        num_groups = min(32, channels)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply conditioned normalisation.

        Parameters
        ----------
        x : Tensor
            Feature map ``(B, C, T)``.
        cond : Tensor
            Conditioning vector ``(B, cond_dim)``.

        Returns
        -------
        Tensor
            Modulated feature map ``(B, C, T)``.
        """
        scale, shift = self.proj(cond).chunk(2, dim=-1)  # each (B, C)
        return self.norm(x) * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)


class ResBlock1D(nn.Module):
    """Conditioning-aware 1-D residual block.

    Architecture::

        AdaLN -> GELU -> Conv1d(3)
        AdaLN -> GELU -> Dropout -> Conv1d(3)
        + skip connection (Conv1d(1) if channel change, else identity)

    Parameters
    ----------
    channels : int
        Number of input channels.
    cond_dim : int
        Dimension of the conditioning vector.
    dropout : float
        Dropout probability (default 0.1).
    out_channels : int or None
        Output channels; defaults to *channels* (no change).
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        dropout: float = 0.1,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        out_channels = out_channels or channels

        # --- first sub-block ---
        self.norm1 = AdaptiveLayerNorm(channels, cond_dim)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv1d(channels, out_channels, kernel_size=3, padding=1)

        # --- second sub-block ---
        self.norm2 = AdaptiveLayerNorm(out_channels, cond_dim)
        self.act2 = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # --- skip connection ---
        if channels != out_channels:
            self.skip = nn.Conv1d(channels, out_channels, kernel_size=1)
            _init_conv(self.skip)
        else:
            self.skip = nn.Identity()

        # Stable initialisation
        _init_conv(self.conv1)
        _init_conv(self.conv2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply conditioned residual block.

        Parameters
        ----------
        x : Tensor
            Input ``(B, C, T)``.
        cond : Tensor
            Conditioning vector ``(B, cond_dim)``.

        Returns
        -------
        Tensor
            Output ``(B, out_C, T)``.
        """
        h = self.conv1(self.act1(self.norm1(x, cond)))
        h = self.conv2(self.drop(self.act2(self.norm2(h, cond))))
        return self.skip(x) + h
