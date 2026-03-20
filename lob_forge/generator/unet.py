"""1-D U-Net denoiser backbone for LOB diffusion generation.

Provides the encoder-decoder U-Net architecture with skip connections,
AdaLN conditioning injection, and self-attention at selected resolution
levels.  Built on top of :pymod:`lob_forge.generator.blocks`.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------


class Downsample1D(nn.Module):
    """Halve temporal dimension with a stride-2 convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """Double temporal dimension with nearest upsample + convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.up(x))


class SelfAttention1D(nn.Module):
    """Global self-attention over the temporal axis.

    Applies GroupNorm then multi-head attention with a residual connection,
    adding global context at bottleneck resolution.

    Parameters
    ----------
    channels : int
        Number of input channels.
    n_heads : int
        Number of attention heads (default 4).
    """

    def __init__(self, channels: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply self-attention with residual.

        Parameters
        ----------
        x : Tensor
            Feature map ``(B, C, T)``.

        Returns
        -------
        Tensor
            ``(B, C, T)`` with global context mixed in.
        """
        h = self.norm(x)
        # (B, C, T) -> (B, T, C) for MHA
        h = h.transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        # (B, T, C) -> (B, C, T)
        h = h.transpose(1, 2)
        return x + h
