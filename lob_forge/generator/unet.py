"""1-D U-Net denoiser backbone for LOB diffusion generation.

Provides the encoder-decoder U-Net architecture with skip connections,
AdaLN conditioning injection, and self-attention at selected resolution
levels.  Built on top of :pymod:`lob_forge.generator.blocks`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from lob_forge.generator.blocks import ResBlock1D

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


# ---------------------------------------------------------------------------
# UNet1D
# ---------------------------------------------------------------------------


class UNet1D(nn.Module):
    """1-D U-Net denoiser for LOB sequence generation.

    Encoder-decoder with skip connections, AdaLN conditioning injection at
    every :class:`ResBlock1D`, and optional self-attention at selected
    resolution levels.

    Parameters
    ----------
    in_channels : int
        Number of LOB feature channels (default 40).
    d_model : int
        Base channel width (default 128).
    channel_mults : tuple[int, ...]
        Channel multipliers per resolution level (default ``(1, 2, 4, 4)``).
    n_res_blocks : int
        Number of residual blocks per level (default 2).
    cond_dim : int
        Conditioning vector dimension (default 128).
    dropout : float
        Dropout probability (default 0.1).
    attention_levels : tuple[int, ...]
        Which resolution levels receive self-attention (default ``(2, 3)``).
    n_heads : int
        Number of attention heads (default 4).
    """

    def __init__(
        self,
        in_channels: int = 40,
        d_model: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4, 4),
        n_res_blocks: int = 2,
        cond_dim: int = 128,
        dropout: float = 0.1,
        attention_levels: tuple[int, ...] = (2, 3),
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_res_blocks = n_res_blocks

        # --- input projection ---
        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)

        # --- encoder ---
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch_in = d_model
        encoder_channels: list[int] = []  # track channels at each skip point

        for level, mult in enumerate(channel_mults):
            ch_out = d_model * mult
            level_blocks = nn.ModuleList()
            for _ in range(n_res_blocks):
                level_blocks.append(
                    ResBlock1D(ch_in, cond_dim, dropout, out_channels=ch_out)
                )
                if level in attention_levels:
                    level_blocks.append(SelfAttention1D(ch_out, n_heads))
                encoder_channels.append(ch_out)
                ch_in = ch_out
            self.encoder_blocks.append(level_blocks)

            if level < len(channel_mults) - 1:
                self.downsamples.append(Downsample1D(ch_out))
            else:
                self.downsamples.append(nn.Identity())  # placeholder

        # --- bottleneck ---
        self.bottleneck = nn.ModuleList(
            [
                ResBlock1D(ch_in, cond_dim, dropout),
                SelfAttention1D(ch_in, n_heads),
                ResBlock1D(ch_in, cond_dim, dropout),
            ]
        )

        # --- decoder ---
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level, mult in enumerate(reversed(channel_mults)):
            ch_out = d_model * mult
            level_blocks = nn.ModuleList()
            for _i in range(n_res_blocks):
                # pop skip channel count from encoder_channels
                ch_skip = encoder_channels.pop()
                block_in = ch_in + ch_skip
                level_blocks.append(
                    ResBlock1D(block_in, cond_dim, dropout, out_channels=ch_out)
                )
                if (len(channel_mults) - 1 - level) in attention_levels:
                    level_blocks.append(SelfAttention1D(ch_out, n_heads))
                ch_in = ch_out
            self.decoder_blocks.append(level_blocks)

            if level < len(channel_mults) - 1:
                self.upsamples.append(Upsample1D(ch_out))
            else:
                self.upsamples.append(nn.Identity())  # placeholder

        # --- output projection ---
        self.out_norm = nn.GroupNorm(min(32, d_model), d_model)
        self.out_act = nn.GELU()
        self.out_proj = nn.Conv1d(d_model, in_channels, kernel_size=1)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Predict noise from noisy input and conditioning.

        Parameters
        ----------
        x : Tensor
            Noisy LOB sequences ``(B, C_in, T)``.
        cond : Tensor
            Conditioning vector ``(B, cond_dim)``.

        Returns
        -------
        Tensor
            Noise prediction ``(B, C_in, T)`` matching input shape.
        """
        h = self.input_proj(x)

        # --- encoder: collect skip connections ---
        skips: list[Tensor] = []
        for level, level_blocks in enumerate(self.encoder_blocks):
            idx = 0
            for _ in range(self.n_res_blocks):
                h = level_blocks[idx](h, cond)
                idx += 1
                # check for attention block
                if idx < len(level_blocks) and isinstance(
                    level_blocks[idx], SelfAttention1D
                ):
                    h = level_blocks[idx](h)
                    idx += 1
                skips.append(h)
            # downsample (Identity at last level)
            if not isinstance(self.downsamples[level], nn.Identity):
                h = self.downsamples[level](h)

        # --- bottleneck ---
        h = self.bottleneck[0](h, cond)
        h = self.bottleneck[1](h)
        h = self.bottleneck[2](h, cond)

        # --- decoder: consume skip connections in reverse ---
        for level, level_blocks in enumerate(self.decoder_blocks):
            # upsample first (except first decoder level)
            if level > 0 and not isinstance(self.upsamples[level - 1], nn.Identity):
                h = self.upsamples[level - 1](h)

            idx = 0
            for _ in range(self.n_res_blocks):
                skip = skips.pop()
                # handle odd-length mismatch from downsample/upsample
                if h.shape[-1] != skip.shape[-1]:
                    h = h[..., : skip.shape[-1]]
                h = torch.cat([h, skip], dim=1)
                h = level_blocks[idx](h, cond)
                idx += 1
                if idx < len(level_blocks) and isinstance(
                    level_blocks[idx], SelfAttention1D
                ):
                    h = level_blocks[idx](h)
                    idx += 1

            # upsample after last decoder level block only for the last level
            # (handled by level > 0 check above for other levels)

        # final upsample for last decoder level
        if not isinstance(self.upsamples[-1], nn.Identity):
            h = self.upsamples[-1](h)

        # --- output projection ---
        h = self.out_proj(self.out_act(self.out_norm(h)))
        return h
