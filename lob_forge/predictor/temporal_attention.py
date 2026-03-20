"""Temporal attention mechanism for sequential LOB state modeling.

After spatial attention has captured cross-level relationships at each
time step, temporal attention applies causal multi-head self-attention
*across* time steps.  A causal mask prevents each position *t* from
attending to future positions *t + 1 .. T*, ensuring the model cannot
leak future information during training or inference.

Architecture follows TLOB (arXiv 2502.15757):
  - Pre-LayerNorm (norm_first=True) for training stability
  - GELU activation in the feed-forward network
  - Boolean upper-triangular causal mask registered as a buffer
  - Mask pre-computed for ``max_seq_len`` and sliced in ``forward``
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TemporalAttentionBlock(nn.Module):
    """Causal multi-head self-attention across time steps.

    Parameters
    ----------
    d_model : int
        Embedding / feature dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of stacked transformer encoder layers.
    feedforward_dim : int
        Hidden dimension of the position-wise FFN.
    dropout : float
        Dropout rate applied inside attention and FFN.
    max_seq_len : int
        Maximum sequence length for causal-mask pre-computation.

    Input shape
    -----------
    ``(batch, T, d_model)`` — one feature vector per time step after
    spatial attention and level pooling.

    Output shape
    ------------
    Same as input: ``(batch, T, d_model)``.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Boolean causal mask: True = masked (position must NOT attend).
        # Upper-triangular with diagonal=1 blocks each position from
        # attending to any future position.
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal temporal self-attention across time steps.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, T, d_model)``.

        Returns
        -------
        Tensor
            Same shape as *x*.
        """
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]
        return self.encoder(x, mask=mask)
