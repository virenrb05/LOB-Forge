"""Spatial attention mechanism for cross-level LOB feature extraction.

For each time step independently, spatial attention applies multi-head
self-attention across LOB price levels.  This allows the model to learn
cross-level relationships — e.g., how level-1 bid-ask imbalance relates
to deeper-book liquidity at level 5 or 10 — without mixing information
across time steps.

Architecture follows TLOB (arXiv 2502.15757):
  - Pre-LayerNorm (norm_first=True) for training stability
  - GELU activation in the feed-forward network
  - Wrapped ``nn.TransformerEncoder`` for multi-layer stacking
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpatialAttentionBlock(nn.Module):
    """Multi-head self-attention across LOB levels per time step.

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

    Input shape
    -----------
    ``(batch * T, L, d_model)`` where *L* is the number of LOB levels.

    Output shape
    ------------
    Same as input: ``(batch * T, L, d_model)``.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
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

    def forward(self, x: Tensor) -> Tensor:
        """Apply spatial self-attention across LOB levels.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch * T, L, d_model)``.

        Returns
        -------
        Tensor
            Same shape as *x*.
        """
        return self.encoder(x)
