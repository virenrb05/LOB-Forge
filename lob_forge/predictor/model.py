"""End-to-end spatial-temporal transformer model for LOB prediction.

The ``DualAttentionTransformer`` (TLOB architecture) composes:

1. **Input embedding** — projects per-level features into *d_model* space
   with learnable level and temporal positional encodings.
2. **Spatial attention** — cross-level self-attention at each time step
   (via :class:`SpatialAttentionBlock`).
3. **Level pooling** — mean pool across levels.
4. **Temporal attention** — causal self-attention across time steps
   (via :class:`TemporalAttentionBlock`).
5. **Per-horizon classification heads** — predict {DOWN, STATIONARY, UP}
   for each forecast horizon.
6. **Optional VPIN regression head** — predict volume-clock probability
   of informed trading in [0, 1].
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lob_forge.predictor.spatial_attention import SpatialAttentionBlock
from lob_forge.predictor.temporal_attention import TemporalAttentionBlock


class DualAttentionTransformer(nn.Module):
    """TLOB dual-attention transformer for LOB mid-price prediction.

    Parameters
    ----------
    n_levels : int
        Number of LOB price levels (default 10).
    features_per_level : int
        Features per level: bid_price, bid_size, ask_price, ask_size (default 4).
    d_model : int
        Embedding dimension.
    n_heads : int
        Number of attention heads.
    n_spatial_layers : int
        Number of spatial transformer encoder layers.
    n_temporal_layers : int
        Number of temporal transformer encoder layers.
    feedforward_dim : int
        Hidden dimension of position-wise FFN.
    dropout : float
        Dropout rate.
    n_classes : int
        Number of classification targets (default 3: DOWN, STATIONARY, UP).
    n_horizons : int
        Number of prediction horizons.
    max_seq_len : int
        Maximum sequence length for temporal positional encoding.
    vpin_head : bool
        Whether to include the VPIN regression head.
    """

    def __init__(
        self,
        n_levels: int = 10,
        features_per_level: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_spatial_layers: int = 2,
        n_temporal_layers: int = 2,
        feedforward_dim: int = 128,
        dropout: float = 0.1,
        n_classes: int = 3,
        n_horizons: int = 4,
        max_seq_len: int = 512,
        vpin_head: bool = True,
    ) -> None:
        super().__init__()

        self.n_levels = n_levels
        self.features_per_level = features_per_level
        self.d_model = d_model
        self.n_horizons = n_horizons
        self.has_vpin_head = vpin_head

        # 1. Input embedding
        self.input_proj = nn.Linear(features_per_level, d_model)
        self.level_pe = nn.Parameter(torch.randn(1, 1, n_levels, d_model) * 0.02)
        self.time_pe = nn.Parameter(torch.randn(1, max_seq_len, 1, d_model) * 0.02)

        # 2. Spatial attention
        self.spatial_attn = SpatialAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_spatial_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

        # 4. Temporal attention
        self.temporal_attn = TemporalAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_temporal_layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # 5. Per-horizon classification heads
        self.cls_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, n_classes),
                )
                for _ in range(n_horizons)
            ]
        )

        # 6. Optional VPIN head
        if vpin_head:
            self.vpin = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, T, n_levels * features_per_level)``.
            Column ordering: bid_price_1..10, bid_size_1..10,
            ask_price_1..10, ask_size_1..10 (4 groups of 10).

        Returns
        -------
        dict[str, Tensor]
            ``"logits"``: ``(batch, n_horizons, n_classes)``
            ``"vpin"``: ``(batch, 1)`` (only if ``vpin_head=True``)
            ``"embedding"``: ``(batch, d_model)`` — last-timestep embedding
        """
        B, T, _ = x.shape
        L = self.n_levels

        # 1. Reshape flat features → (B, T, L, features_per_level)
        # Input is 4 groups of 10: reshape to (B, T, 4, 10) then permute
        x = x.view(B, T, self.features_per_level, L).permute(0, 1, 3, 2)

        # Project to d_model
        x = self.input_proj(x)  # (B, T, L, d_model)

        # Add positional encodings
        x = x + self.level_pe + self.time_pe[:, :T, :, :]

        # 2. Spatial attention: (B, T, L, d_model) → (B*T, L, d_model)
        x = x.reshape(B * T, L, self.d_model)
        x = self.spatial_attn(x)
        x = x.reshape(B, T, L, self.d_model)

        # 3. Level pooling: mean across levels
        x = x.mean(dim=2)  # (B, T, d_model)

        # 4. Temporal attention
        x = self.temporal_attn(x)  # (B, T, d_model)

        # Take last time step
        embedding = x[:, -1, :]  # (B, d_model)

        # 5. Classification heads
        logits = torch.stack(
            [head(embedding) for head in self.cls_heads], dim=1
        )  # (B, n_horizons, n_classes)

        out: dict[str, Tensor] = {
            "logits": logits,
            "embedding": embedding,
        }

        # 6. VPIN head
        if self.has_vpin_head:
            out["vpin"] = self.vpin(embedding)  # (B, 1)

        return out
