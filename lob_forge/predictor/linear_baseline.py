"""Linear (logistic regression) baseline for LOB classification.

Provides a performance floor — if the TLOB transformer or DeepLOB
cannot beat this model, something is wrong with the setup.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearBaseline(nn.Module):
    """Pure logistic regression baseline for LOB mid-price direction.

    Takes the last time step of the input sequence and applies a separate
    linear projection per prediction horizon. No hidden layers, no dropout,
    no activation — intentionally minimal.

    Parameters
    ----------
    n_levels : int
        Number of LOB price levels (default 10).
    features_per_level : int
        Features per level: bid_price, bid_size, ask_price, ask_size (default 4).
    n_classes : int
        Number of output classes — DOWN, STATIONARY, UP (default 3).
    n_horizons : int
        Number of prediction horizons; one head per horizon (default 4).
    """

    def __init__(
        self,
        n_levels: int = 10,
        features_per_level: int = 4,
        n_classes: int = 3,
        n_horizons: int = 4,
    ) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.features_per_level = features_per_level
        self.n_classes = n_classes
        self.n_horizons = n_horizons

        self.heads = nn.ModuleList(
            [
                nn.Linear(n_levels * features_per_level, n_classes)
                for _ in range(n_horizons)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, T, n_levels * features_per_level)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, n_horizons, n_classes)``.
        """
        # Take only the last time step: (B, T, 40) -> (B, 40)
        x = x[:, -1, :]

        # Per-horizon linear heads
        logits = torch.stack([head(x) for head in self.heads], dim=1)
        return logits  # (B, n_horizons, n_classes)
