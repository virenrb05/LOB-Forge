"""DeepLOB baseline model (simplified Zhang et al. 2019).

CNN + Inception + LSTM architecture for LOB mid-price direction prediction.
Used as a comparison baseline against the TLOB dual-attention transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepLOB(nn.Module):
    """DeepLOB: CNN + Inception + LSTM baseline for LOB classification.

    Architecture
    ------------
    1. Two convolutional blocks with stride-2 spatial reduction
    2. Inception module (three parallel branches at different kernel widths)
    3. Spatial collapse via 1x10 convolution
    4. Single-layer LSTM over the temporal axis
    5. Per-horizon linear classification heads

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
    lstm_hidden : int
        Hidden size of the LSTM layer (default 64).
    """

    def __init__(
        self,
        n_levels: int = 10,
        features_per_level: int = 4,
        n_classes: int = 3,
        n_horizons: int = 4,
        lstm_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.features_per_level = features_per_level
        self.n_classes = n_classes
        self.n_horizons = n_horizons
        self.lstm_hidden = lstm_hidden

        # --- Conv block 1: (B, 1, T, 40) -> (B, 16, T, 20) ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(16),
        )

        # --- Conv block 2: (B, 16, T, 20) -> (B, 32, T, 10) ---
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )

        # --- Inception module: (B, 32, T, 10) -> (B, 96, T, 10) ---
        self.inception_a = nn.Conv2d(32, 32, kernel_size=(1, 1))
        self.inception_b = nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0))
        self.inception_c = nn.Conv2d(32, 32, kernel_size=(5, 1), padding=(2, 0))

        # --- Spatial collapse: (B, 96, T, 10) -> (B, 32, T, 1) ---
        self.collapse = nn.Conv2d(96, 32, kernel_size=(1, n_levels))

        # --- LSTM: (B, T, 32) -> (B, T, lstm_hidden) ---
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # --- Per-horizon classification heads ---
        self.heads = nn.ModuleList(
            [nn.Linear(lstm_hidden, n_classes) for _ in range(n_horizons)]
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
        # (B, T, 40) -> (B, 1, T, 40)
        x = x.unsqueeze(1)

        # Conv blocks
        x = self.conv1(x)  # (B, 16, T, 20)
        x = self.conv2(x)  # (B, 32, T, 10)

        # Inception module — three parallel branches
        a = self.inception_a(x)  # (B, 32, T, 10)
        b = self.inception_b(x)  # (B, 32, T, 10)
        c = self.inception_c(x)  # (B, 32, T, 10)
        x = torch.cat([a, b, c], dim=1)  # (B, 96, T, 10)

        # Spatial collapse
        x = self.collapse(x)  # (B, 32, T, 1)
        x = x.squeeze(-1)  # (B, 32, T)
        x = x.permute(0, 2, 1)  # (B, T, 32)

        # LSTM
        x, _ = self.lstm(x)  # (B, T, lstm_hidden)
        x = x[:, -1, :]  # (B, lstm_hidden) — last time step

        # Per-horizon heads
        logits = torch.stack([head(x) for head in self.heads], dim=1)
        return logits  # (B, n_horizons, n_classes)
