"""PyTorch Dataset and DataLoader utilities for LOB time-series data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class LOBDataset(Dataset):
    """Sliding-window dataset for LOB predictor training.

    Each sample is a ``(features, labels)`` tuple where *features* is a
    ``(sequence_length, num_features)`` float32 tensor and *labels* is a
    ``(num_horizons,)`` int64 tensor taken at the end of the window.

    Parameters
    ----------
    parquet_path : str | Path
        Path to a preprocessed Parquet file (output of :func:`preprocess`).
    sequence_length : int
        Number of time steps per sample.
    horizons : list[int]
        Label horizons; the dataset expects columns ``label_h{h}`` for each.
    feature_cols : list[str] | None
        Explicit feature column list.  When *None*, all float64 columns
        excluding ``label_*``, ``timestamp``, and ``trade_side`` are used.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        sequence_length: int = 100,
        horizons: list[int] | None = None,
        feature_cols: list[str] | None = None,
        vpin_col: str | None = None,
    ) -> None:
        if horizons is None:
            horizons = [10, 20, 50, 100]

        table = pq.read_table(str(parquet_path), memory_map=True)
        schema = table.schema

        # Resolve feature columns
        if feature_cols is None:
            exclude_prefixes = ("label_",)
            exclude_names = {"timestamp", "trade_side"}
            feature_cols = [
                f.name
                for f in schema
                if f.name not in exclude_names
                and not any(f.name.startswith(p) for p in exclude_prefixes)
                and str(f.type) in ("double", "float")
            ]

        label_cols = [f"label_h{h}" for h in horizons]

        df = table.to_pandas()

        self.features = df[feature_cols].to_numpy(dtype=np.float32)
        self.labels = df[label_cols].to_numpy(dtype=np.float64)
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.vpin_col = vpin_col
        self.vpin: np.ndarray | None = None
        if vpin_col is not None and vpin_col in df.columns:
            self.vpin = df[vpin_col].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.features) - self.sequence_length

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        features = torch.from_numpy(
            self.features[idx : idx + self.sequence_length].copy()
        )
        # Label at the END of the window (the prediction target)
        raw_label = self.labels[idx + self.sequence_length]
        # NaN labels become 0 (safe default); real training masks NaN rows
        label_clean = np.where(np.isnan(raw_label), 0, raw_label)
        labels = torch.tensor(label_clean, dtype=torch.int64)
        if self.vpin is not None:
            vpin_target = torch.tensor(
                self.vpin[idx + self.sequence_length], dtype=torch.float32
            )
            return features, labels, vpin_target
        return features, labels


class LOBSequenceDataset(Dataset):
    """Sliding-window dataset for LOB generator (diffusion) training.

    Each sample is a ``(sequence, regime)`` tuple where *sequence* is a
    ``(sequence_length, num_features)`` float32 tensor and *regime* is a
    scalar int64 tensor classifying the window's volatility regime.

    Regime labels:
        0 = low-vol (realized vol < 33rd percentile)
        1 = normal  (33rd--67th percentile)
        2 = high-vol (> 67th percentile)

    Parameters
    ----------
    parquet_path : str | Path
        Path to a preprocessed Parquet file.
    sequence_length : int
        Number of time steps per sample.
    feature_cols : list[str] | None
        Explicit feature column list (same auto-detect logic as LOBDataset).
    vol_quantiles : tuple[float, float]
        Quantile boundaries for regime classification.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        sequence_length: int = 100,
        feature_cols: list[str] | None = None,
        vol_quantiles: tuple[float, float] = (0.33, 0.67),
    ) -> None:
        table = pq.read_table(str(parquet_path), memory_map=True)
        schema = table.schema

        # Resolve feature columns
        if feature_cols is None:
            exclude_prefixes = ("label_",)
            exclude_names = {"timestamp", "trade_side"}
            feature_cols = [
                f.name
                for f in schema
                if f.name not in exclude_names
                and not any(f.name.startswith(p) for p in exclude_prefixes)
                and str(f.type) in ("double", "float")
            ]

        df = table.to_pandas()

        self.features = df[feature_cols].to_numpy(dtype=np.float32)
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols

        # Compute mid returns for realized volatility
        if "mid_return_1" in df.columns:
            mid_returns = df["mid_return_1"].to_numpy(dtype=np.float64)
        elif "mid_price" in df.columns:
            mid = df["mid_price"].to_numpy(dtype=np.float64)
            mid_returns = np.empty(len(mid), dtype=np.float64)
            mid_returns[0] = np.nan
            mid_returns[1:] = np.diff(mid) / mid[:-1]
        else:
            raise ValueError(
                "Cannot compute regime labels: no mid_return_1 or mid_price column"
            )

        # Precompute realized volatility per window
        n_windows = len(self.features) - sequence_length
        realized_vols = np.empty(n_windows, dtype=np.float64)
        for i in range(n_windows):
            window_returns = mid_returns[i : i + sequence_length]
            realized_vols[i] = np.nanstd(window_returns)

        # Classify into regimes
        q_low, q_high = np.quantile(realized_vols, vol_quantiles)
        regimes = np.ones(n_windows, dtype=np.int64)  # default: normal (1)
        regimes[realized_vols < q_low] = 0  # low-vol
        regimes[realized_vols > q_high] = 2  # high-vol
        self.regimes = regimes

    def __len__(self) -> int:
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = torch.from_numpy(
            self.features[idx : idx + self.sequence_length].copy()
        )
        regime = torch.tensor(self.regimes[idx], dtype=torch.int64)
        return sequence, regime
