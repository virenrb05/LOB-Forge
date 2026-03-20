"""Tests for LOBDataset and LOBSequenceDataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from lob_forge.data.dataset import LOBDataset, LOBSequenceDataset


@pytest.fixture()
def sample_parquet(tmp_path):
    """Create a minimal preprocessed Parquet file for testing."""
    n = 500
    rng = np.random.default_rng(42)
    mid = 100.0 + np.cumsum(rng.standard_normal(n) * 0.001)

    data = {
        "timestamp": np.arange(n, dtype=np.int64),
        "mid_price": mid,
        "spread_bps": rng.random(n).astype(np.float64) * 10,
    }
    for i in range(1, 11):
        data[f"bid_price_{i}"] = mid - 0.005 * i
        data[f"bid_size_{i}"] = rng.random(n).astype(np.float64) * 100
        data[f"ask_price_{i}"] = mid + 0.005 * i
        data[f"ask_size_{i}"] = rng.random(n).astype(np.float64) * 100
    data["trade_price"] = mid
    data["trade_size"] = rng.random(n).astype(np.float64) * 10
    data["trade_side"] = rng.choice([-1, 0, 1], n).astype(np.int8)
    data["mid_return_1"] = np.concatenate([[np.nan], np.diff(mid) / mid[:-1]]).astype(
        np.float64
    )
    data["order_imbalance"] = rng.standard_normal(n).astype(np.float64)

    for h in [10, 20, 50, 100]:
        labels = rng.choice([0.0, 1.0, 2.0], n)
        labels[-h:] = np.nan
        data[f"label_h{h}"] = labels

    df = pd.DataFrame(data)
    path = tmp_path / "test_data.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))
    return path, n


class TestLOBDataset:
    def test_len(self, sample_parquet):
        path, n = sample_parquet
        seq_len = 50
        ds = LOBDataset(path, sequence_length=seq_len, horizons=[10])
        assert len(ds) == n - seq_len

    def test_getitem_shapes(self, sample_parquet):
        path, _ = sample_parquet
        seq_len = 50
        horizons = [10, 20]
        ds = LOBDataset(path, sequence_length=seq_len, horizons=horizons)
        features, labels = ds[0]

        assert features.shape[0] == seq_len
        assert features.ndim == 2
        assert features.dtype == torch.float32
        assert labels.shape == (len(horizons),)
        assert labels.dtype == torch.int64

    def test_getitem_last_valid(self, sample_parquet):
        path, n = sample_parquet
        seq_len = 50
        ds = LOBDataset(path, sequence_length=seq_len, horizons=[10])
        features, labels = ds[len(ds) - 1]
        assert features.shape[0] == seq_len

    def test_label_values_valid(self, sample_parquet):
        path, _ = sample_parquet
        ds = LOBDataset(path, sequence_length=50, horizons=[10])
        for i in range(min(10, len(ds))):
            _, labels = ds[i]
            assert all(val.item() in {0, 1, 2} for val in labels)

    def test_auto_feature_detection(self, sample_parquet):
        path, _ = sample_parquet
        ds = LOBDataset(path, sequence_length=50, horizons=[10])
        assert "timestamp" not in ds.feature_cols
        assert "trade_side" not in ds.feature_cols
        assert all(not c.startswith("label_") for c in ds.feature_cols)

    def test_explicit_feature_cols(self, sample_parquet):
        path, _ = sample_parquet
        cols = ["mid_price", "spread_bps"]
        ds = LOBDataset(path, sequence_length=50, horizons=[10], feature_cols=cols)
        features, _ = ds[0]
        assert features.shape[1] == 2

    def test_dataloader(self, sample_parquet):
        path, _ = sample_parquet
        ds = LOBDataset(path, sequence_length=50, horizons=[10])
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
        batch_features, batch_labels = next(iter(loader))
        assert batch_features.shape == (8, 50, len(ds.feature_cols))
        assert batch_labels.shape == (8, 1)


class TestLOBSequenceDataset:
    def test_len(self, sample_parquet):
        path, n = sample_parquet
        seq_len = 50
        ds = LOBSequenceDataset(path, sequence_length=seq_len)
        assert len(ds) == n - seq_len

    def test_getitem_shapes(self, sample_parquet):
        path, _ = sample_parquet
        seq_len = 50
        ds = LOBSequenceDataset(path, sequence_length=seq_len)
        sequence, regime = ds[0]

        assert sequence.shape[0] == seq_len
        assert sequence.ndim == 2
        assert sequence.dtype == torch.float32
        assert regime.shape == ()
        assert regime.dtype == torch.int64

    def test_regime_values(self, sample_parquet):
        path, _ = sample_parquet
        ds = LOBSequenceDataset(path, sequence_length=50)
        regimes = set()
        for i in range(len(ds)):
            _, regime = ds[i]
            regimes.add(regime.item())
        assert regimes.issubset({0, 1, 2})
        assert len(regimes) >= 2  # at least 2 distinct regimes

    def test_regime_distribution(self, sample_parquet):
        path, _ = sample_parquet
        ds = LOBSequenceDataset(path, sequence_length=50)
        counts = {0: 0, 1: 0, 2: 0}
        for i in range(len(ds)):
            _, regime = ds[i]
            counts[regime.item()] += 1
        # Each regime should have at least some samples with 500 rows
        for r in [0, 1, 2]:
            assert counts[r] > 0, f"Regime {r} has no samples"

    def test_dataloader(self, sample_parquet):
        path, _ = sample_parquet
        ds = LOBSequenceDataset(path, sequence_length=50)
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)
        batch_seq, batch_regime = next(iter(loader))
        assert batch_seq.shape == (8, 50, len(ds.feature_cols))
        assert batch_regime.shape == (8,)

    def test_auto_feature_detection(self, sample_parquet):
        path, _ = sample_parquet
        ds = LOBSequenceDataset(path, sequence_length=50)
        assert "timestamp" not in ds.feature_cols
        assert "trade_side" not in ds.feature_cols


class TestMPSLoading:
    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available",
    )
    def test_lob_dataset_mps(self, sample_parquet):
        path, _ = sample_parquet
        ds = LOBDataset(path, sequence_length=50, horizons=[10])
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        features, labels = next(iter(loader))
        features = features.to("mps")
        labels = labels.to("mps")
        assert features.device.type == "mps"
        assert labels.device.type == "mps"

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available",
    )
    def test_lob_sequence_dataset_mps(self, sample_parquet):
        path, _ = sample_parquet
        ds = LOBSequenceDataset(path, sequence_length=50)
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        sequence, regime = next(iter(loader))
        sequence = sequence.to("mps")
        regime = regime.to("mps")
        assert sequence.device.type == "mps"
        assert regime.device.type == "mps"
