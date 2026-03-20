"""Integration tests for training loop on synthetic data.

Verifies model factory, training pipeline, VPIN loss, early stopping,
checkpoint saving, and LOBDataset VPIN extension.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf

from lob_forge.data.dataset import LOBDataset
from lob_forge.data.schema import (
    ASK_PRICE_COLS,
    ASK_SIZE_COLS,
    BID_PRICE_COLS,
    BID_SIZE_COLS,
)
from lob_forge.predictor.deeplob import DeepLOB
from lob_forge.predictor.linear_baseline import LinearBaseline
from lob_forge.predictor.model import DualAttentionTransformer
from lob_forge.predictor.trainer import build_model, train_model

BOOK_COLS = BID_PRICE_COLS + BID_SIZE_COLS + ASK_PRICE_COLS + ASK_SIZE_COLS


def _create_synthetic_parquet(tmp_path: Path, n_rows: int = 500) -> tuple[Path, Path]:
    """Generate minimal synthetic Parquet files for train and val."""
    rng = np.random.RandomState(42)

    data: dict[str, np.ndarray] = {}
    data["timestamp"] = np.arange(n_rows, dtype=np.int64)

    # 40 book columns
    for col in BOOK_COLS:
        data[col] = rng.rand(n_rows).astype(np.float64) * 100

    # Derived columns
    data["mid_price"] = (data["bid_price_1"] + data["ask_price_1"]) / 2
    data["spread"] = data["ask_price_1"] - data["bid_price_1"]
    data["trade_price"] = data["mid_price"] + rng.randn(n_rows) * 0.01
    data["trade_size"] = rng.rand(n_rows).astype(np.float64) * 10
    data["trade_side"] = rng.choice([0, 1], size=n_rows).astype(np.int8)

    # Labels for 4 horizons
    for h in [10, 20, 50, 100]:
        labels = rng.choice([0.0, 1.0, 2.0], size=n_rows).astype(np.float64)
        data[f"label_h{h}"] = labels

    # VPIN column
    data["vpin_50"] = rng.rand(n_rows).astype(np.float64)

    df = pd.DataFrame(data)
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(train_path))
    pq.write_table(table, str(val_path))
    return train_path, val_path


def _make_test_cfg(tmp_path: Path, model: str = "dual_attention") -> OmegaConf:
    """Create minimal test config."""
    return OmegaConf.create(
        {
            "project": {"name": "test", "seed": 42, "device": "cpu"},
            "training": {
                "epochs": 2,
                "batch_size": 32,
                "gradient_accumulation": 1,
                "early_stopping_patience": 5,
                "num_workers": 0,
            },
            "predictor": {
                "model": model,
                "d_model": 16,
                "n_heads": 2,
                "n_spatial_layers": 1,
                "n_temporal_layers": 1,
                "feedforward_dim": 32,
                "dropout": 0.0,
                "n_levels": 10,
                "features_per_level": 4,
                "n_classes": 3,
                "n_horizons": 4,
                "max_seq_len": 64,
                "focal_loss_gamma": 2.0,
                "vpin_head": model == "dual_attention",
                "vpin_loss_weight": 0.1,
                "optimizer": {"lr": 1e-3, "weight_decay": 0},
                "scheduler": {"max_lr": 1e-2, "pct_start": 0.3},
            },
            "data": {
                "sequence_length": 20,
                "horizons": [10, 20, 50, 100],
                "data_dir": str(tmp_path),
            },
            "wandb": {"project": "test", "enabled": False},
        }
    )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
class TestBuildModel:
    def test_build_dual_attention(self, tmp_path: Path) -> None:
        cfg = _make_test_cfg(tmp_path, model="dual_attention")
        m = build_model(cfg)
        assert isinstance(m, DualAttentionTransformer)

    def test_build_deeplob(self, tmp_path: Path) -> None:
        cfg = _make_test_cfg(tmp_path, model="deeplob")
        m = build_model(cfg)
        assert isinstance(m, DeepLOB)

    def test_build_linear(self, tmp_path: Path) -> None:
        cfg = _make_test_cfg(tmp_path, model="linear")
        m = build_model(cfg)
        assert isinstance(m, LinearBaseline)

    def test_build_unknown_raises(self, tmp_path: Path) -> None:
        cfg = _make_test_cfg(tmp_path, model="unknown")
        with pytest.raises(ValueError, match="Unknown model"):
            build_model(cfg)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
class TestTrainModel:
    def test_train_dual_attention_2_epochs(self, tmp_path: Path) -> None:
        train_path, val_path = _create_synthetic_parquet(tmp_path)
        cfg = _make_test_cfg(tmp_path, model="dual_attention")
        out_dir = tmp_path / "output"
        result = train_model(cfg, train_path, val_path, out_dir)

        assert "best_metrics" in result
        assert "best_epoch" in result
        assert (out_dir / "best_model.pt").exists()

    def test_train_deeplob_2_epochs(self, tmp_path: Path) -> None:
        train_path, val_path = _create_synthetic_parquet(tmp_path)
        cfg = _make_test_cfg(tmp_path, model="deeplob")
        out_dir = tmp_path / "output"
        result = train_model(cfg, train_path, val_path, out_dir)

        assert "best_metrics" in result
        assert (out_dir / "best_model.pt").exists()

    def test_train_linear_2_epochs(self, tmp_path: Path) -> None:
        train_path, val_path = _create_synthetic_parquet(tmp_path)
        cfg = _make_test_cfg(tmp_path, model="linear")
        out_dir = tmp_path / "output"
        result = train_model(cfg, train_path, val_path, out_dir)

        assert "best_metrics" in result
        assert (out_dir / "best_model.pt").exists()

    def test_vpin_loss_only_for_tlob(self, tmp_path: Path) -> None:
        train_path, val_path = _create_synthetic_parquet(tmp_path)

        # TLOB with VPIN head → should have vpin metrics
        cfg_tlob = _make_test_cfg(tmp_path, model="dual_attention")
        result_tlob = train_model(cfg_tlob, train_path, val_path, tmp_path / "out_tlob")
        assert any(k.startswith("vpin_") for k in result_tlob["best_metrics"])

        # DeepLOB → no vpin metrics
        cfg_dl = _make_test_cfg(tmp_path, model="deeplob")
        result_dl = train_model(cfg_dl, train_path, val_path, tmp_path / "out_dl")
        assert not any(k.startswith("vpin_") for k in result_dl["best_metrics"])

    def test_early_stopping(self, tmp_path: Path) -> None:
        train_path, val_path = _create_synthetic_parquet(tmp_path)
        cfg = _make_test_cfg(tmp_path, model="linear")
        # Set patience=1 and many epochs — should stop early
        cfg.training.early_stopping_patience = 1
        cfg.training.epochs = 50
        out_dir = tmp_path / "output"
        result = train_model(cfg, train_path, val_path, out_dir)

        # Should not have trained all 50 epochs
        assert result["best_epoch"] < 49

    def test_checkpoint_saved(self, tmp_path: Path) -> None:
        train_path, val_path = _create_synthetic_parquet(tmp_path)
        cfg = _make_test_cfg(tmp_path, model="linear")
        out_dir = tmp_path / "output"
        train_model(cfg, train_path, val_path, out_dir)

        model_path = out_dir / "best_model.pt"
        assert model_path.exists()
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        assert isinstance(state_dict, dict)


# ---------------------------------------------------------------------------
# LOBDataset VPIN extension
# ---------------------------------------------------------------------------
class TestLOBDatasetVpin:
    def test_dataset_without_vpin(self, tmp_path: Path) -> None:
        train_path, _ = _create_synthetic_parquet(tmp_path)
        ds = LOBDataset(
            train_path,
            sequence_length=20,
            horizons=[10, 20, 50, 100],
            feature_cols=BOOK_COLS,
        )
        sample = ds[0]
        assert len(sample) == 2
        features, labels = sample
        assert features.shape == (20, 40)
        assert labels.shape == (4,)

    def test_dataset_with_vpin(self, tmp_path: Path) -> None:
        train_path, _ = _create_synthetic_parquet(tmp_path)
        ds = LOBDataset(
            train_path,
            sequence_length=20,
            horizons=[10, 20, 50, 100],
            feature_cols=BOOK_COLS,
            vpin_col="vpin_50",
        )
        sample = ds[0]
        assert len(sample) == 3
        features, labels, vpin_target = sample
        assert features.shape == (20, 40)
        assert labels.shape == (4,)
        assert vpin_target.shape == ()
        assert 0.0 <= vpin_target.item() <= 1.0
