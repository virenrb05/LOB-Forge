"""Unit tests for classification and VPIN evaluation metrics.

Covers perfect predictions, all-wrong, single/multi horizon, class imbalance,
output key validation, known hand-computed values, and VPIN regression metrics.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from lob_forge.predictor.metrics import (
    compute_classification_metrics,
    compute_vpin_metrics,
)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
class TestClassificationMetrics:
    def test_perfect_predictions(self) -> None:
        """All correct predictions → F1=1.0 for every class and horizon."""
        n = 100
        n_horizons = 4
        y_true = (
            np.random.RandomState(0).randint(0, 3, (n, n_horizons)).astype(np.int64)
        )
        y_pred = y_true.copy()
        metrics = compute_classification_metrics(y_true, y_pred, n_horizons)

        for h in range(n_horizons):
            assert metrics[f"horizon_{h}_f1_weighted"] == pytest.approx(1.0)
            assert metrics[f"horizon_{h}_f1_macro"] == pytest.approx(1.0)
            for c in range(3):
                assert metrics[f"horizon_{h}_f1_class_{c}"] == pytest.approx(1.0)
        assert metrics["f1_weighted_mean"] == pytest.approx(1.0)
        assert metrics["f1_macro_mean"] == pytest.approx(1.0)

    def test_all_wrong(self) -> None:
        """All wrong predictions → low F1 values."""
        n = 100
        n_horizons = 4
        rng = np.random.RandomState(1)
        y_true = rng.randint(0, 3, (n, n_horizons)).astype(np.int64)
        # Shift predictions so they never match
        y_pred = ((y_true + 1) % 3).astype(np.int64)
        metrics = compute_classification_metrics(y_true, y_pred, n_horizons)

        for h in range(n_horizons):
            assert metrics[f"horizon_{h}_f1_weighted"] < 0.5

    def test_single_horizon(self) -> None:
        """Works correctly with n_horizons=1."""
        n = 50
        y_true = np.random.RandomState(2).randint(0, 3, (n, 1)).astype(np.int64)
        y_pred = y_true.copy()
        metrics = compute_classification_metrics(y_true, y_pred, n_horizons=1)
        assert metrics["horizon_0_f1_weighted"] == pytest.approx(1.0)
        assert "horizon_1_f1_weighted" not in metrics

    def test_multi_horizon(self) -> None:
        """Returns per-horizon keys for n_horizons=4."""
        n = 50
        n_horizons = 4
        y_true = (
            np.random.RandomState(3).randint(0, 3, (n, n_horizons)).astype(np.int64)
        )
        y_pred = y_true.copy()
        metrics = compute_classification_metrics(y_true, y_pred, n_horizons)

        for h in range(n_horizons):
            assert f"horizon_{h}_f1_weighted" in metrics
            assert f"horizon_{h}_f1_macro" in metrics
            assert f"horizon_{h}_precision_macro" in metrics
            assert f"horizon_{h}_recall_macro" in metrics
            for c in range(3):
                assert f"horizon_{h}_f1_class_{c}" in metrics

    def test_class_imbalance(self) -> None:
        """Majority-class predictions → weighted F1 != macro F1."""
        n = 200
        # Imbalanced ground truth: 80% class 0
        y_true = np.zeros((n, 1), dtype=np.int64)
        y_true[:40] = 1
        # Predict all 0 (majority)
        y_pred = np.zeros((n, 1), dtype=np.int64)

        metrics = compute_classification_metrics(y_true, y_pred, n_horizons=1)
        # Weighted should be higher than macro because majority class is correct
        assert metrics["horizon_0_f1_weighted"] != pytest.approx(
            metrics["horizon_0_f1_macro"], abs=1e-3
        )

    def test_output_keys(self) -> None:
        """Verify all expected keys are present in returned dict."""
        n_horizons = 2
        y_true = np.ones((10, n_horizons), dtype=np.int64)
        y_pred = np.ones((10, n_horizons), dtype=np.int64)
        metrics = compute_classification_metrics(y_true, y_pred, n_horizons)

        expected_keys = {"f1_weighted_mean", "f1_macro_mean"}
        for h in range(n_horizons):
            expected_keys |= {
                f"horizon_{h}_f1_weighted",
                f"horizon_{h}_f1_macro",
                f"horizon_{h}_precision_macro",
                f"horizon_{h}_recall_macro",
            }
            for c in range(3):
                expected_keys.add(f"horizon_{h}_f1_class_{c}")

        assert expected_keys == set(metrics.keys())

    def test_known_values(self) -> None:
        """Hand-computed example matches sklearn directly."""
        # y_true = [0,0,1,1,2,2], y_pred = [0,1,1,1,2,0]
        y_true = np.array([[0, 0, 1, 1, 2, 2]], dtype=np.int64).T.reshape(6, 1)
        y_pred = np.array([[0, 1, 1, 1, 2, 0]], dtype=np.int64).T.reshape(6, 1)

        metrics = compute_classification_metrics(y_true, y_pred, n_horizons=1)

        # Verify against sklearn directly
        yt = y_true[:, 0]
        yp = y_pred[:, 0]
        expected_f1_w = f1_score(yt, yp, labels=[0, 1, 2], average="weighted")
        expected_f1_m = f1_score(yt, yp, labels=[0, 1, 2], average="macro")
        per_class = f1_score(yt, yp, labels=[0, 1, 2], average=None)

        assert metrics["horizon_0_f1_weighted"] == pytest.approx(expected_f1_w)
        assert metrics["horizon_0_f1_macro"] == pytest.approx(expected_f1_m)
        for c in range(3):
            assert metrics[f"horizon_0_f1_class_{c}"] == pytest.approx(per_class[c])


# ---------------------------------------------------------------------------
# VPIN metrics
# ---------------------------------------------------------------------------
class TestVpinMetrics:
    def test_perfect_predictions(self) -> None:
        """y_true == y_pred → MSE=0, MAE=0, corr=1.0."""
        y = np.random.RandomState(10).rand(100).astype(np.float32)
        metrics = compute_vpin_metrics(y, y)
        assert metrics["vpin_mse"] == pytest.approx(0.0, abs=1e-7)
        assert metrics["vpin_mae"] == pytest.approx(0.0, abs=1e-7)
        assert metrics["vpin_corr"] == pytest.approx(1.0, abs=1e-5)

    def test_known_mse(self) -> None:
        """Hand-computed MSE for simple arrays."""
        y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_pred = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        # MSE = mean((0.5)^2, (0.5)^2, (0.5)^2) = 0.25
        metrics = compute_vpin_metrics(y_true, y_pred)
        assert metrics["vpin_mse"] == pytest.approx(0.25, abs=1e-6)
        assert metrics["vpin_mae"] == pytest.approx(0.5, abs=1e-6)

    def test_correlation(self) -> None:
        """Linearly related arrays → correlation ≈ 1.0."""
        y_true = np.arange(100, dtype=np.float32)
        y_pred = 2.0 * y_true + 5.0
        metrics = compute_vpin_metrics(y_true, y_pred)
        assert metrics["vpin_corr"] == pytest.approx(1.0, abs=1e-5)

    def test_output_keys(self) -> None:
        """Verify vpin_mse, vpin_mae, vpin_corr keys."""
        y = np.ones(10, dtype=np.float32)
        metrics = compute_vpin_metrics(y, y + 0.1)
        assert set(metrics.keys()) == {"vpin_mse", "vpin_mae", "vpin_corr"}
