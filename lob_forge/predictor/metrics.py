"""Evaluation metrics for LOB predictor models.

Provides per-horizon classification metrics (F1, precision, recall) and
VPIN regression metrics (MSE, MAE, Pearson correlation).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_classification_metrics(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    n_horizons: int,
    n_classes: int = 3,
) -> dict[str, float]:
    """Compute per-horizon classification metrics.

    Parameters
    ----------
    y_true : ndarray
        Ground-truth labels of shape ``(N, n_horizons)`` with dtype int64.
    y_pred : ndarray
        Predicted labels of shape ``(N, n_horizons)`` with dtype int64.
    n_horizons : int
        Number of prediction horizons.
    n_classes : int
        Number of classes (default 3: DOWN, STATIONARY, UP).

    Returns
    -------
    dict[str, float]
        Metric dictionary with keys like ``horizon_{h}_f1_weighted``,
        ``horizon_{h}_f1_class_{c}``, ``f1_weighted_mean``, etc.
    """
    labels = list(range(n_classes))
    metrics: dict[str, float] = {}

    f1_weighted_list: list[float] = []
    f1_macro_list: list[float] = []

    for h in range(n_horizons):
        yt = y_true[:, h]
        yp = y_pred[:, h]

        f1_w = float(
            f1_score(yt, yp, labels=labels, average="weighted", zero_division=0)
        )
        f1_m = float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
        prec_m = float(
            precision_score(yt, yp, labels=labels, average="macro", zero_division=0)
        )
        rec_m = float(
            recall_score(yt, yp, labels=labels, average="macro", zero_division=0)
        )

        metrics[f"horizon_{h}_f1_weighted"] = f1_w
        metrics[f"horizon_{h}_f1_macro"] = f1_m
        metrics[f"horizon_{h}_precision_macro"] = prec_m
        metrics[f"horizon_{h}_recall_macro"] = rec_m

        # Per-class F1
        f1_per_class = f1_score(yt, yp, labels=labels, average=None, zero_division=0)
        for c in range(n_classes):
            metrics[f"horizon_{h}_f1_class_{c}"] = float(f1_per_class[c])

        f1_weighted_list.append(f1_w)
        f1_macro_list.append(f1_m)

    # Mean across horizons
    metrics["f1_weighted_mean"] = float(np.mean(f1_weighted_list))
    metrics["f1_macro_mean"] = float(np.mean(f1_macro_list))

    return metrics


def compute_vpin_metrics(
    y_true: NDArray[np.float32],
    y_pred: NDArray[np.float32],
) -> dict[str, float]:
    """Compute VPIN regression metrics.

    Parameters
    ----------
    y_true : ndarray
        Ground-truth VPIN values of shape ``(N,)`` with dtype float32.
    y_pred : ndarray
        Predicted VPIN values of shape ``(N,)`` with dtype float32.

    Returns
    -------
    dict[str, float]
        ``{"vpin_mse": float, "vpin_mae": float, "vpin_corr": float}``
    """
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Pearson correlation; handle degenerate case (constant predictions)
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])

    return {"vpin_mse": mse, "vpin_mae": mae, "vpin_corr": corr}
