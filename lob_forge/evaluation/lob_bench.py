"""LOB-Bench quantitative evaluation metrics.

Implements Wasserstein distances, MLP discriminator scores, and conditional
statistics comparison for evaluating synthetic LOB data quality.
Satisfies GEN-08.

Reference: LOB-Bench (ICML 2025).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance

__all__ = [
    "compute_wasserstein_metrics",
    "compute_conditional_stats",
    "train_discriminator",
    "run_lob_bench",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPS = 1e-12


def _extract_spread(data: NDArray[np.floating]) -> NDArray[np.floating]:
    """Spread = ask_1 (col 0) - bid_1 (col 2) assuming standard 40-col LOB layout.

    Standard layout: [ask_price_1, ask_size_1, bid_price_1, bid_size_1, ...].
    """
    return data[:, 0] - data[:, 2]


def _extract_mid(data: NDArray[np.floating]) -> NDArray[np.floating]:
    """Mid-price = (ask_1 + bid_1) / 2."""
    return (data[:, 0] + data[:, 2]) / 2.0


def _extract_log_returns(data: NDArray[np.floating]) -> NDArray[np.floating]:
    """Log returns of mid-price series."""
    mid = _extract_mid(data)
    mid = np.maximum(mid, _EPS)  # guard against log(0)
    return np.diff(np.log(mid))


# ---------------------------------------------------------------------------
# 1. Wasserstein Metrics
# ---------------------------------------------------------------------------


def compute_wasserstein_metrics(
    real: NDArray[np.floating],
    synthetic: NDArray[np.floating],
) -> dict[str, Any]:
    """Compute 1-D Wasserstein distances between real and synthetic marginals.

    Parameters
    ----------
    real : ndarray of shape ``(N, C)``
        Real LOB data (C features, typically 40 columns).
    synthetic : ndarray of shape ``(M, C)``
        Synthetic LOB data with the same number of columns.

    Returns
    -------
    dict[str, Any]
        ``wd_mean``, ``wd_max``, ``wd_per_feature`` (list), and derived
        quantities ``wd_spread``, ``wd_mid``, ``wd_returns``.
    """
    n_features = real.shape[1]
    per_feature: list[float] = []
    for c in range(n_features):
        per_feature.append(float(wasserstein_distance(real[:, c], synthetic[:, c])))

    # Derived quantities
    wd_spread = float(
        wasserstein_distance(_extract_spread(real), _extract_spread(synthetic))
    )
    wd_mid = float(wasserstein_distance(_extract_mid(real), _extract_mid(synthetic)))

    real_ret = _extract_log_returns(real)
    synth_ret = _extract_log_returns(synthetic)
    wd_returns = float(wasserstein_distance(real_ret, synth_ret))

    return {
        "wd_mean": float(np.mean(per_feature)),
        "wd_max": float(np.max(per_feature)),
        "wd_per_feature": per_feature,
        "wd_spread": wd_spread,
        "wd_mid": wd_mid,
        "wd_returns": wd_returns,
    }


# ---------------------------------------------------------------------------
# 2. Conditional Statistics
# ---------------------------------------------------------------------------


def compute_conditional_stats(
    real: NDArray[np.floating],
    synthetic: NDArray[np.floating],
    real_regimes: NDArray[np.integer],
    synthetic_regimes: NDArray[np.integer],
) -> dict[str, Any]:
    """Compare distributional properties across volatility regimes.

    Parameters
    ----------
    real, synthetic : ndarray of shape ``(N, C)``
        LOB snapshots.
    real_regimes, synthetic_regimes : ndarray of shape ``(N,)``
        Regime labels (0=low-vol, 1=normal, 2=high-vol).

    Returns
    -------
    dict[str, Any]
        Per-regime relative errors and ``mean_relative_error`` across all
        regime-stat pairs.
    """
    results: dict[str, Any] = {}
    all_errors: list[float] = []

    for regime in (0, 1, 2):
        r_mask = real_regimes == regime
        s_mask = synthetic_regimes == regime

        # Skip regime if either set is empty
        if r_mask.sum() < 2 or s_mask.sum() < 2:
            results[str(regime)] = {"skipped": True}
            continue

        r_data = real[r_mask]
        s_data = synthetic[s_mask]

        r_spread = _extract_spread(r_data)
        s_spread = _extract_spread(s_data)

        # Total book depth: sum of all size columns (odd indices in 40-col layout)
        r_depth = r_data[:, 1::2].sum(axis=1)
        s_depth = s_data[:, 1::2].sum(axis=1)

        r_ret = _extract_log_returns(r_data)
        s_ret = _extract_log_returns(s_data)

        def _rel_err(a: float, b: float) -> float:
            return abs(a - b) / (abs(a) + _EPS)

        regime_stats: dict[str, float] = {}
        pairs = [
            ("spread_mean_err", float(np.mean(r_spread)), float(np.mean(s_spread))),
            ("spread_std_err", float(np.std(r_spread)), float(np.std(s_spread))),
            ("depth_mean_err", float(np.mean(r_depth)), float(np.mean(s_depth))),
            ("depth_std_err", float(np.std(r_depth)), float(np.std(s_depth))),
            ("return_mean_err", float(np.mean(r_ret)), float(np.mean(s_ret))),
            ("return_std_err", float(np.std(r_ret)), float(np.std(s_ret))),
        ]

        for name, rv, sv in pairs:
            err = _rel_err(rv, sv)
            regime_stats[name] = err
            all_errors.append(err)

        results[str(regime)] = regime_stats

    results["mean_relative_error"] = float(np.mean(all_errors)) if all_errors else 0.0
    return results


# ---------------------------------------------------------------------------
# 3. MLP Discriminator
# ---------------------------------------------------------------------------


def train_discriminator(
    real: NDArray[np.floating],
    synthetic: NDArray[np.floating],
    hidden_dim: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict[str, float]:
    """Train an MLP to distinguish real from synthetic data.

    A score near 0.5 accuracy indicates the generator produces realistic data.

    Parameters
    ----------
    real, synthetic : ndarray of shape ``(N, C)`` or ``(N, T, C)``
        Real and synthetic LOB data.  3-D inputs are flattened to 2-D.
    hidden_dim : int
        Hidden layer width.
    epochs : int
        Training epochs.
    lr : float
        Adam learning rate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, float]
        ``accuracy`` (test), ``auc`` (ROC AUC on test), ``train_loss``.
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    # Flatten 3-D inputs
    if real.ndim == 3:
        real = real.reshape(real.shape[0], -1)
    if synthetic.ndim == 3:
        synthetic = synthetic.reshape(synthetic.shape[0], -1)

    # Determinism
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    input_dim = real.shape[1]

    # Labels: real=1, synthetic=0
    X = np.concatenate([real, synthetic], axis=0).astype(np.float32)
    y = np.concatenate(
        [
            np.ones(len(real), dtype=np.float32),
            np.zeros(len(synthetic), dtype=np.float32),
        ]
    )

    # 80/20 deterministic split
    n = len(X)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = torch.tensor(X[train_idx]), torch.tensor(y[train_idx]).unsqueeze(
        1
    )
    X_test, y_test = torch.tensor(X[test_idx]), torch.tensor(y[test_idx]).unsqueeze(1)

    # MLP
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    model.train()
    train_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        train_loss = float(loss.item())

    # Evaluate
    model.eval()
    with torch.no_grad():
        probs = model(X_test).numpy()

    preds = (probs >= 0.5).astype(np.float32)
    y_test_np = y_test.numpy()
    accuracy = float(np.mean(preds == y_test_np))

    # AUC — handle degenerate case
    unique_labels = np.unique(y_test_np)
    auc = 0.5 if len(unique_labels) < 2 else float(roc_auc_score(y_test_np, probs))

    return {"accuracy": accuracy, "auc": auc, "train_loss": train_loss}


# ---------------------------------------------------------------------------
# 4. Orchestrator
# ---------------------------------------------------------------------------


def run_lob_bench(
    real: NDArray[np.floating],
    synthetic: NDArray[np.floating],
    real_regimes: NDArray[np.integer] | None = None,
    synthetic_regimes: NDArray[np.integer] | None = None,
) -> dict[str, Any]:
    """Run the full LOB-Bench evaluation suite.

    Parameters
    ----------
    real, synthetic : ndarray
        LOB data arrays (2-D or 3-D).
    real_regimes, synthetic_regimes : ndarray or None
        Optional regime labels for conditional statistics.

    Returns
    -------
    dict[str, Any]
        Namespaced metrics: ``wasserstein/*``, ``discriminator/*``,
        and optionally ``conditional/*``.
    """
    # Flatten 3-D for Wasserstein / conditional (keep original for discriminator)
    real_2d = real.reshape(real.shape[0], -1) if real.ndim == 3 else real
    synth_2d = (
        synthetic.reshape(synthetic.shape[0], -1) if synthetic.ndim == 3 else synthetic
    )

    results: dict[str, Any] = {}

    # Wasserstein
    ws = compute_wasserstein_metrics(real_2d, synth_2d)
    for k, v in ws.items():
        results[f"wasserstein/{k}"] = v

    # Discriminator
    disc = train_discriminator(real, synthetic)
    for k, v in disc.items():
        results[f"discriminator/{k}"] = v

    # Conditional stats (optional)
    if real_regimes is not None and synthetic_regimes is not None:
        cs = compute_conditional_stats(
            real_2d, synth_2d, real_regimes, synthetic_regimes
        )
        for k, v in cs.items():
            results[f"conditional/{k}"] = v

    logger.info(
        "LOB-Bench summary: WD_mean=%.4f, disc_acc=%.3f, disc_auc=%.3f",
        results.get("wasserstein/wd_mean", float("nan")),
        results.get("discriminator/accuracy", float("nan")),
        results.get("discriminator/auc", float("nan")),
    )

    return results
