"""Predictor training loop with model factory, wandb, and checkpointing."""

from __future__ import annotations

import logging
from math import ceil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from lob_forge.data.dataset import LOBDataset
from lob_forge.data.schema import (
    ASK_PRICE_COLS,
    ASK_SIZE_COLS,
    BID_PRICE_COLS,
    BID_SIZE_COLS,
)
from lob_forge.predictor.deeplob import DeepLOB
from lob_forge.predictor.linear_baseline import LinearBaseline
from lob_forge.predictor.losses import FocalLoss
from lob_forge.predictor.metrics import (
    compute_classification_metrics,
    compute_vpin_metrics,
)
from lob_forge.predictor.model import DualAttentionTransformer

log = logging.getLogger(__name__)

# 40 book columns in correct order for model input
BOOK_FEATURE_COLS: list[str] = (
    BID_PRICE_COLS + BID_SIZE_COLS + ASK_PRICE_COLS + ASK_SIZE_COLS
)


def _get(cfg: DictConfig, key: str, default=None):
    """Retrieve nested config key supporting both dict and OmegaConf access."""
    keys = key.split(".")
    obj = cfg
    for k in keys:
        try:
            obj = obj[k]
        except (KeyError, TypeError, AttributeError):
            try:
                obj = getattr(obj, k)
            except AttributeError:
                return default
    return obj


def resolve_device(cfg: DictConfig) -> torch.device:
    """Resolve training device from config."""
    device_str = _get(cfg, "project.device", "auto")
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def build_model(cfg: DictConfig) -> nn.Module:
    """Create a predictor model from config.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with ``predictor.*`` keys.

    Returns
    -------
    nn.Module
        One of DualAttentionTransformer, DeepLOB, or LinearBaseline.
    """
    p = cfg.predictor
    model_name = p.model

    n_levels = p.n_levels
    features_per_level = p.features_per_level
    n_classes = p.n_classes
    n_horizons = p.n_horizons

    if model_name == "dual_attention":
        return DualAttentionTransformer(
            n_levels=n_levels,
            features_per_level=features_per_level,
            d_model=p.d_model,
            n_heads=p.n_heads,
            n_spatial_layers=p.n_spatial_layers,
            n_temporal_layers=p.n_temporal_layers,
            feedforward_dim=p.feedforward_dim,
            dropout=p.dropout,
            n_classes=n_classes,
            n_horizons=n_horizons,
            max_seq_len=p.max_seq_len,
            vpin_head=p.vpin_head,
        )
    elif model_name == "deeplob":
        return DeepLOB(
            n_levels=n_levels,
            features_per_level=features_per_level,
            n_classes=n_classes,
            n_horizons=n_horizons,
            lstm_hidden=_get(p, "lstm_hidden", 64),
        )
    elif model_name == "linear":
        return LinearBaseline(
            n_levels=n_levels,
            features_per_level=features_per_level,
            n_classes=n_classes,
            n_horizons=n_horizons,
        )
    else:
        raise ValueError(f"Unknown model: {model_name!r}")


def _compute_class_weights(
    dataset: LOBDataset, n_classes: int, device: torch.device
) -> torch.Tensor:
    """Compute inverse-frequency class weights from training labels."""
    counts = np.zeros(n_classes, dtype=np.float64)
    labels = dataset.labels  # (N, n_horizons)
    for c in range(n_classes):
        counts[c] = np.nansum(labels == c)
    # Inverse frequency, normalized
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _extract_logits(output: dict | torch.Tensor) -> torch.Tensor:
    """Extract logits from model output (dict for TLOB, tensor for baselines)."""
    if isinstance(output, dict):
        return output["logits"]
    return output


def train_model(
    cfg: DictConfig,
    train_path: str | Path,
    val_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Train a predictor model.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.
    train_path : str | Path
        Path to training Parquet file.
    val_path : str | Path
        Path to validation Parquet file.
    output_dir : str | Path
        Directory for checkpoints and artifacts.

    Returns
    -------
    dict
        Training results with best_metrics, best_epoch, model_path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup ---
    device = resolve_device(cfg)
    seed = _get(cfg, "project.seed", 42)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    log.info("Device: %s, Seed: %d", device, seed)

    model = build_model(cfg).to(device)
    log.info(
        "Model: %s (%d params)",
        cfg.predictor.model,
        sum(p.numel() for p in model.parameters()),
    )

    # --- Datasets ---
    horizons = list(_get(cfg, "data.horizons", [10, 20, 50, 100]))
    seq_len = _get(cfg, "data.sequence_length", 100)
    model_wants_vpin = (
        isinstance(model, DualAttentionTransformer) and model.has_vpin_head
    )
    vpin_col = "vpin" if model_wants_vpin else None

    train_ds = LOBDataset(
        train_path,
        sequence_length=seq_len,
        horizons=horizons,
        feature_cols=BOOK_FEATURE_COLS,
        vpin_col=vpin_col,
    )
    val_ds = LOBDataset(
        val_path,
        sequence_length=seq_len,
        horizons=horizons,
        feature_cols=BOOK_FEATURE_COLS,
        vpin_col=vpin_col,
    )
    # has_vpin is True only when both model has the head AND the column exists in data
    has_vpin = model_wants_vpin and train_ds.vpin is not None

    batch_size = _get(cfg, "training.batch_size", 256)
    num_workers = _get(cfg, "training.num_workers", 0)
    pin_memory = device.type not in ("mps", "cpu")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # --- Loss ---
    n_classes = _get(cfg, "predictor.n_classes", 3)
    class_weights = _compute_class_weights(train_ds, n_classes, device)
    focal_gamma = _get(cfg, "predictor.focal_loss_gamma", 2.0)
    criterion = FocalLoss(gamma=focal_gamma, class_weights=class_weights).to(device)
    vpin_criterion = nn.MSELoss() if has_vpin else None
    vpin_loss_weight = _get(cfg, "predictor.vpin_loss_weight", 0.1)

    # --- Optimizer & Scheduler ---
    lr = _get(cfg, "predictor.optimizer.lr", 1e-4)
    wd = _get(cfg, "predictor.optimizer.weight_decay", 1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    epochs = _get(cfg, "training.epochs", 50)
    grad_accum = _get(cfg, "training.gradient_accumulation", 2)
    steps_per_epoch = ceil(len(train_ds) / batch_size / grad_accum)

    max_lr = _get(cfg, "predictor.scheduler.max_lr", 1e-3)
    pct_start = _get(cfg, "predictor.scheduler.pct_start", 0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        pct_start=pct_start,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )

    # --- wandb ---
    wandb_enabled = _get(cfg, "wandb.enabled", False)
    wandb_run = None
    if wandb_enabled:
        import wandb

        wandb_run = wandb.init(
            project=_get(cfg, "wandb.project", "lob-forge"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # --- Training loop ---
    patience = _get(cfg, "training.early_stopping_patience", 10)
    best_val_loss = float("inf")
    best_epoch = -1
    best_metrics: dict = {}
    patience_counter = 0
    n_horizons = _get(cfg, "predictor.n_horizons", 4)

    for epoch in range(epochs):
        # --- Train epoch ---
        model.train()
        train_loss_accum = 0.0
        train_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            if has_vpin:
                features, labels, vpin_target = batch
                vpin_target = vpin_target.to(device)
            else:
                features, labels = batch
                vpin_target = None

            features = features.to(device)
            labels = labels.to(device)

            output = model(features)
            logits = _extract_logits(output)
            loss = criterion(logits, labels)

            # VPIN regression loss
            if (
                has_vpin
                and vpin_target is not None
                and isinstance(output, dict)
                and "vpin" in output
            ):
                vpin_pred = output["vpin"].squeeze(-1)
                loss = loss + vpin_loss_weight * vpin_criterion(vpin_pred, vpin_target)

            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss_accum += loss.item() * grad_accum
            train_steps += 1

        avg_train_loss = train_loss_accum / max(train_steps, 1)

        # --- Val epoch ---
        model.eval()
        val_loss_accum = 0.0
        val_steps = 0
        all_preds = []
        all_labels = []
        all_vpin_true = []
        all_vpin_pred = []

        with torch.no_grad():
            for batch in val_loader:
                if has_vpin:
                    features, labels, vpin_target = batch
                    vpin_target = vpin_target.to(device)
                else:
                    features, labels = batch
                    vpin_target = None

                features = features.to(device)
                labels = labels.to(device)

                output = model(features)
                logits = _extract_logits(output)
                loss = criterion(logits, labels)

                if (
                    has_vpin
                    and vpin_target is not None
                    and isinstance(output, dict)
                    and "vpin" in output
                ):
                    vpin_pred = output["vpin"].squeeze(-1)
                    loss = loss + vpin_loss_weight * vpin_criterion(
                        vpin_pred, vpin_target
                    )
                    all_vpin_true.append(vpin_target.cpu().numpy())
                    all_vpin_pred.append(vpin_pred.cpu().numpy())

                val_loss_accum += loss.item()
                val_steps += 1

                preds = logits.argmax(dim=-1).cpu().numpy()  # (B, n_horizons)
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss_accum / max(val_steps, 1)

        # --- Metrics ---
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_labels, axis=0)
        cls_metrics = compute_classification_metrics(
            y_true, y_pred, n_horizons, n_classes
        )

        vpin_metrics: dict[str, float] = {}
        if all_vpin_true:
            vt = np.concatenate(all_vpin_true)
            vp = np.concatenate(all_vpin_pred)
            vpin_metrics = compute_vpin_metrics(vt, vp)

        # --- Logging ---
        log_dict = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **cls_metrics,
            **vpin_metrics,
        }
        log.info(
            "Epoch %d/%d — train_loss=%.4f val_loss=%.4f f1_macro=%.4f",
            epoch + 1,
            epochs,
            avg_train_loss,
            avg_val_loss,
            cls_metrics.get("f1_macro_mean", 0.0),
        )

        if wandb_run is not None:
            wandb_run.log(log_dict)

        # --- Early stopping & checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_metrics = {**cls_metrics, **vpin_metrics}
            patience_counter = 0
            model_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), model_path)
            log.info(
                "Saved best model (epoch %d, val_loss=%.4f)", epoch + 1, avg_val_loss
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(
                    "Early stopping at epoch %d (patience=%d)", epoch + 1, patience
                )
                break

    # --- Cleanup ---
    if wandb_run is not None:
        import wandb

        wandb_run.log({"best_epoch": best_epoch, "best_val_loss": best_val_loss})
        wandb.finish()

    return {
        "best_metrics": best_metrics,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "model_path": str(output_dir / "best_model.pt"),
    }
