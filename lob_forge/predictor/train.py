"""Convenience entry points for predictor training and model comparison."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from lob_forge.predictor.trainer import train_model

log = logging.getLogger(__name__)

_MODEL_TYPES = ("dual_attention", "deeplob", "linear")


def train_predictor(cfg: DictConfig) -> dict:
    """Resolve data paths from config and invoke the training loop.

    A convenience wrapper that reads ``cfg.data.data_dir`` and calls
    :func:`~lob_forge.predictor.trainer.train_model` with the resolved
    train/val Parquet paths.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.

    Returns
    -------
    dict
        Training results from :func:`train_model`.
    """
    data_dir = Path(OmegaConf.select(cfg, "data.data_dir", default="data"))
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    output_dir = Path(OmegaConf.select(cfg, "project.output_dir", default="outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    return train_model(cfg, train_path, val_path, output_dir)


def compare_models(
    cfg: DictConfig,
    train_path: str | Path,
    val_path: str | Path,
    output_dir: str | Path,
) -> dict[str, dict]:
    """Train all 3 model types sequentially and compare metrics.

    Trains ``dual_attention``, ``deeplob``, and ``linear`` models in
    sequence, each with its own output sub-directory. Logs a comparison
    table to wandb when enabled.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config. ``predictor.model`` is overridden per iteration.
    train_path : str | Path
        Path to training Parquet file.
    val_path : str | Path
        Path to validation Parquet file.
    output_dir : str | Path
        Root output directory; each model gets a sub-directory.

    Returns
    -------
    dict[str, dict]
        Mapping of model_name to its training results dict.
    """
    output_dir = Path(output_dir)
    results: dict[str, dict] = {}

    for model_name in _MODEL_TYPES:
        log.info("=== Training model: %s ===", model_name)
        model_cfg = copy.deepcopy(cfg)
        OmegaConf.update(model_cfg, "predictor.model", model_name)

        model_output = output_dir / model_name
        model_output.mkdir(parents=True, exist_ok=True)

        model_results = train_model(model_cfg, train_path, val_path, model_output)
        results[model_name] = model_results

    # Log comparison to wandb
    wandb_enabled = OmegaConf.select(cfg, "wandb.enabled", default=False)
    if wandb_enabled:
        try:
            import wandb

            comparison: dict[str, float] = {}
            for name, res in results.items():
                comparison[f"compare/{name}_val_loss"] = res.get(
                    "best_val_loss", float("inf")
                )
                metrics = res.get("best_metrics", {})
                comparison[f"compare/{name}_f1_macro"] = metrics.get(
                    "f1_macro_mean", 0.0
                )
            wandb.log(comparison)
            log.info("Logged model comparison to wandb: %s", comparison)
        except ImportError:
            log.warning("wandb not installed; skipping comparison logging")

    return results
