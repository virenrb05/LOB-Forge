"""Hydra entry point for LOB-Forge training pipeline."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from lob_forge.predictor.trainer import train_model

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the LOB-Forge training pipeline.

    Dispatches to the predictor or generator trainer based on the ``trainer``
    config key.  Pass ``trainer=generator`` on the CLI to route to the
    diffusion model training loop; the default (or ``trainer=predictor``)
    routes to the predictor training loop.

    Invoked via ``python -m lob_forge.train`` with optional config overrides.
    """
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # Determine which trainer to dispatch to.
    # ``trainer`` is NOT defined in config.yaml — it is passed exclusively as a
    # CLI override (e.g. trainer=generator), so use OmegaConf.select with a
    # default to avoid KeyError.
    trainer_key = OmegaConf.select(cfg, "trainer", default="predictor")
    log.info("Dispatcher: trainer=%s", trainer_key)

    if trainer_key == "generator":
        from lob_forge.generator.train import train_generator  # noqa: PLC0415

        log.info("Routing to generator trainer (train_generator)")
        train_generator(cfg)
        return

    # --- Predictor path (default) ---
    # Resolve output directory (Hydra changes cwd to its output dir)
    output_dir = Path(os.getcwd())
    log.info("Output directory: %s", output_dir)

    # Resolve data paths
    data_dir = OmegaConf.select(cfg, "data.data_dir", default="data")
    train_path = Path(data_dir) / "train.parquet"
    val_path = Path(data_dir) / "val.parquet"

    if not train_path.exists():
        log.error("Training data not found: %s", train_path)
        print(f"ERROR: Training data not found at {train_path}")
        print("Run data preprocessing first to generate train/val splits.")
        sys.exit(1)

    if not val_path.exists():
        log.error("Validation data not found: %s", val_path)
        print(f"ERROR: Validation data not found at {val_path}")
        print("Run data preprocessing first to generate train/val splits.")
        sys.exit(1)

    log.info("Train data: %s", train_path)
    log.info("Val data: %s", val_path)

    results = train_model(cfg, train_path, val_path, output_dir)

    # Print summary
    print("\n=== Training Complete ===")
    print(f"Best epoch: {results['best_epoch'] + 1}")
    print(f"Best val loss: {results['best_val_loss']:.4f}")
    print(f"Model saved: {results['model_path']}")
    if results.get("best_metrics"):
        for key, val in sorted(results["best_metrics"].items()):
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    main()
