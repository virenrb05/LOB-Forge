"""Generator training loop with EMA, wandb, and checkpointing."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from lob_forge.data.dataset import LOBSequenceDataset
from lob_forge.data.schema import (
    ASK_PRICE_COLS,
    ASK_SIZE_COLS,
    BID_PRICE_COLS,
    BID_SIZE_COLS,
)
from lob_forge.generator.ema import ExponentialMovingAverage
from lob_forge.generator.model import DiffusionModel

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


def _resolve_device(cfg: DictConfig) -> torch.device:
    """Resolve training device from config (MPS > CUDA > CPU)."""
    device_str = _get(cfg, "project.device", "auto")
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def train_generator(cfg: DictConfig) -> Path:
    """Train a diffusion generator model.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config with ``generator.*`` keys.

    Returns
    -------
    Path
        Path to the final checkpoint file.
    """
    g = cfg.generator

    # --- Device & seed ---
    device = _resolve_device(cfg)
    seed = _get(cfg, "project.seed", 42)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    log.info("Device: %s, Seed: %d", device, seed)

    # --- Model ---
    channel_mults = tuple(_get(g, "channel_mults", [1, 2, 4, 4]))
    attention_levels = tuple(_get(g, "attention_levels", [2, 3]))

    model = DiffusionModel(
        in_channels=_get(g, "in_channels", 40),
        d_model=_get(g, "d_model", 128),
        channel_mults=channel_mults,
        n_res_blocks=_get(g, "n_res_blocks", 2),
        num_timesteps=_get(g, "noise_steps", 1000),
        ddim_steps=_get(g, "ddim_steps", 50),
        n_regimes=_get(g, "n_regimes", 3),
        dropout=_get(g, "dropout", 0.1),
        attention_levels=attention_levels,
        n_heads=_get(g, "n_heads", 4),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info("DiffusionModel: %d params", n_params)

    # --- EMA ---
    ema = ExponentialMovingAverage(model, decay=_get(g, "ema_decay", 0.9999))

    # --- Dataset ---
    data_dir = Path(_get(cfg, "data.data_dir", "data"))
    parquet_path = data_dir / "preprocessed.parquet"
    seq_len = _get(cfg, "data.sequence_length", 100)

    dataset = LOBSequenceDataset(
        parquet_path=parquet_path,
        sequence_length=seq_len,
        feature_cols=BOOK_FEATURE_COLS,
    )

    batch_size = _get(g, "training.batch_size", 64)
    pin_memory = device.type not in ("mps", "cpu")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=True,
    )

    # --- Optimizer ---
    lr = _get(g, "optimizer.lr", 1e-4)
    weight_decay = _get(g, "optimizer.weight_decay", 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Training config ---
    epochs = _get(g, "training.epochs", 100)
    save_every = _get(g, "training.save_every", 10)
    sample_every = _get(g, "training.sample_every", 20)

    # --- Output dir ---
    output_dir = Path(_get(cfg, "project.output_dir", "outputs")) / "generator"
    output_dir.mkdir(parents=True, exist_ok=True)

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
    log.info(
        "Starting generator training: %d epochs, %d batches/epoch",
        epochs,
        len(loader),
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for sequence, regime in loader:
            sequence = sequence.to(device)
            regime = regime.to(device)

            loss = model.training_loss(sequence, regime)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        log.info("Epoch %d/%d — loss=%.6f", epoch + 1, epochs, avg_loss)

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": avg_loss})

        # --- Checkpoint ---
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                ckpt_path,
            )
            log.info("Saved checkpoint: %s", ckpt_path)

        # --- Sample ---
        if (epoch + 1) % sample_every == 0:
            model.eval()
            ema.apply_shadow(model)
            with torch.no_grad():
                sample_regime = torch.zeros(4, dtype=torch.long, device=device)
                sample = model.generate(
                    n_samples=4,
                    seq_len=min(seq_len, 32),
                    regime=sample_regime,
                    method="ddim",
                    ddim_steps=10,
                )
                log.info("Sample shape: %s", sample.shape)
                if wandb_run is not None:
                    wandb_run.log({"sample_shape": list(sample.shape)})
            ema.restore(model)

    # --- Cleanup ---
    if wandb_run is not None:
        import wandb

        wandb.finish()

    final_ckpt = output_dir / f"checkpoint_epoch{epochs:04d}.pt"
    log.info("Training complete. Final checkpoint: %s", final_ckpt)
    return final_ckpt
