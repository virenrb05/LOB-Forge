"""End-to-end generator validation pipeline.

Loads a trained diffusion model checkpoint, generates synthetic LOB data
conditioned on volatility regimes, and runs all validation suites:
stylized facts, LOB-Bench metrics, and regime conditioning tests.

Usage::

    python -m lob_forge.evaluation.validate_generator \\
        validation.checkpoint_path=outputs/generator/checkpoint_epoch0100.pt \\
        validation.data_path=data/preprocessed.parquet

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from lob_forge.evaluation.lob_bench import run_lob_bench
from lob_forge.evaluation.regime_validation import validate_regime_conditioning
from lob_forge.evaluation.stylized_facts import (
    run_all_stylized_tests,
    summary_figure,
)
from lob_forge.generator.model import DiffusionModel

logger = logging.getLogger(__name__)


def _resolve_device() -> torch.device:
    """Resolve best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model(
    cfg: DictConfig,
    checkpoint_path: str | Path,
    device: torch.device,
) -> DiffusionModel:
    """Instantiate a DiffusionModel and load checkpoint weights.

    Uses EMA weights when available (``ema_state_dict`` key in checkpoint).
    """
    gen_cfg = cfg.get("generator", {})
    model = DiffusionModel(
        in_channels=gen_cfg.get("in_channels", 40),
        d_model=gen_cfg.get("d_model", 128),
        channel_mults=tuple(gen_cfg.get("channel_mults", [1, 2, 4, 4])),
        n_res_blocks=gen_cfg.get("n_res_blocks", 2),
        num_timesteps=gen_cfg.get("noise_steps", 1000),
        ddim_steps=gen_cfg.get("ddim_steps", 50),
        n_regimes=gen_cfg.get("n_regimes", 3),
        dropout=gen_cfg.get("dropout", 0.1),
        attention_levels=tuple(gen_cfg.get("attention_levels", [2, 3])),
        n_heads=gen_cfg.get("n_heads", 4),
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Prefer EMA weights when available
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        logger.info("Loaded EMA weights from checkpoint")
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded model weights from checkpoint")
    else:
        model.load_state_dict(ckpt)
        logger.info("Loaded raw state dict from checkpoint")

    model.to(device)
    model.eval()
    return model


def _load_real_data(
    data_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed Parquet and extract book data, mid-prices, and regimes.

    Returns
    -------
    book : ndarray of shape (N, 40)
        LOB book columns.
    mid : ndarray of shape (N,)
        Mid-price series.
    regimes : ndarray of shape (N,)
        Regime labels (0=low-vol, 1=normal, 2=high-vol) from realized
        volatility quantiles.
    """
    import pandas as pd

    df = pd.read_parquet(data_path)

    # Book columns: first 40 numeric columns (ask_price * 10, ask_size * 10,
    # bid_price * 10, bid_size * 10)
    book_cols = [c for c in df.columns if c.startswith(("ask_", "bid_"))][:40]
    if len(book_cols) < 40:
        raise ValueError(
            f"Expected 40 book columns, found {len(book_cols)} in {data_path}"
        )
    book = df[book_cols].to_numpy(dtype=np.float64)
    mid = (book[:, 0] + book[:, 20]) / 2.0

    # Compute regime labels from realized volatility quantiles
    returns = np.diff(mid) / (mid[:-1] + 1e-12)
    window = min(50, len(returns) // 2)
    if window < 2:
        regimes = np.zeros(len(book), dtype=np.int64)
    else:
        # Pad returns to match book length
        padded_ret = np.concatenate([[0.0], returns])
        rolling_vol = np.array(
            [
                np.std(padded_ret[max(0, i - window) : i + 1])
                for i in range(len(padded_ret))
            ]
        )
        q33, q67 = np.percentile(rolling_vol, [33, 67])
        regimes = np.where(
            rolling_vol < q33, 0, np.where(rolling_vol < q67, 1, 2)
        ).astype(np.int64)

    return book, mid, regimes


def _generate_by_regime(
    model: DiffusionModel,
    n_samples: int,
    seq_len: int,
    ddim_steps: int,
    device: torch.device,
    n_regimes: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic samples for each regime.

    Returns
    -------
    synthetic_book : ndarray of shape (N_total, 40)
        Flattened synthetic book data.
    synthetic_regimes : ndarray of shape (N_total,)
        Regime labels for each row.
    """
    samples_per_regime = max(1, n_samples // n_regimes)
    all_books: list[np.ndarray] = []
    all_regimes: list[np.ndarray] = []

    for regime_id in range(n_regimes):
        regime_tensor = torch.full(
            (samples_per_regime,), regime_id, dtype=torch.long, device=device
        )
        generated = model.generate(
            n_samples=samples_per_regime,
            seq_len=seq_len,
            regime=regime_tensor,
            method="ddim",
            ddim_steps=ddim_steps,
        )
        # generated shape: (B, T, C) -> flatten to (B*T, C)
        gen_np = generated.cpu().numpy()
        flat = gen_np.reshape(-1, gen_np.shape[-1])
        all_books.append(flat)
        all_regimes.append(np.full(flat.shape[0], regime_id, dtype=np.int64))

        logger.info(
            "Generated %d sequences for regime %d (%d rows)",
            samples_per_regime,
            regime_id,
            flat.shape[0],
        )

    return np.concatenate(all_books), np.concatenate(all_regimes)


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def validate_generator(cfg: DictConfig) -> dict[str, Any]:
    """Run the full generator validation pipeline.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config containing ``validation`` and ``generator`` groups.

    Returns
    -------
    dict[str, Any]
        Combined results from all validation suites.
    """
    val_cfg = cfg.validation

    # ---- Seed ----
    seed = val_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Validate required paths ----
    checkpoint_path = val_cfg.get("checkpoint_path")
    data_path = val_cfg.get("data_path")
    if checkpoint_path is None:
        raise ValueError("validation.checkpoint_path is required")
    if data_path is None:
        raise ValueError("validation.data_path is required")

    checkpoint_path = Path(checkpoint_path)
    data_path = Path(data_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # ---- Device ----
    device = _resolve_device()
    logger.info("Using device: %s", device)

    # ---- Load model ----
    model = _load_model(cfg, checkpoint_path, device)
    logger.info("Model loaded from %s", checkpoint_path)

    # ---- Load real data ----
    real_book, _real_mid, real_regimes = _load_real_data(data_path)
    logger.info("Loaded real data: %d rows from %s", len(real_book), data_path)

    # ---- Generate synthetic data ----
    n_samples = val_cfg.get("n_samples", 1000)
    seq_len = val_cfg.get("seq_len", 100)
    ddim_steps = val_cfg.get("ddim_steps", 50)

    synthetic_book, synthetic_regimes = _generate_by_regime(
        model, n_samples, seq_len, ddim_steps, device
    )
    logger.info("Generated %d synthetic rows", len(synthetic_book))

    # ---- Output directory ----
    output_dir = Path(val_cfg.get("output_dir", "outputs/validation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run validation suites ----
    results: dict[str, Any] = {}

    # Stylized facts
    if val_cfg.get("run_stylized_facts", True):
        logger.info("Running stylized fact tests...")
        sf_results = run_all_stylized_tests(real_book, synthetic_book)
        results["stylized_facts"] = sf_results

        n_passed = sum(1 for v in sf_results.values() if v.get("passed", False))
        logger.info("Stylized facts: %d/%d passed", n_passed, len(sf_results))

        # Generate and save summary figure
        fig = summary_figure(sf_results, real_book, synthetic_book)
        fig_path = output_dir / "stylized_facts.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        logger.info("Saved summary figure: %s", fig_path)

        import matplotlib.pyplot as plt

        plt.close(fig)

    # LOB-Bench
    if val_cfg.get("run_lob_bench", True):
        logger.info("Running LOB-Bench metrics...")
        lb_results = run_lob_bench(
            real_book,
            synthetic_book,
            real_regimes=real_regimes,
            synthetic_regimes=synthetic_regimes,
        )
        results["lob_bench"] = lb_results
        logger.info(
            "LOB-Bench: WD_mean=%.4f, disc_acc=%.3f",
            lb_results.get("wasserstein/wd_mean", float("nan")),
            lb_results.get("discriminator/accuracy", float("nan")),
        )

    # Regime validation
    if val_cfg.get("run_regime_validation", True):
        logger.info("Running regime conditioning validation...")
        # Group real and synthetic data by regime
        real_by_regime: dict[int, np.ndarray] = {}
        syn_by_regime: dict[int, np.ndarray] = {}
        for r in (0, 1, 2):
            r_mask = real_regimes == r
            s_mask = synthetic_regimes == r
            if r_mask.sum() > 0:
                real_by_regime[r] = real_book[r_mask]
            if s_mask.sum() > 0:
                syn_by_regime[r] = synthetic_book[s_mask]

        rv_results = validate_regime_conditioning(real_by_regime, syn_by_regime)
        results["regime_validation"] = rv_results
        logger.info(
            "Regime validation: distinct=%s, matched=%s, ordering=%s",
            rv_results.get("regime_distinct"),
            rv_results.get("regime_matched"),
            rv_results.get("ordering_preserved"),
        )

    # ---- Save JSON results ----
    json_path = output_dir / "validation_results.json"
    serializable = _make_json_serializable(results)
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved validation results: %s", json_path)

    # ---- Log summary ----
    logger.info("=== Validation Summary ===")
    if "stylized_facts" in results:
        for name, r in results["stylized_facts"].items():
            logger.info("  %s: %s", name, "PASS" if r.get("passed") else "FAIL")
    if "lob_bench" in results:
        logger.info(
            "  LOB-Bench WD_mean: %.4f",
            results["lob_bench"].get("wasserstein/wd_mean", float("nan")),
        )
    if "regime_validation" in results:
        logger.info(
            "  Regime all_passed: %s",
            results["regime_validation"].get("all_passed"),
        )

    return results


def _main() -> None:
    """Hydra CLI entry point."""
    import hydra

    @hydra.main(
        version_base=None,
        config_path="../../configs",
        config_name="validation",
    )
    def _run(cfg: DictConfig) -> None:
        logging.basicConfig(level=logging.INFO)
        validate_generator(cfg)

    _run()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    _main()
