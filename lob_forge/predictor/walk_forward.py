"""Rolling-window walk-forward evaluation for temporal cross-validation.

Walk-forward evaluation trains on an expanding window and evaluates on
the next temporal segment. This is the standard approach in financial
ML to avoid look-ahead bias. Purge gaps between train/val prevent
label leakage across boundaries.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf

from lob_forge.predictor.trainer import train_model

log = logging.getLogger(__name__)


def _compute_window_boundaries(
    n_rows: int,
    n_windows: int,
    purge_gap: int,
) -> list[tuple[int, int, int, int]]:
    """Compute expanding-window train/val boundaries.

    Splits *n_rows* into ``n_windows + 1`` equal segments. Window *i*
    trains on segments ``[0 .. i]`` and validates on segment ``i + 1``,
    with *purge_gap* rows removed between the train end and val start.

    Parameters
    ----------
    n_rows : int
        Total rows in the dataset.
    n_windows : int
        Number of evaluation windows (minimum 2).
    purge_gap : int
        Rows to skip between train and val.

    Returns
    -------
    list of (train_start, train_end, val_start, val_end) tuples.
    """
    n_windows = max(n_windows, 2)
    n_segments = n_windows + 1
    segment_size = n_rows // n_segments

    if segment_size < 1:
        raise ValueError(f"Dataset too small ({n_rows} rows) for {n_windows} windows.")

    windows: list[tuple[int, int, int, int]] = []
    for i in range(n_windows):
        train_start = 0
        train_end = segment_size * (i + 1)
        val_start = train_end + purge_gap
        val_end = segment_size * (i + 2)
        # Last window may extend to end of data
        if i == n_windows - 1:
            val_end = n_rows
        # Ensure val range is valid
        if val_start >= val_end or val_start >= n_rows:
            log.warning(
                "Window %d: val range invalid (start=%d, end=%d, n_rows=%d). "
                "Skipping.",
                i,
                val_start,
                val_end,
                n_rows,
            )
            continue
        windows.append((train_start, train_end, val_start, val_end))

    return windows


def walk_forward_eval(
    cfg: DictConfig,
    data_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Run walk-forward (expanding-window) evaluation.

    Trains the model on progressively larger training windows and
    evaluates on the subsequent temporal segment. Returns per-window
    and aggregated (mean/std) metrics.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config with ``predictor.walk_forward.*`` keys.
    data_path : str | Path
        Path to the full dataset Parquet file.
    output_dir : str | Path
        Root output directory; each window gets a sub-directory.

    Returns
    -------
    dict
        ``per_window``: list of per-window metrics dicts.
        ``mean``: mean across windows for each metric.
        ``std``: std across windows for each metric.
    """
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read walk-forward config
    wf = OmegaConf.select(cfg, "predictor.walk_forward", default={})
    n_windows = int(OmegaConf.select(wf, "n_windows", default=3))
    purge_gap = int(OmegaConf.select(wf, "purge_gap", default=10))

    # Get total rows without loading full data
    pf = pq.ParquetFile(str(data_path))
    n_rows = pf.metadata.num_rows
    log.info(
        "Walk-forward: %d rows, %d windows, purge_gap=%d",
        n_rows,
        n_windows,
        purge_gap,
    )

    windows = _compute_window_boundaries(n_rows, n_windows, purge_gap)
    if len(windows) < 2:
        raise ValueError(
            f"Need at least 2 valid windows, got {len(windows)}. "
            "Dataset may be too small or purge_gap too large."
        )

    log.info(
        "Walk-forward windows: %d valid out of %d requested", len(windows), n_windows
    )

    # Read full table once for slicing
    table = pq.read_table(str(data_path))

    per_window_results: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="wf_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        for i, (tr_start, tr_end, vl_start, vl_end) in enumerate(windows):
            log.info(
                "Window %d/%d: train[%d:%d] (%d rows), val[%d:%d] (%d rows)",
                i + 1,
                len(windows),
                tr_start,
                tr_end,
                tr_end - tr_start,
                vl_start,
                vl_end,
                vl_end - vl_start,
            )

            # Slice and write temporary Parquet files
            train_slice = table.slice(tr_start, tr_end - tr_start)
            val_slice = table.slice(vl_start, vl_end - vl_start)

            tmp_train = tmp_path / f"window_{i}_train.parquet"
            tmp_val = tmp_path / f"window_{i}_val.parquet"

            pq.write_table(train_slice, str(tmp_train))
            pq.write_table(val_slice, str(tmp_val))

            # Train on this window
            window_output = output_dir / f"window_{i}"
            window_output.mkdir(parents=True, exist_ok=True)

            results = train_model(cfg, tmp_train, tmp_val, window_output)
            results["window"] = i
            results["train_rows"] = tr_end - tr_start
            results["val_rows"] = vl_end - vl_start
            per_window_results.append(results)

            # Log per-window to wandb
            wandb_enabled = OmegaConf.select(cfg, "wandb.enabled", default=False)
            if wandb_enabled:
                try:
                    import wandb

                    window_log: dict[str, float] = {
                        f"wf_w{i}/val_loss": results.get("best_val_loss", float("inf")),
                        f"wf_w{i}/best_epoch": results.get("best_epoch", -1),
                    }
                    for k, v in results.get("best_metrics", {}).items():
                        if isinstance(v, (int, float)):
                            window_log[f"wf_w{i}/{k}"] = v
                    wandb.log(window_log)
                except ImportError:
                    pass

    # Aggregate metrics across windows
    metric_keys: set[str] = set()
    for r in per_window_results:
        metric_keys.update(r.get("best_metrics", {}).keys())

    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}

    for key in sorted(metric_keys):
        values = [
            r["best_metrics"][key]
            for r in per_window_results
            if key in r.get("best_metrics", {})
            and isinstance(r["best_metrics"][key], (int, float))
        ]
        if values:
            mean_metrics[key] = float(np.mean(values))
            std_metrics[key] = float(np.std(values))

    # Also aggregate val_loss
    val_losses = [
        r["best_val_loss"] for r in per_window_results if "best_val_loss" in r
    ]
    if val_losses:
        mean_metrics["val_loss"] = float(np.mean(val_losses))
        std_metrics["val_loss"] = float(np.std(val_losses))

    log.info("Walk-forward mean metrics: %s", mean_metrics)
    log.info("Walk-forward std metrics: %s", std_metrics)

    # Log aggregated to wandb
    wandb_enabled = OmegaConf.select(cfg, "wandb.enabled", default=False)
    if wandb_enabled:
        try:
            import wandb

            agg_log: dict[str, float] = {}
            for k, v in mean_metrics.items():
                agg_log[f"wf_mean/{k}"] = v
            for k, v in std_metrics.items():
                agg_log[f"wf_std/{k}"] = v
            wandb.log(agg_log)
        except ImportError:
            pass

    return {
        "per_window": per_window_results,
        "mean": mean_metrics,
        "std": std_metrics,
    }
