"""Normalize, clean, and transform raw LOB snapshots into model-ready tensors.

Provides resampling to a uniform time grid, rolling z-score normalization,
and a full end-to-end preprocessing pipeline that orchestrates all Phase 3
modules (features, labels, splits).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lob_forge.data.features import compute_all_features
from lob_forge.data.labels import compute_labels
from lob_forge.data.schema import TIMESTAMP, TRADE_SIDE, read_lob_parquet
from lob_forge.data.splits import temporal_split
from lob_forge.data.validation import validate_lob_dataframe

logger = logging.getLogger(__name__)


def resample_to_grid(df: pd.DataFrame, interval_ms: int = 100) -> pd.DataFrame:
    """Resample LOB data to a uniform time grid using forward-fill.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``timestamp`` column with int64 microseconds.
    interval_ms : int
        Grid interval in milliseconds.  Default 100 ms.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with int64 microsecond timestamps.
    """
    original_rows = len(df)
    result = df.copy()

    # Convert int64 microseconds to DatetimeIndex
    result.index = pd.to_datetime(result[TIMESTAMP], unit="us")

    # Resample with forward-fill
    interval_str = f"{interval_ms}ms"
    result = result.resample(interval_str).ffill()

    # Drop rows that are entirely NaN (before the first observation)
    result = result.dropna(how="all")

    resampled_rows = len(result)
    fill_rate = original_rows / resampled_rows if resampled_rows > 0 else 0.0
    logger.info(
        "Resampled %d -> %d rows (fill rate: %.4f)",
        original_rows,
        resampled_rows,
        fill_rate,
    )

    # Convert DatetimeIndex back to int64 microseconds in the timestamp column
    result[TIMESTAMP] = result.index.astype(np.int64) // 1000  # ns -> us
    result = result.reset_index(drop=True)

    return result


def rolling_zscore(
    df: pd.DataFrame,
    feature_cols: list[str],
    window: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply rolling z-score normalization (causal, backward-looking only).

    For each column: ``z = (x - rolling_mean) / rolling_std``

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_cols : list[str]
        Column names to normalize.
    window : int
        Rolling window size.  Uses ``min_periods=1`` so the first rows get
        partial statistics.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(normalized_df, stats_df)`` where *stats_df* contains
        ``{col}_mean`` and ``{col}_std`` columns for inverse transformation.
    """
    normalized = df.copy()
    stats_data: dict[str, np.ndarray] = {}

    for col in feature_cols:
        series = df[col].astype(np.float64)
        roll = series.rolling(window, min_periods=1)
        roll_mean = roll.mean()
        roll_std = roll.std()

        # Replace zero/NaN std with 1.0 to avoid division by zero
        roll_std = roll_std.fillna(1.0)
        roll_std = roll_std.replace(0.0, 1.0)

        normalized[col] = (series - roll_mean) / roll_std
        stats_data[f"{col}_mean"] = roll_mean.values
        stats_data[f"{col}_std"] = roll_std.values

    stats_df = pd.DataFrame(stats_data, index=df.index)
    return normalized, stats_df


def preprocess(
    input_path: str | Path,
    output_dir: str | Path,
    cfg: Any,
) -> dict[str, Any]:
    """Run the full preprocessing pipeline: load, resample, featurize, label, normalize, split, save.

    Parameters
    ----------
    input_path : str | Path
        Path to raw LOB Parquet file.
    output_dir : str | Path
        Directory to save processed files.
    cfg : dict or OmegaConf
        Configuration with keys: ``resample_interval_ms``, ``horizons``,
        ``label_threshold``, ``normalization_window``, ``split.train``,
        ``split.val``, ``split.test``, ``purge_gap``.

    Returns
    -------
    dict
        Row counts per split and file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Attribute-style or dict-style access helper
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        """Get a value from dict or attribute-style config."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # 1. Load raw LOB data
    logger.info("Loading raw LOB data from %s", input_path)
    df = read_lob_parquet(input_path)

    # 2. Resample to uniform grid
    resample_ms = _get(cfg, "resample_interval_ms", 100)
    df = resample_to_grid(df, interval_ms=resample_ms)

    # 3. Compute derived features
    lookbacks = _get(cfg, "lookbacks", [1, 5, 10, 50])
    df = compute_all_features(df, lookbacks=lookbacks)

    # 4. Compute labels
    horizons = _get(cfg, "horizons", [10, 20, 50, 100])
    label_threshold = _get(cfg, "label_threshold", 0.00002)
    df = compute_labels(df, horizons=horizons, threshold=label_threshold)

    # 5. Identify feature columns (numeric, excluding metadata)
    label_cols = [c for c in df.columns if c.startswith("label_")]
    exclude = {TIMESTAMP, TRADE_SIDE} | set(label_cols)
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    # 6. Rolling z-score normalization
    norm_window = _get(cfg, "normalization_window", 5000)
    df_normed, stats = rolling_zscore(df, feature_cols, window=norm_window)

    # 7. Temporal split
    split_cfg = _get(cfg, "split", {})
    train_ratio = _get(split_cfg, "train", 0.7)
    val_ratio = _get(split_cfg, "val", 0.15)
    test_ratio = _get(split_cfg, "test", 0.15)
    purge_gap = _get(split_cfg, "purge_gap", _get(cfg, "purge_gap", 0))

    train_idx, val_idx, test_idx = temporal_split(
        len(df_normed),
        ratios=(train_ratio, val_ratio, test_ratio),
        purge_gap=purge_gap,
    )

    # 8-10. Save splits as Parquet
    paths: dict[str, Path] = {}
    counts: dict[str, int] = {}

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_df = df_normed.iloc[idx]
        split_path = output_dir / f"{name}.parquet"
        table = pa.Table.from_pandas(split_df, preserve_index=False)
        pq.write_table(table, split_path)
        paths[name] = split_path
        counts[name] = len(split_df)
        logger.info("Saved %s split: %d rows -> %s", name, len(split_df), split_path)

    # 11. Save normalization stats
    stats_path = output_dir / "normalization_stats.parquet"
    stats_table = pa.Table.from_pandas(stats, preserve_index=False)
    pq.write_table(stats_table, stats_path)
    paths["stats"] = stats_path
    logger.info("Saved normalization stats -> %s", stats_path)

    # 12. Return summary
    return {
        "counts": counts,
        "paths": {k: str(v) for k, v in paths.items()},
        "total_rows": len(df_normed),
        "feature_cols": feature_cols,
        "label_cols": label_cols,
    }


if __name__ == "__main__":
    from hydra import compose, initialize_config_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # Locate configs/ directory relative to project root (two levels up from this file)
    _project_root = Path(__file__).parents[2]
    _config_dir = str(_project_root / "configs")
    _data_dir = _project_root / "data"

    # Determine config name from argv (--config-name <name>) or default to "data"
    _config_name = "data"
    for _i, _arg in enumerate(sys.argv[1:], 1):
        if _arg == "--config-name" and _i < len(sys.argv):
            _config_name = sys.argv[_i + 1]
            break
        if _arg.startswith("--config-name="):
            _config_name = _arg.split("=", 1)[1]
            break

    with initialize_config_dir(config_dir=_config_dir, version_base=None):
        _cfg = compose(config_name=_config_name)

    # Flatten: the data YAML uses `# @package _global_` so keys live under cfg.data
    _data_cfg = _cfg.data if hasattr(_cfg, "data") else _cfg

    # Discover the most recently modified raw Parquet file in data/
    _raw_files = sorted(
        _data_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    # Exclude split outputs from Stage 2 itself
    _raw_files = [
        p
        for p in _raw_files
        if p.name
        not in (
            "train.parquet",
            "val.parquet",
            "test.parquet",
            "normalization_stats.parquet",
        )
    ]

    if not _raw_files:
        logger.error(
            "No raw Parquet files found in %s. Run Stage 1 (coinbase_downloader) first.",
            _data_dir,
        )
        sys.exit(1)

    _input_path = _raw_files[0]
    logger.info("Using raw input: %s", _input_path)

    # Run validation on raw data before preprocessing
    _raw_df = read_lob_parquet(_input_path)
    _issues = validate_lob_dataframe(_raw_df)
    if _issues:
        for _issue in _issues:
            logger.warning("Validation issue: %s", _issue)
    else:
        logger.info("Raw data validation: OK (%d rows)", len(_raw_df))
    del _raw_df  # free memory before preprocessing

    _result = preprocess(_input_path, _data_dir, _data_cfg)
    logger.info(
        "Preprocessing complete. Splits: train=%d, val=%d, test=%d",
        _result["counts"].get("train", 0),
        _result["counts"].get("val", 0),
        _result["counts"].get("test", 0),
    )
    for _split_name, _split_path in _result["paths"].items():
        logger.info("  %s -> %s", _split_name, _split_path)
