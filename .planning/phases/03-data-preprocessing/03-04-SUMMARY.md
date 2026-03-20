# Plan 03-04 Summary: Preprocessing Pipeline

## Status: Complete

## What Was Built

`lob_forge/data/preprocessor.py` (221 lines) — the glue module that orchestrates all Phase 3 components into a full preprocessing pipeline.

### Functions

1. **`resample_to_grid(df, interval_ms=100)`** — Resamples LOB data to uniform time grid using forward-fill. Logs fill rate.
2. **`rolling_zscore(df, feature_cols, window=5000)`** — Causal rolling z-score normalization. Returns (normalized_df, stats_df) for inverse transformation.
3. **`preprocess(input_path, output_dir, cfg)`** — End-to-end pipeline: load → resample → features → labels → normalize → split → save Parquet.

### Pipeline Flow

```
read_lob_parquet → resample_to_grid → compute_all_features → compute_labels → rolling_zscore → temporal_split → write Parquet splits + stats
```

## Commits

| # | Hash | Description |
|---|------|-------------|
| 1 | `ce1189a` | feat(03-04): implement resampling, rolling z-score, and preprocessing pipeline |

## Decisions

- `_get()` helper supports both dict and OmegaConf attribute-style access
- Zero/NaN std replaced with 1.0 in rolling z-score to avoid division by zero
- Feature columns auto-detected: all numeric columns except timestamp, trade_side, and label_* columns
- Uses plain `pq.write_table` (not schema-constrained `write_lob_parquet`) since output has extra feature/label columns

## Deviations

- configs/data.yaml already had purge_gap=10 from plan 03-03; left as-is rather than changing to 100 since 10 was the intentional production value

## Verification

- [x] All 3 functions importable
- [x] Rolling z-score produces normalized output
- [x] black + ruff pass
