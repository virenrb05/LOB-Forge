# Plan 03-05 Summary: PyTorch Datasets

## Status: Complete

## What Was Built

`lob_forge/data/dataset.py` (171 lines) — Two PyTorch Dataset classes for predictor and generator training.

### Classes

1. **`LOBDataset`** — Sliding-window dataset for predictor training. Returns `(features, labels)` tuples where features is `(seq_len, num_features)` float32 and labels is `(num_horizons,)` int64.

2. **`LOBSequenceDataset`** — Sliding-window dataset for generator training. Returns `(sequence, regime)` tuples where regime classifies volatility as low-vol(0), normal(1), or high-vol(2) from realized volatility quantiles.

### Tests

15 tests in `tests/test_dataset.py`:
- LOBDataset: length, shapes, dtypes, label values, auto/explicit feature detection, DataLoader
- LOBSequenceDataset: length, shapes, regime values/distribution, DataLoader, feature detection
- MPS loading: both datasets load correctly on MPS device

## Commits

| # | Hash | Description |
|---|------|-------------|
| 1 | `8a22818` | feat(03-05): implement LOBDataset and LOBSequenceDataset with tests |
| 2 | `06a8b78` | fix(03): resolve lint issues from wave 1 agents |

## Decisions

- NaN labels mapped to 0 in LOBDataset (real training should mask NaN rows)
- Regime labels computed from realized volatility (nanstd of mid_return_1 per window)
- Quantile boundaries (0.33, 0.67) configurable via `vol_quantiles` parameter
- Memory-mapped Parquet reading for efficiency; tensors created in __getitem__

## Deviations

- None
