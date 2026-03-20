---
status: passed
---

# Phase 03 -- Data Preprocessing Verification Report

**Date:** 2026-03-20
**Phase goal:** "Raw LOB snapshots become model-ready tensors with features, labels, and temporal splits"
**Test run:** 89/89 passed (3.65s)

---

## Criterion 1: Rolling z-score normalization produces zero-mean unit-variance features (per-feature, causal)

**Code exists:** YES
- `lob_forge/data/preprocessor.py` -- `rolling_zscore()` (lines 71-114)
- Uses backward-looking `pd.Series.rolling(window, min_periods=1)` -- strictly causal
- Per-feature normalization: iterates `feature_cols`, computes `(x - rolling_mean) / rolling_std`
- Handles zero/NaN std by replacing with 1.0

**Tests exist:** YES -- 8 tests in `tests/test_preprocessor.py`
- `TestRollingZscoreZeroMean::test_zero_mean_after_warmup` -- verifies post-warmup mean is near 0
- `TestRollingZscoreUnitVariance::test_unit_variance_after_warmup` -- verifies post-warmup std is near 1
- `TestRollingZscoreCausality::test_causal_independence` -- mutates row 500, confirms rows 0-499 are unchanged
- `TestRollingZscoreStatsReturned` -- verifies `{col}_mean` and `{col}_std` columns returned with finite values
- `TestRollingZscoreConstantSeries::test_constant_gives_zero` -- constant input yields 0.0 output
- `TestRollingZscoreSingleRow::test_single_row_no_crash` -- single-row edge case
- `TestRollingZscoreWindowParameter::test_different_windows_differ` -- different windows produce different results

**Tests pass:** YES (8/8)

**Verdict:** FULLY SATISFIED

---

## Criterion 2: Mid-price labels are strictly causal (no future data leakage -- unit tested)

**Code exists:** YES
- `lob_forge/data/labels.py` -- `compute_labels()` (lines 22-88)
- Label at row t uses `mid_price[t+1..t+h]` via cumsum trick -- strictly causal
- Last h rows are NaN (insufficient future data)

**Tests exist:** YES -- 12 tests in `tests/test_labels.py`
- `TestCausality::test_modifying_future_does_not_change_past_labels` -- directly tests causality by mutating a future price and verifying earlier labels are unchanged
- `TestCausality::test_modifying_past_may_change_nearby_labels` -- verifies boundary behavior
- `TestManualComputation::test_known_trajectory` -- hand-calculated expected values verified against code output
- `TestMultiHorizon::test_each_horizon_has_correct_nan_tail` -- verifies last h rows are NaN per horizon
- `TestMonotonicPrices` -- increasing/decreasing/flat prices produce correct UP/DOWN/STATIONARY labels

**Tests pass:** YES (12/12)

**Verdict:** FULLY SATISFIED

---

## Criterion 3: OFI, MLOFI, VPIN, and microprice features compute correctly on sample data

**Code exists:** YES -- all four features implemented in `lob_forge/data/features.py`
- `compute_ofi()` (lines 213-253) -- level-1 Order Flow Imbalance following Cont, Kukanov & Stoikov (2014); measures changes in order book quantities between consecutive snapshots
- `compute_mlofi()` (lines 256-306) -- Multi-Level OFI with exponential decay weighting across 10 book levels
- `compute_vpin()` (lines 119-210) -- Volume-Synchronized Probability of Informed Trading with bulk volume classification
- `compute_microprice()` (lines 69-84) -- size-weighted mid-price
- `compute_all_features()` (lines 309-366) -- orchestrator that adds all 20 derived features including ofi and mlofi columns

**Tests exist:** YES -- 27 tests in `tests/test_features.py`
- `TestOfi` (4 tests) -- unchanged book gives zero OFI, known value (+10 from bid size increase), first row NaN, price level change zeroes contribution
- `TestMlofi` (4 tests) -- returns named Series, decay weighting verified (level 2 = 0.5 * level 1), first row NaN, single-level MLOFI equals OFI
- `TestVpin` (3 tests) -- returns correct-length Series, values in [0,1], NaN trades forward-filled
- `TestMicroprice` (2 tests) -- equal sizes yields mid-price, known value verified to 4 decimal places
- `TestComputeAllFeatures` (3 tests) -- adds exactly 20 columns, all expected columns present (including ofi, mlofi), original columns preserved

**Tests pass:** YES (27/27)

**Verdict:** FULLY SATISFIED

---

## Criterion 4: Train/val/test splits are temporal with purge gaps (no data leakage -- unit tested)

**Code exists:** YES
- `lob_forge/data/splits.py` -- `temporal_split()` (lines 13-62)
- Temporal ordering enforced: train < purge < val < purge < test
- Purge gap rows are excluded (not assigned to any split)

**Tests exist:** YES -- 27 tests in `tests/test_splits.py`
- `TestNoLeakage::test_no_overlap_between_segments` -- parametrized over 5 purge_gap values (0, 1, 5, 10, 50)
- `TestNoLeakage::test_temporal_ordering` -- max(train) < min(val) < max(val) < min(test)
- `TestNoLeakage::test_purge_gap_maintained_between_train_and_val` -- min(val) - max(train) >= purge_gap + 1
- `TestNoLeakage::test_purge_gap_maintained_between_val_and_test` -- min(test) - max(val) >= purge_gap + 1
- Edge cases: large purge gaps, empty segments, contiguous coverage at gap=0

**Tests pass:** YES (27/27)

**Verdict:** FULLY SATISFIED

---

## Criterion 5: LOBDataset and LOBSequenceDataset load batches correctly on MPS device

**Code exists:** YES
- `lob_forge/data/dataset.py` -- `LOBDataset` (lines 13-80) and `LOBSequenceDataset` (lines 83-170)
- LOBDataset: sliding window, returns `(features: float32[seq_len, n_feat], labels: int64[n_horizons])` tuples
- LOBSequenceDataset: sliding window with volatility-based regime labels (0/1/2)
- Both support DataLoader batching and auto-detect feature columns (excluding timestamps, trade_side, labels)

**Tests exist:** YES -- 15 tests in `tests/test_dataset.py`
- `TestLOBDataset` (7 tests) -- length, shapes, last valid index, label values in {0,1,2}, auto feature detection, explicit feature cols, DataLoader batching
- `TestLOBSequenceDataset` (6 tests) -- length, shapes, regime values in {0,1,2}, regime distribution (all 3 regimes present), DataLoader batching, auto feature detection
- `TestMPSLoading::test_lob_dataset_mps` -- loads batch, transfers to MPS, asserts `device.type == "mps"`
- `TestMPSLoading::test_lob_sequence_dataset_mps` -- loads batch, transfers to MPS, asserts `device.type == "mps"`

**MPS available on this machine:** YES
**Tests pass:** YES (15/15, including both MPS tests)

**Verdict:** FULLY SATISFIED

---

## Summary

| # | Criterion | Code | Tests | Pass | Status |
|---|-----------|------|-------|------|--------|
| 1 | Rolling z-score normalization (zero-mean, unit-var, causal) | YES | YES (8) | YES | **PASS** |
| 2 | Causal mid-price labels (unit tested) | YES | YES (12) | YES | **PASS** |
| 3 | OFI, MLOFI, VPIN, microprice features | YES | YES (27) | YES | **PASS** |
| 4 | Temporal splits with purge gaps (unit tested) | YES | YES (27) | YES | **PASS** |
| 5 | LOBDataset + LOBSequenceDataset on MPS | YES | YES (15) | YES | **PASS** |

**Overall: ALL 5 CRITERIA PASSED -- Phase 03 is complete.**
