---
phase: 05-predictor-training
plan: 01
status: complete
started: 2026-03-20
completed: 2026-03-20
---

## Summary

Built core predictor training infrastructure: evaluation metrics and the full training loop with model factory, wandb integration, early stopping, and checkpointing.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Create predictor evaluation metrics module | `91d7b9f` | lob_forge/predictor/metrics.py |
| 2 | Create training loop with model factory, wandb, and checkpointing | `23aa3db` | lob_forge/predictor/trainer.py, lob_forge/data/dataset.py, lob_forge/predictor/__init__.py, pyproject.toml |

## Deliverables

- **lob_forge/predictor/metrics.py** — Per-class F1/precision/recall per horizon, weighted/macro F1, VPIN regression metrics (MSE, MAE, correlation)
- **lob_forge/predictor/trainer.py** — `build_model()` factory for all 3 model types, `train_model()` with gradient accumulation, OneCycleLR scheduling, early stopping, wandb logging, and best-model checkpointing
- **lob_forge/data/dataset.py** — LOBDataset extended with optional `vpin_col` parameter (backward-compatible 2-tuple or 3-tuple return)
- **lob_forge/predictor/__init__.py** — Updated exports: `build_model`, `train_model`, `compute_classification_metrics`, `compute_vpin_metrics`
- **pyproject.toml** — Added `scikit-learn` dependency

## Decisions

- LOBDataset returns 2-tuple `(features, labels)` when `vpin_col=None`, 3-tuple `(features, labels, vpin_target)` when set — backward compatible
- Trainer explicitly passes 40 book columns (BOOK_FEATURE_COLS from schema) to LOBDataset to avoid picking up derived features
- Model output handling: dict for DualAttentionTransformer, plain Tensor for baselines — `_extract_logits()` helper handles both
- Class weights computed as inverse frequency from training label distribution
- pin_memory=False for MPS/CPU devices

## Issues

None.
