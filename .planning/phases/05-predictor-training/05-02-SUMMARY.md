---
phase: 05-predictor-training
plan: 02
status: complete
started: 2026-03-20
completed: 2026-03-20
---

## Summary

Wired Hydra CLI entry point to training loop and implemented walk-forward rolling-window evaluation for temporal cross-validation.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Wire Hydra entry points to predictor training | `8f68f1a` | lob_forge/train.py, lob_forge/predictor/train.py, configs/predictor.yaml |
| 2 | Implement walk-forward rolling-window evaluation | `25b0961` | lob_forge/predictor/walk_forward.py, lob_forge/predictor/__init__.py |

## Deliverables

- **lob_forge/train.py** — Hydra `@hydra.main` entry point dispatching to `train_model` with data path resolution and error handling
- **lob_forge/predictor/train.py** — `train_predictor()` convenience wrapper + `compare_models()` for sequential 3-model comparison with wandb logging
- **lob_forge/predictor/walk_forward.py** — `walk_forward_eval()` with expanding-window strategy, purge gaps, per-window and aggregated metrics, wandb logging
- **configs/predictor.yaml** — Added `walk_forward` config section (enabled, n_windows, train_ratio, purge_gap)
- **lob_forge/predictor/__init__.py** — Updated exports: walk_forward_eval, train_predictor, compare_models

## Decisions

- Walk-forward uses expanding window strategy (standard in finance): window i trains on segments [0..i], evaluates on segment [i+1]
- Purge gap rows removed between train end and val start to prevent label leakage
- Temporary Parquet files used per window, cleaned up after evaluation
- compare_models trains all 3 types sequentially with wandb comparison logging

## Issues

None.
