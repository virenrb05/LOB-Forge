---
phase: 07-generator-validation
plan: 05
status: complete
started: 2026-03-22
completed: 2026-03-22
---

# Plan 07-05: E2E Validation Pipeline

## What was built

- `validate_generator(cfg)` — end-to-end pipeline that loads checkpoint, generates samples, runs all validation suites
- `configs/validation.yaml` — Hydra config for validation parameters (n_samples, ddim_steps, discriminator settings)
- Updated `lob_forge/evaluation/__init__.py` with 16 public exports

## Tasks

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Validation pipeline + Hydra config | `575ca70` | lob_forge/evaluation/validate_generator.py, configs/validation.yaml |
| 2 | Evaluation __init__.py public API | `79c9425` | lob_forge/evaluation/__init__.py |

## Verification

- All 16 evaluation functions importable from `lob_forge.evaluation`
- ruff check clean
- black --check clean
- No circular imports

## Decisions

- EMA weights loaded from checkpoint when available (key: "ema_state_dict")
- Per-regime generation: n_samples // 3 per regime (0, 1, 2)
- JSON results + PNG figure saved to configurable output_dir

## Issues

None.
