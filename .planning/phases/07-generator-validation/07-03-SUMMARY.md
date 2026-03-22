---
phase: 07-generator-validation
plan: 03
subsystem: evaluation
tags: [wasserstein, discriminator, mlp, scipy, torch, sklearn, lob-bench]

requires:
  - phase: 06-generator-core
    provides: DiffusionModel generating (B, T, 40) LOB sequences
provides:
  - compute_wasserstein_metrics for per-feature distributional comparison
  - train_discriminator MLP-based real vs synthetic scoring
  - compute_conditional_stats per-regime statistical comparison
  - run_lob_bench orchestrator combining all metrics
affects: [08-rl-environment, 09-integration, 10-evaluation]

tech-stack:
  added: [scipy.stats.wasserstein_distance, sklearn.metrics.roc_auc_score]
  patterns: [namespaced metric dict, lazy torch import]

key-files:
  created:
    - lob_forge/evaluation/lob_bench.py
    - tests/test_lob_bench.py

key-decisions:
  - "Lazy import of torch/sklearn inside train_discriminator to keep scipy-only functions lightweight"
  - "Discriminator identical-data test asserts accuracy < 0.8 rather than near 0.5 (small N makes exact threshold fragile)"
  - "run_lob_bench namespaces keys with / separator (wasserstein/wd_mean, discriminator/accuracy, etc.)"

patterns-established:
  - "Evaluation functions return flat dict[str, float|Any] for easy logging"
  - "Namespaced metric keys with / for multi-metric orchestrators"

duration: 5min
completed: 2026-03-22
---

# Plan 07-03: LOB-Bench Metrics Summary

**Wasserstein distances, MLP discriminator, and conditional stats for quantitative LOB generation evaluation (GEN-08)**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-22
- **Completed:** 2026-03-22
- **Tasks:** 3
- **Files created:** 2

## Accomplishments
- Per-feature Wasserstein distances plus derived spread/mid/returns metrics
- MLP discriminator (3-layer) returning accuracy, AUC, and training loss
- Per-regime conditional statistics with relative error aggregation
- run_lob_bench orchestrator combining all three metric families
- 13-test suite covering all four public functions

## Task Commits

Each task was committed atomically:

1. **Task 1: Wasserstein distances and conditional statistics** - `3455e92` (feat)
2. **Task 2: MLP discriminator and run_lob_bench** - `c5bb41d` (feat)
3. **Task 3: Unit tests for LOB-Bench metrics** - `a548920` (test)

## Files Created/Modified
- `lob_forge/evaluation/lob_bench.py` - LOB-Bench evaluation metrics (349 lines, 4 public functions)
- `tests/test_lob_bench.py` - 13 unit tests across 4 test classes (184 lines)

## Decisions Made
- Lazy torch/sklearn imports in train_discriminator to avoid heavy deps for Wasserstein-only usage
- Discriminator identical-data test uses `accuracy < 0.8` bound (not near-0.5) since small N + few epochs produces noisy results
- Namespaced output keys with `/` separator for clean multi-metric aggregation

## Deviations from Plan

### Auto-fixed Issues

**1. [Lint] SIM108 ternary operator in train_discriminator**
- **Found during:** Task 2
- **Issue:** ruff flagged if/else block for AUC degenerate case
- **Fix:** Converted to ternary operator
- **Committed in:** c5bb41d (part of Task 2 commit)

---

**Total deviations:** 1 auto-fixed (lint)
**Impact on plan:** Trivial style fix, no scope change.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- LOB-Bench metrics ready for integration into generator training/eval pipeline
- run_lob_bench can be called directly with numpy arrays from any evaluation script

---
*Phase: 07-generator-validation*
*Completed: 2026-03-22*
