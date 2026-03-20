---
phase: 04-predictor-architecture
plan: 02
subsystem: predictor
tags: [pytorch, cnn, lstm, inception, deeplob, baseline, logistic-regression]

# Dependency graph
requires:
  - phase: 04-predictor-architecture/01
    provides: TLOB model interface (input/output shapes for baseline matching)
provides:
  - DeepLOB CNN+Inception+LSTM baseline model
  - LinearBaseline logistic regression performance floor
  - Both baselines share (batch, n_horizons, n_classes) output interface
affects: [04-predictor-architecture, 05-training-loop]

# Tech tracking
tech-stack:
  added: []
  patterns: [per-horizon ModuleList heads, matching input/output interface across models]

key-files:
  created:
    - lob_forge/predictor/deeplob.py
    - lob_forge/predictor/linear_baseline.py
  modified: []

key-decisions:
  - "DeepLOB uses stride-2 Conv2d for spatial reduction (not pooling)"
  - "Inception branches use kernel sizes (1,1), (3,1), (5,1) for multi-scale temporal features"
  - "LinearBaseline uses only last time step — no sequence processing"
  - "Both models use nn.ModuleList for per-horizon heads (not Python list)"

patterns-established:
  - "Baseline interface: (batch, T, 40) input -> (batch, n_horizons, n_classes) logits"
  - "Per-horizon heads via nn.ModuleList for proper parameter registration"

# Metrics
duration: 2min
completed: 2026-03-20
---

# Phase 04, Plan 02: Baseline Models Summary

**DeepLOB CNN+Inception+LSTM baseline (67K params) and LinearBaseline logistic regression floor (492 params) with matching output interfaces**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-20T18:45:13Z
- **Completed:** 2026-03-20T18:47:30Z
- **Tasks:** 2
- **Files created:** 2

## Accomplishments
- DeepLOB: 2 conv blocks, 3-branch inception module, spatial collapse, LSTM, per-horizon linear heads (67,132 params)
- LinearBaseline: last-time-step logistic regression with per-horizon heads (492 params)
- Both produce identical output shape (batch, n_horizons, n_classes) for fair comparison with TLOB

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement DeepLOB baseline** - `60d54f9` (feat)
2. **Task 2: Implement LinearBaseline** - `1b88886` (feat)

**Lint fixes:** `1b88886` (refactor: remove unused variables, black formatting)

## Files Created/Modified
- `lob_forge/predictor/deeplob.py` - DeepLOB CNN+Inception+LSTM baseline model
- `lob_forge/predictor/linear_baseline.py` - Pure logistic regression performance floor

## Decisions Made
- DeepLOB architecture follows simplified Zhang et al. 2019 as specified in plan
- No hidden layers in LinearBaseline — intentionally minimal for performance floor

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Unused variable flagged by ruff**
- **Found during:** Final verification
- **Issue:** `in_features` assigned but never used in both files (F841)
- **Fix:** Removed unused variable in deeplob.py, inlined expression in linear_baseline.py
- **Files modified:** lob_forge/predictor/deeplob.py, lob_forge/predictor/linear_baseline.py
- **Verification:** `ruff check` passes, `black --check` passes
- **Committed in:** `1b88886`

---

**Total deviations:** 1 auto-fixed (1 blocking lint error)
**Impact on plan:** Trivial cleanup. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Both baselines ready for training loop integration
- All three models (TLOB, DeepLOB, LinearBaseline) share matching input/output interface
- Ready for plan 04-03 (model factory / registry)

---
*Phase: 04-predictor-architecture*
*Completed: 2026-03-20*
