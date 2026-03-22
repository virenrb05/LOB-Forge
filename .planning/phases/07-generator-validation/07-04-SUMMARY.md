---
phase: 07-generator-validation
plan: 04
subsystem: evaluation
tags: [scipy, ks-test, kl-divergence, regime-validation]

requires:
  - phase: 06-generator-core
    provides: "Generator with regime conditioning (3 volatility regimes)"
provides:
  - "Regime distribution comparison (KS tests, vol ratios)"
  - "KL divergence regime separability metric"
  - "Orchestrated regime conditioning validator"
affects: [08-rl-agent, 09-integration]

tech-stack:
  added: []
  patterns: [pairwise-regime-comparison, discrete-kl-divergence]

key-files:
  created:
    - lob_forge/evaluation/regime_validation.py
    - tests/test_regime_validation.py
  modified: []

key-decisions:
  - "Mid-price computed from columns 0 (ask_1) and 20 (bid_1) as (ask+bid)/2"
  - "Epsilon 1e-10 added to histogram bins for KL stability; re-normalised after"
  - "Regime distinctness requires KS p < 0.05 on returns for all pairs"
  - "Regime fidelity requires KS p > 0.05 (cannot reject same distribution)"
  - "Separability threshold: mean_kl > 0.1"

patterns-established:
  - "Pairwise regime comparison via itertools.combinations on sorted regime keys"
  - "Shared histogram bin edges across all regimes for fair KL comparison"

duration: 4min
completed: 2026-03-22
---

# Plan 07-04: Regime Validation Summary

**KS-test and KL-divergence validation proving regime-conditioned generation produces statistically distinct LOB distributions**

## Performance

- **Duration:** 4 min
- **Completed:** 2026-03-22
- **Tasks:** 2
- **Files created:** 2

## Accomplishments
- Three regime validation functions: comparison, divergence, and orchestrated conditioning check
- 8 unit tests covering distinct/identical regimes, KL separability, vol ordering, and collapsed-regime detection
- All lint and format checks pass clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement regime distribution comparison functions** - `1a2a36f` (feat)
2. **Task 2: Unit tests for regime validation** - `8d9af69` (test)

## Files Created/Modified
- `lob_forge/evaluation/regime_validation.py` - compare_regime_distributions, compute_regime_divergence, validate_regime_conditioning
- `tests/test_regime_validation.py` - 8 tests across 3 test classes

## Decisions Made
- Mid-price derived from columns 0/20 matching LOB schema convention
- KL divergence uses shared bin edges across regimes for fair comparison
- Test data uses cumulative log-normal returns with controlled std for reliable statistical separation
- Identical-regime fixture uses same RNG seed per regime (not slicing one array) to ensure true distributional identity

## Deviations from Plan

### Auto-fixed Issues

**1. [Test data generator] Fixed synthetic data to produce controlled return volatility**
- **Found during:** Task 2 (unit tests)
- **Issue:** Original cumsum-of-normal approach produced near-identical return volatilities across regimes because random walk dynamics dominated
- **Fix:** Switched to cumulative log-returns (exp of cumsum of normal) with direct return_std parameter
- **Files modified:** tests/test_regime_validation.py
- **Verification:** All 8 tests pass with clear statistical separation
- **Committed in:** 8d9af69

---

**Total deviations:** 1 auto-fixed (test data generation)
**Impact on plan:** Necessary for test correctness. No scope creep.

## Issues Encountered
None beyond the data generator fix noted above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Regime validation module ready for integration with generator training pipeline
- All 4 plans of phase 07 now complete pending confirmation

---
*Phase: 07-generator-validation*
*Completed: 2026-03-22*
