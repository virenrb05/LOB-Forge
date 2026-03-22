---
phase: 07-generator-validation
plan: 01
subsystem: evaluation
tags: [scipy, ks-test, autocorrelation, stylized-facts, market-microstructure]

requires:
  - phase: 06-generator-core
    provides: diffusion model that generates (B, T, 40) LOB snapshots
provides:
  - Six stylized fact statistical test functions for validating synthetic LOB data
affects: [07-generator-validation, evaluation, training-loop-validation]

tech-stack:
  added: []
  patterns: [consistent dict return type with passed/statistic keys, pure numpy/scipy statistical functions]

key-files:
  created: []
  modified:
    - lob_forge/evaluation/stylized_facts.py

key-decisions:
  - "40-col layout: ask_price(0-9), ask_size(10-19), bid_price(20-29), bid_size(30-39)"
  - "Volume proxy for market impact: sum of absolute bid-size changes across levels"
  - "Book shape test uses combined ask+bid depth per level for KS comparison"

patterns-established:
  - "Stylized fact test pattern: fn(real, synthetic, **kwargs) -> dict with passed key"
  - "Helper functions (_log_returns, _acf, _lag1_autocorrelation) for reuse across tests"

duration: 3min
completed: 2026-03-22
---

# Plan 07-01: Stylized Facts Summary

**Six scipy-based statistical tests for validating generated LOB data against real market microstructure properties**

## Performance

- **Duration:** 3 min
- **Completed:** 2026-03-22
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Implemented return_distribution_test (KS test on log-returns with kurtosis reporting)
- Implemented volatility_clustering_test (ACF of absolute returns with ratio check)
- Implemented bid_ask_bounce_test (negative lag-1 autocorrelation detection)
- Implemented spread_cdf_test (KS test on bid-ask spread distributions)
- Implemented book_shape_test (per-level depth profile KS comparison)
- Implemented market_impact_test (log-log regression for square-root law concavity)

## Task Commits

1. **Task 1: Return, volatility, and bid-ask bounce tests** - `ef9fa72` (feat)
2. **Task 2: Spread, book shape, and market impact tests** - included in `ef9fa72` (same file)
3. **Lint fix** - `2ed0e38` (fix)

## Files Created/Modified
- `lob_forge/evaluation/stylized_facts.py` - Six stylized fact test functions with helper utilities

## Decisions Made
- Used 40-column layout (ask_price, ask_size, bid_price, bid_size) consistent with plan specification
- Volume proxy for market impact uses bid-size changes (observable order flow)
- Book shape combines ask and bid depth at each level for a single depth profile comparison

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Ruff SIM108 lint violation**
- **Found during:** Post-implementation verification
- **Issue:** if/else block for ratio assignment flagged by ruff
- **Fix:** Replaced with ternary operator
- **Files modified:** lob_forge/evaluation/stylized_facts.py
- **Verification:** ruff check and black --check both pass
- **Committed in:** `2ed0e38`

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Trivial style fix. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All six stylized fact primitives ready for integration into GEN-06 validation pipeline
- Functions are pure statistical (no model or data path dependencies)

---
*Phase: 07-generator-validation*
*Completed: 2026-03-22*
