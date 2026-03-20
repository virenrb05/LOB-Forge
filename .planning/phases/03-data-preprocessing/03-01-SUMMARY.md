---
phase: 03-data-preprocessing
plan: 01
subsystem: data
tags: [pandas, scipy, features, lob, vpin, microprice]

# Dependency graph
requires:
  - phase: 02-data-ingestion
    provides: Unified LOB schema with column constants and Parquet read/write
provides:
  - 6 vectorized LOB feature functions (mid returns, order imbalance, microprice, depth imbalance, spread bps, VPIN)
  - compute_all_features orchestrator producing 18 derived columns
  - Full test suite with 19 unit tests
affects: [03-data-preprocessing, 04-predictor]

# Tech tracking
tech-stack:
  added: [scipy.stats.norm]
  patterns: [vectorized pandas feature computation, epsilon guard for division by zero]

key-files:
  created: [lob_forge/data/features.py, tests/test_features.py]
  modified: [lob_forge/data/__init__.py]

key-decisions:
  - "Used epsilon (1e-12) in denominators to guard against division by zero in imbalance and microprice calculations"
  - "VPIN uses bulk volume classification with scipy.stats.norm.cdf and rolling sigma window of 50"
  - "VPIN forward-fills for rows without trades, clips to [0, 1]"

patterns-established:
  - "Feature functions accept pd.DataFrame, return pd.DataFrame or pd.Series"
  - "Column naming: mid_return_K, depth_imb_i, order_imbalance, microprice, spread_bps, vpin"

# Metrics
duration: 3min
completed: 2026-03-19
---

# Plan 03-01: LOB Derived Features Summary

**6 vectorized LOB feature functions with VPIN bulk volume classification using scipy.stats.norm**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-19
- **Completed:** 2026-03-19
- **Tasks:** 3 (RED, GREEN, REFACTOR)
- **Files modified:** 3

## Accomplishments
- Implemented all 6 derived feature functions from tech spec section 2.4 step 2
- Mid-price returns, order imbalance, microprice, depth imbalance (10 levels), spread in bps, and VPIN
- compute_all_features orchestrator adds 18 derived columns to any LOB DataFrame
- 19 unit tests covering correctness with known values, range constraints, column structure, and edge cases

## Task Commits

Each task was committed atomically:

1. **RED: Failing tests** - `7e07544` (test)
2. **GREEN: Implementation** - `f8a8f1b` (feat)
3. **REFACTOR: Package exports** - handled by concurrent agent in `7672344`

## Files Created/Modified
- `lob_forge/data/features.py` - All 6 feature functions + compute_all_features orchestrator (264 lines)
- `tests/test_features.py` - 19 unit tests across 6 test classes (258 lines)
- `lob_forge/data/__init__.py` - Added 7 feature function exports

## Decisions Made
- Used epsilon (1e-12) denominator guard rather than np.where for division-by-zero protection -- simpler, same effect at float64 precision
- VPIN implementation uses bulk volume classification per Easley/Lopez de Prado: trade_size * CDF(log_return / rolling_sigma), with rolling window of 50 for sigma estimation
- Forward-fill VPIN for non-trade rows, clip to [0, 1] range

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All 6 feature functions tested and exported
- Ready for downstream consumers (normalizer, predictor transformer)
- VPIN depends on scipy which is already in pyproject.toml dependencies

---
*Phase: 03-data-preprocessing*
*Completed: 2026-03-19*
