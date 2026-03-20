---
phase: 03-data-preprocessing
plan: 02
subsystem: data
tags: [pandas, numpy, labeling, cumsum, causality]

requires:
  - phase: 01-scaffold
    provides: project structure, schema.py with MID_PRICE constant
provides:
  - compute_labels() function for multi-horizon mid-price movement classification
  - Causality-proven labeling (UP/STATIONARY/DOWN) at configurable horizons
affects: [04-feature-engineering, 05-transformer]

tech-stack:
  added: []
  patterns: [cumsum trick for O(n) rolling future mean]

key-files:
  created: [lob_forge/data/labels.py, tests/test_labels.py]
  modified: [lob_forge/data/__init__.py]

key-decisions:
  - "Used cumsum trick (not rolling/shift) for correct O(n) future mean computation"
  - "Causality boundary: modifying row k affects labels at rows [k-h, k-1], not rows >= k"
  - "Label dtype float64 to support NaN for insufficient-horizon tail rows"

patterns-established:
  - "Cumsum-based future windowing: cs = concat([0], cumsum(arr)), future_sum = cs[t+h+1] - cs[t+1]"
  - "Causality test pattern: compute, mutate one row, recompute, assert safe rows unchanged"

duration: 4min
completed: 2026-03-19
---

# Plan 03-02: Mid-Price Movement Labels Summary

**Cumsum-based multi-horizon mid-price movement labeler with 12 tests proving causality, correctness, and dtype guarantees**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-03-19
- **Completed:** 2026-03-19
- **Tasks:** 3 (RED/GREEN/REFACTOR)
- **Files modified:** 3

## Accomplishments
- `compute_labels()` classifies future mid-price movement as UP(2)/STATIONARY(1)/DOWN(0) at configurable horizons
- O(n) cumsum trick avoids error-prone rolling/shift pandas operations
- 12 unit tests covering monotonic prices, manual computation, causality, dtype, NaN tails, and input preservation

## Task Commits

Each TDD phase was committed atomically:

1. **RED: Failing tests** - `8b0f78c` (test)
2. **GREEN: Implementation** - `3f99707` (feat)
3. **REFACTOR: Package export** - `7672344` (refactor)

## Files Created/Modified
- `lob_forge/data/labels.py` - compute_labels() with cumsum-based future mean and threshold classification
- `tests/test_labels.py` - 12 tests across 6 test classes
- `lob_forge/data/__init__.py` - Added compute_labels to public API

## Decisions Made
- Used cumsum trick instead of pandas rolling/shift (plan explored several rolling approaches but cumsum is simplest and correct)
- Fixed causality test boundary from plan's specification: label at row t uses mid[t+1..t+h], so modifying row k affects rows [k-h, k-1] not rows [k, k+max_h]
- Label dtype is float64 (not int) to accommodate NaN in tail rows

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Correctness] Fixed causality test boundary**
- **Found during:** GREEN phase (test execution)
- **Issue:** Plan specified "Labels at rows 0..49 must NOT change" after modifying row 50, but label at row 40 with h=10 uses mid[41..50], which includes modified row 50
- **Fix:** Corrected safe boundary to `modify_row - h` per horizon, increased modify_row to 100 for cleaner margins
- **Verification:** All 12 tests pass including corrected causality assertions
- **Committed in:** `3f99707`

---

**Total deviations:** 1 auto-fixed (correctness)
**Impact on plan:** Essential fix for mathematically correct causality boundary. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Labels ready for downstream feature engineering and transformer training
- compute_labels() exported from lob_forge.data package
- All horizons from data.yaml (10, 20, 50, 100) supported by default

---
*Phase: 03-data-preprocessing*
*Completed: 2026-03-19*
