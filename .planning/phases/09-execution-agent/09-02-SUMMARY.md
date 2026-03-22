---
phase: 09-execution-agent
plan: 02
subsystem: executor
tags: [twap, vwap, almgren-chriss, baselines, gymnasium, execution, rl]

# Dependency graph
requires:
  - phase: 08-execution-environment
    provides: LOBExecutionEnv gymnasium environment with CostModel and action space

provides:
  - ExecutionResult dataclass with cost, IS, remaining, steps, actions
  - TWAPBaseline: uniform market orders every step
  - VWAPBaseline: sinusoidal volume-proxy schedule
  - AlmgrenChrissBaseline: closed-form optimal liquidation trajectory
  - RandomBaseline: uniform random action sampling
  - Shared run_episode(env, seed) interface for all 4 baselines
  - 25 pytest tests verifying all baselines

affects: [09-04-comparison, dqn-training, evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns: [dataclass result object, BaselineAgent ABC with run_episode hook]

key-files:
  created:
    - lob_forge/executor/baselines.py
    - tests/test_baselines.py
  modified: []

key-decisions:
  - "AlmgrenChriss action threshold uses 50% of TWAP rate (inventory/horizon/2) rather than 1% of inventory to handle small kappa regimes"
  - "VWAPBaseline lazily recomputes volume schedule when env.horizon differs from constructor horizon"
  - "RandomBaseline stores env reference via run_episode() override rather than constructor to stay environment-agnostic"
  - "TYPE_CHECKING guard for LOBExecutionEnv import to avoid circular dependencies"

patterns-established:
  - "Baseline pattern: BaselineAgent subclasses override select_action(); run_episode() is shared via super()"
  - "Pre-episode setup: override run_episode(), call super() after setup (AlmgrenChriss reset_episode, VWAP ensure_horizon)"

# Metrics
duration: 8min
completed: 2026-03-22
---

# Plan 09-02: Execution Baselines Summary

**TWAP, VWAP, Almgren-Chriss, and Random baselines sharing run_episode(env, seed) -> ExecutionResult interface for apples-to-apples comparison with DQN**

## Performance

- **Duration:** ~8 min
- **Completed:** 2026-03-22
- **Tasks:** 2 of 2
- **Files modified:** 2 created

## Accomplishments

- Implemented `ExecutionResult` dataclass capturing episode_cost, implementation_shortfall, remaining_inventory, n_steps, and actions
- Built 4 execution baselines (TWAP, VWAP, AlmgrenChriss, Random) all sharing the `run_episode(env, seed) -> ExecutionResult` interface via `BaselineAgent` ABC
- Wrote 25 tests covering field types, inventory execution, action validity, AC trajectory properties, and same-seed reproducibility

## Task Commits

1. **Task 1: Implement 4 execution baselines** — `6f70691` (feat)
2. **Task 2: Tests for all 4 baselines** — `52ed0a9` (test)

## Files Created/Modified

- `lob_forge/executor/baselines.py` — BaselineAgent, ExecutionResult, TWAPBaseline, VWAPBaseline, AlmgrenChrissBaseline, RandomBaseline (311 lines)
- `tests/test_baselines.py` — 25 tests across 6 test classes (198 lines)

## Decisions Made

- AlmgrenChriss action threshold: with default parameters (eta=0.1, sigma=0.3, lam=1e-5, horizon=200), kappa≈0.003 produces per-step deltas of ~5.58 against a 1% threshold of 10.0, so nothing executed. Fixed by using 50% of TWAP rate (inventory/horizon/2 = 2.5) as threshold — deltas ~5.58 exceed this and execute properly.
- VWAPBaseline computes volume schedule lazily via `_ensure_horizon()` to support different environment configurations without requiring constructor re-instantiation.

## Deviations from Plan

### Auto-fixed Issues

**1. AlmgrenChriss threshold too high — baseline executed 0 inventory**
- **Found during:** Task 1 verification
- **Issue:** Plan specified threshold as `0.01 * inventory` (10.0), but with default kappa the per-step deltas are ~5.58, all below threshold — resulted in zero execution and zero cost
- **Fix:** Changed threshold to `0.5 * inventory_total / horizon` (i.e., half the uniform TWAP rate = 2.5), which the AC deltas exceed at every active step
- **Verification:** AlmgrenChriss: cost=272.58, remaining=0.0 — full liquidation achieved
- **Committed in:** `6f70691` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (threshold calibration)
**Impact on plan:** Necessary fix — plan's threshold made AC baseline degenerate. No scope creep.

## Issues Encountered

None beyond the threshold calibration above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- All 4 baselines ready; run_episode() interface is stable for Plan 09-04 comparison
- `lob_forge/executor/baselines.py` exports TWAPBaseline, VWAPBaseline, AlmgrenChrissBaseline, RandomBaseline, ExecutionResult
- Baselines can be used independently of DQN (Wave 1 parallel with 09-01 DQN architecture)

---
*Phase: 09-execution-agent*
*Completed: 2026-03-22*
