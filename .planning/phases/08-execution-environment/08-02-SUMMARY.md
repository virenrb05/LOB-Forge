---
phase: 08-execution-environment
plan: 02
subsystem: executor
tags: [gymnasium, rl-environment, lob, execution, 7-action-space, numpy, check_env]

# Dependency graph
requires:
  - phase: 08-01
    provides: CostModel.compute() for reward calculation in step()
  - phase: 07-generator-validation
    provides: validated synthetic LOB data format (40 book columns, z-score normalized)

provides:
  - LOBExecutionEnv(gymnasium.Env) passing check_env() with no errors
  - 7-action discrete space: WAIT, MARKET x3, LIMIT x3 with Bernoulli fill sampling
  - Sliding-window observation space (seq_len, 40) with zero-padding at episode start
  - Running VWAP tracking across executed trades
  - Updated executor.yaml with seq_len, inventory, horizon, cost_model config

affects:
  - 09-dqn-agent (trains DQN in LOBExecutionEnv)
  - Any phase using LOBExecutionEnv for backtesting or evaluation

# Tech tracking
tech-stack:
  added: [gymnasium]
  patterns: [gymnasium.Env subclass, numpy-only environment (no torch), Bernoulli fill sampling via np_random]

key-files:
  created:
    - lob_forge/executor/environment.py
  modified:
    - lob_forge/executor/__init__.py
    - configs/executor.yaml

key-decisions:
  - "LOBExecutionEnv takes pre-loaded np.ndarray (not a file path) — caller handles data loading; real mode built by loading parquet then passing array"
  - "Limit orders reuse order_sizes fractions for exec_size (same as market actions at same index) — keeps 3-level size logic consistent"
  - "np_random.random() for Bernoulli fill sampling — uses gymnasium's seeded RNG for reproducibility"
  - "Unbounded Box observation space (-inf, +inf) by design — z-score normalized data; check_env warnings are informational only"
  - "_get_obs() pads with zeros at episode start rather than starting at seq_len offset — allows episodes to start from any valid position"

patterns-established:
  - "Gymnasium pattern: super().reset(seed=seed) before any episode state reset for proper RNG seeding"
  - "Episode start sampling: np_random.integers(min_start, max_start+1) to leave full horizon+seq_len window"
  - "Reward normalization: -cost / inventory for scale-consistent rewards across episodes"

# Metrics
duration: 12min
completed: 2026-03-22
---

# Phase 08-02: LOBExecutionEnv Summary

**Gymnasium-compliant LOBExecutionEnv with 7-action discrete space, CostModel integration, and check_env() passing with no errors**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-03-22
- **Completed:** 2026-03-22
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Implemented LOBExecutionEnv inheriting from gymnasium.Env with full API compliance (reset, step, observation_space, action_space)
- 7-action space: WAIT(0), MARKET_SMALL/MED/LARGE(1-3), LIMIT_AGGRESSIVE/MID/PASSIVE(4-6) with Bernoulli fill probabilities (50%/30%/10%)
- gymnasium.utils.env_checker.check_env() passes with no errors; warnings are informational (unbounded obs space by design)
- Updated executor.yaml with environment parameters: seq_len, inventory, horizon, order_sizes, limit_offsets_bps, cost_model

## Task Commits

Each task was committed atomically:

1. **Task 1: LOBExecutionEnv implementation** - `7e3c447` (feat(08-02))
2. **Task 2: executor.yaml env config** - `71d2e96` (feat(08-02))

## Files Created/Modified
- `lob_forge/executor/environment.py` - LOBExecutionEnv with full gymnasium API, 7-action space, VWAP tracking
- `lob_forge/executor/__init__.py` - Re-exports LOBExecutionEnv alongside CostModel
- `configs/executor.yaml` - Added seq_len, inventory, horizon, order_sizes, limit_offsets_bps, avg_daily_volume, cost_model

## Decisions Made
- `LOBExecutionEnv` takes `lob_data: np.ndarray` directly (caller loads parquet, passes 40-col array) — keeps environment pure, file loading is caller concern
- Limit orders reuse `order_sizes` fractions indexed by limit level (0/1/2) — consistent 3-level size logic across market and limit orders
- Bernoulli fill sampling uses `self.np_random.random()` (gymnasium's seeded RNG) — reproducible episode replay via `reset(seed=...)`
- Observation space is unbounded Box(-inf, inf) by design — z-score normalized data has no natural bounds; check_env warnings are expected
- `_get_obs()` uses zero-padding at episode start rather than requiring `seq_len` rows of history — allows episodes to start anywhere in `lob_data[seq_len:]`

## Deviations from Plan

None - plan executed exactly as written. The plan does not specify exec_size for limit orders explicitly; used same `order_sizes` fractions as the corresponding market action level (0/1/2), which is the most consistent interpretation.

## Issues Encountered
- `black --check` flagged one line for reformatting (multi-line ValueError message) — auto-fixed by running `black`. No logic change.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- LOBExecutionEnv is ready for Phase 09 DQN agent training via `from lob_forge.executor.environment import LOBExecutionEnv`
- `env.reset(seed=N)` provides reproducible episodes; `env.step(action)` returns full gymnasium tuple
- Config accessible via `cfg.executor.seq_len`, `cfg.executor.cost_model.fee_bps`, etc.
- No blockers

---
*Phase: 08-execution-environment*
*Completed: 2026-03-22*
