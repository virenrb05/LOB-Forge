---
phase: 08-execution-environment
plan: 03
subsystem: executor
tags: [gymnasium, rl-environment, lob, synthetic-mode, diffusion-model, curriculum-learning, public-api]

# Dependency graph
requires:
  - phase: 08-02
    provides: LOBExecutionEnv base implementation passing check_env()
  - phase: 06-generator-core
    provides: DiffusionModel.generate() for synthetic LOB generation

provides:
  - LOBExecutionEnv with mode="synthetic" using DiffusionModel.generate() on reset()
  - ValueError guard when mode="synthetic" and generator=None
  - lob_forge.executor public API: LOBExecutionEnv, CostModel, ACTION_NAMES

affects:
  - 09-dqn-agent (can now use synthetic curriculum via regime param)
  - Any downstream consumer importing from lob_forge.executor

# Tech tracking
tech-stack:
  added: []
  patterns:
    - TYPE_CHECKING guard for DiffusionModel type annotation (no circular import at runtime)
    - Lazy torch import inside reset() for synthetic mode (keeps module torch-free)
    - Duck-typed generator at runtime (Any); strongly typed under TYPE_CHECKING only
    - Module-level ACTION_NAMES alias sourced from class attribute for clean API surface

key-files:
  modified:
    - lob_forge/executor/environment.py
    - lob_forge/executor/__init__.py

key-decisions:
  - "TYPE_CHECKING guard for DiffusionModel import — prevents heavy torch import at module load time; generator duck-typed as Any at runtime"
  - "Lazy `import torch` inside reset() synthetic branch — env module stays importable without torch on PATH"
  - "ACTION_NAMES exposed as module-level alias (LOBExecutionEnv.ACTION_NAMES) in __init__.py — avoids adding a standalone module-level list to environment.py"
  - "Synthetic reset() always starts at self._start = self.seq_len (no randomization) — generator produces exactly the needed length; no need for episode-start sampling"
  - "Do NOT export agent.py or train.py yet — those are Phase 9"

# Metrics
duration: 8min
completed: 2026-03-22
---

# Phase 08-03: Synthetic Mode + Public API Summary

**LOBExecutionEnv extended with synthetic mode (DiffusionModel.generate() on reset) and executor public API finalized with ACTION_NAMES**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-03-22
- **Completed:** 2026-03-22
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Extended `LOBExecutionEnv.__init__()` with four new params: `mode`, `generator`, `regime`, `device`
- `reset()` branches on `mode`: synthetic path generates a fresh `(n_steps, 40)` LOB array via `DiffusionModel.generate()` on every call; real path unchanged
- `ValueError("generator required for synthetic mode")` raised at construction time when `mode="synthetic"` and `generator=None`
- `TYPE_CHECKING` guard for `DiffusionModel` annotation; lazy `import torch` inside `reset()` keeps module import lightweight and torch-free
- Updated `lob_forge/executor/__init__.py` to export `ACTION_NAMES`, `CostModel`, `LOBExecutionEnv` as the complete public API
- All verification checks pass: `check_env()`, ruff, black, no circular imports

## Task Commits

Each task was committed atomically:

1. **Task 1: Synthetic mode** — `d8511f2` (feat(08-03))
2. **Task 2: Public API exports** — `7ca9ee4` (feat(08-03))

## Files Modified

- `lob_forge/executor/environment.py` — synthetic mode logic in `__init__` and `reset()`, TYPE_CHECKING guard, docstring updated (279 lines)
- `lob_forge/executor/__init__.py` — ACTION_NAMES module-level alias, updated `__all__`

## Decisions Made

- `TYPE_CHECKING` guard for `DiffusionModel` import — prevents heavy torch dependency at module load time; generator is duck-typed (`Any`) at runtime
- Lazy `import torch` inside `reset()` synthetic branch — executor module stays importable in environments without torch
- `ACTION_NAMES` exposed as a module-level alias from the class attribute rather than adding a standalone list to `environment.py` — single source of truth
- Synthetic `reset()` sets `self._start = self.seq_len` with no randomization; the generator produces exactly `horizon + seq_len + 10` rows so deterministic start is safe
- `agent.py` and `train.py` deliberately excluded from public API — Phase 9 concern

## Deviations from Plan

None — plan executed exactly as written. `ACTION_NAMES` was a class attribute (not a module-level name) in the existing implementation, so the `__init__.py` import was adapted to use a module-level alias `ACTION_NAMES = LOBExecutionEnv.ACTION_NAMES` rather than a direct `from ... import ACTION_NAMES`.

## Issues Encountered

- Initial `__init__.py` import `from lob_forge.executor.environment import ACTION_NAMES` failed because `ACTION_NAMES` is a class attribute, not a module-level name. Resolved by adding `ACTION_NAMES = LOBExecutionEnv.ACTION_NAMES` in `__init__.py`.

## Next Phase Readiness

- Phase 09 (DQN agent) can import `from lob_forge.executor import LOBExecutionEnv, ACTION_NAMES` and use either `mode="real"` or `mode="synthetic"` with any trained `DiffusionModel`
- Regime-conditioned curriculum: instantiate with `mode="synthetic", generator=model, regime=2` for high-vol adversarial episodes
- No blockers

---
*Phase: 08-execution-environment*
*Completed: 2026-03-22*
