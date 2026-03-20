---
phase: 06-generator-core
plan: 01
subsystem: generator
tags: [diffusion, cosine-schedule, ema, pytorch]

# Dependency graph
requires:
  - phase: 01-scaffold
    provides: project structure and lob_forge package
provides:
  - CosineNoiseSchedule with q_sample and all diffusion coefficients as buffers
  - ExponentialMovingAverage wrapper with update/apply/restore/serialization
affects: [06-generator-core]

# Tech tracking
tech-stack:
  added: []
  patterns: [register_buffer for device-portable schedule tensors, float64 intermediate precision]

key-files:
  created:
    - lob_forge/generator/noise_schedule.py
    - lob_forge/generator/ema.py
  modified: []

key-decisions:
  - "Compute cosine schedule in float64, store buffers as float32 for numerical precision"
  - "EMA is a plain class (not nn.Module) — shadow params stored as dict, not registered buffers"

patterns-established:
  - "Schedule coefficients as registered buffers: all tensors move with .to(device) automatically"
  - "_extract() pattern: gather schedule values at timestep indices and reshape for broadcasting"

# Metrics
duration: 2min
completed: 2026-03-20
---

# Phase 06-01: Noise Schedule & EMA Utilities

**Cosine noise schedule (Nichol & Dhariwal 2021) and EMA wrapper for diffusion generator inference stability**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-20T23:43:00Z
- **Completed:** 2026-03-20T23:44:38Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- CosineNoiseSchedule with 10 registered buffers and monotonically decreasing alpha_bar from ~1 to ~0
- Forward diffusion q_sample producing correctly shaped noised data at any timestep
- ExponentialMovingAverage with update, apply_shadow, restore, and state_dict serialization

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement cosine noise schedule** - `70e732f` (feat)
2. **Task 2: Implement EMA wrapper** - `abc19c4` (feat)

## Files Created/Modified
- `lob_forge/generator/noise_schedule.py` - CosineNoiseSchedule (129 lines): cosine schedule, q_sample, _extract helper
- `lob_forge/generator/ema.py` - ExponentialMovingAverage (73 lines): shadow param tracking with save/load

## Decisions Made
- Computed cosine schedule intermediates in float64 for numerical precision, then stored as float32 buffers
- EMA implemented as plain class (not nn.Module) per plan specification -- simpler, no buffer registration overhead

## Deviations from Plan

None - plan executed exactly as written

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Noise schedule and EMA utilities ready for use by diffusion model training loop
- Both modules pass all verification checks and lint cleanly

---
*Phase: 06-generator-core*
*Completed: 2026-03-20*
