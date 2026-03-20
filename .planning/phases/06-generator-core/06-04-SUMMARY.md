---
phase: 06-generator-core
plan: 04
subsystem: generator
tags: [diffusion, ddpm, ddim, sampling, pytorch]

# Dependency graph
requires:
  - phase: 06-01
    provides: CosineNoiseSchedule with q_sample and posterior coefficients
  - phase: 06-02
    provides: ConditioningModule combining timestep, regime, time-of-day embeddings
  - phase: 06-03
    provides: UNet1D encoder-decoder denoiser backbone
provides:
  - DiffusionModel composing schedule + UNet + conditioning
  - Training loss via forward diffusion + MSE noise prediction
  - DDPM full reverse sampling (1000 steps)
  - DDIM accelerated sampling (50 steps, deterministic with eta=0)
  - generate() convenience router for both methods
affects: [06-generator-core, 07-generator-training]

# Tech tracking
tech-stack:
  added: []
  patterns: [diffusion-model-composition, ddpm-reverse-process, ddim-accelerated-sampling]

key-files:
  created: [lob_forge/generator/model.py]
  modified: []

key-decisions:
  - "Both DDPM and DDIM implemented in single DiffusionModel class (not separate samplers)"
  - "DDIM uses linspace timestep subsequence for uniform spacing"
  - "generate() convenience method defaults to DDIM for faster inference"

patterns-established:
  - "DiffusionModel as top-level user-facing interface: training_loss() + generate()"
  - "Input (B,T,C) permuted to (B,C,T) for U-Net, permuted back on output"

# Metrics
duration: 3min
completed: 2026-03-20
---

# Phase 06-04: DiffusionModel Summary

**DiffusionModel with DDPM/DDIM sampling composing CosineNoiseSchedule + UNet1D + ConditioningModule**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-20
- **Completed:** 2026-03-20
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- DiffusionModel wraps schedule, U-Net, and conditioning into single trainable module
- training_loss computes MSE between predicted and actual noise via forward diffusion
- DDPM full reverse process generates (B, T, C) LOB sequences from pure noise
- DDIM accelerated sampling with configurable steps and deterministic mode (eta=0)
- generate() convenience method routes to either sampler

## Task Commits

Both tasks targeted the same file and class, implemented atomically:

1. **Task 1: DiffusionModel with training loss and DDPM sampling** - `6b102a1` (feat)
2. **Task 2: DDIM sampling** - included in `6b102a1` (same class, same file)

**Plan metadata:** see below

## Files Created/Modified
- `lob_forge/generator/model.py` - DiffusionModel with training_loss, ddpm_sample, ddim_sample, generate (379 lines)

## Decisions Made
- Both DDPM and DDIM implemented in same commit since they share the same class and file -- splitting would have required artificial separation
- DDIM timestep subsequence via torch.linspace for uniform spacing
- generate() defaults to DDIM for practical inference speed

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- DiffusionModel ready for training loop integration
- EMA wrapper (06-02) can wrap DiffusionModel for stable generation
- All generator core components complete: schedule, blocks, conditioning, UNet, model

---
*Phase: 06-generator-core*
*Completed: 2026-03-20*
