---
phase: 04-predictor-architecture
plan: 01
subsystem: predictor
tags: [pytorch, transformer, attention, causal-mask, tlob]

# Dependency graph
requires:
  - phase: 03-data-preprocessing
    provides: feature pipeline producing per-level feature vectors
provides:
  - SpatialAttentionBlock nn.Module for cross-level attention
  - TemporalAttentionBlock nn.Module with causal masking for time-step attention
affects: [04-predictor-architecture]

# Tech tracking
tech-stack:
  added: []
  patterns: [Pre-LN TransformerEncoder, boolean causal mask as buffer]

key-files:
  created:
    - lob_forge/predictor/spatial_attention.py
    - lob_forge/predictor/temporal_attention.py
  modified: []

key-decisions:
  - "Boolean upper-triangular causal mask registered as buffer and sliced in forward for efficiency"
  - "Pre-LN (norm_first=True) with GELU activation following TLOB paper"

patterns-established:
  - "Attention blocks accept reshaped tensors — caller handles 4D→3D reshaping"
  - "Causal mask pre-computed at max_seq_len and sliced to actual T in forward"

# Metrics
duration: 2min
completed: 2026-03-20
---

# Phase 4, Plan 01: Attention Blocks Summary

**Spatial and temporal attention building blocks using Pre-LN TransformerEncoder with GELU and boolean causal masking**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-20T18:45:10Z
- **Completed:** 2026-03-20T18:47:25Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- SpatialAttentionBlock wraps nn.TransformerEncoder for cross-level LOB attention per time step
- TemporalAttentionBlock applies causal self-attention across time steps with pre-computed boolean mask
- Causal masking verified: modifying future inputs does not affect past outputs

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement SpatialAttentionBlock** - `75feeb5` (feat)
2. **Task 2: Implement TemporalAttentionBlock** - `657b1c2` (feat)

## Files Created/Modified
- `lob_forge/predictor/spatial_attention.py` - SpatialAttentionBlock nn.Module (cross-level attention)
- `lob_forge/predictor/temporal_attention.py` - TemporalAttentionBlock nn.Module (causal temporal attention)

## Decisions Made
- Boolean upper-triangular mask registered as buffer, sliced to actual sequence length in forward — avoids recomputing each call
- Pre-LN (norm_first=True) chosen per TLOB paper for stable training without learning rate sensitivity

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Removed unused torch import in spatial_attention.py**
- **Found during:** Task 1 (SpatialAttentionBlock)
- **Issue:** `import torch` unused — ruff flagged F401
- **Fix:** Removed the unused import line
- **Files modified:** lob_forge/predictor/spatial_attention.py
- **Verification:** `ruff check` passes
- **Committed in:** 75feeb5 (part of task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Lint fix only. No scope creep.

## Issues Encountered
- Causal masking verification fails in training mode due to dropout non-determinism; verified correctly in eval mode with torch.no_grad()

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Both attention blocks ready for composition in DualAttentionTransformer (plan 04-02 or later)
- Blocks follow standalone nn.Module pattern — caller handles 4D tensor reshaping

---
*Phase: 04-predictor-architecture*
*Completed: 2026-03-20*
