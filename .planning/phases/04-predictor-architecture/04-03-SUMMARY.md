---
phase: 04-predictor-architecture
plan: 03
subsystem: predictor
tags: [pytorch, transformer, focal-loss, attention, tlob]

# Dependency graph
requires:
  - phase: 04-01
    provides: SpatialAttentionBlock and TemporalAttentionBlock building blocks
provides:
  - FocalLoss with class weights and configurable gamma
  - DualAttentionTransformer composing spatial + temporal attention with per-horizon heads
  - VPIN regression head (optional)
  - Embedding output for downstream generator conditioning
affects: [04-predictor-architecture, 05-training-loop, 06-diffusion-generator]

# Tech tracking
tech-stack:
  added: []
  patterns: [dual-attention-composition, focal-loss-imbalance, per-horizon-heads]

key-files:
  created:
    - lob_forge/predictor/losses.py
  modified:
    - lob_forge/predictor/model.py

key-decisions:
  - "FocalLoss implemented from scratch (no external library) for full control"
  - "class_weights registered as buffer (not parameter) for automatic device movement"
  - "3D logits flattened before loss computation, reshaped back for 'none' reduction"
  - "Input reshape via .view(B, T, 4, 10).permute(0, 1, 3, 2) to group per-level features"
  - "Forward returns dict with logits, embedding, and optional vpin for flexible downstream use"

patterns-established:
  - "Model forward returns dict[str, Tensor] for multi-output architecture"
  - "Optional heads gated by constructor bool, absent from output dict when disabled"

# Metrics
duration: 2min
completed: 2026-03-20
---

# Phase 04, Plan 03: DualAttentionTransformer & FocalLoss Summary

**FocalLoss with gamma-controlled focusing and DualAttentionTransformer (TLOB) composing spatial/temporal attention with per-horizon classification heads and optional VPIN regression**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-20T18:50:12Z
- **Completed:** 2026-03-20T18:52:24Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- FocalLoss from scratch: gamma=0 degrades to cross-entropy, supports 2D/3D logits, class weight buffers
- DualAttentionTransformer: input embedding with level/temporal PE, spatial attention, level pooling, causal temporal attention, per-horizon classification heads
- Optional VPIN regression head with sigmoid activation producing [0, 1] output
- Embedding output exposed for downstream generator conditioning (Phase 6)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement FocalLoss** - `90b4775` (feat)
2. **Task 2: Implement DualAttentionTransformer** - `295dd2a` (feat)

**Plan metadata:** pending (docs: complete plan)

## Files Created/Modified
- `lob_forge/predictor/losses.py` - FocalLoss with gamma, class weights, 2D/3D support
- `lob_forge/predictor/model.py` - DualAttentionTransformer (TLOB) with 187,789 parameters

## Decisions Made
- FocalLoss from scratch for full control over the computation
- class_weights as buffer for automatic .to(device) movement
- 3D logits flatten/reshape approach for multi-horizon focal loss
- Input reshape: 4 groups of 10 permuted to 10 levels of 4 features
- Dict-based forward output for flexible multi-head architecture

## Deviations from Plan

None - plan executed exactly as written

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- DualAttentionTransformer ready for training loop integration (Phase 5)
- FocalLoss ready as training objective with class-imbalance handling
- Embedding output ready for generator conditioning (Phase 6)

---
*Phase: 04-predictor-architecture*
*Completed: 2026-03-20*
