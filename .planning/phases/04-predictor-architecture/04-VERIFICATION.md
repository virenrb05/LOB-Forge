---
status: passed
date: 2026-03-20
phase_goal: "TLOB and baseline models forward-pass correctly on LOB data"
test_command: "python -m pytest tests/test_predictor.py -v"
test_result: "31 passed, 0 failed (2.28s)"
---

# Phase 04 Verification: Predictor Architecture

## Summary

All 5 must-have criteria **PASS**. The predictor module implements three models
(TLOB, DeepLOB, LinearBaseline) and focal loss, all verified by 31 passing tests
including MPS device tests and determinism checks.

---

## Criterion 1: TLOB dual-attention transformer produces 3-class logits from LOB input tensors

**PASS**

- `lob_forge/predictor/model.py` — `DualAttentionTransformer` implements:
  - Input projection from `(B, T, 40)` to per-level embeddings
  - `SpatialAttentionBlock` (cross-level attention per time step)
  - Mean pooling across levels
  - `TemporalAttentionBlock` (causal self-attention across time steps)
  - Per-horizon classification heads outputting `(B, n_horizons, 3)` logits
  - Optional VPIN regression head outputting `(B, 1)`
- Tests `TestDualAttentionTransformer::test_output_shapes` confirms logits shape `(4, 4, 3)`.
- Causal masking verified: `TestTemporalAttentionBlock::test_causal_masking` confirms
  modifying future inputs does not change past outputs.

## Criterion 2: DeepLOB (CNN + Inception + LSTM) produces matching output shape

**PASS**

- `lob_forge/predictor/deeplob.py` — `DeepLOB` implements:
  - Two conv blocks with stride-2 spatial reduction
  - Inception module (3 parallel branches: 1x1, 3x1, 5x1 kernels)
  - Spatial collapse via `(1, n_levels)` convolution
  - Single-layer LSTM with last-timestep extraction
  - Per-horizon linear heads
- Output shape `(B, n_horizons, n_classes)` matches TLOB.
- Tests `TestDeepLOB::test_output_shape` confirms `(4, 4, 3)`.

## Criterion 3: Linear/logistic baseline produces matching output shape

**PASS**

- `lob_forge/predictor/linear_baseline.py` — `LinearBaseline` implements:
  - Last-timestep extraction from `(B, T, 40)` to `(B, 40)`
  - Per-horizon linear projections to `(B, n_horizons, n_classes)`
- `TestLinearBaseline::test_uses_last_timestep` confirms only last timestep is used.
- `TestModelConsistency::test_all_models_same_output_shape` confirms all three models
  produce identical output shape `(4, 4, 3)` on the same input.

## Criterion 4: Focal loss computes correctly with class weights (verified against manual calculation)

**PASS**

- `lob_forge/predictor/losses.py` — `FocalLoss` implements:
  - Log-softmax, gather true-class log-prob, compute pt, apply `(1 - pt)^gamma` weight
  - Optional per-class weights registered as buffer
  - 3D logit support (reshapes `(B, H, C)` to `(B*H, C)`)
  - `"mean"`, `"sum"`, `"none"` reduction modes
- Test `TestFocalLoss::test_manual_calculation`: hand-computes focal loss for specific
  logits `[[2.0, 1.0, 0.5], [0.0, 3.0, 1.0]]` with targets `[0, 2]` at `gamma=2.0`,
  verifies match within `1e-5`.
- Test `TestFocalLoss::test_gamma_zero_matches_ce`: confirms `gamma=0` degrades to
  standard cross-entropy (verified against `F.cross_entropy`).
- Test `TestFocalLoss::test_class_weights`: confirms class weights change loss values.

## Criterion 5: All models train deterministically with fixed seeds on MPS

**PASS**

- All 4 MPS tests pass (`test_tlob_on_mps`, `test_deeplob_on_mps`,
  `test_linear_on_mps`, `test_focal_loss_on_mps`) confirming forward pass on MPS device.
- `TestDualAttentionTransformer::test_deterministic_with_seed` confirms same seed
  produces identical logits and embeddings.
- Manual MPS determinism verification: all three models initialized with `manual_seed(42)`
  produce bit-identical outputs on MPS device.
- Training step (forward + focal loss + backward + optimizer step) completes
  successfully on MPS.

---

## Files Verified

| File | Lines | Role |
|------|-------|------|
| `lob_forge/predictor/model.py` | 188 | TLOB DualAttentionTransformer |
| `lob_forge/predictor/spatial_attention.py` | 83 | SpatialAttentionBlock |
| `lob_forge/predictor/temporal_attention.py` | 100 | TemporalAttentionBlock with causal mask |
| `lob_forge/predictor/deeplob.py` | 125 | DeepLOB CNN+Inception+LSTM baseline |
| `lob_forge/predictor/linear_baseline.py` | 70 | LinearBaseline logistic regression |
| `lob_forge/predictor/losses.py` | 107 | FocalLoss with class weights |
| `lob_forge/predictor/__init__.py` | 17 | Public API exports |
| `tests/test_predictor.py` | 337 | 31 tests covering all criteria |

## Gaps

None.
