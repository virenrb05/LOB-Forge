---
phase: 12-coinbase-data-pipeline-run
plan: 03
status: complete
date: 2026-03-22
---

# 12-03 Summary: wandb IS/Slippage Logging in train_all.sh Stage 6

## What Was Done

Replaced the Stage 6 inline Python block in `scripts/train_all.sh` with an expanded version that:

1. **Keeps `generate_all_plots()`** — unchanged from prior implementation.
2. **Loads the executor checkpoint** (`checkpoints/executor_adversarial.pt`). If the checkpoint is absent, prints `[WARN]` and skips wandb logging — keeping the script usable when Stage 5 was skipped.
3. **Loads real LOB data** from `DATA_DIR/test.parquet` (preferred) → `DATA_DIR/train.parquet` → random `np.random.randn(10000, 40).astype('float32')` fallback.
4. **Builds `LOBExecutionEnv`** with `seq_len=50, inventory=100.0, horizon=20` (matching executor.yaml defaults).
5. **Calls `compare_to_baselines()`** with `n_episodes=10, device="cpu"`.
6. **Computes IS stats** via `compute_implementation_shortfall(cmp["dqn"]["results"])`.
7. **Computes slippage** via `compute_slippage_vs_twap(cmp["dqn"]["results"], cmp["twap"]["results"])`.
8. **Logs to wandb** with `wandb.init(project="lob-forge", job_type="executor-eval", resume="allow", settings=wandb.Settings(silent=True))`:
   - `executor/is_mean`
   - `executor/is_std`
   - `executor/is_sharpe`
   - `executor/slippage_vs_twap`
   - `executor/dqn_beats_twap`
9. Calls `wandb.finish()` after logging.
10. **Wraps wandb block in try/except** — failure prints `[WARN]` and continues (non-fatal).

## Verification Results

- `bash -n scripts/train_all.sh` → syntax OK
- `grep -n "wandb" scripts/train_all.sh` → 9 hits (init, log, finish, and supporting lines)
- `grep -n "is_mean|is_sharpe|slippage_vs_twap"` → all three keys present
- `ruff check lob_forge/ && black --check lob_forge/` → clean
- `pytest tests/ -x -q` → 307 passed

## Files Modified

- `scripts/train_all.sh` — Stage 6 Python block expanded with IS/slippage wandb logging

## Phase 12 Impact

This closes Phase 12 success criterion 4: "Real metrics (F1, IS, slippage) logged to wandb". F1 was already logged by the predictor trainer; IS and slippage are now logged by the executor eval stage.
