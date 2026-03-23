#!/usr/bin/env bash
# train_all.sh — Reproduce LOB-Forge results end-to-end
#
# Usage:
#   bash scripts/train_all.sh
#   bash scripts/train_all.sh --skip-data
#   bash scripts/train_all.sh --device=cuda
#   DEVICE=cuda bash scripts/train_all.sh
#
# Requires:
#   pip install -e ".[dev]"
#
# Stages:
#   0 - Validate environment
#   1 - Data ingestion (Coinbase BTC-USD LOB recorder)
#   2 - Preprocessing (features, labels, datasets)
#   3 - Train predictor (DualAttentionTransformer)
#   4 - Train generator (DDPM/DDIM diffusion model)
#   5 - Train execution agent (Dueling DQN)
#   6 - Evaluate and generate plots

set -euo pipefail

# ---------------------------------------------------------------------------
# Config defaults (environment variables take precedence)
# ---------------------------------------------------------------------------
DEVICE=${DEVICE:-mps}     # Override: DEVICE=cuda bash scripts/train_all.sh
DATA_DIR=${DATA_DIR:-data}
RECORD_DURATION=${RECORD_DURATION:-300}  # Override: RECORD_DURATION=60 bash scripts/train_all.sh

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
SKIP_DATA=false

for arg in "$@"; do
  case $arg in
    --skip-data)
      SKIP_DATA=true
      ;;
    --device=*)
      DEVICE="${arg#*=}"
      ;;
    --help|-h)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log_stage() {
  echo ""
  echo "=================================================================="
  echo "  $1"
  echo "=================================================================="
}

# ---------------------------------------------------------------------------
# Stage 0: Validate environment
# ---------------------------------------------------------------------------
log_stage "Stage 0: Environment Validation"

python -c "import lob_forge; print('lob_forge version:', lob_forge.__version__)"
python scripts/validate_mps.py || echo "[WARN] MPS validation skipped (non-fatal)"

echo "Device: $DEVICE"
echo "Data directory: $DATA_DIR"

# ---------------------------------------------------------------------------
# Stage 1: Data ingestion
# ---------------------------------------------------------------------------
if [ "$SKIP_DATA" = false ]; then
  log_stage "Stage 1: Data Ingestion (Coinbase BTC-USD)"

  # Record 5 minutes of live LOB data from Coinbase public API (no auth required)
  if python -m lob_forge.data.coinbase_downloader \
      --duration "$RECORD_DURATION" \
      --output "$DATA_DIR" \
      --symbol BTC-USD \
      --depth 10; then
    echo "[OK] Coinbase LOB recording complete (${RECORD_DURATION}s)"
  else
    echo "[WARN] Coinbase recording failed; continuing with existing data if present"
  fi
else
  echo "[SKIP] Data ingestion skipped (--skip-data)"
fi

# ---------------------------------------------------------------------------
# Stage 2: Preprocessing
# ---------------------------------------------------------------------------
log_stage "Stage 2: Preprocessing"

python -m lob_forge.data.preprocessor --config-name data

# ---------------------------------------------------------------------------
# Stage 3: Train predictor
# ---------------------------------------------------------------------------
log_stage "Stage 3: Predictor Training (DualAttentionTransformer)"

python -m lob_forge.train --config-name predictor +trainer=predictor +device="$DEVICE"

# ---------------------------------------------------------------------------
# Stage 4: Train generator
# ---------------------------------------------------------------------------
log_stage "Stage 4: Generator Training (DDPM/DDIM)"

python -m lob_forge.train --config-name generator +trainer=generator +device="$DEVICE"

# ---------------------------------------------------------------------------
# Stage 5: Train execution agent
# ---------------------------------------------------------------------------
log_stage "Stage 5: Execution Agent Training (Dueling DQN)"

python - <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] if "__file__" in dir() else Path.cwd()))
import os
from hydra import compose, initialize_config_dir
from lob_forge.executor.train import train_agent, STAGE_CONFIG

config_dir = str(Path.cwd() / "configs")
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name="executor")

if os.environ.get("SMOKE_TEST"):
    for stage in STAGE_CONFIG:
        STAGE_CONFIG[stage]["steps"] = 500

train_agent(cfg)
PYEOF

# ---------------------------------------------------------------------------
# Stage 6: Evaluation
# ---------------------------------------------------------------------------
log_stage "Stage 6: Evaluation and Plot Generation"

python - "$DATA_DIR" <<'PYEOF'
import sys
import os
from pathlib import Path

import numpy as np

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")

from lob_forge.executor import compare_to_baselines
from lob_forge.evaluation import generate_all_plots
from lob_forge.evaluation.metrics import (
    compute_implementation_shortfall,
    compute_slippage_vs_twap,
)

# ------------------------------------------------------------------
# Plot generation
# ------------------------------------------------------------------
plots = generate_all_plots()
print(f"Generated {len(plots)} plots:")
for p in plots:
    print(f"  {p}")

# ------------------------------------------------------------------
# IS / slippage logging to wandb
# ------------------------------------------------------------------
ckpt_path = Path("checkpoints/executor_adversarial.pt")
if not ckpt_path.exists():
    print(f"[WARN] Checkpoint not found: {ckpt_path} — skipping wandb IS/slippage logging")
else:
    # Load LOB data for eval env (test > train > dummy fallback)
    lob_data = None
    for candidate in ["test.parquet", "train.parquet"]:
        parquet_path = data_dir / candidate
        if parquet_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(parquet_path)
                book_cols = [c for c in df.columns if c not in (
                    "timestamp", "mid_price", "spread",
                    "trade_price", "trade_size", "trade_side",
                )][:40]
                if len(book_cols) >= 40:
                    lob_data = df[book_cols[:40]].values.astype("float32")
                    print(f"[INFO] Loaded LOB data from {parquet_path}: {lob_data.shape}")
                    break
            except Exception as exc:
                print(f"[WARN] Could not load {parquet_path}: {exc}")

    if lob_data is None:
        print("[WARN] No parquet data found — using random fallback array for eval env")
        lob_data = np.random.randn(10000, 40).astype("float32")

    # Build evaluation environment
    from lob_forge.executor.environment import LOBExecutionEnv

    eval_env = LOBExecutionEnv(
        lob_data=lob_data,
        seq_len=50,
        inventory=100.0,
        horizon=20,
    )

    # Run comparison
    cmp = compare_to_baselines(ckpt_path, eval_env, n_episodes=10, device="cpu")

    # Compute IS stats and slippage
    is_stats = compute_implementation_shortfall(cmp["dqn"]["results"])
    slippage = compute_slippage_vs_twap(cmp["dqn"]["results"], cmp["twap"]["results"])
    dqn_beats_twap = bool(cmp["dqn_beats_twap"])

    # Log to wandb (non-fatal on failure)
    try:
        import wandb

        run = wandb.init(
            project="lob-forge",
            job_type="executor-eval",
            resume="allow",
            settings=wandb.Settings(silent=True),
        )
        wandb.log(
            {
                "executor/is_mean": is_stats["is_mean"],
                "executor/is_std": is_stats["is_std"],
                "executor/is_sharpe": is_stats["is_sharpe"],
                "executor/slippage_vs_twap": slippage,
                "executor/dqn_beats_twap": dqn_beats_twap,
            }
        )
        wandb.finish()
        print("[OK] IS/slippage metrics logged to wandb (executor-eval)")
    except Exception as exc:
        print(f"[WARN] wandb logging failed (non-fatal): {exc}")

print("")
print("Evaluation complete.")
print("For full quantitative results, open the notebooks:")
print("  notebooks/02_predictor_results.ipynb")
print("  notebooks/03_generator_quality.ipynb")
print("  notebooks/04_execution_backtest.ipynb")
PYEOF

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "  LOB-Forge Pipeline Complete"
echo "=================================================================="
echo ""
echo "Outputs:"
echo "  Plots       -> outputs/plots/"
echo "  Checkpoints -> checkpoints/"
echo "  Notebooks   -> notebooks/"
echo ""
