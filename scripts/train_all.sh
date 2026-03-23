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

python -m lob_forge.train --config-name predictor trainer=predictor device="$DEVICE"

# ---------------------------------------------------------------------------
# Stage 4: Train generator
# ---------------------------------------------------------------------------
log_stage "Stage 4: Generator Training (DDPM/DDIM)"

python -m lob_forge.train --config-name generator trainer=generator device="$DEVICE"

# ---------------------------------------------------------------------------
# Stage 5: Train execution agent
# ---------------------------------------------------------------------------
log_stage "Stage 5: Execution Agent Training (Dueling DQN)"

python -m lob_forge.executor.train --config-name executor device="$DEVICE"

# ---------------------------------------------------------------------------
# Stage 6: Evaluation
# ---------------------------------------------------------------------------
log_stage "Stage 6: Evaluation and Plot Generation"

python - <<'PYEOF'
from lob_forge.executor import compare_to_baselines
from lob_forge.evaluation import generate_all_plots

plots = generate_all_plots()
print(f"Generated {len(plots)} plots:")
for p in plots:
    print(f"  {p}")

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
