# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-19)

**Core value:** The three-component pipeline works end-to-end: transformer embeddings condition the diffusion model, which generates unlimited training environments for the RL agent that beats TWAP on real data.
**Current focus:** Phase 6 in progress — generator core building blocks

## Current Position

Phase: 6 of 10 (Generator Core) — IN PROGRESS
Plan: 06-02 complete (conditioning embeddings + AdaLN blocks)
Status: Plan 06-02 done, ready for plan 06-03
Last activity: 2026-03-20 — Conditioning module, AdaLN, ResBlock1D

Progress: █████▌░░░░ 55%

## Performance Metrics

**Velocity:**
- Total plans completed: 21
- Average duration: ~2.9 min
- Total execution time: ~61 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-scaffold | 2/2 | ~10 min | ~5 min |
| 02-data-ingestion | 3/3 | ~8 min | ~2.7 min |
| 03-data-preprocessing | 7/7 | ~22 min | ~3.1 min |
| 04-predictor-architecture | 4/4 | ~8 min | ~2 min |
| 05-predictor-training | 3/3 | ~9 min | ~3 min |
| 06-generator-core | 2/? | ~4 min | ~2 min |

**Recent Trend:**
- Last 5 plans: 05-02, 05-03, 06-01, 06-02
- Trend: Steady

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Added `[tool.setuptools.packages.find]` include directive to pyproject.toml (setuptools auto-discovery failed with .planning/ directory present)
- Used `/data/` (root-anchored) in .gitignore to avoid ignoring `lob_forge/data/`
- Sub-config YAML files use `# @package _global_` with nested keys to avoid key collisions in Hydra flat config structure
- LOB schema has 46 columns (not 45): plan enumeration yields 3 header + 40 book + 3 trade = 46
- Trade columns (trade_price, trade_size) allow NaN; book columns do not
- Validation returns list[str] of issues rather than raising exceptions
- LOBSTER prices divided by 10000 (stored as integer cents x 100)
- Trade events from LOBSTER message file: event_types 4, 5 are executions
- Historical Bybit archives provide trade data only; book columns NaN with warning
- WebSocket recorder uses async internally, sync API via asyncio.run()
- temporal_split returns index arrays (not data slices); empty np.ndarray for segments that don't fit
- purge_gap defaults to 0 in function; configs/data.yaml sets 10 for production
- Cumsum trick for O(n) future mean: cs = concat([0], cumsum(mid)), future_mean = (cs[t+h+1] - cs[t+1]) / h
- Causality boundary: label at row t uses mid[t+1..t+h], so modifying row k affects labels at rows [k-h, k-1]
- Label dtype float64 to support NaN; values in {0.0, 1.0, 2.0, NaN}
- Feature functions use epsilon (1e-12) denominator guard for division-by-zero protection
- VPIN uses bulk volume classification with scipy.stats.norm.cdf and rolling sigma window of 50
- VPIN forward-fills for non-trade rows, clips to [0, 1]
- _get() helper in preprocessor supports both dict and OmegaConf attribute-style access
- NaN labels mapped to 0 in LOBDataset; real training should mask NaN rows
- Regime labels from realized vol quantiles (0.33, 0.67): low-vol(0), normal(1), high-vol(2)
- rolling_zscore tolerance for zero-mean: abs(mean) < 0.15; unit-variance: std in [0.5, 2.0] (accounts for rolling window statistical variance)
- OFI/MLOFI are additive flow-based features; existing static imbalance functions preserved
- compute_all_features produces 20 columns (was 18) after adding ofi and mlofi
- Attention blocks use Pre-LN (norm_first=True) with GELU per TLOB paper
- Boolean causal mask registered as buffer, sliced to actual T in forward for efficiency
- Attention blocks accept pre-reshaped tensors; caller handles 4D-to-3D reshaping
- DeepLOB uses stride-2 Conv2d for spatial reduction (not pooling)
- LinearBaseline uses only last time step — no sequence processing, no hidden layers
- Both baselines use nn.ModuleList for per-horizon heads (matching TLOB interface)
- FocalLoss from scratch: class_weights as buffer, gamma=0 degrades to CE, supports 2D/3D logits
- DualAttentionTransformer input reshape: .view(B, T, 4, 10).permute(0, 1, 3, 2) for per-level grouping
- Model forward returns dict[str, Tensor] with logits, embedding, and optional vpin
- Optional heads gated by constructor bool, absent from output dict when disabled
- predictor.yaml completed with features_per_level=4, n_horizons=4, max_seq_len=512
- __init__.py re-exports all 6 public classes (DualAttentionTransformer, DeepLOB, LinearBaseline, FocalLoss, SpatialAttentionBlock, TemporalAttentionBlock)
- LOBDataset returns 2-tuple (features, labels) when vpin_col=None, 3-tuple when set — backward compatible
- Trainer passes explicit 40 book columns (BOOK_FEATURE_COLS) to LOBDataset to avoid derived features
- Model output: dict for DualAttentionTransformer, plain Tensor for baselines; _extract_logits() handles both
- Class weights computed as inverse frequency; pin_memory=False for MPS/CPU
- AdaLN uses (1+scale) modulation for identity-initialised conditioning injection
- GroupNorm with min(32, channels) groups in AdaLN for flexible channel counts
- Xavier-uniform init on Conv1d weights, zero biases in ResBlock1D for stable training

### Pending Todos

None yet.

### Blockers/Concerns

- Bybit REST API returns 403 from some network locations (geo-restriction); does not affect code correctness

## Session Continuity

Last session: 2026-03-20
Stopped at: Plan 06-02 complete — conditioning embeddings and AdaLN blocks implemented
Resume file: .planning/phases/06-generator-core/06-02-SUMMARY.md
