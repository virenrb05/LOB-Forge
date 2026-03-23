# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-19)

**Core value:** The three-component pipeline works end-to-end: transformer embeddings condition the diffusion model, which generates unlimited training environments for the RL agent that beats TWAP on real data.
**Current focus:** Phase 12 in progress — CoinbaseDownloader implemented and wired into pipeline (12-01 and 12-02 complete); full pipeline uses Coinbase BTC-USD data; 307 tests passing

## Current Position

Phase: 12 (Coinbase Data & Full Pipeline Run) — IN PROGRESS
Plan: 12-03 complete (wandb IS/slippage logging)
Status: 12-03 done — wandb IS/slippage logging added to train_all.sh Stage 6; executor/is_mean, executor/is_std, executor/is_sharpe, executor/slippage_vs_twap, executor/dqn_beats_twap logged; non-fatal on missing checkpoint or wandb failure; 307 tests pass
Last activity: 2026-03-22 — Phase 12-03 execution complete; Phase 12 success criterion 4 met (IS and slippage logged to wandb)

Progress: ██████████░ ~93% — Phase 12 in progress (plans 01-02 of N complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 37
- Average duration: ~3.0 min
- Total execution time: ~119 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-scaffold | 2/2 | ~10 min | ~5 min |
| 02-data-ingestion | 3/3 | ~8 min | ~2.7 min |
| 03-data-preprocessing | 7/7 | ~22 min | ~3.1 min |
| 04-predictor-architecture | 4/4 | ~8 min | ~2 min |
| 05-predictor-training | 3/3 | ~9 min | ~3 min |
| 06-generator-core | 5/5 | ~15 min | ~3.0 min |
| 07-generator-validation | 5/5 | ~15 min | ~3.0 min |
| 08-execution-environment | 3/3 | ~28 min | ~9.3 min |
| 09-execution-agent | 4/4 | ~29 min | ~7.3 min |
| 10-evaluation-polish | 4/4 | ~19 min | ~4.8 min |

**Recent Trend:**
- Last 5 plans: 09-04, 10-01, 10-02, 10-03, 10-04
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
- Cosine schedule computed in float64 intermediates, stored as float32 buffers for numerical precision
- EMA is plain class (not nn.Module): shadow params as dict, simpler than buffer registration
- UNet1D: ~40M params with default config (d_model=128, channel_mults=(1,2,4,4))
- Self-attention at levels 2,3 plus bottleneck sandwich for global context
- Skip connections stored per-ResBlock, popped in reverse during decode
- DiffusionModel composes schedule + UNet + conditioning; input (B,T,C) permuted to (B,C,T) for UNet, back on output
- DDIM uses torch.linspace timestep subsequence; eta=0 is deterministic
- generate() convenience defaults to DDIM for practical inference speed
- No early stopping for diffusion training — loss behavior differs from classification
- train_generator(cfg) returns Path to final checkpoint; uses DDIM 10 steps for periodic samples
- Generator __init__.py re-exports 9 public symbols (all classes + train_generator)
- Stylized fact test pattern: fn(real, synthetic, **kwargs) -> dict with passed key
- 40-col LOB layout for evaluation: ask_price(0-9), ask_size(10-19), bid_price(20-29), bid_size(30-39)
- Market impact volume proxy: sum of absolute bid-size changes across levels
- Book shape test combines ask+bid depth per level for KS comparison
- Mid-price for evaluation: (col 0 + col 20) / 2 = (ask_1 + bid_1) / 2
- KL divergence uses shared bin edges across regimes; epsilon 1e-10 for stability
- Regime separability threshold: mean_kl > 0.1
- Regime distinctness: KS p < 0.05 on returns for all pairs; fidelity: KS p > 0.05
- LOB-Bench lazy-imports torch/sklearn in train_discriminator to keep scipy-only functions lightweight
- run_lob_bench namespaces keys with / separator (wasserstein/wd_mean, discriminator/accuracy, etc.)
- LOB spread uses grouped layout: col 0 (ask_1) - col 20 (bid_1); depth from cols 10-19 + 30-39
- TYPE_CHECKING guard for matplotlib.figure import in stylized_facts.py
- evaluation __init__.py re-exports 16 public symbols from all 4 submodules
- validate_generator uses EMA weights from checkpoint when available (key: "ema_state_dict")
- Per-regime generation: n_samples // 3 per regime (0, 1, 2)
- CostModel as dataclass: fee_bps and impact_eta as constructor params; compute() returns float
- exec_size == 0.0 early-return avoids division-by-zero in participation_rate computation
- Plan 08-01 example fee numbers inconsistent with stated formula; formula implemented as written
- LOBExecutionEnv takes np.ndarray directly (caller loads parquet, passes 40-col array); env is pure numpy, no torch
- Limit orders use order_sizes fractions indexed by limit level (0/1/2) for exec_size — consistent with market order levels
- Bernoulli fill sampling uses self.np_random.random() for reproducible episodes via reset(seed=N)
- Observation space unbounded Box(-inf, inf) by design for z-score normalized LOB data; check_env warnings are informational only
- _get_obs() zero-pads at episode start rather than requiring seq_len rows of prior history
- LOBExecutionEnv mode="synthetic" generates fresh LOB data via DiffusionModel.generate() on every reset(); raises ValueError when generator=None
- TYPE_CHECKING guard for DiffusionModel import in environment.py; lazy import torch inside reset() synthetic branch (keeps module torch-free)
- ACTION_NAMES exposed as module-level alias in executor/__init__.py (LOBExecutionEnv.ACTION_NAMES) — single source of truth
- Synthetic reset() always starts at self._start = self.seq_len (no randomization); generator produces exactly horizon+seq_len+10 rows
- AlmgrenChriss action threshold uses 50% of TWAP rate (inventory/horizon/2); plan's 1% of inventory threshold was too high for default kappa≈0.003
- VWAPBaseline lazily recomputes sinusoidal volume schedule via _ensure_horizon() when env.horizon differs from constructor horizon
- RandomBaseline stores env reference via run_episode() override to stay environment-agnostic at construction time
- DuelingDQN flattens 3D input (B,seq_len,40) inside forward() — callers pass raw obs tensors without reshape
- PrioritizedReplayBuffer uses deque(maxlen=capacity) for O(1) capacity-eviction; numpy random.choice for priority-weighted sampling (sufficient for 100k buffer)
- PER beta annealed in sample() via self._beta = min(beta_end, beta + increment); priorities updated as |td_error|+1e-6
- DuelingDQN and PrioritizedReplayBuffer exported from lob_forge.executor.__all__ for Plan 09-03 import
- train_agent() uses mode="real" for all curriculum stages; regime label logged but doesn't change env behavior in real mode
- STAGE_CONFIG exposed as module-level dict so tests can override steps without mocking
- Dummy LOB data: np.random.randn(10_000, 40).astype(float32) when data_path is None
- Checkpoint format: {stage, online_net, target_net, optimizer, epsilon, step}; saved to checkpoints/executor_{stage}.pt
- Double-DQN: online net selects next_action=argmax(Q_online(s')), target net evaluates Q_target(s', next_action)
- _select_device() helper: MPS > CUDA > CPU
- evaluate_agent() seeds episodes 0..n_episodes-1; greedy policy epsilon=0 with torch.no_grad()
- compare_to_baselines() returns dict with 5 agent entries + dqn_beats_twap bool (strict < on mean_cost)
- VWAPBaseline instantiated with horizon=env.horizon to pre-compute correct schedule
- ACTION_NAMES still aliased as LOBExecutionEnv.ACTION_NAMES in __init__.py (not module-level in environment.py)
- train.py dispatch uses OmegaConf.select(cfg, "trainer", default="predictor") — trainer is a CLI-only override key, never in config.yaml; direct cfg.trainer raises KeyError in struct mode
- train_generator imported lazily inside trainer=generator branch to avoid unconditional diffusion/torch imports on predictor training path
- notebooks/ excluded from ruff via pyproject.toml [tool.ruff] exclude directive — E402 in notebook cells is a false positive (mid-notebook imports are idiomatic)
- CoinbaseDownloader REST_BASE uses api.exchange.coinbase.com (public, no auth); api.coinbase.com/v3/brokerage requires JWT — Exchange API returns identical [price, size, num_orders] format with "time" field
- CoinbaseDownloader WS uses wss://advanced-trade-ws.coinbase.com with max_size=10MB (default 1MB limit causes 1009 on large snapshots); protocol: channel="level2", messages have channel="l2_data", events[].type in ("snapshot","update"), updates[].side in ("bid","offer")
- compute_implementation_shortfall sets slippage_vs_twap=NaN; caller fills via compute_slippage_vs_twap(agent, twap)
- run_backtest lazy-imports torch only for DQN checkpoint branch; delegates to evaluate_agent() when seed_offset==0
- training_loss_curve plot emits a placeholder when checkpoints/training_log.csv is absent (safe for CI)
- generate_all_plots creates output_dir with mkdir(parents=True, exist_ok=True); returns list[Path] of 6 PNG files
- matplotlib.use("Agg") set at module level in plots.py for non-interactive/CI-safe rendering
- IS Sharpe uses ddof=0 (population std) in metrics.py; test values for [10,20,30] are std≈8.165, sharpe≈2.449
- MockEnv pattern for executor tests: minimal class with action_space, seq_len, horizon, inventory; terminates after first step
- test_eval_metrics.py used instead of test_metrics.py (which already contains predictor classification metric tests)
- temporal_split uses positional ratios tuple (0.7, 0.15, 0.15), not keyword args train_frac/val_frac
- LOB_SCHEMA is pyarrow.Schema; access fields via .field(i), not .columns attribute
- Schema column order: timestamp, mid_price, spread, bid_price_1..10, bid_size_1..10, ask_price_1..10, ask_size_1..10, trade_price, trade_size, trade_side
- LinearBaseline/DeepLOB constructors use n_levels + features_per_level (not in_features)
- CostModel.compute(exec_price, exec_size, mid_price, spread, avg_daily_volume) — not arrival_price/total_volume
- validate_regime_conditioning(real_by_regime, synthetic_by_regime) requires two dicts
- CosineNoiseSchedule.alphas_cumprod has shape (num_timesteps,) = 1000, not 1001
- MockEnv non-zero ask/bid (100.05/99.95) needed for baseline arrival_price computation

### Roadmap Evolution

- Phase 12 added: Coinbase Data & Full Pipeline Run — Coinbase public API downloader (REST + WebSocket), record BTC-USD LOB data, full training pipeline with real data, real metrics with wandb

### Pending Todos

- None — Phase 12-02 complete; next: remaining 12-xx plans (full pipeline run with real Coinbase data, wandb metrics)

### Blockers/Concerns

- Bybit REST API returns 403 from US (geo-restriction) — pivoted to Coinbase Exchange public API (api.exchange.coinbase.com) — works without auth
- User has WRDS access but LOBSTER not available in their subscription

## Session Continuity

Last session: 2026-03-22
Stopped at: Plan 12-03 complete — wandb IS/slippage logging added to train_all.sh Stage 6; Phase 12 success criterion 4 met; 307 tests pass; ruff + black clean
Resume file: .planning/phases/12-coinbase-data-pipeline-run/12-03-SUMMARY.md
