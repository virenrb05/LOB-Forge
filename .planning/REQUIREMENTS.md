# Requirements: LOB-Forge

**Defined:** 2026-03-18
**Core Value:** The three-component pipeline works end-to-end: transformer embeddings condition the diffusion model, which generates unlimited training environments for the RL agent that beats TWAP on real data.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Data Pipeline

- [ ] **DATA-01**: Bybit BTC-USDT downloader (WebSocket recorder + historical archives)
- [ ] **DATA-02**: LOBSTER adapter for NASDAQ equity data
- [ ] **DATA-03**: Unified Parquet schema (10-level LOB snapshots, 40 features per snapshot)
- [ ] **DATA-04**: Rolling z-score normalization (per-feature, rolling window, non-causal)
- [ ] **DATA-05**: Mid-price movement labeling (3-class: up/down/stationary, smoothed future mid)
- [ ] **DATA-06**: Multi-level order flow imbalance (OFI/MLOFI) features
- [ ] **DATA-07**: VPIN (Volume-Synchronized Probability of Informed Trading) derived feature
- [ ] **DATA-08**: Microprice (volume-weighted mid using best bid/ask sizes)
- [ ] **DATA-09**: Strict temporal train/val/test split with purge gaps (no lookahead bias)
- [ ] **DATA-10**: Multiple prediction horizons (1s, 2s, 5s, 10s)
- [ ] **DATA-11**: PyTorch Datasets: LOBDataset (predictor) and LOBSequenceDataset (generator)

### Predictor (Module A)

- [ ] **PRED-01**: Dual-attention transformer (spatial across levels, temporal across time) — TLOB architecture
- [ ] **PRED-02**: DeepLOB baseline (CNN + Inception + LSTM) for comparison
- [ ] **PRED-03**: Linear/logistic regression baseline as performance floor
- [ ] **PRED-04**: Focal loss with class-weight rebalancing for imbalanced labels
- [ ] **PRED-05**: Multi-horizon prediction (1s, 2s, 5s, 10s)
- [ ] **PRED-06**: Auxiliary VPIN regression head (multi-task learning)
- [ ] **PRED-07**: F1, precision, recall per class (weighted and macro)
- [ ] **PRED-08**: Walk-forward / rolling window evaluation (at least 2 windows)
- [ ] **PRED-09**: wandb experiment tracking with logged metrics and configs
- [ ] **PRED-10**: Reproducibility: fixed seeds, deterministic training, config-driven

### Generator (Module B)

- [ ] **GEN-01**: Cosine noise schedule (Nichol & Dhariwal 2021)
- [ ] **GEN-02**: 1D U-Net denoiser with AdaLN conditioning (DiT-style)
- [ ] **GEN-03**: Conditioning on diffusion timestep + volatility regime + time-of-day
- [ ] **GEN-04**: DDPM sampling (1000-step) and DDIM sampling (50-step)
- [ ] **GEN-05**: EMA model weights
- [ ] **GEN-06**: 7 stylized-fact validation tests (return distribution, vol clustering, bid-ask bounce, spread CDF, book shape, market impact concavity, summary figure)
- [ ] **GEN-07**: Regime-conditioned generation (volatility, trend, liquidity regimes)
- [ ] **GEN-08**: LOB-Bench quantitative evaluation (Wasserstein distances, discriminator scores, conditional statistics)

### Executor (Module C)

- [ ] **EXEC-01**: Gymnasium LOBExecutionEnv (real mode + synthetic mode)
- [ ] **EXEC-02**: 7-action discrete space (wait, 3 market order sizes, 3 limit order types)
- [ ] **EXEC-03**: Dueling Double-DQN with prioritized experience replay (from scratch, no SB3)
- [ ] **EXEC-04**: 3-stage curriculum learning (low-vol → mixed → adversarial)
- [ ] **EXEC-05**: TWAP baseline
- [ ] **EXEC-06**: VWAP baseline
- [ ] **EXEC-07**: Almgren-Chriss analytical baseline
- [ ] **EXEC-08**: Random baseline
- [ ] **EXEC-09**: Realistic cost model (spread costs + exchange fees + market impact)

### Evaluation & Polish

- [ ] **EVAL-01**: Implementation shortfall metrics, IS Sharpe, slippage vs TWAP
- [ ] **EVAL-02**: 6 publication-ready visualization plots
- [ ] **EVAL-03**: 4 Jupyter notebooks (data exploration, predictor results, generator quality, execution backtest)
- [ ] **EVAL-04**: Comprehensive README with architecture diagram, results tables, citations
- [ ] **EVAL-05**: Full test suite with meaningful coverage
- [ ] **EVAL-06**: Linting (black + ruff) passing across entire codebase
- [ ] **EVAL-07**: train_all.sh script (single-command reproducible pipeline)
- [ ] **EVAL-08**: Type hints throughout all modules
- [ ] **EVAL-09**: Hydra/OmegaConf hierarchical config management

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Predictor Enhancements

- **PRED-V2-01**: Ablation studies (attention heads, level importance, horizon sensitivity)
- **PRED-V2-02**: Attention/gradient visualization (temporal and feature heatmaps)
- **PRED-V2-03**: Financial P&L evaluation (Sharpe, Sortino, max drawdown from predictions)
- **PRED-V2-04**: Statistical significance testing (bootstrap CIs, permutation tests)

### Cross-Component

- **CROSS-V2-01**: Synthetic data augmentation for predictor training
- **CROSS-V2-02**: Generator as configurable RL environment with regime control
- **CROSS-V2-03**: Latency-aware design (inference profiling, model size comparison)

### Research-Grade

- **FUTURE-01**: Multi-horizon joint prediction (all horizons simultaneously)
- **FUTURE-02**: Counterfactual generation ("what if" scenarios, DiffLOB-style)
- **FUTURE-03**: Multi-instrument extension (second instrument for generalization)
- **FUTURE-04**: ArXiv-style technical report
- **FUTURE-05**: Adaptive execution (RL agent conditioned on predictor output)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Live/paper trading | Research portfolio project, not production trading |
| Multi-asset support | Single instrument (BTC-USDT) — depth over breadth |
| Web UI or dashboard | CLI and notebooks only |
| Distributed training | Single-GPU/MPS only |
| Custom market making | Execution (liquidation) only |
| Real-time streaming pipeline | Engineering distraction from ML quality |
| GAN-based generation | Diffusion models strictly better for LOB domain |
| Complex RL (PPO, SAC, multi-agent) | Double-DQN sufficient, mention as extensions |
| Serving infrastructure (FastAPI, Docker, K8s) | Not differentiating for quant roles |
| Tick-level backtester from scratch | Use simplified simulation, focus ML effort on models |

## Traceability

Which phases cover which requirements. Updated by create-roadmap.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 2 | Pending |
| DATA-02 | Phase 2 | Pending |
| DATA-03 | Phase 2 | Pending |
| DATA-04 | Phase 3 | Pending |
| DATA-05 | Phase 3 | Pending |
| DATA-06 | Phase 3 | Pending |
| DATA-07 | Phase 3 | Pending |
| DATA-08 | Phase 3 | Pending |
| DATA-09 | Phase 3 | Pending |
| DATA-10 | Phase 3 | Pending |
| DATA-11 | Phase 3 | Pending |
| PRED-01 | Phase 4 | Pending |
| PRED-02 | Phase 4 | Pending |
| PRED-03 | Phase 4 | Pending |
| PRED-04 | Phase 4 | Pending |
| PRED-05 | Phase 5 | Pending |
| PRED-06 | Phase 5 | Pending |
| PRED-07 | Phase 5 | Pending |
| PRED-08 | Phase 5 | Pending |
| PRED-09 | Phase 5 | Pending |
| PRED-10 | Phase 4 | Pending |
| GEN-01 | Phase 6 | Pending |
| GEN-02 | Phase 6 | Pending |
| GEN-03 | Phase 6 | Pending |
| GEN-04 | Phase 6 | Pending |
| GEN-05 | Phase 6 | Pending |
| GEN-06 | Phase 7 | Pending |
| GEN-07 | Phase 7 | Pending |
| GEN-08 | Phase 7 | Pending |
| EXEC-01 | Phase 8 | Pending |
| EXEC-02 | Phase 8 | Pending |
| EXEC-03 | Phase 9 | Pending |
| EXEC-04 | Phase 9 | Pending |
| EXEC-05 | Phase 9 | Pending |
| EXEC-06 | Phase 9 | Pending |
| EXEC-07 | Phase 9 | Pending |
| EXEC-08 | Phase 9 | Pending |
| EXEC-09 | Phase 8 | Pending |
| EVAL-01 | Phase 10 | Pending |
| EVAL-02 | Phase 10 | Pending |
| EVAL-03 | Phase 10 | Pending |
| EVAL-04 | Phase 10 | Pending |
| EVAL-05 | Phase 10 | Pending |
| EVAL-06 | Phase 1 | Complete |
| EVAL-07 | Phase 10 | Pending |
| EVAL-08 | Phase 1 | Complete |
| EVAL-09 | Phase 1 | Complete |

**Coverage:**
- v1 requirements: 47 total
- Mapped to phases: 47 ✓
- Unmapped: 0

---
*Requirements defined: 2026-03-18*
*Last updated: 2026-03-19 after roadmap creation*
