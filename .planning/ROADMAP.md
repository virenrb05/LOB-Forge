# Roadmap: LOB-Forge

## Overview

LOB-Forge builds an end-to-end market microstructure ML pipeline in dependency order: project scaffold with MPS validation, then a bulletproof data pipeline, then three ML modules (predictor → generator → executor) where each conditions on the previous, finishing with cross-component evaluation and polish. Ten phases, each delivering one verifiable capability.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Scaffold** - Project structure, Hydra configs, linting, MPS validation
- [x] **Phase 2: Data Ingestion** - Bybit downloader, LOBSTER adapter, unified Parquet schema
- [ ] **Phase 3: Data Preprocessing** - Features, labeling, normalization, splits, PyTorch Datasets
- [x] **Phase 4: Predictor Architecture** - TLOB transformer, DeepLOB baseline, focal loss
- [x] **Phase 5: Predictor Training** - Multi-horizon prediction, VPIN head, walk-forward eval, wandb
- [ ] **Phase 6: Generator Core** - Conditional DDPM/DDIM, 1D U-Net with AdaLN, EMA
- [ ] **Phase 7: Generator Validation** - Stylized facts, regime conditioning, LOB-Bench metrics
- [x] **Phase 8: Execution Environment** - Gymnasium LOB env, action space, cost model
- [x] **Phase 9: Execution Agent** - Double-DQN, curriculum learning, TWAP/VWAP/AC/Random baselines
- [x] **Phase 10: Evaluation & Polish** - IS metrics, plots, notebooks, README, test suite, train_all.sh
- [x] **Phase 11: Fix Generator Training Dispatch & Lint Sweep** - Fix train.py dispatch, train_all.sh Stage 4, linting sweep

## Phase Details

### Phase 1: Scaffold
**Goal**: Project structure builds, configs load, linting passes, MPS training works
**Depends on**: Nothing (first phase)
**Requirements**: EVAL-06, EVAL-08, EVAL-09
**Success Criteria** (what must be TRUE):
  1. `python -c "import lob_forge"` succeeds (package installable)
  2. Hydra config loads and overrides work (`python -m lob_forge.train --help`)
  3. `black --check . && ruff check .` pass on entire codebase
  4. MPS validation script confirms float32 training works (forward + backward pass)
**Research**: Unlikely (established Hydra + PyTorch patterns)
**Plans**: 2 (01-01, 01-02)

### Phase 2: Data Ingestion
**Goal**: Raw LOB data flows from Bybit API and LOBSTER files into unified Parquet format
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-02, DATA-03
**Success Criteria** (what must be TRUE):
  1. Bybit downloader fetches 10-level BTC-USDT LOB snapshots and saves to Parquet
  2. LOBSTER adapter reads NASDAQ equity LOB files and converts to same Parquet schema
  3. Both sources produce identical column layout (10 levels × 4 fields = 40 features + timestamp)
  4. Data integrity checks pass (no NaNs, monotonic timestamps, positive prices/sizes)
**Research**: Likely (external API — Bybit WebSocket specifics)
**Research topics**: Bybit WebSocket API endpoints, rate limits, snapshot vs delta format, historical data archives
**Plans**: 3 (02-01, 02-02, 02-03)

### Phase 3: Data Preprocessing
**Goal**: Raw LOB snapshots become model-ready tensors with features, labels, and temporal splits
**Depends on**: Phase 2
**Requirements**: DATA-04, DATA-05, DATA-06, DATA-07, DATA-08, DATA-09, DATA-10, DATA-11
**Success Criteria** (what must be TRUE):
  1. Rolling z-score normalization produces zero-mean unit-variance features (per-feature, causal)
  2. Mid-price labels are strictly causal (no future data leakage — unit tested)
  3. OFI, MLOFI, VPIN, and microprice features compute correctly on sample data
  4. Train/val/test splits are temporal with purge gaps (no data leakage — unit tested)
  5. LOBDataset and LOBSequenceDataset load batches correctly on MPS device
**Research**: Unlikely (well-documented in LOBFrame, LOBCAST references)
**Plans**: 7 (03-01, 03-02, 03-03, 03-04, 03-05, 03-06, 03-07)

### Phase 4: Predictor Architecture
**Goal**: TLOB and baseline models forward-pass correctly on LOB data
**Depends on**: Phase 3
**Requirements**: PRED-01, PRED-02, PRED-03, PRED-04, PRED-10
**Success Criteria** (what must be TRUE):
  1. TLOB dual-attention transformer produces 3-class logits from LOB input tensors
  2. DeepLOB (CNN + Inception + LSTM) produces matching output shape
  3. Linear/logistic baseline produces matching output shape
  4. Focal loss computes correctly with class weights (verified against manual calculation)
  5. All models train deterministically with fixed seeds on MPS
**Research**: Unlikely (TLOB has official repo with working code)
**Plans**: 4 (04-01, 04-02, 04-03, 04-04)

### Phase 5: Predictor Training
**Goal**: Trained predictor beats baselines on held-out data with proper evaluation
**Depends on**: Phase 4
**Requirements**: PRED-05, PRED-06, PRED-07, PRED-08, PRED-09
**Success Criteria** (what must be TRUE):
  1. Multi-horizon predictions (1s, 2s, 5s, 10s) each produce per-class F1 scores
  2. Auxiliary VPIN regression head trains jointly and produces reasonable VPIN estimates
  3. Walk-forward evaluation runs on at least 2 rolling windows
  4. TLOB F1 exceeds DeepLOB F1 on at least one horizon
  5. All experiments logged to wandb with metrics, configs, and model checkpoints
**Research**: Unlikely (standard training loop patterns)
**Plans**: 3 (05-01, 05-02, 05-03)

### Phase 6: Generator Core
**Goal**: Conditional diffusion model generates LOB sequences from noise
**Depends on**: Phase 5 (needs frozen predictor embeddings)
**Requirements**: GEN-01, GEN-02, GEN-03, GEN-04, GEN-05
**Success Criteria** (what must be TRUE):
  1. Cosine noise schedule produces correct alpha/beta values (verified against reference)
  2. 1D U-Net with AdaLN accepts LOB sequences and conditioning inputs
  3. DDPM 1000-step sampling produces LOB-shaped output tensors
  4. DDIM 50-step sampling produces LOB-shaped output tensors (faster inference)
  5. EMA model weights update correctly and can be loaded for inference
**Research**: Likely (FiLM conditioning on predictor embeddings is frontier)
**Research topics**: Diffusion-TS architecture, FiLM modulation dimensionality, predictor embedding extraction, AdaLN vs cross-attention conditioning
**Plans**: 5 (06-01, 06-02, 06-03, 06-04, 06-05)

### Phase 7: Generator Validation
**Goal**: Generated LOB sequences are statistically indistinguishable from real data
**Depends on**: Phase 6
**Requirements**: GEN-06, GEN-07, GEN-08
**Success Criteria** (what must be TRUE):
  1. All 7 stylized-fact tests pass (return dist, vol clustering, bid-ask bounce, spread CDF, book shape, market impact, summary figure)
  2. Regime-conditioned generation produces distinct distributions for high-vol vs low-vol regimes
  3. LOB-Bench quantitative metrics (Wasserstein distances, discriminator scores) computed and reported
**Research**: Likely (LOB-Bench evaluation methodology)
**Research topics**: LOB-Bench metric implementation, stylized fact test thresholds, regime labeling methodology
**Plans**: 5 (07-01, 07-02, 07-03, 07-04, 07-05)

### Phase 8: Execution Environment
**Goal**: Gymnasium-compatible LOB environment simulates realistic order execution
**Depends on**: Phase 7 (needs generator for synthetic mode)
**Requirements**: EXEC-01, EXEC-02, EXEC-09
**Success Criteria** (what must be TRUE):
  1. LOBExecutionEnv passes Gymnasium API check (reset/step/render)
  2. Real mode replays historical LOB data; synthetic mode uses generator output
  3. 7-action discrete space (wait + 3 market + 3 limit) executes correctly
  4. Cost model includes spread costs, exchange fees, and market impact
**Research**: Likely (market impact model calibration, fill simulation)
**Research topics**: Market impact coefficients for BTC-USDT, realistic fill models, TorchRL + custom Gym env integration
**Plans**: TBD

### Phase 9: Execution Agent
**Goal**: Double-DQN agent trained via curriculum beats TWAP on real data
**Depends on**: Phase 8
**Requirements**: EXEC-03, EXEC-04, EXEC-05, EXEC-06, EXEC-07, EXEC-08
**Success Criteria** (what must be TRUE):
  1. Dueling Double-DQN with prioritized replay trains without divergence
  2. 3-stage curriculum (low-vol → mixed → adversarial) completes all stages
  3. TWAP, VWAP, Almgren-Chriss, and Random baselines produce execution metrics
  4. DQN agent achieves lower implementation shortfall than TWAP on test data
**Research**: Unlikely (TorchRL DQN well-documented)
**Plans**: TBD

### Phase 10: Evaluation & Polish
**Goal**: Publication-ready outputs, comprehensive tests, reproducible pipeline
**Depends on**: Phase 9
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-07
**Success Criteria** (what must be TRUE):
  1. Implementation shortfall, IS Sharpe, and slippage-vs-TWAP metrics computed across all agents
  2. 6 publication-ready plots generated (specified in technical spec)
  3. 4 Jupyter notebooks run end-to-end without errors
  4. README includes architecture diagram, results tables, and citations
  5. Test suite runs with `pytest` and covers critical paths
  6. `train_all.sh` reproduces full pipeline from data download to final results
**Research**: Unlikely (internal patterns, visualization)
**Plans**: TBD

### Phase 11: Fix Generator Training Dispatch & Lint Sweep
**Goal**: Fix broken E2E pipeline (train_all.sh Stage 4) and verify linting across full codebase
**Depends on**: Phase 10
**Requirements**: EVAL-06, EVAL-09
**Gap Closure**: Closes gaps from v1 audit — EVAL-09 (Hydra dispatch), EVAL-06 (linting), train_all.sh integration gap, E2E flow gap
**Success Criteria** (what must be TRUE):
  1. `train.py` dispatches to generator trainer when `--config-name generator` is used
  2. `train_all.sh` Stage 4 successfully invokes generator training (not predictor)
  3. `black --check . && ruff check .` passes across entire codebase
  4. Full `train_all.sh` pipeline executes all stages without dispatch errors
**Research**: Unlikely (internal fix, existing patterns)
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Scaffold | 2/2 | Complete | 2026-03-19 |
| 2. Data Ingestion | 3/3 | Complete | 2026-03-19 |
| 3. Data Preprocessing | 7/7 | Complete | 2026-03-20 |
| 4. Predictor Architecture | 4/4 | Complete | 2026-03-20 |
| 5. Predictor Training | 3/3 | Complete | 2026-03-20 |
| 6. Generator Core | 5/5 | Complete | 2026-03-20 |
| 7. Generator Validation | 5/5 | Complete | 2026-03-22 |
| 8. Execution Environment | 3/3 | Complete | 2026-03-22 |
| 9. Execution Agent | 4/4 | Complete | 2026-03-22 |
| 10. Evaluation & Polish | 4/4 | Complete | 2026-03-22 |
| 11. Fix Generator Dispatch & Lint | 2/2 | Complete | 2026-03-22 |
