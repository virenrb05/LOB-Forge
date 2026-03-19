# Project Research Summary

**Project:** LOB-Forge
**Domain:** Quantitative Finance / Market Microstructure ML
**Researched:** 2026-03-18
**Confidence:** MEDIUM-HIGH

## Executive Summary

LOB-Forge is a three-component ML pipeline (transformer predictor → conditional diffusion generator → Double-DQN execution agent) for limit order book market microstructure. Research confirms this is a well-established domain with clear best practices, strong academic references (TLOB 2025, DeepLOB 2019, Diffusion-TS ICLR 2024, LOB-Bench ICML 2025), and well-documented failure modes. The end-to-end integration across all three components is the single biggest differentiator — no existing paper or open-source project does this.

The recommended approach is sequential training: predictor first (supervised), then generator conditioned on frozen predictor embeddings via FiLM modulation (self-supervised diffusion), then executor in synthetic+real environments (RL). The stack is Python 3.11 + PyTorch 2.6-2.9 on MPS, with TorchRL for the DQN agent and Hydra for config management. Critical constraint: MPS has no float64 support and torch.compile doesn't work — use float32 everywhere and eager mode.

The highest-risk pitfalls are data pipeline integrity issues: lookahead bias in label construction, non-causal normalization, and random train/test splits. These are "silent killers" that inflate metrics without obvious errors. The diffusion model's temporal coherence (not just marginal distributions) and the RL environment's realism (market impact, realistic fills) are the next biggest risks. Statistical rigor — not flashy results — is what impresses quant interviewers most.

## Key Findings

### Recommended Stack

Python 3.11 + PyTorch 2.6.0-2.9.x on MPS backend. TorchRL 0.11.1 for the Double-DQN agent (provides DQNLoss, prioritized replay natively). Hydra 1.3.2 + OmegaConf for hierarchical config composition. wandb for experiment tracking.

**Core technologies:**
- **PyTorch 2.6+**: MPS backend mature enough for training; avoid 2.10+ (macOS 26 bug). Float32 only.
- **TorchRL 0.11.1**: Official PyTorch RL library with DQN losses and replay buffers. Requires gymnasium==0.29.1 (incompatible with 1.x).
- **Hydra 1.3.2**: Industry standard for ML experiment configs. `_target_` instantiation pattern.
- **rotary-embedding-torch**: RoPE for the dual-attention transformer (2025 standard, better than learned/sinusoidal).
- **einops**: Readable tensor reshaping for attention code (prevents shape bugs).

**MPS critical constraints:**
- No float64 — cast all data to float32 at load time
- No torch.compile — use eager mode
- detect_anomaly is ~100x slower on MPS — debug on CPU
- No distributed training — use gradient accumulation for larger effective batch sizes

### Expected Features

**Must have (table stakes):**
- 10-level LOB ingestion with rolling z-score normalization and OFI features
- Strict temporal train/val/test splits with purge gaps
- F1 per class (not just accuracy), walk-forward evaluation
- Multiple prediction horizons (at least 3)
- DeepLOB + linear baselines for comparison
- Stylized fact validation for generated LOB data (spread, volume, autocorrelation)
- Realistic execution cost model (spread + fees + market impact)
- Clean, reproducible code with configs and type hints

**Should have (competitive):**
- TLOB dual-attention architecture (3.7% F1 improvement over prior SOTA)
- Conditional diffusion with DiT-style adaLN conditioning
- LOB-Bench quantitative evaluation metrics
- Ablation studies with attention visualization
- Financial P&L evaluation (Sharpe, drawdown, implementation shortfall)
- Cross-component integration (generator augments predictor; generator provides RL environment)

**Defer (v2+):**
- Multi-instrument modeling (depth > breadth)
- Counterfactual generation and stress testing
- Paper write-up

### Architecture Approach

Sequential three-phase training pipeline following the lightning-hydra-template pattern. Each module uses a different learning paradigm (supervised, self-supervised diffusion, RL) and trains independently. The predictor produces frozen embeddings that condition the generator via FiLM modulation (analogous to CLIP → Stable Diffusion). The generator produces synthetic LOB trajectories wrapped in Gym-compatible environments for RL training.

**Major components:**
1. **Data Layer** — Raw LOB ingestion, rolling z-score normalization, mid-price labeling, PyTorch Datasets
2. **Module A: Predictor** — Dual-attention transformer (temporal + spatial), focal loss, multi-horizon prediction
3. **Module B: Generator** — Conditional DDPM with transformer backbone, FiLM conditioning from predictor embeddings + regime variables
4. **Module C: Executor** — Double-DQN with prioritized replay via TorchRL, Gym-compatible LOB environment
5. **Experiment Layer** — Hydra configs, wandb tracking, checkpointing

### Critical Pitfalls

1. **Lookahead bias in labels** — Symmetric labeling windows leak future prices. Use strictly causal labels (only forward returns). Unit test: no feature/label at t uses data from t+1..T.
2. **Non-causal normalization** — Z-scoring over the full dataset leaks test info. Use rolling or per-snapshot relative normalization.
3. **Random train/test splits** — Autocorrelation makes random splits catastrophic. Use temporal splits with purge gaps.
4. **Diffusion temporal incoherence** — Synthetic data may match marginal distributions but fail temporal dynamics. Validate autocorrelation, volatility clustering, and TSTR.
5. **Unrealistic RL environment** — Zero market impact and mid-price fills make the agent learn fantasy strategies. Implement impact models, partial fills, and latency.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Project Scaffold & Environment
**Rationale:** MPS has documented silent failure modes (float64, non-contiguous tensors). Must validate the training environment works correctly before building anything on top of it.
**Delivers:** Project structure, Hydra configs, dependency management, MPS validation test suite
**Avoids:** Pitfall 10 (MPS numerical issues)

### Phase 2: Data Pipeline
**Rationale:** Every pitfall in the data pipeline (lookahead bias, non-causal normalization, random splits) silently corrupts all downstream results. This must be bulletproof before any model training. Bybit data quality issues require explicit validation.
**Delivers:** LOB ingestion (Bybit + LOBSTER adapter), preprocessing (z-score, OFI, VPIN), labeling, temporal splits, PyTorch Datasets
**Addresses:** Table stakes features (10-level LOB, normalization, OFI, temporal splits)
**Avoids:** Pitfalls 1-3, 9 (lookahead bias, non-causal normalization, random splits, Bybit data quality)

### Phase 3: Predictor (Module A)
**Rationale:** The predictor must be trained first because its frozen embeddings condition the generator. TLOB architecture is well-documented (2025 paper with code).
**Delivers:** TLOB dual-attention transformer, DeepLOB baseline, focal loss, multi-horizon evaluation, wandb tracking
**Uses:** PyTorch + rotary-embedding-torch + einops
**Addresses:** Differentiator features (TLOB architecture, ablation studies, attention visualization)
**Avoids:** Pitfalls 4-5 (overfitting, focal loss miscalibration)

### Phase 4: Generator (Module B)
**Rationale:** Generator depends on predictor embeddings for conditioning. Diffusion model quality must be validated with stylized facts before the RL agent can train in synthetic environments.
**Delivers:** Conditional DDPM with transformer backbone, FiLM conditioning, DDPM/DDIM sampling, stylized fact validation suite
**Implements:** Embedding-conditioned generation pattern (predictor → generator via FiLM)
**Avoids:** Pitfall 6 (statistically plausible but dynamically wrong synthetic data)

### Phase 5: Executor (Module C)
**Rationale:** The executor is the final downstream consumer. It needs both the realistic execution environment and the synthetic environment from the generator.
**Delivers:** Gymnasium LOB environment (real + synthetic mode), Double-DQN with TorchRL, baselines (TWAP, VWAP, Almgren-Chriss), curriculum learning
**Uses:** TorchRL + gymnasium 0.29.1
**Avoids:** Pitfalls 7-8 (unrealistic environment, reward hacking)

### Phase 6: Evaluation & Polish
**Rationale:** Cross-component integration and publication-ready outputs come after all individual modules work.
**Delivers:** Implementation shortfall metrics, visualization plots, Jupyter notebooks, README with results, test suite, linting
**Addresses:** Anti-pitfall 11 (portfolio red flags — baselines, limitations, confidence intervals)

### Phase Ordering Rationale

- **Scaffold → Data → Predictor → Generator → Executor → Polish** follows the dependency chain: each module consumes the output of the previous
- Data pipeline must be first because all three modules depend on clean, correctly labeled LOB data
- Predictor before generator because predictor embeddings condition the generator
- Generator before executor because the RL agent trains in synthetic environments
- Polish last because it requires all components working to produce meaningful results and cross-component experiments

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (Generator):** Diffusion conditioning on predictor embeddings is frontier research with few open-source references. May need to adapt Diffusion-TS or CoFinDiff patterns.
- **Phase 5 (Executor):** Realistic LOB execution environment design (market impact model, fill simulation) has sparse open-source references. TorchRL + custom Gym env integration needs careful design.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Scaffold):** Well-documented Hydra + PyTorch project setup.
- **Phase 2 (Data Pipeline):** LOBFrame, LOBCAST, and multiple academic codebases provide clear reference implementations.
- **Phase 3 (Predictor):** TLOB has an official repository with working code.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM-HIGH | PyTorch/Hydra/wandb verified via PyPI and official docs. MPS constraints verified via Apple docs and issue trackers. TorchRL + gymnasium compatibility verified. |
| Features | MEDIUM | Based on academic papers (TLOB, DeepLOB, LOB-Bench) and industry hiring practices. Interview expectations based on practitioner discussions. |
| Architecture | MEDIUM | Sequential training pattern well-established in literature. Project structure follows lightning-hydra-template convention. FiLM conditioning is standard but LOB-specific application has fewer references. |
| Pitfalls | MEDIUM-HIGH | Data pipeline pitfalls well-documented in recent LOB benchmarking studies. MPS issues verified via PyTorch issue tracker. RL pitfalls based on established RL failure mode literature. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **Bybit WebSocket data collection specifics**: Need to research exact API endpoints, rate limits, and data format during Phase 2 planning
- **FiLM conditioning dimensionality**: How to extract and size predictor embeddings for generator conditioning — needs experimentation during Phase 4
- **Market impact model calibration**: What impact coefficients to use for BTC-USDT execution simulation — needs Phase 5 research
- **LOBSTER data access**: Requires paid WRDS academic license — confirm availability or plan Bybit-only path

## Sources

### Primary (HIGH confidence)
- TLOB (Berti et al., 2025) — Dual-attention transformer, SOTA on FI-2010
- DeepLOB (Zhang et al., 2018) — Foundational LOB DL, CNN+LSTM baseline
- LOB-Bench (ICML 2025) — Quantitative evaluation framework for generative LOB models
- Diffusion-TS (ICLR 2024) — Diffusion architecture for time series generation
- TorchRL official docs — DQN implementation, replay buffers, gymnasium compatibility
- PyTorch MPS backend docs — Float64 limitation, non-contiguous tensor bug

### Secondary (MEDIUM confidence)
- CoFinDiff (IJCAI 2025) — Conditional financial diffusion model
- DiffLOB (2026) — Regime-conditioned diffusion for counterfactual LOB generation
- TRADES (Berti et al., 2025) — Transformer DDPM for LOB order flow
- LOB deep learning benchmark (AI Review 2024) — All 15 models degrade on new data

### Tertiary (LOW confidence)
- >70% accuracy on short-horizon 3-class LOB prediction indicates data leakage — practitioner consensus, needs validation
- Optimal focal loss gamma for LOB — no systematic study found, requires empirical tuning
- MPS addcmul_ bug specifics — may be version-dependent

---
*Research completed: 2026-03-18*
*Ready for roadmap: yes*
