# LOB-Forge

## What This Is

LOB-Forge is an end-to-end market microstructure ML system: a dual-attention transformer for LOB mid-price prediction, a conditional diffusion model generating synthetic LOB sequences, and a Double-DQN execution agent trained in the synthetic environment. Built as a single cohesive Python repository targeting quant finance roles at firms like Citadel Securities, Jane Street, and HRT.

## Core Value

The three-component pipeline works end-to-end: transformer embeddings condition the diffusion model, which generates unlimited training environments for the RL agent that beats TWAP on real data.

## Requirements

### Validated

(None yet — ship to validate)

### Active

**Data Pipeline**
- [ ] Bybit BTC-USDT downloader (WebSocket recorder + historical archives)
- [ ] LOBSTER adapter for NASDAQ equity data
- [ ] Unified Parquet schema (10-level LOB snapshots)
- [ ] Preprocessor: resampling, derived features (imbalance, VPIN, weighted mid), rolling z-score normalization
- [ ] PyTorch Datasets: LOBDataset (predictor) and LOBSequenceDataset (generator)

**Module A: Predictor**
- [ ] Dual-attention transformer (spatial across levels, temporal across time) inspired by TLOB
- [ ] Focal loss with class-weight rebalancing
- [ ] Multi-horizon prediction (1s, 2s, 5s, 10s)
- [ ] Auxiliary VPIN regression head
- [ ] DeepLOB baseline for comparison
- [ ] wandb experiment tracking

**Module B: Generator**
- [ ] Cosine noise schedule (Nichol & Dhariwal 2021)
- [ ] 1D U-Net denoiser with AdaLN conditioning (DiT-style)
- [ ] Conditioning on diffusion timestep + volatility regime + time-of-day
- [ ] DDPM (1000-step) and DDIM (50-step) sampling
- [ ] EMA model weights
- [ ] 7 stylized-fact validation tests (return distribution, vol clustering, bid-ask bounce, spread CDF, book shape, market impact concavity, summary figure)

**Module C: Executor**
- [ ] Gymnasium LOBExecutionEnv (real mode + synthetic mode)
- [ ] 7-action discrete space (wait, 3 market order sizes, 3 limit order types)
- [ ] Dueling Double-DQN with prioritized experience replay
- [ ] 3-stage curriculum learning (low-vol → mixed → adversarial)
- [ ] Baselines: TWAP, VWAP, Almgren-Chriss, Random

**Evaluation & Polish**
- [ ] Implementation shortfall metrics, IS Sharpe, slippage vs TWAP
- [ ] 6 publication-ready visualization plots
- [ ] 4 Jupyter notebooks (data exploration, predictor results, generator quality, execution backtest)
- [ ] Comprehensive README with architecture diagram, results, citations
- [ ] Full test suite, linting (black + ruff), train_all.sh script

### Out of Scope

- Live trading or paper trading — this is a research/portfolio project, not production trading
- Multi-asset support — single instrument (BTC-USDT) only
- Web UI or dashboard — CLI and notebooks only
- Distributed training — single-GPU/MPS only
- Custom market making strategies — execution (liquidation) only

## Context

- Technical spec: `LOB-Forge_Technical_Specification.docx` contains exact architecture dimensions, hyperparameters, and implementation details for all modules
- Primary data source: Bybit BTC-USDT perpetual futures (free, public API)
- Key papers: TLOB (2025), DeepLOB (2019), TRADES (2025), DDPM (2020), DDIM (2021), DiT (2023), Dueling DQN (2016), PER (2016), Almgren-Chriss (2001)
- Implementation order: Scaffold → Data → Predictor → Generator → Executor → Polish (dependency-aware)

## Constraints

- **Hardware**: Apple Silicon (MPS backend) — batch sizes may need reduction, some PyTorch ops may need CPU fallback
- **Tech stack**: Python 3.10+, PyTorch 2.1+, OmegaConf/Hydra for configs, wandb for tracking
- **Data**: Default to free Bybit data; LOBSTER requires paid WRDS access
- **Fidelity**: Must follow spec exactly — all architectures, hyperparameters, and evaluation criteria as documented

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Bybit as default data source | Free, no licensing issues, high liquidity | — Pending |
| From-scratch Double-DQN (no SB3) | Demonstrates deeper understanding for interviews | — Pending |
| Focal loss over cross-entropy | LOB labels heavily imbalanced (STATIONARY dominates) | — Pending |
| MPS training backend | User's hardware is Apple Silicon | — Pending |

---
*Last updated: 2026-03-18 after initialization*
