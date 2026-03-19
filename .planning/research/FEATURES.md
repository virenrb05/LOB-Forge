# Feature Research

**Domain:** Quantitative Finance / Market Microstructure ML
**Researched:** 2026-03-18
**Confidence:** MEDIUM (composite; individual items marked below)

## Feature Landscape

### Table Stakes (Interviewers Expect These)

Features that any serious LOB ML portfolio project must have. Missing these signals lack of rigor.

| Feature | Why Expected | Complexity | Confidence | Notes |
|---------|--------------|------------|------------|-------|
| **Raw LOB snapshot ingestion (10+ levels)** | Standard in all LOB ML literature (DeepLOB, TLOB, FI-2010). Anything less looks toy-scale. | MEDIUM | HIGH | 10 levels of bid/ask price + volume = 40 features per snapshot. BTC-USDT from exchange WS feed or historical parquet. |
| **Proper train/val/test temporal split** | Non-negotiable in financial ML. Random splits cause lookahead bias, which is an instant red flag for any quant reviewer. | LOW | HIGH | Must be strictly chronological. No shuffling across time. |
| **Z-score normalization (rolling window)** | Standard preprocessing for LOB data. Static normalization fails on non-stationary financial data. | LOW | HIGH | Rolling 1-day or N-snapshot window z-score, applied feature-wise. FI-2010 literature uses 5-day rolling windows. |
| **Mid-price movement labeling (3-class)** | The canonical LOB prediction task: up/down/stationary over horizon k. Established by FI-2010 benchmark. | LOW | HIGH | Use smoothed future mid-price (mean over horizon) vs current. Multiple horizons (short/medium/long) expected. |
| **Order flow imbalance (OFI) features** | Fundamental microstructure signal. Any LOB project without OFI looks naive. Multi-level OFI (MLOFI) is even better. | LOW | HIGH | OFI = net difference between buy/sell order events. Extend to multiple price levels. |
| **Classification metrics: F1, precision, recall per class** | Accuracy alone is meaningless with imbalanced classes (stationary dominates). Every LOB ML paper reports F1. | LOW | HIGH | Weighted and per-class F1. Cohen's kappa also common. |
| **Reproducibility: seed control, config files, requirements** | Quant firms value engineering rigor. Irreproducible results are worthless. | LOW | HIGH | Fixed random seeds, YAML/JSON configs, pinned dependencies, deterministic training where possible. |
| **Multiple prediction horizons** | All LOB papers evaluate on multiple horizons (e.g., k=1,5,10,20,50,100 ticks). Single horizon is incomplete. | LOW | HIGH | Shows how model degrades/improves with horizon. Short horizons harder, long horizons more useful for execution. |
| **Baseline comparisons** | Without baselines, results are uninterpretable. Need at least linear model + simple DL (MLP or LSTM). | MEDIUM | HIGH | DeepLOB is the standard baseline. Linear/logistic regression as floor. |
| **Walk-forward / rolling window evaluation** | Standard in financial ML to prevent overfitting. Static splits give false confidence. ~95% of backtested strategies fail live. | MEDIUM | HIGH | Expanding or sliding window retraining + evaluation. Shows temporal stability. |
| **Clean project structure and documentation** | Code quality signals engineering competence. Sloppy repos get closed in 30 seconds. | LOW | HIGH | Type hints, docstrings, modular design, clear README with results. |
| **Execution agent: realistic cost model** | An execution agent without transaction costs, slippage, and market impact is fantasy. | MEDIUM | HIGH | Must include spread costs, exchange fees, and some form of market impact (even linear). |
| **Generative model: stylized facts validation** | Generated LOB data must reproduce known statistical properties. Otherwise generation is just noise. | MEDIUM | HIGH | Spread distribution, volume profiles, autocorrelation structure, volatility clustering, fat tails. |

### Differentiators (Competitive Advantage)

Features that elevate a project from "competent" to "impressive." These are what make interviewers lean in.

| Feature | Value Proposition | Complexity | Confidence | Notes |
|---------|-------------------|------------|------------|-------|
| **Dual-attention transformer (TLOB-style)** | Shows you can implement cutting-edge architecture, not just copy DeepLOB. Temporal + feature attention captures richer LOB dynamics. | HIGH | HIGH | TLOB exceeds SOTA by 3.7 F1 on FI-2010. Dual attention (temporal self-attention + feature self-attention) is architecturally novel. |
| **Conditional diffusion for LOB generation (DiT-style)** | Generative modeling of LOB is frontier research (DiffLOB, TRADES, DiffVolume all 2025+). Using DiT architecture shows deep understanding of both diffusion and transformers. | HIGH | MEDIUM | Condition on regime variables (volatility, trend, liquidity, OFI). adaLN conditioning from DiT. Very few open-source implementations exist. |
| **LOB-Bench style quantitative evaluation for generator** | Goes beyond "the plots look similar." Wasserstein distances, discriminator scores, conditional statistics. Published at ICML 2025. | HIGH | MEDIUM | Implement subset of LOB-Bench metrics: distributional divergence, cross-correlation, price response functions. |
| **Regime-conditioned generation** | Allows counterfactual analysis ("what if volatility doubled?"). DiffLOB conditions on trend, volatility, liquidity, OFI. | HIGH | MEDIUM | Conditioning on future regime variables enables stress testing and scenario analysis. Directly useful for risk management. |
| **Microprice and queue-informed features** | Shows deeper microstructure knowledge beyond basic OFI. Microprice is a better fair-value estimator than mid-price. | LOW | HIGH | Microprice = volume-weighted mid using best bid/ask sizes. Queue position dynamics inform execution probability. |
| **Ablation studies** | Shows scientific rigor. Which attention head matters? Which LOB levels are informative? What horizon is predictable? | MEDIUM | HIGH | Attention visualization, feature importance via ablation, layer-wise analysis. Interviewers love seeing you understand WHY your model works. |
| **Financial P&L evaluation (not just ML metrics)** | Bridges gap between ML accuracy and trading reality. Shows you understand that prediction != profit. | MEDIUM | HIGH | Convert predictions to positions, apply transaction costs, compute Sharpe/Sortino/max drawdown. Deflated Sharpe Ratio corrects for selection bias. |
| **Double-DQN with prioritized experience replay** | Shows RL depth beyond vanilla DQN. PER improves sample efficiency on rare but important transitions. | MEDIUM | HIGH | Standard improvement over basic DQN. LSTM-based Q-network for sequential state. |
| **Execution benchmarks (TWAP/VWAP/Almgren-Chriss)** | RL agent must beat meaningful baselines. TWAP/VWAP are industry standard. Almgren-Chriss is the classic optimal execution model. | MEDIUM | HIGH | Without these baselines, RL results are uninterpretable. Implementation shortfall as primary metric. |
| **Cross-component integration** | Using synthetic data to augment predictor training or to train the execution agent shows system-level thinking. | MEDIUM | MEDIUM | Generator -> augmented training data -> better predictor. Generator -> sim environment -> RL agent training. This is the "end-to-end" story. |
| **Latency-aware design** | Shows you understand real HFT constraints. Even if not real-time, demonstrating awareness of inference cost matters. | LOW | MEDIUM | Profile inference time, compare model sizes, discuss deployment considerations. |
| **Attention / gradient visualization** | Makes model interpretable. Quant interviewers want to know your model isn't learning noise. | LOW | HIGH | Temporal attention weights show which past snapshots matter. Feature attention shows which LOB levels are informative. |
| **Statistical significance testing** | Prevents "lucky split" conclusions. Permutation tests, confidence intervals, multiple-comparison corrections. | LOW | HIGH | Bootstrap confidence intervals on F1, paired t-tests across horizons. |

### Anti-Features (Commonly Attempted, Often Problematic)

Features that seem impressive but create more problems than they solve in a portfolio project context.

| Feature | Why Tempting | Why Problematic | Alternative |
|---------|-------------|-----------------|-------------|
| **Multi-instrument modeling** | "Scale" looks impressive | Massively increases data pipeline complexity, cross-instrument dynamics are a whole separate research area, and BTC-USDT alone is sufficient to demonstrate competence. | Deep single-instrument analysis with multiple horizons and regimes. Show depth, not breadth. |
| **Real-time streaming pipeline** | Feels production-ready | Engineering distraction from ML quality. Kafka/Redis infra is not what quant interviewers evaluate. Adds ops burden without ML value. | Batch processing with realistic latency simulation. Mention real-time as future work. |
| **GAN-based LOB generation** | GANs are well-known | Training instability, mode collapse, and difficulty evaluating quality. Diffusion models are strictly better for this domain (LOB-Bench results confirm). | Conditional diffusion models (DDPM/DiT). More stable training, better distributional coverage. |
| **Full live trading integration** | Proves "it works" | Regulatory risk, exchange API complexity, real money risk, and live results over short periods prove nothing statistically. | Paper trading simulation with realistic fills. Walk-forward backtesting is more rigorous anyway. |
| **Overly complex RL (PPO, SAC, multi-agent)** | More advanced algorithms | Double-DQN is well-established for optimal execution. More complex algorithms add hyperparameter tuning burden without clear benefit for single-instrument execution. | Double-DQN with clear ablations. Mention PPO/SAC as extensions. |
| **Serving infrastructure (FastAPI, Docker, K8s)** | "Production-ready" | Interviewer time is spent on infra, not ML. These skills are assumed for SWE roles, not differentiating for quant roles. | Clean CLI with config management. Docker for reproducibility only. |
| **Tick-level backtester from scratch** | Full control | Massive engineering effort, many subtle bugs (queue priority, partial fills, latency). Use established tools. | Use or adapt existing frameworks (e.g., hftbacktest). Focus ML effort on the models. |

## Feature Dependencies

```
[LOB Data Pipeline]
    |
    +--requires--> [Raw LOB Snapshot Ingestion]
    |                  +--requires--> [Z-score Normalization]
    |                  +--requires--> [Mid-price Labeling]
    |                  +--requires--> [OFI Feature Engineering]
    |
    +--enables--> [Prediction Module (TLOB)]
    |                 +--requires--> [Temporal Split]
    |                 +--requires--> [Baselines (DeepLOB, Linear)]
    |                 +--requires--> [Walk-forward Evaluation]
    |                 +--enhances--> [Ablation Studies]
    |                 +--enhances--> [Attention Visualization]
    |                 +--enhances--> [Financial P&L Evaluation]
    |
    +--enables--> [Generation Module (DiT Diffusion)]
    |                 +--requires--> [LOB Data Pipeline]
    |                 +--requires--> [Stylized Facts Validation]
    |                 +--enhances--> [Regime-conditioned Generation]
    |                 +--enhances--> [LOB-Bench Quantitative Eval]
    |
    +--enables--> [Execution Module (Double-DQN)]
                      +--requires--> [Realistic Cost Model]
                      +--requires--> [Execution Baselines (TWAP/VWAP)]
                      +--enhances--> [Prioritized Experience Replay]

[Generation Module] --augments--> [Prediction Module] (synthetic data augmentation)
[Generation Module] --provides-env--> [Execution Module] (simulated environment)
[Prediction Module] --provides-signal--> [Execution Module] (alpha signal)
```

### Dependency Notes

- **Prediction requires Data Pipeline:** All downstream modeling depends on clean, properly normalized LOB data with correct temporal splits. This must be bulletproof first.
- **Generation requires Data Pipeline:** Diffusion model trains on the same LOB snapshots. Quality of generated data inherits data pipeline quality.
- **Execution requires Cost Model:** Without realistic costs, RL agent learns unrealistic policies. Cost model is prerequisite, not enhancement.
- **Cross-component integration enhances all three:** The real "end-to-end" value comes from connecting generation -> prediction augmentation and generation -> execution environment. But each component must work standalone first.
- **Ablation studies enhance Prediction:** Only meaningful after base model is working. Cannot ablate what doesn't exist.
- **Financial P&L evaluation enhances Prediction:** Converts ML metrics to trading metrics. Requires prediction model + cost assumptions.

## MVP Definition

### Launch With (v1) -- "Reviewable Portfolio Project"

The minimum that would not embarrass you in a quant interview.

- [ ] **LOB data pipeline** -- BTC-USDT, 10 levels, rolling z-score normalization, OFI features, temporal split
- [ ] **TLOB predictor** -- Dual-attention transformer, 3-class mid-price prediction, multiple horizons (at least 3)
- [ ] **Baselines** -- Linear model + DeepLOB, same evaluation protocol
- [ ] **Proper evaluation** -- F1 per class, walk-forward on at least 2 windows, confusion matrices
- [ ] **Conditional diffusion generator (basic)** -- Generate LOB snapshots, validate with at least 3 stylized facts (spread, volume distribution, autocorrelation)
- [ ] **Double-DQN execution agent** -- Single instrument, TWAP baseline comparison, realistic spread + fee costs
- [ ] **Clean repo** -- Type hints, configs, requirements, README with results tables, reproducible with single command

### Add After Validation (v1.x) -- "Impressive Project"

- [ ] **Regime conditioning for generator** -- Condition on volatility/trend/liquidity regimes
- [ ] **LOB-Bench quantitative evaluation** -- Wasserstein distances, discriminator scores
- [ ] **Ablation studies for TLOB** -- Attention head analysis, level importance, horizon sensitivity
- [ ] **Financial P&L evaluation** -- Sharpe ratio, max drawdown, implementation shortfall from predictions
- [ ] **Cross-component integration** -- Synthetic data augmentation for predictor, generator as RL environment
- [ ] **Attention visualization** -- Temporal and feature attention heatmaps with interpretation
- [ ] **Prioritized experience replay** -- For Double-DQN agent
- [ ] **Almgren-Chriss baseline** -- Analytical optimal execution benchmark
- [ ] **Statistical significance** -- Bootstrap CIs, permutation tests on key results

### Future Consideration (v2+) -- "Research-Grade"

- [ ] **Multi-horizon joint prediction** -- Predict all horizons simultaneously
- [ ] **Counterfactual generation** -- "What if" scenarios for stress testing (DiffLOB-style)
- [ ] **Multi-instrument extension** -- Second instrument to show generalization
- [ ] **Paper write-up** -- ArXiv-style technical report documenting methodology and results
- [ ] **Adaptive execution** -- RL agent that conditions on prediction model output

## Feature Prioritization Matrix

| Feature | Interviewer Value | Implementation Cost | Priority |
|---------|-------------------|---------------------|----------|
| LOB data pipeline (10-level, normalized, OFI) | HIGH | MEDIUM | P1 |
| Temporal train/val/test split | HIGH | LOW | P1 |
| TLOB dual-attention predictor | HIGH | HIGH | P1 |
| DeepLOB + linear baselines | HIGH | MEDIUM | P1 |
| F1 + walk-forward evaluation | HIGH | MEDIUM | P1 |
| Multiple prediction horizons | HIGH | LOW | P1 |
| Conditional diffusion generator (basic) | HIGH | HIGH | P1 |
| Stylized facts validation | HIGH | MEDIUM | P1 |
| Double-DQN execution agent | HIGH | HIGH | P1 |
| TWAP/VWAP baselines | HIGH | LOW | P1 |
| Realistic cost model | HIGH | MEDIUM | P1 |
| Clean repo + reproducibility | HIGH | LOW | P1 |
| Regime-conditioned generation | HIGH | HIGH | P2 |
| LOB-Bench quantitative eval | HIGH | MEDIUM | P2 |
| Ablation studies | HIGH | MEDIUM | P2 |
| Financial P&L evaluation | HIGH | MEDIUM | P2 |
| Attention visualization | MEDIUM | LOW | P2 |
| Cross-component integration | HIGH | MEDIUM | P2 |
| Statistical significance testing | MEDIUM | LOW | P2 |
| Almgren-Chriss baseline | MEDIUM | MEDIUM | P2 |
| Prioritized experience replay | MEDIUM | LOW | P2 |
| Counterfactual generation | MEDIUM | HIGH | P3 |
| Multi-instrument | LOW | HIGH | P3 |
| Paper write-up | MEDIUM | MEDIUM | P3 |

**Priority key:**
- P1: Must have for launch -- without these, project is incomplete
- P2: Should have -- each one meaningfully improves interview impact
- P3: Nice to have -- defer unless core is polished

## Competitor / Reference Project Analysis

| Feature | DeepLOB (Zhang 2018) | TLOB (Berti 2025) | DiffLOB (2026) | TRADES/DeepMarket (Berti 2025) | LOB-Forge (Our Plan) |
|---------|---------------------|-------------------|----------------|-------------------------------|---------------------|
| Architecture | CNN + Inception + LSTM | Dual-attention Transformer | Regime-conditioned Diffusion | Transformer DDPM | All three integrated |
| Data | FI-2010 (equities) | FI-2010 + LOBSTER | LOBSTER | LOBSTER | BTC-USDT (crypto) |
| Prediction | 3-class mid-price | 3-class mid-price | N/A | N/A | 3-class mid-price |
| Generation | N/A | N/A | Counterfactual LOB | Realistic order flow | Conditional LOB snapshots |
| Execution | N/A | N/A | N/A | N/A | Double-DQN |
| Evaluation | F1, accuracy | F1, new labeling | Stylized facts + WD | Stylized facts | F1 + stylized facts + P&L + LOB-Bench |
| End-to-end | No | No | No | No | Yes (key differentiator) |
| Open source | Yes | Yes | No | Yes | Yes |

## Key Insight: What Actually Impresses

Based on research, the hierarchy of what quant interviewers care about:

1. **Statistical rigor** -- Proper evaluation methodology, awareness of overfitting, walk-forward validation, significance testing. This is #1 because it signals you think like a researcher, not a tutorial-follower.

2. **Financial intuition** -- Transaction cost awareness, understanding that prediction != profit, Sharpe ratio thinking, awareness of market impact. Shows you understand the business.

3. **Depth over breadth** -- Deep analysis of one instrument with ablations, multiple horizons, and regime analysis beats shallow analysis of many instruments.

4. **Architectural understanding** -- Not just "I used a transformer" but "here's why dual attention captures cross-level LOB dynamics that CNN misses, and here's the ablation proving it."

5. **Code quality** -- Clean, typed, tested, reproducible code signals professional engineering. Messy notebooks signal a student project.

6. **End-to-end thinking** -- Connecting prediction -> generation -> execution in a coherent system (even loosely) shows systems thinking that individual papers lack.

## Sources

### Academic Papers (HIGH confidence)
- [TLOB: Dual Attention Transformer for LOB (Berti et al., 2025)](https://arxiv.org/abs/2502.15757) -- SOTA on FI-2010, dual attention architecture
- [DeepLOB: Deep CNNs for LOB (Zhang et al., 2018)](https://arxiv.org/pdf/1808.03668) -- Foundational LOB DL paper, IEEE TSP
- [Double Deep Q-Learning for Optimal Execution (Ning et al., 2018)](https://arxiv.org/pdf/1812.06600) -- Double-DQN for execution
- [DiffLOB: Diffusion for Counterfactual LOB Generation (2026)](https://arxiv.org/abs/2602.03776) -- Regime-conditioned diffusion
- [TRADES: Market Simulations with Diffusion Models (Berti et al., 2025)](https://arxiv.org/html/2502.07071v2) -- Transformer DDPM for LOB
- [DiffVolume: Diffusion for LOB Volume Generation (2025)](https://arxiv.org/abs/2508.08698) -- Conditional diffusion for volumes
- [LOB-Bench: Benchmarking Generative AI for LOB (2025)](https://arxiv.org/abs/2502.09172) -- ICML 2025, quantitative evaluation framework
- [Scalable Diffusion Models with Transformers / DiT (Peebles & Xie, 2023)](https://www.wpeebles.com/DiT) -- DiT architecture, adaLN conditioning
- [Deep Limit Order Book Forecasting: A Microstructural Guide (2025)](https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2522911) -- Comprehensive LOB feature engineering guide
- [Multi-Level Order-Flow Imbalance (Xu et al.)](https://ora.ox.ac.uk/objects/uuid:9b7d0422-4ef1-48e7-a2d4-4eaa8a0a7ec1) -- MLOFI for LOB prediction
- [Deep Learning for Fill Probabilities in LOB (Columbia, 2021)](https://business.columbia.edu/sites/default/files-efs/citation_file_upload/deep-lob-2021.pdf) -- Fill probability estimation

### Open Source References (HIGH confidence)
- [DeepLOB PyTorch Implementation](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)
- [TLOB Official Repository](https://github.com/LeonardoBerti00/TLOB)
- [DeepMarket / TRADES Framework](https://github.com/LeonardoBerti00/DeepMarket)
- [LOBCAST Benchmarking Framework](https://github.com/matteoprata/LOBCAST)
- [LOB-Bench Library](https://github.com/peernagy/lob_bench)
- [lob-deep-learning Multi-Model Implementation](https://github.com/Jeonghwan-Cheon/lob-deep-learning)

### Industry / Interview Sources (MEDIUM confidence)
- [Top 10 Projects for Quantitative Finance Roles](https://dataloopr.com/blog/top-10-projects-for-quantitative-finance-roles-121/)
- [Quantitative Finance Portfolio Projects (OpenQuant)](https://openquant.co/blog/quantitative-finance-portfolio-projects)
- [Ultimate Guide to Landing a Quant Job 2025](https://www.quantblueprint.com/post/the-ultimate-guide-to-landing-a-quant-job-in-2025)
- [Walk-Forward Optimization (IBKR)](https://www.interactivebrokers.com/campus/ibkr-quant-news/the-future-of-backtesting-a-deep-dive-into-walk-forward-analysis/)
- [Hypothesis Testing in Quant Finance](https://reasonabledeviations.com/2021/06/17/hypothesis-testing-quant/)
- [hftbacktest Documentation](https://hftbacktest.readthedocs.io/en/latest/tutorials/Market%20Making%20with%20Alpha%20-%20Order%20Book%20Imbalance.html)

---
*Feature research for: Quantitative Finance / Market Microstructure ML*
*Researched: 2026-03-18*
