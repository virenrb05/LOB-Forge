# Pitfalls Research

**Domain:** Quantitative Finance / Market Microstructure ML (LOB-Forge)
**Researched:** 2026-03-18
**Confidence:** MEDIUM (synthesized from academic papers, practitioner discussions, and documented post-mortems)

---

## Critical Pitfalls

### Pitfall 1: Lookahead Bias in Label Construction

**What goes wrong:**
The labeling scheme for mid-price direction (up/down/stationary) uses future information that would not be available at prediction time. The most common form: computing a label based on a symmetric window around time t (e.g., averaging prices from t-k to t+k and comparing to another window), which leaks future price information into the label itself. The FI-2010 benchmark's own labeling method has been shown to be "prone to instability" and introduces what researchers call "horizon bias."

**Why it happens:**
Academic LOB papers (DeepLOB, TransLOB) use label definitions that seem reasonable mathematically but violate strict temporal causality. Practitioners copy these without questioning. The label looks backward and forward symmetrically, but during live inference you only have the past.

**How to avoid:**
- Use strictly causal labeling: label at time t should only depend on prices at t+h (pure forward return), never on a symmetric window
- Implement the "horizon bias free" labeling from recent LOB forecasting literature
- Write a unit test that verifies no feature at time t references any data after t
- Use an assertion-based data pipeline that flags temporal violations

**Warning signs:**
- Accuracy that seems too high (>70% on 3-class mid-price direction is suspicious for short horizons)
- Model performance drops dramatically when switching from offline evaluation to simulated live inference
- Labels are computed using pandas rolling windows without explicit future-exclusion

**Phase to address:** Phase 1 (Data Pipeline) -- must be foundational before any model training

---

### Pitfall 2: Non-Causal Feature Normalization (Z-Score Lookahead)

**What goes wrong:**
Normalizing LOB features (prices, volumes) using statistics (mean, std) computed over the entire dataset or the entire test set. This is a subtle but devastating form of data leakage because the normalization parameters contain information about future price levels and volume distributions.

**Why it happens:**
The FI-2010 benchmark ships pre-normalized data, so researchers never confront the normalization question. When working with raw data (like Bybit), practitioners default to sklearn's StandardScaler fitted on the full dataset, or compute z-scores per-column across all timestamps.

**How to avoid:**
- Use a rolling window z-score computed only on past data (e.g., trailing 1-hour or 1-day window)
- Alternatively, normalize per-snapshot relative to the current mid-price (price levels as bps from mid, volumes as fraction of total visible depth)
- Never fit a scaler on test data; use walk-forward normalization
- Per-snapshot relative normalization has the added benefit of being stationary across price regimes

**Warning signs:**
- Model works well on in-distribution test data but fails on data from a different price regime
- Normalization code uses `fit_transform` on the full dataset before splitting
- Features contain absolute price levels rather than relative measures

**Phase to address:** Phase 1 (Data Pipeline)

---

### Pitfall 3: Random Train/Test Split on Time Series Data

**What goes wrong:**
Using random shuffled splits (or k-fold cross-validation) instead of temporal splits. This creates massive data leakage because LOB data has strong autocorrelation -- a sample at time t is nearly identical to t+1. Random splitting means the model "memorizes" the test set through its temporal neighbors in the training set.

**Why it happens:**
Default sklearn/PyTorch patterns use random splits. Tutorials for image classification transfer directly. Even practitioners who know better sometimes use random splits "just to check" and then get attached to the inflated numbers.

**How to avoid:**
- Use strict temporal splits: train on [0, T1], validate on [T1+gap, T2], test on [T2+gap, T3]
- Include a gap (purge window) between train/val/test to eliminate autocorrelation leakage
- Use walk-forward validation for robust evaluation: train on expanding or sliding windows, test on subsequent non-overlapping periods
- Report walk-forward results, not single-split results

**Warning signs:**
- Test accuracy is significantly higher than what walk-forward validation shows
- Model performs equally well regardless of which temporal period is used for testing
- No gap/purge between train and test splits

**Phase to address:** Phase 1 (Data Pipeline) and Phase 2 (Model Training)

---

### Pitfall 4: Transformer Overfitting on Non-Stationary Financial Data

**What goes wrong:**
Transformers memorize regime-specific patterns (e.g., a particular volatility cluster or trending period) rather than learning generalizable microstructure dynamics. The model achieves excellent in-sample metrics but performance "degenerates terribly on non-stationary real-world data" (Liu et al., NeurIPS 2022). All 15 state-of-the-art models benchmarked in recent LOB studies show "significant performance drop when exposed to new data."

**Why it happens:**
- Transformers have enormous capacity and will memorize small financial datasets
- LOB data has regime changes (trending, mean-reverting, volatile, calm) that change the joint distribution over time
- "Over-stationarization" -- aggressively normalizing data removes the non-stationary signals the model needs for real-world prediction
- Attention patterns become "indistinguishable" across different market states

**How to avoid:**
- Use aggressive regularization: dropout (0.3+), weight decay, early stopping on temporal validation set
- Train on data spanning multiple market regimes (not just one calm period)
- Consider Non-stationary Transformer techniques (Series Stationarization + De-stationary Attention)
- Use regime-aware evaluation: report metrics separately for trending/mean-reverting/volatile periods
- Keep model small; a 2-layer transformer may generalize better than a 6-layer one on LOB data

**Warning signs:**
- Validation loss diverges from training loss after few epochs
- Model confidence is uniformly high regardless of market regime
- Attention maps show no interpretable pattern (attending to everything equally)
- Performance varies wildly across different test windows

**Phase to address:** Phase 2 (Transformer Training)

---

### Pitfall 5: Focal Loss Miscalibration and Class Imbalance Artifacts

**What goes wrong:**
Focal loss with poorly tuned gamma creates a vanishing gradient problem where "the learning gradient becomes significantly smaller than that of the original CE function when the predicted output prematurely approaches the actual output." Additionally, the 3-class label distribution (up/down/stationary) shifts dramatically with prediction horizon -- short horizons are dominated by "stationary" labels (80%+), making focal loss essential but tricky.

**Why it happens:**
- Default focal loss gamma (2.0 from the object detection paper) is not calibrated for financial 3-class problems
- The class distribution changes with the prediction horizon (k=1: ~80% stationary; k=50: ~40% stationary), but practitioners use the same gamma across horizons
- Focal loss can suppress learning on the majority class so aggressively that the model never learns baseline patterns

**How to avoid:**
- Tune gamma separately for each prediction horizon
- Combine focal loss with class-weighted loss (alpha parameter) calibrated to actual class frequencies
- Monitor per-class precision/recall, not just aggregate accuracy or F1
- Start with standard cross-entropy as baseline, then add focal loss only if stationary class dominates predictions

**Warning signs:**
- Model predicts almost exclusively the majority class despite using focal loss
- Training loss decreases but per-class F1 for minority classes stays near zero
- Changing gamma from 2.0 to 0.5 dramatically changes behavior (indicates sensitivity, not robustness)

**Phase to address:** Phase 2 (Transformer Training)

---

### Pitfall 6: Diffusion Model Generates Statistically Plausible but Dynamically Wrong LOB Sequences

**What goes wrong:**
The diffusion model produces synthetic LOB snapshots that match marginal distributions (spread, depth, imbalance) individually but fail to reproduce temporal dynamics: autocorrelation structure, volatility clustering, cross-level correlations, and realistic order flow patterns. The data "looks right" snapshot-by-snapshot but is dynamically incoherent as a sequence.

**Why it happens:**
- Most diffusion models treat each sample independently; temporal conditioning is hard to get right
- Evaluation focuses on marginal statistics (KS test, MMD on single snapshots) rather than joint temporal statistics
- LOB data has complex cross-feature constraints (bid < ask, volume monotonicity away from best prices) that are easy to violate
- Mode collapse in conditional generation: the model learns a "generic" LOB shape rather than diverse market states

**How to avoid:**
- Validate with temporal metrics: autocorrelation of returns, volatility clustering (GARCH-like properties), Hurst exponent
- Check cross-feature constraints: bid-ask ordering, volume profiles, spread distribution
- Use LOB-Bench metrics: spread, depth, imbalance, inter-arrival times, cross-correlation, price response functions
- Train on Synthetic / Test on Real (TSTR): if a downstream model trained on synthetic data performs significantly worse than one trained on real data, the synthetic data is missing critical structure
- Validate stylized facts explicitly: fat tails in returns, negative autocorrelation of returns at short lags, long memory in absolute returns

**Warning signs:**
- Synthetic spread distribution matches real but autocorrelation of spread changes is flat
- Generated sequences have unrealistic smoothness (no sudden liquidity events)
- TSTR downstream performance is >10% worse than real-data training
- Bid-ask inversions or negative volumes appear in generated samples

**Phase to address:** Phase 3 (Diffusion Model)

---

### Pitfall 7: RL Agent Trained in Unrealistic Market Environment

**What goes wrong:**
The execution agent learns to exploit simulator artifacts rather than learning genuine execution strategies. Common simulator flaws: (1) assuming execution at mid-price instead of walking the book, (2) no market impact -- the agent's trades don't move the price, (3) infinite liquidity at each price level, (4) no latency between decision and execution. The agent "performs consistently close to arrival price" in simulation but fails catastrophically in any realistic setting.

**Why it happens:**
- Building a realistic LOB simulator is extremely hard; practitioners take shortcuts
- Historical replay doesn't react to the agent's trades -- the agent learns that it can "trade for free"
- Reward functions that only measure execution price vs. arrival price ignore slippage, market impact, and timing risk
- "RL methods with only high profit on backtesting are likely to overfit on historical data and fail in real-world deployment"

**How to avoid:**
- Implement a market impact model (even simple: linear permanent + temporary impact)
- Use order book replay with realistic fill simulation: match against actual book depth, partial fills
- Include realistic latency (even 100ms changes execution dynamics significantly in crypto)
- Validate agent behavior makes economic sense: does it split large orders? Does it use limit orders? Does it avoid trading during low liquidity?
- Compare against simple baselines: TWAP, VWAP. If the RL agent can't consistently beat TWAP, something is wrong with the setup, not the algorithm

**Warning signs:**
- Agent achieves near-zero implementation shortfall (unrealistic)
- Agent always executes at or better than mid-price
- Agent strategy doesn't change with order size (no size-awareness)
- Agent performs identically regardless of market volatility or liquidity
- Training reward increases monotonically without plateaus

**Phase to address:** Phase 4 (RL Agent)

---

### Pitfall 8: Reward Hacking in Double-DQN Execution Agent

**What goes wrong:**
The agent discovers and exploits flaws in the reward function rather than learning genuine execution skill. Common hacks: (1) the agent learns to "wait" because the reward penalizes market impact but doesn't penalize time risk, so doing nothing scores well, (2) the agent learns to trade in specific historical patterns it memorized, (3) larger neural networks find "more complex reward hacking exploits." With Double-DQN specifically, Q-value overestimation (though reduced vs. DQN) can still create phantom value in unexplored state-action regions.

**Why it happens:**
- Reward function is a proxy for what we actually want (good execution quality), and "agents exploit flaws in proxy reward functions, deviating from true objectives"
- Financial environments have low signal-to-noise, making it easy for the agent to find spurious correlations
- Curriculum learning can mask the problem: easy environments have exploitable structure that doesn't transfer

**How to avoid:**
- Use vector rewards: separate components for price improvement, completion rate, time penalty, impact penalty
- Monitor the agent's actual behavior, not just its reward: visualize trade trajectories
- Test on out-of-distribution market conditions (high volatility, low liquidity, flash crashes)
- Implement sanity checks: the agent must complete the order within the time window
- Use curriculum learning carefully: verify the agent's strategy transfers between difficulty levels

**Warning signs:**
- Agent achieves high reward but doesn't actually trade (or trades all at once)
- Agent's strategy is identical across different market conditions
- Q-values grow unbounded during training
- Agent performs worse when you improve the simulator's realism

**Phase to address:** Phase 4 (RL Agent)

---

### Pitfall 9: Crypto-Specific LOB Data Quality Issues (Bybit)

**What goes wrong:**
Bybit LOB data has unique quality problems that don't exist in traditional equity LOB data: intermittent order book updates due to exchange/communication delays, timestamp drift, silent data gaps, WebSocket reconnection artifacts, and the 24/7 nature means there's no natural session boundary for normalization. "Discrepancies between actual and expected outcomes on Bybit are strongly correlated with market factors such as volatility, latency, and LOB liquidity."

**Why it happens:**
- Crypto exchanges have no regulatory requirement for data quality
- WebSocket feeds can drop updates silently, creating artificial gaps in the LOB
- Bybit's API rate limits and update frequency differ from what academic LOB papers assume
- "There is no such thing as standardized real-time crypto data unless someone normalizes it for you"

**How to avoid:**
- Implement data quality checks: detect gaps >1s, validate bid-ask ordering, check for frozen orderbooks
- Use sequence numbering from the exchange to detect dropped messages
- Build a gap-filling strategy (interpolation vs. dropping sequences with gaps)
- Validate data against a second source (e.g., compare Bybit snapshots against Tardis.dev historical data)
- Log and report data quality metrics alongside model metrics

**Warning signs:**
- Sudden jumps in mid-price without corresponding order book changes
- Periods where the order book doesn't change for seconds (likely dropped updates, not genuine)
- Bid-ask spreads of zero or negative values
- Volume at best bid/ask jumps discontinuously

**Phase to address:** Phase 1 (Data Pipeline)

---

### Pitfall 10: MPS/Apple Silicon Training Produces Non-Reproducible or Incorrect Results

**What goes wrong:**
PyTorch's MPS backend has documented numerical issues: (1) no float64 support -- silently falls back to CPU or truncates, (2) addcmul_/addcdiv_ operations "silently fail when writing to non-contiguous output tensors" causing weights to freeze during training, (3) scaled dot-product attention has memory issues with long sequences, (4) FP16 rounding errors accumulate over long sequences, (5) mixed precision on MPS "may not provide consistent speedups or numerical stability compared to CUDA AMP."

**Why it happens:**
- MPS backend is still maturing compared to CUDA
- Many PyTorch operations are "emulated" on MPS (upcast to BF16/FP32 internally), creating a "correctness gap"
- Bugs manifest as silent incorrect results, not crashes -- the most dangerous kind
- Models may converge on MPS (due to hidden higher-precision math) but diverge on other hardware

**How to avoid:**
- Stick with float32 on MPS (avoid mixed precision)
- Run periodic validation against CPU results to catch MPS-specific numerical drift
- Test critical operations (attention, custom loss functions) on both MPS and CPU, compare outputs
- Use `.contiguous()` explicitly before in-place operations
- Pin PyTorch version and document known MPS issues for your specific version
- Consider that results on MPS may not transfer to CUDA/cloud GPU -- document this limitation

**Warning signs:**
- Weights not updating (gradients are zero) despite non-zero loss
- Training loss suddenly becomes NaN
- Results differ between MPS and CPU for the same model/data/seed
- Model trains fine but produces garbage on inference

**Phase to address:** Phase 0 (Environment Setup) and ongoing throughout all phases

---

### Pitfall 11: Portfolio Project Red Flags That Get You Immediately Dismissed

**What goes wrong:**
The project looks like a homework assignment, not a demonstration of genuine understanding. Specific dismissal triggers: (1) showing backtest results without discussing overfitting risk, (2) claiming unrealistic accuracy/returns without uncertainty quantification, (3) inability to explain design choices when questioned, (4) no comparison against simple baselines, (5) copying architecture from a paper without understanding why it works, (6) no discussion of failure modes or limitations.

**Why it happens:**
- "If you just copy and paste work without understanding it, and an interviewer asks about it and you can't explain anything about it, then you fail"
- Projects optimize for looking impressive rather than being rigorous
- Quant firms value intellectual honesty and understanding of limitations more than flashy results

**How to avoid:**
- Include a "Limitations" section prominently in the README and presentation
- Show baseline comparisons: TWAP vs. RL agent, logistic regression vs. transformer
- Report metrics with confidence intervals, not point estimates
- Document what didn't work and why -- this demonstrates genuine understanding
- Be prepared to explain every architectural choice: why dual attention? Why focal loss? Why diffusion and not GAN?
- Show you understand market microstructure, not just ML: can you explain the bid-ask bounce? Why LOB imbalance predicts short-term returns?

**Warning signs:**
- You can't explain your results without looking at your code
- Your model "works" but you don't know why
- You haven't tested what happens when assumptions are violated
- You don't have baseline comparisons

**Phase to address:** Phase 5 (Documentation/Presentation) but must be built into every phase

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Using FI-2010 preprocessed data instead of raw Bybit data | Skip data pipeline, start modeling immediately | Cannot demonstrate data engineering skill; model may not transfer to live data; "original LOB cannot be backtracked" | Never for this project (raw data handling is a selling point) |
| Single train/test split instead of walk-forward validation | Faster iteration, simpler code | Overfitting goes undetected; inflated metrics; reviewers will ask about this | Early prototyping only; must switch before any metrics are reported |
| Hardcoded normalization constants | Quick to implement | Breaks when price regime changes; non-transferable | Never |
| Using mid-price for RL fill simulation | Simpler environment, faster training | Agent learns unrealistic execution; zero market impact; will be questioned in interviews | Never for final results; acceptable for debugging environment mechanics |
| Skipping stylized fact validation for diffusion model | Faster iteration on diffusion model | Cannot claim synthetic data is "realistic"; downstream RL agent trains on unrealistic data | Never for final results |
| Training on single market condition | Faster convergence, better in-sample metrics | Model fails on regime change; overfitting | Early prototyping only |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Bybit WebSocket API | Assuming continuous, gap-free data stream | Implement reconnection logic, sequence number tracking, gap detection, and backfill from REST API snapshots |
| PyTorch MPS Backend | Using same training code as CUDA without modification | Avoid mixed precision, use `.contiguous()` before in-place ops, validate against CPU periodically |
| Diffusion -> RL Pipeline | Treating synthetic LOB data as drop-in replacement for real data | Validate synthetic data with TSTR metrics before using for RL training; blend real and synthetic data |
| Transformer -> Diffusion Pipeline | Using transformer predictions as diffusion conditioning without calibration | Ensure transformer output probabilities are well-calibrated (use temperature scaling); condition on logits not argmax |
| Walk-Forward Evaluation | Using different normalization windows for train vs. test within each fold | Normalization must be re-computed per fold using only that fold's training data |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading full LOB history into memory | OOM on Apple Silicon (16-64GB), slow startup | Use memory-mapped files or chunked data loading; process in temporal windows | >2 weeks of tick-level Bybit BTC-USDT data (~5-10GB) |
| Attention computation on full LOB depth | Quadratic memory growth with sequence length; MPS SDPA crashes >12K tokens | Limit attention window; use sliding window attention or sparse attention patterns | Sequences >4K tokens on 16GB M-series |
| Storing all RL replay buffer in RAM | Memory exhaustion during long training runs | Use prioritized replay with fixed buffer size; disk-backed buffer for experience storage | >1M transitions with full LOB state |
| Per-tick diffusion inference | Each generated LOB snapshot requires full denoising chain | Generate in batches; cache intermediate denoising steps; use DDIM for fewer steps | When generating >10K synthetic snapshots |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Committing Bybit API keys to GitHub | Account compromise; unauthorized trading | Use `.env` files with `.gitignore`; use environment variables; never hardcode keys |
| Training on live exchange data without rate limiting | IP ban from Bybit; account suspension | Implement rate limiting; use historical data providers (Tardis.dev) for bulk data |
| Storing raw exchange data without access controls | Data licensing violations (exchange ToS) | Review Bybit's data redistribution policy; store processed/aggregated data only in public repos |

## "Looks Done But Isn't" Checklist

- [ ] **Label Construction:** Labels may use future data -- verify by checking that label at time t only depends on prices at times > t, with no symmetric windows
- [ ] **Normalization:** Stats may be computed on test data -- verify that normalization parameters are computed only on training data, per walk-forward fold
- [ ] **Train/Test Split:** May use random split -- verify temporal ordering is preserved with a purge gap between splits
- [ ] **Accuracy Metrics:** May report aggregate accuracy -- verify per-class precision/recall/F1 for all three classes (up/down/stationary)
- [ ] **Focal Loss:** May suppress minority class learning -- verify that changing gamma doesn't cause model to predict only majority class
- [ ] **Diffusion Output:** May pass marginal distribution tests but fail temporal tests -- verify autocorrelation structure, volatility clustering, and bid-ask ordering in generated sequences
- [ ] **Synthetic Data Quality:** May look like LOB data snapshot-by-snapshot -- verify by running TSTR (Train on Synthetic, Test on Real) and comparing to real-data baseline
- [ ] **RL Environment:** May have zero market impact -- verify that large orders move the price and that the agent's fill rate is < 100% for large orders
- [ ] **RL Agent Behavior:** May achieve high reward by not trading -- verify the agent actually executes the full order within the time window
- [ ] **Baseline Comparisons:** May show transformer beats "nothing" -- verify comparison against meaningful baselines (logistic regression, TWAP for RL)
- [ ] **MPS Numerics:** May train without errors but produce wrong results -- verify key computations match CPU output to within tolerance
- [ ] **Walk-Forward Results:** May report single-split results -- verify that reported metrics are averaged across multiple walk-forward folds with standard deviations
- [ ] **Presentation:** May show impressive numbers -- verify you can explain every result, every design choice, and every limitation without looking at code

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Lookahead bias in labels | MEDIUM | Re-label all data with causal labels; retrain all models; compare old vs. new metrics to quantify bias magnitude |
| Non-causal normalization | MEDIUM | Re-implement normalization pipeline; retrain; existing model architectures can be reused |
| Random train/test split | LOW | Re-split temporally; re-evaluate (no retraining needed if model was saved) |
| Transformer overfitting | HIGH | Requires architecture changes, regularization tuning, possibly more training data from different regimes |
| Focal loss miscalibration | LOW | Hyperparameter sweep on gamma/alpha; only affects loss function, not data pipeline |
| Unrealistic diffusion output | HIGH | May require architecture changes to temporal conditioning; need comprehensive evaluation framework first |
| Unrealistic RL environment | HIGH | Requires rebuilding simulation environment with market impact; all agent training must be repeated |
| Reward hacking | MEDIUM | Redesign reward function; add constraints; retrain agent (environment can be reused) |
| Bybit data quality issues | HIGH | Must re-collect or re-process all raw data; downstream retraining required |
| MPS numerical issues | MEDIUM | Add CPU validation checks; may need to retrain if results were corrupted; pin PyTorch version |
| Poor project presentation | LOW | Add baselines, limitations section, and confidence intervals; no retraining needed |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Lookahead bias in labels | Phase 1: Data Pipeline | Unit test: assert no feature/label at t uses data from t+1..T |
| Non-causal normalization | Phase 1: Data Pipeline | Compare normalized values with/without future data; should be identical |
| Random train/test split | Phase 1: Data Pipeline | Assert all training timestamps < all validation timestamps < all test timestamps |
| Crypto data quality (Bybit) | Phase 1: Data Pipeline | Data quality report: gap frequency, bid-ask violations, frozen book detection |
| MPS numerical issues | Phase 0: Environment Setup | CPU vs. MPS comparison test suite; run before and after every PyTorch upgrade |
| Transformer overfitting | Phase 2: Transformer | Walk-forward validation; per-regime performance breakdown |
| Focal loss miscalibration | Phase 2: Transformer | Per-class F1 scores; gamma sensitivity analysis |
| Diffusion temporal incoherence | Phase 3: Diffusion Model | Stylized fact test suite: autocorrelation, volatility clustering, TSTR |
| Unrealistic RL environment | Phase 4: RL Agent | Market impact verification; fill rate < 100% for large orders |
| Reward hacking | Phase 4: RL Agent | Behavior visualization; out-of-distribution evaluation |
| Poor presentation | Phase 5: Documentation | Limitations section exists; baseline comparisons included; can explain all results verbally |

## Sources

### Academic Papers (HIGH confidence)
- [Non-stationary Transformers (NeurIPS 2022)](https://arxiv.org/abs/2205.14415) -- over-stationarization problem, attention degeneracy on non-stationary data
- [LOB-based deep learning benchmark study (AI Review 2024)](https://link.springer.com/article/10.1007/s10462-024-10715-4) -- all 15 models show significant performance drop on new data; FI-2010 limitations
- [Deep limit order book forecasting: a microstructural guide](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/) -- FI-2010 pre-processed limitations, labeling instability, normalization issues
- [LOB-Bench: Benchmarking Generative AI for LOB](https://arxiv.org/html/2502.09172v1) -- comprehensive evaluation metrics for synthetic LOB data
- [Generation of synthetic financial time series by diffusion models](https://arxiv.org/abs/2410.18897) -- diffusion vs. GAN/VAE for stylized fact reproduction
- [CoFinDiff: Controllable Financial Diffusion Model](https://arxiv.org/abs/2503.04164) -- conditional diffusion for financial data with stylized fact validation
- [Optimal Execution with RL in Multi-Agent Market Simulator](https://arxiv.org/html/2411.06389v2) -- RL execution strategies, sim-to-real considerations
- [Guidelines for Building a Realistic Market Simulator (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3429261) -- market impact modeling, backtesting assumptions
- [TLOB: Transformer with Dual Attention for LOB](https://arxiv.org/html/2502.15757v1) -- dual attention architecture for LOB price prediction

### Practitioner Discussions (MEDIUM confidence)
- [Deep Reinforcement Learning Doesn't Work Yet (Alex Irpan)](https://www.alexirpan.com/2018/02/14/rl-hard.html) -- fundamental RL challenges: reward sensitivity, reproducibility, local optima
- [The Alpha Scientist: Walk-Forward Modeling](https://alphascientist.com/walk_forward_model_building.html) -- walk-forward validation methodology for financial ML
- [Look-ahead Bias: Silent Killer of Trading Strategies](https://medium.com/funny-ai-quant/look-ahead-bias-in-quantitative-finance-the-silent-killer-of-trading-strategies-bbbbb31d943a) -- lookahead bias patterns and detection
- [Data Leakage, Lookahead Bias, and Causality in Time Series](https://medium.com/@kyle-t-jones/data-leakage-lookahead-bias-and-causality-in-time-series-analytics-76e271ba2f6b) -- temporal data leakage taxonomy
- [Bybit and Binance: Exploratory Trading (Quant Finance journal)](https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2515933) -- Bybit LOB data discrepancies, latency issues
- [Why Real-Time Crypto Data Is Harder Than It Looks (CoinAPI)](https://www.coinapi.io/blog/why-real-time-crypto-data-is-harder-than-it-looks) -- crypto data standardization challenges

### Technical Resources (MEDIUM confidence)
- [The bug that taught me more about PyTorch than years of using it](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/) -- MPS backend silent failures
- [Apple Silicon PyTorch MPS Setup and Speed](https://tillcode.com/apple-silicon-pytorch-mps-setup-and-speed-expectations/) -- MPS numerical precision limitations
- [State of PyTorch Hardware Acceleration 2025](https://tunguz.github.io/PyTorch_Hardware_2025/) -- MPS vs CUDA correctness gap
- [Reward Hacking in RL](https://www.emergentmind.com/topics/reward-hacking) -- taxonomy of reward exploitation
- [Focal Loss for Class Imbalance](https://medium.com/data-science-ecom-express/focal-loss-for-handling-the-issue-of-class-imbalance-be7addebd856) -- vanishing gradient problem with high gamma

### Portfolio/Hiring (MEDIUM confidence)
- [Quantitative Finance Portfolio Projects (OpenQuant)](https://openquant.co/blog/quantitative-finance-portfolio-projects) -- what makes quant projects stand out
- [Top 5 Resume Mistakes in Quant Finance (LinkedIn)](https://www.linkedin.com/pulse/top-5-resume-mistakes-quant-finance-dimitri-bianco-frm) -- presentation anti-patterns

### Unverified but Plausible (LOW confidence)
- Claim that >70% accuracy on 3-class LOB prediction at short horizons indicates data leakage -- based on practitioner consensus, not formal study
- MPS addcmul_/addcdiv_ bug on non-contiguous tensors -- reported in PyTorch issues, may be version-specific
- Specific gamma values for focal loss on LOB data -- no systematic study found; requires empirical tuning

---
*Pitfalls research for: Quantitative Finance / Market Microstructure ML (LOB-Forge)*
*Researched: 2026-03-18*
