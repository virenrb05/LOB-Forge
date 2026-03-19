# Architecture Research

**Domain:** Quantitative Finance / Market Microstructure ML
**Researched:** 2026-03-18
**Confidence:** MEDIUM (composite -- individual sections rated separately)

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │  Raw LOB     │  │  FI-2010 /   │  │  Preprocessed│                   │
│  │  Messages    │  │  LOBSTER     │  │  Tensors     │                   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │
│         └──────────────────┴─────────────────┘                          │
│                            │                                            │
│                    ┌───────▼───────┐                                     │
│                    │ Normalization │                                     │
│                    │ & Labelling   │                                     │
│                    └───────┬───────┘                                     │
├────────────────────────────┼────────────────────────────────────────────┤
│                     MODEL LAYER                                         │
│                            │                                            │
│         ┌──────────────────┼──────────────────┐                         │
│         ▼                  ▼                  ▼                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ MODULE A     │  │ MODULE B     │  │ MODULE C     │                   │
│  │ Dual-Attn    │  │ Conditional  │  │ Double-DQN   │                   │
│  │ Transformer  │  │ Diffusion    │  │ Execution    │                   │
│  │ (Predictor)  │  │ (Generator)  │  │ (Agent)      │                   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │
│         │                 │                  │                          │
│         │   embeddings    │   synthetic      │                          │
│         └────────────────►│   LOB envs       │                          │
│                           └─────────────────►│                          │
│                                              │                          │
├──────────────────────────────────────────────┼──────────────────────────┤
│                  EXPERIMENT LAYER             │                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────▼──────┐                   │
│  │ Hydra Configs│  │ W&B Tracking │  │ Checkpoints  │                   │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Predictor -> Generator -> Executor

```
                 TRAINING PHASE
                 ==============

 ┌────────────┐   LOB snapshots    ┌────────────────┐
 │ Historical │──────────────────►│ Module A:       │
 │ LOB Data   │                   │ Transformer     │
 │ (T x 4L)   │                   │ Predictor       │
 └────────────┘                   └───────┬─────────┘
                                          │
                              learned embeddings
                              + price predictions
                                          │
                                          ▼
                                  ┌───────────────┐
                                  │ Module B:      │
      past LOB snapshots ────────►│ Conditional    │
      + order history            │ Diffusion      │
                                  │ Generator      │
                                  └───────┬────────┘
                                          │
                              synthetic LOB trajectories
                              (realistic market environments)
                                          │
                                          ▼
                                  ┌───────────────┐
                                  │ Module C:      │
           reward signal ◄────────│ Double-DQN     │
           (execution cost)       │ Execution      │
                                  │ Agent          │
                                  └────────────────┘

                 INFERENCE PHASE
                 ===============

 Live LOB ──► Predictor ──► (optional conditioning) ──► Agent ──► Orders
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation | Confidence |
|-----------|----------------|------------------------|------------|
| **Data Preprocessor** | Ingest raw LOB (LOBSTER/FI-2010), normalize, label mid-price movements | Rolling z-score normalization (5-day window), 3-class labelling (up/down/stationary), train/val/test splits | HIGH |
| **Module A: Predictor** | Predict mid-price movement direction from LOB snapshots | Dual-attention transformer (temporal + spatial self-attention), bilinear normalization, 4 encoder layers, sequence length ~128, input shape [B, T, 4L] | HIGH |
| **Module B: Generator** | Generate realistic synthetic LOB trajectories conditioned on market state | DDPM with transformer or WaveNet backbone, conditioned on historical LOB + predictor embeddings + regime variables, sliding-window autoregressive generation | MEDIUM |
| **Module C: Executor** | Learn optimal trade execution policy minimizing market impact | Double DQN with experience replay, state = (LOB features, inventory, time remaining), discrete action space (order sizes/timing), reward = negative execution cost vs. VWAP/TWAP | HIGH |
| **Config System** | Manage hyperparameters and experiment composition | Hydra/OmegaConf hierarchical YAML configs with `_target_` instantiation | HIGH |
| **Experiment Tracker** | Log metrics, hyperparameters, model artifacts | W&B integration: log train/val loss, prediction accuracy, diffusion sample quality, RL cumulative reward | HIGH |

## Recommended Project Structure

```
lob_forge/
├── configs/                        # Hydra configuration root
│   ├── config.yaml                 # Main defaults composition
│   ├── module_a/                   # Predictor configs
│   │   ├── tlob.yaml               # TLOB transformer config
│   │   └── deeplob.yaml            # DeepLOB baseline config
│   ├── module_b/                   # Generator configs
│   │   ├── diffusion.yaml          # Diffusion model config
│   │   └── scheduler.yaml          # Noise schedule config
│   ├── module_c/                   # Executor configs
│   │   ├── double_dqn.yaml         # DQN architecture + RL config
│   │   └── env.yaml                # Environment parameters
│   ├── data/                       # Dataset configs
│   │   ├── fi2010.yaml
│   │   └── lobster.yaml
│   ├── trainer/                    # Training loop configs
│   │   ├── default.yaml
│   │   └── debug.yaml
│   ├── logger/                     # W&B / tensorboard configs
│   │   └── wandb.yaml
│   ├── callbacks/                  # Checkpoint, early stopping
│   │   └── default.yaml
│   └── experiment/                 # Full experiment overrides
│       ├── predictor_fi2010.yaml
│       ├── generator_lobster.yaml
│       └── full_pipeline.yaml
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── data/                       # Data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── datasets.py             # PyTorch Dataset classes
│   │   ├── datamodules.py          # Lightning DataModules
│   │   ├── preprocessing.py        # Raw LOB -> normalized tensors
│   │   ├── normalization.py        # Z-score, min-max, bilinear
│   │   └── labelling.py            # Mid-price movement labels
│   │
│   ├── models/                     # Neural network definitions
│   │   ├── __init__.py
│   │   ├── predictor/              # Module A
│   │   │   ├── __init__.py
│   │   │   ├── tlob.py             # Dual-attention transformer
│   │   │   ├── attention.py        # Temporal + spatial attention
│   │   │   └── deeplob.py          # CNN-LSTM baseline
│   │   ├── generator/              # Module B
│   │   │   ├── __init__.py
│   │   │   ├── diffusion.py        # DDPM forward/reverse process
│   │   │   ├── unet.py             # Denoising network backbone
│   │   │   └── conditioning.py     # Embedding injection / FiLM
│   │   └── executor/               # Module C
│   │       ├── __init__.py
│   │       ├── double_dqn.py       # Double DQN agent
│   │       ├── replay_buffer.py    # Experience replay
│   │       └── networks.py         # Q-network architectures
│   │
│   ├── environments/               # RL environment wrappers
│   │   ├── __init__.py
│   │   ├── lob_env.py              # Gym-compatible LOB environment
│   │   ├── synthetic_env.py        # Env backed by diffusion samples
│   │   └── rewards.py              # Reward functions (IS, VWAP)
│   │
│   ├── training/                   # Training loops & orchestration
│   │   ├── __init__.py
│   │   ├── train_predictor.py      # Module A training
│   │   ├── train_generator.py      # Module B training
│   │   ├── train_executor.py       # Module C training (RL loop)
│   │   └── pipeline.py             # End-to-end orchestration
│   │
│   └── utils/                      # Shared utilities
│       ├── __init__.py
│       ├── metrics.py              # Accuracy, F1, execution cost
│       ├── visualization.py        # LOB heatmaps, training curves
│       └── io.py                   # Checkpoint save/load helpers
│
├── scripts/                        # Entry points
│   ├── train.py                    # Hydra main entry point
│   ├── evaluate.py                 # Model evaluation
│   ├── generate.py                 # Generate synthetic LOB data
│   └── backtest.py                 # Strategy backtesting
│
├── tests/                          # Unit + integration tests
│   ├── test_data/
│   ├── test_models/
│   └── test_environments/
│
├── notebooks/                      # Exploration & visualization
│
├── data/                           # Data storage (gitignored)
│   ├── raw/                        # Original LOB files
│   ├── processed/                  # Normalized tensors
│   └── synthetic/                  # Generated LOB trajectories
│
├── outputs/                        # Hydra run outputs (gitignored)
├── logs/                           # W&B logs (gitignored)
├── checkpoints/                    # Model checkpoints (gitignored)
│
├── pyproject.toml                  # Project metadata & dependencies
└── Makefile                        # Common commands
```

### Structure Rationale

- **`configs/`:** Mirrors the three-module architecture. Each module has its own config group, enabling independent hyperparameter sweeps. Experiment configs compose across modules for full-pipeline runs. Follows the lightning-hydra-template pattern.
- **`src/models/`:** Three subdirectories enforce clear module boundaries. Each module is independently trainable, testable, and versionable. Shared layers (e.g., attention primitives) can be promoted to a `src/models/common/` directory if needed.
- **`src/environments/`:** Separated from models because the RL environment is a distinct abstraction -- it wraps either historical or synthetic LOB data into a Gym-compatible interface that Module C consumes.
- **`src/training/`:** Keeps training orchestration separate from model definitions. Each module has its own training script because they use fundamentally different paradigms (supervised, self-supervised diffusion, RL).
- **`scripts/`:** Thin entry points that compose Hydra configs and call training functions. Keeps the `src/` package importable without side effects.

## Architectural Patterns

### Pattern 1: Hierarchical Hydra Config Composition

**What:** Each architectural choice (model, dataset, trainer, logger) is a separate config group. A main `config.yaml` declares defaults that can be overridden per-experiment.
**When to use:** Always -- this is the standard for multi-module ML projects.
**Trade-offs:** Powerful composability, but config debugging can be opaque. Use `--cfg job` to inspect resolved configs.

**Example:**
```yaml
# configs/config.yaml
defaults:
  - data: lobster
  - module_a: tlob
  - module_b: diffusion
  - module_c: double_dqn
  - trainer: default
  - logger: wandb
  - callbacks: default
  - _self_

seed: 42
```

```yaml
# configs/experiment/full_pipeline.yaml
# @package _global_
defaults:
  - override /data: lobster
  - override /module_a: tlob
  - override /module_b: diffusion
  - override /module_c: double_dqn

module_a:
  hidden_dim: 144
  n_layers: 4
  seq_len: 128

module_b:
  diffusion_steps: 1000
  conditioning: ["embeddings", "regime"]
```

### Pattern 2: Embedding-Conditioned Generation (Predictor -> Generator coupling)

**What:** The predictor's learned representations (intermediate embeddings, not just classification outputs) condition the diffusion model's denoising process. This transfers market-state understanding to the generator without requiring end-to-end training.
**When to use:** When the predictor and generator are trained sequentially, and you want the generator to respect the predictor's learned market dynamics.
**Trade-offs:** Decoupled training is simpler and more stable than end-to-end, but there is an information bottleneck at the embedding interface. Embedding dimensionality and what layer to extract from become hyperparameters.

**Example:**
```python
# src/models/generator/conditioning.py
class EmbeddingConditioner(nn.Module):
    """Injects predictor embeddings into diffusion denoising via FiLM."""

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.scale_proj = nn.Linear(embed_dim, hidden_dim)
        self.shift_proj = nn.Linear(embed_dim, hidden_dim)

    def forward(self, h: Tensor, embedding: Tensor) -> Tensor:
        scale = self.scale_proj(embedding)
        shift = self.shift_proj(embedding)
        return h * (1 + scale) + shift  # FiLM modulation
```

### Pattern 3: Synthetic Environment for RL Training

**What:** The diffusion generator produces synthetic LOB trajectories that are wrapped in a Gym-compatible environment. The RL agent trains against these synthetic environments, optionally mixed with historical replay.
**When to use:** When real LOB data is limited, or when you need counterfactual scenarios (e.g., different liquidity regimes) for robust policy learning.
**Trade-offs:** Agent performance is bounded by generator fidelity. Validate synthetic environments against real LOB statistical properties (autocorrelation, spread distribution, volume profiles) before training the agent.

**Example:**
```python
# src/environments/synthetic_env.py
class SyntheticLOBEnv(gym.Env):
    """LOB execution environment backed by diffusion-generated data."""

    def __init__(self, generator: DiffusionGenerator, config: DictConfig):
        self.generator = generator
        self.trajectory = None

    def reset(self):
        # Generate fresh synthetic LOB trajectory
        self.trajectory = self.generator.sample(
            n_steps=self.config.episode_length,
            regime=self.config.regime,
        )
        self.step_idx = 0
        return self._get_obs()

    def step(self, action):
        # Execute action against synthetic LOB state
        fill_price, fill_qty = self._simulate_execution(action)
        reward = self._compute_reward(fill_price, fill_qty)
        self.step_idx += 1
        done = self.step_idx >= len(self.trajectory)
        return self._get_obs(), reward, done, {}
```

## Data Flow

### Training Flow (Three-Phase Sequential)

```
Phase 1: Train Predictor (Supervised)
──────────────────────────────────────
Historical LOB Data
    ↓ (DataModule)
[batch: B x T x 4L] ──► TLOB Transformer ──► [B x 3] class logits
    ↓                         ↓
Cross-entropy loss       Save embeddings extractor
    ↓                         ↓
W&B log metrics          Checkpoint best model

Phase 2: Train Generator (Self-Supervised Diffusion)
────────────────────────────────────────────────────
Historical LOB Data + Frozen Predictor Embeddings
    ↓
[LOB trajectory + embeddings + regime] ──► Conditional DDPM
    ↓                                           ↓
Denoising loss (MSE on predicted noise)    Sample quality metrics
    ↓                                           ↓
W&B log metrics                            Checkpoint best model

Phase 3: Train Executor (Reinforcement Learning)
─────────────────────────────────────────────────
Synthetic LOB Env (from Generator) + Historical LOB Env
    ↓
[state: LOB snapshot + inventory + time] ──► Double DQN
    ↓                                             ↓
Bellman loss (TD error)                      Execution cost metrics
    ↓                                             ↓
W&B log metrics                              Checkpoint best model
```

### Preprocessing Pipeline

```
Raw LOBSTER Files
    ↓
┌───────────────────────────────┐
│ 1. Filter auction periods     │
│ 2. Remove crossed quotes      │
│ 3. Collapse duplicate stamps  │
│ 4. Extract top-K levels       │
└───────────┬───────────────────┘
            ↓
┌───────────────────────────────┐
│ 5. Rolling z-score normalize  │
│    (5-day window per feature) │
│ 6. Compute mid-price labels   │
│    (3-class: up/down/stable)  │
│ 7. Sliding window sequences   │
│    [T x 4L] per sample        │
└───────────┬───────────────────┘
            ↓
     PyTorch Tensors
     (saved to data/processed/)
```

### Key Data Flows

1. **Predictor Embeddings -> Generator:** The predictor is frozen after Phase 1. Its intermediate representations (pre-classification-head activations) are extracted for each LOB window and passed as conditioning vectors to the diffusion model. This is analogous to CLIP embeddings conditioning Stable Diffusion.
2. **Generator Samples -> RL Environment:** The trained generator produces variable-length LOB trajectories on demand. These are wrapped in a Gym environment that simulates order matching, slippage, and market impact against the synthetic book state.
3. **Config -> All Modules:** Hydra composes a single DictConfig object that is passed to each module's constructor via `hydra.utils.instantiate()`. W&B receives the resolved config for full experiment reproducibility.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Single GPU, small dataset (FI-2010) | Monolithic training scripts, in-memory data loading, no distributed training needed |
| Single GPU, large dataset (full LOBSTER) | Memory-mapped datasets, gradient accumulation, mixed precision (AMP), streaming data loading |
| Multi-GPU | PyTorch DDP for predictor/generator. RL agent typically single-GPU but with parallel environment workers (vectorized envs) |
| Multi-node | Predictor benefits most from distributed training. Generator and agent rarely need multi-node. Consider generating synthetic data offline as a batch job |

### Scaling Priorities

1. **First bottleneck: Data preprocessing.** Raw LOBSTER files are large (GBs per day). Pre-process once, save as memory-mapped arrays (numpy .npy) or HDF5. The sliding window operation can be done lazily in the DataLoader.
2. **Second bottleneck: Diffusion sampling speed.** Generating synthetic trajectories is slow (many denoising steps). Use DDIM or DPM-Solver for faster sampling. Pre-generate and cache synthetic environments before RL training.
3. **Third bottleneck: RL sample efficiency.** Double DQN with experience replay is sample-efficient, but environment step throughput matters. Use vectorized environments and consider prioritized experience replay.

## Anti-Patterns

### Anti-Pattern 1: End-to-End Training of All Three Modules

**What people do:** Attempt to backpropagate gradients from the RL reward through the generator and into the predictor.
**Why it is wrong:** The three modules use fundamentally different loss functions (cross-entropy, denoising MSE, Bellman TD). Joint optimization creates unstable gradients and mode collapse. The diffusion sampling process is not straightforwardly differentiable.
**Do this instead:** Train sequentially (predictor -> generator -> executor). Freeze upstream modules when training downstream ones. Iterate if needed.

### Anti-Pattern 2: Using Raw LOB Data Without Normalization

**What people do:** Feed raw price/volume data directly into neural networks.
**Why it is wrong:** LOB features span vastly different scales (prices in hundreds vs. volumes in thousands). Financial time series are non-stationary -- distributions shift across days. Models fail to generalize.
**Do this instead:** Apply rolling z-score normalization with a multi-day window. Use bilinear normalization (as in TLOB) for batch-adaptive scaling. Normalize prices relative to mid-price.

### Anti-Pattern 3: Training RL Agent Only on Historical Data

**What people do:** Train the execution agent exclusively on historical LOB replays.
**Why it is wrong:** Historical data provides a single trajectory -- the agent cannot explore counterfactuals. The agent's actions would have changed the book in reality (market impact), but replay does not reflect this. Leads to overfitting to historical patterns.
**Do this instead:** Mix synthetic environments (from the generator) with historical replay. The generator can produce diverse market conditions and regime-specific scenarios. Validate that synthetic environments reproduce key LOB statistical properties.

### Anti-Pattern 4: Monolithic Config Files

**What people do:** Put all hyperparameters in a single YAML file or hardcode them in Python.
**Why it is wrong:** Makes it impossible to run parameter sweeps, compare experiments, or compose different module configurations. Breaks reproducibility.
**Do this instead:** Use Hydra config groups with one group per module. Store experiment-specific overrides in `configs/experiment/`. Log resolved configs to W&B.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| W&B | `wandb.init()` with Hydra resolved config; `.log()` per training step; `.watch()` for gradient tracking | Use W&B Artifacts for model versioning. Group runs by module (predictor/generator/executor) using W&B groups |
| LOBSTER Data Service | Download raw message + orderbook files per ticker-day | Requires academic license. Files follow naming convention `{date}_{start}_{end}_{type}` |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Predictor -> Generator | Frozen predictor checkpoint + embedding extraction function | Define a stable interface: `predictor.extract_embeddings(lob_window) -> Tensor[B, embed_dim]`. Version the embedding schema |
| Generator -> Executor | Generator produces LOB trajectories consumed by Gym env wrapper | Define trajectory format: `Dict[str, Tensor]` with keys `prices`, `volumes`, `timestamps`. Validate statistical properties before RL training |
| Configs -> All Modules | Hydra DictConfig passed to constructors via `_target_` instantiation | Each module defines a dataclass or attrs schema for its config subset. Validate configs at startup |
| Data Pipeline -> All Modules | Shared DataModule classes with consistent tensor shapes | All modules consume `[B, T, 4L]` LOB tensors. Predictor adds labels; generator adds noise; executor wraps in env |

## Sources

### Academic Papers (HIGH confidence)
- [DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://arxiv.org/abs/1808.03668) -- Zhang, Zohren, Roberts (2018). CNN+LSTM architecture for LOB prediction. IEEE Transactions on Signal Processing.
- [TLOB: A Novel Transformer Model with Dual Attention for Price Trend Prediction](https://arxiv.org/abs/2502.15757) -- Berti et al. (2025). Dual temporal+spatial self-attention transformer for LOB.
- [Double Deep Q-Learning for Optimal Execution](https://arxiv.org/abs/1812.06600) -- Ning, Ling, et al. (2018). Model-free Double DQN for trade execution. Applied Mathematical Finance.
- [TRADES: Generating Realistic Market Simulations with Diffusion Models](https://arxiv.org/html/2502.07071v2) -- Transformer-based DDPM for conditional LOB order flow generation.

### Academic Papers (MEDIUM confidence)
- [DiffLOB: Diffusion Models for Counterfactual Generation in Limit Order Books](https://arxiv.org/abs/2602.03776) -- Regime-conditioned DDPM with WaveNet backbone and ControlNet-style conditioning.
- [DiffVolume: Diffusion Models for Volume Generation in Limit Order Books](https://arxiv.org/abs/2508.08698) -- Conditional diffusion for LOB volume snapshot generation.
- [Asynchronous Deep Double Dueling Q-learning for trading-signal execution](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1151003/full) -- APEX + Dueling Double DQN variant for LOB execution.
- [LiT: Limit Order Book Transformer](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full) -- Patch-based transformer for LOB forecasting.

### Frameworks and Templates (HIGH confidence)
- [Lightning-Hydra Template](https://github.com/ashleve/lightning-hydra-template) -- Standard project structure for PyTorch + Hydra ML projects.
- [TLOB Official Repository](https://github.com/LeonardoBerti00/TLOB) -- Reference implementation showing Hydra config + model organization for LOB ML.
- [LOBCAST Framework](https://github.com/matteoprata/LOBCAST) -- Open-source LOB forecasting framework with preprocessing, training, and evaluation.
- [LOBster Preprocessing Tools](https://github.com/Jeonghwan-Cheon/LOBster) -- LOB data normalization, downsampling, and labelling utilities.

### Data Sources (HIGH confidence)
- FI-2010 Dataset -- 10 days, 5 Finnish stocks, NASDAQ Nordic. Standard benchmark.
- LOBSTER Data -- Academic-licensed tick-level LOB reconstruction from NASDAQ.

---
*Architecture research for: LOB-Forge -- Multi-component ML pipeline for limit order book market microstructure*
*Researched: 2026-03-18*
