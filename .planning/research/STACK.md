# Stack Research

**Domain:** Quantitative Finance / Market Microstructure ML
**Researched:** 2026-03-18
**Confidence:** MEDIUM-HIGH (versions verified via PyPI/GitHub; MPS compatibility verified via Apple docs and issue trackers)

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.10+ | Runtime | Required by PyTorch 2.6+, TorchRL, and Gymnasium. 3.11 recommended for performance improvements. |
| PyTorch | >=2.6.0, <2.10 | Deep learning framework | 2.6.0 (Jan 2025) is first stable 2025 release. MPS backend mature enough for training. Avoid 2.10+ due to macOS 26 Tahoe MPS availability bug. Pin to 2.6.x-2.9.x for Apple Silicon stability. |
| hydra-core | 1.3.2 | Config management | Industry standard for ML experiment configuration. Hierarchical YAML configs with CLI overrides. Stable release, no breaking changes expected. |
| omegaconf | 2.3.0 | Config backend | Required by Hydra. Supports custom resolvers for dynamic config values (e.g., computed paths, device selection). |
| wandb | >=0.19.0 | Experiment tracking | De facto standard for ML experiment tracking. Logs metrics, hyperparameters, model artifacts. Free tier sufficient for personal projects. |
| TorchRL | 0.11.1 | RL framework | Official PyTorch RL library. Provides DQN/Double-DQN losses, replay buffers (including prioritized), environment wrappers. C++ binaries for prioritized replay require PyTorch >=2.7. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| einops | >=0.7.0 | Tensor operations | Readable attention reshaping in transformer and diffusion model code. Used by rotary-embedding-torch. |
| rotary-embedding-torch | >=0.6.0 | Rotary positional embeddings (RoPE) | For the dual-attention transformer. RoPE is the 2025 standard for positional encoding -- better length extrapolation than learned/sinusoidal. |
| gymnasium | 0.29.1 | RL environments | Custom trading environment for the Double-DQN agent. Pin to 0.29.1 -- TorchRL is incompatible with Gymnasium 1.x as of Jan 2025. |
| numpy | >=1.26.0, <2.0 | Numerical computing | LOB data preprocessing. Pin <2.0 to avoid breaking changes with older scientific libraries. |
| pandas | >=2.1.0 | Data manipulation | LOBSTER message/orderbook file parsing, time-series alignment, rolling window calculations. |
| scikit-learn | >=1.4.0 | ML utilities | Classification metrics (F1, precision, recall) for mid-price prediction evaluation. Train/test splitting. |
| matplotlib | >=3.8.0 | Visualization | LOB depth visualization, training curves, diffusion sample quality plots. |
| seaborn | >=0.13.0 | Statistical plots | Distribution comparisons for synthetic vs real LOB data validation. |
| scipy | >=1.12.0 | Scientific computing | Statistical tests (KS test, Wasserstein distance) for evaluating synthetic LOB quality. |
| tqdm | >=4.66.0 | Progress bars | Training loop progress, data preprocessing progress. |
| tensordict | >=0.11.0 | Structured tensors | Required by TorchRL. Efficient batched tensor storage for RL transitions. |
| h5py | >=3.10.0 | HDF5 file I/O | Efficient storage/loading of large LOB datasets (FI-2010, LOBSTER). Much faster than CSV for multi-GB tick data. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| ruff | Linting + formatting | Replaces flake8/black/isort. Single tool, extremely fast. Use `ruff check` and `ruff format`. |
| pytest | Testing | Standard Python test framework. Use `pytest-cov` for coverage. |
| mypy | Type checking | Optional but recommended for complex tensor shape tracking. |
| pre-commit | Git hooks | Run ruff/mypy on commit. Catches issues early. |
| uv | Package management | Fast pip replacement. Use `uv pip install` or `uv venv` for environment creation. 10-100x faster than pip. |

## Installation

```bash
# Create environment (using uv for speed)
uv venv .venv --python 3.11
source .venv/bin/activate

# Core ML stack
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# RL framework (pin gymnasium for TorchRL compatibility)
uv pip install torchrl==0.11.1 tensordict==0.11.0 gymnasium==0.29.1

# Configuration and experiment tracking
uv pip install hydra-core==1.3.2 omegaconf==2.3.0 wandb>=0.19.0

# Transformer utilities
uv pip install einops>=0.7.0 rotary-embedding-torch>=0.6.0

# Data processing and scientific computing
uv pip install numpy">=1.26.0,<2.0" pandas>=2.1.0 scikit-learn>=1.4.0 scipy>=1.12.0 h5py>=3.10.0

# Visualization
uv pip install matplotlib>=3.8.0 seaborn>=0.13.0

# Utilities
uv pip install tqdm>=4.66.0

# Dev tools
uv pip install ruff pytest pytest-cov mypy pre-commit
```

### MPS Device Setup Pattern

```python
import torch

def get_device() -> torch.device:
    """Get best available device, preferring MPS on Apple Silicon."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# CRITICAL: MPS does not support float64
# Always use float32 for MPS training
DTYPE = torch.float32  # Never torch.float64 on MPS
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| TorchRL + custom env | Stable-Baselines3 | If you want pre-built algorithm implementations and don't need tight PyTorch integration. SB3 is more beginner-friendly but less flexible for custom LOB environments. |
| Hydra/OmegaConf | PyTorch Lightning CLI | If already using Lightning as trainer. Lightning CLI provides similar config management but couples you to Lightning. |
| wandb | MLflow | If you need self-hosted tracking or are in an enterprise environment that restricts cloud services. MLflow is OSS but has weaker visualization. |
| Custom DDPM implementation | Hugging Face diffusers | If generating image-based LOB representations. diffusers is designed for image/audio diffusion, not 1D financial time series. Custom DDPM with transformer backbone is more appropriate for LOB sequences. |
| gymnasium 0.29.1 | gymnasium 1.x | When TorchRL adds 1.x support (check TorchRL release notes). v1.x has 200+ PRs of improvements but breaks TorchRL compatibility as of early 2025. |
| einops | Manual reshape/permute | Never. einops is always clearer and prevents shape bugs in attention code. |
| RoPE (rotary embeddings) | Learned positional embeddings | If sequence lengths are always fixed and you don't need length extrapolation. For LOB data with variable-length sequences, RoPE is strictly better. |
| uv | pip/conda | If you need conda-specific packages (e.g., CUDA toolkit). For pure Python packages, uv is 10-100x faster. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| torch.float64 on MPS | MPS backend does not support float64. Operations silently fail or throw errors. | torch.float32 everywhere. Cast inputs at data loading time. |
| gymnasium >= 1.0 | Incompatible with TorchRL as of Jan 2025. Will cause import errors. | gymnasium==0.29.1 |
| OpenAI Gym (gym) | Deprecated in favor of Gymnasium (Farama Foundation fork). No longer maintained. | gymnasium |
| TensorFlow/Keras | Wrong ecosystem for this project. PyTorch is standard in quant finance research. No MPS-equivalent acceleration. | PyTorch |
| torch.compile on MPS | MPS lacks a mature compiler stack. Complex fusions fall back to CPU or run as unfused Metal kernels. Slowdowns likely. | Eager mode on MPS. Use torch.compile only if you later deploy to CUDA. |
| torch.autograd.detect_anomaly on MPS | Slows execution by ~100x on MPS. Practically unusable for debugging. | Debug on CPU with detect_anomaly, then switch to MPS for training. |
| Distributed training (torchrun) on MPS | NCCL not available on macOS. Distributed training is not supported on Apple Silicon. | Single-device MPS training. Use gradient accumulation for effective larger batch sizes. |
| numpy >= 2.0 | Breaking API changes may affect pandas, scipy, sklearn interop. | numpy >=1.26, <2.0 |

## Stack Patterns by Component

**Transformer (Mid-Price Prediction):**
- Architecture: Dual-attention transformer following TLOB pattern (spatial + temporal attention)
- Positional encoding: RoPE via rotary-embedding-torch
- Data: FI-2010 benchmark or LOBSTER data, z-score normalized with rolling window
- Metrics: F1-score (weighted), accuracy per prediction horizon (10, 20, 50, 100 ticks)
- Reference: TLOB (Berti et al., 2025) -- 3.7% F1 improvement over prior SOTA

**Diffusion Model (Synthetic LOB Generation):**
- Architecture: DDPM with transformer encoder-decoder backbone (not U-Net -- 1D temporal data)
- Conditioning: Cross-attention mechanism for market regime conditioning (trend, volatility)
- Validation: Stylized facts (fat tails, volatility clustering), KS test, Wasserstein distance vs real data
- Reference: Diffusion-TS (ICLR 2024) for architecture patterns; CoFinDiff (IJCAI 2025) for conditional generation
- Training: Predict x_0 directly (not noise) with Fourier-based loss term, per Diffusion-TS

**Double-DQN (Execution Agent):**
- Framework: TorchRL with DQNLoss, prioritized replay buffer
- Environment: Custom gymnasium.Env wrapping LOB simulator
- State: LOB snapshot (bid/ask prices + volumes, N levels) + position + inventory
- Action: Discrete (market buy, market sell, limit buy at levels, limit sell at levels, hold)
- Optimizer: Adam (not RMSProp) -- research shows Adam improves Double-DQN stability more than the Double-DQN modification itself
- Target network: Soft update via TorchRL SoftUpdate

## Version Compatibility Matrix

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| PyTorch 2.6.0 | TorchRL 0.6.x | Minimum viable. Prioritized replay C++ binaries may not work. |
| PyTorch 2.7.0+ | TorchRL 0.11.1 | Full support including C++ prioritized replay buffers. |
| PyTorch 2.6.0-2.9.x | MPS backend | Stable on macOS 15 (Sequoia). Avoid 2.10+ on macOS 26 Tahoe. |
| hydra-core 1.3.2 | omegaconf >=2.2, <2.4 | Pinned dependency range. |
| TorchRL 0.11.1 | gymnasium 0.29.1 | Must pin gymnasium. TorchRL incompatible with gymnasium 1.x. |
| TorchRL 0.11.1 | tensordict 0.11.0 | Version-locked. Always install matching versions. |
| Python 3.11 | All above | Recommended. 3.10 works but 3.11 has 10-25% speedup for CPU-bound code. |

## MPS-Specific Considerations

1. **No float64**: Cast all data to float32 at load time. This affects LOB price data which naturally has high precision -- use price normalization (z-score or min-max) early in the pipeline.

2. **Non-contiguous tensor bug**: Fixed in PyTorch 2.4+ on macOS 15+. Operations like `addcmul_` and `addcdiv_` silently produced wrong results on non-contiguous tensors in earlier versions. Ensure macOS 15+ and PyTorch >=2.4.

3. **No torch.compile**: Use eager mode. Performance is still good for the model sizes in this project (LOB transformers are small compared to LLMs).

4. **Memory**: MPS shares system RAM with CPU. On 16GB machines, batch sizes may need to be smaller than GPU equivalents. Use gradient accumulation to compensate.

5. **Debugging workflow**: Use CPU + detect_anomaly for debugging, then switch to MPS for training runs. detect_anomaly is ~100x slower on MPS.

6. **Diffusion model training**: Use float32, not float16. MPS float16 support exists but is less stable for diffusion noise schedules. Attention slicing can help with memory on longer LOB sequences.

## Sources

- [PyTorch MPS Backend Docs](https://docs.pytorch.org/docs/stable/notes/mps.html) -- MPS limitations and supported ops (HIGH confidence)
- [Apple Metal PyTorch Guide](https://developer.apple.com/metal/pytorch/) -- Official Apple Silicon setup (HIGH confidence)
- [MPS macOS 26 Tahoe Bug](https://github.com/pytorch/pytorch/issues/167679) -- MPS availability issue on macOS 26 (HIGH confidence)
- [Non-contiguous Tensor Bug Post](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/) -- MPS addcmul_ bug details (HIGH confidence)
- [TLOB Paper](https://arxiv.org/abs/2502.15757) -- Dual-attention transformer for LOB, SOTA on FI-2010 (HIGH confidence)
- [LiT: LOB Transformer](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1616485/full) -- Patch-based transformer for LOB (HIGH confidence)
- [LOBFrame GitHub](https://github.com/FinancialComputingUCL/LOBFrame) -- LOB data processing framework (HIGH confidence)
- [lob-deep-learning GitHub](https://github.com/Jeonghwan-Cheon/lob-deep-learning) -- DeepLOB/TransLOB implementations (HIGH confidence)
- [Diffusion-TS (ICLR 2024)](https://github.com/Y-debug-sys/Diffusion-TS) -- Diffusion for time series generation (HIGH confidence)
- [CoFinDiff (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/1040) -- Conditional financial diffusion model (MEDIUM confidence)
- [Synthetic Financial Time Series via Diffusion](https://arxiv.org/abs/2410.18897) -- DDPM for financial data (MEDIUM confidence)
- [TorchRL GitHub](https://github.com/pytorch/rl) -- Official PyTorch RL library (HIGH confidence)
- [TorchRL DQN Tutorial](https://docs.pytorch.org/rl/main/tutorials/coding_dqn.html) -- DQN implementation guide (HIGH confidence)
- [PyTorch DQN Tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) -- Double DQN reference (HIGH confidence)
- [Hydra Docs](https://hydra.cc/docs/intro/) -- Configuration management (HIGH confidence)
- [wandb PyPI](https://pypi.org/project/wandb/) -- Experiment tracking versions (HIGH confidence)
- [torchrl PyPI](https://pypi.org/project/torchrl/) -- TorchRL version info (HIGH confidence)
- [gymnasium PyPI](https://pypi.org/project/Gymnasium/) -- Gymnasium version compatibility (HIGH confidence)
- [Hugging Face MPS Diffusers Guide](https://huggingface.co/docs/diffusers/en/optimization/mps) -- MPS diffusion workarounds (MEDIUM confidence)
- [LOB-Bench (Feb 2025)](https://arxiv.org/html/2502.09172v1) -- Benchmark for generative LOB models (MEDIUM confidence)
- [Deep LOB Forecasting Guide](https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2522911) -- Microstructural forecasting survey (MEDIUM confidence)
- [rotary-embedding-torch GitHub](https://github.com/lucidrains/rotary-embedding-torch) -- RoPE implementation (HIGH confidence)
- [Float64 MPS Discussion](https://discussions.apple.com/thread/256120698) -- float64 limitation confirmed (HIGH confidence)
- [PyTorch Lightning MPS](https://lightning.ai/docs/pytorch/stable/accelerators/mps_basic.html) -- MPS training with Lightning (MEDIUM confidence)

---
*Stack research for: Quantitative Finance / Market Microstructure ML (LOB-Forge)*
*Researched: 2026-03-18*
