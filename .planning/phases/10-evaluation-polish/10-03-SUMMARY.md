---
phase: 10-evaluation-polish
plan: 03
status: complete
date: 2026-03-22
commits:
  - cadd50d  # notebooks 01 + 02
  - c9c19d9  # notebooks 03 + 04
---

# Plan 10-03 Summary — Jupyter Notebooks

## Objective

Create 4 Jupyter notebooks providing end-to-end narrative walkthroughs of the LOB-Forge pipeline. All notebooks execute via `nbconvert --execute` without real data files.

## Tasks Completed

### Task 1: Notebooks 01 and 02 (data + predictor)

**notebooks/01_data_exploration.ipynb** (227 source lines, 16 cells):
- Section 1: LOB schema (pyarrow.Schema, 46 columns, ALL_COLUMNS layout)
- Section 2: Feature engineering — `compute_all_features` on dummy 500-row DataFrame; distribution plots for 8 features
- Section 3: Temporal splits — `temporal_split(n_rows, ratios=(0.7,0.15,0.15), purge_gap=10)` with ASCII split diagram
- Section 4: Data quality — `validate_lob_dataframe` + `compute_quality_metrics` on valid and intentionally broken DataFrames

**notebooks/02_predictor_results.ipynb** (204 source lines, 14 cells):
- Section 1: Model architectures — DualAttentionTransformer (TLOB), DeepLOB, LinearBaseline; parameter counts via `sum(p.numel())`; forward pass shape verification
- Section 2: Training metrics — dummy loss + val-F1 table and training curve plots
- Section 3: Multi-horizon F1 comparison — grouped bar chart across 4 horizons (1s/2s/5s/10s) for all 3 models
- Section 4: Confusion matrix — `sklearn.metrics.confusion_matrix` on dummy 3-class predictions with heatmap

### Task 2: Notebooks 03 and 04 (generator + execution)

**notebooks/03_generator_quality.ipynb** (253 source lines, 18 cells):
- Section 1: Diffusion architecture — `CosineNoiseSchedule` (1000 steps), `UNet1D` small/full parameter counts; alpha-bar + beta schedule plots
- Section 2: Stylized facts — `run_all_stylized_tests(real_lob, synth_lob)` on dummy 1000×40 arrays; results table
- Section 3: LOB-Bench — `run_lob_bench` + `compute_wasserstein_metrics` on dummy data; per-feature WD bar chart
- Section 4: Regime conditioning — `validate_regime_conditioning(real_by_regime, synth_by_regime)` on 3 distinct volatility regimes; return distribution overlay plot

**notebooks/04_execution_backtest.ipynb** (243 source lines, 15 cells):
- Section 1: Environment setup — `ACTION_NAMES` (7 actions), `CostModel` demo, `MockEnv` implementation
- Section 2: Baseline comparison — `run_backtest` for TWAP/VWAP/AlmgrenChriss/Random/DQN (5 episodes each); per-episode cost table
- Section 3: IS metrics — `compute_implementation_shortfall` + `compute_slippage_vs_twap` summary table
- Section 4: Visualization — `generate_all_plots(comparison_dict)` produces 6 PNG files; displayed inline via `IPython.display.Image`

## Verification Results

All 4 notebooks execute end-to-end via `jupyter nbconvert --to notebook --execute`:

| Notebook | Lines | Exit |
|----------|-------|------|
| 01_data_exploration.ipynb | 227 | 0 |
| 02_predictor_results.ipynb | 204 | 0 |
| 03_generator_quality.ipynb | 253 | 0 |
| 04_execution_backtest.ipynb | 243 | 0 |

## Key Decisions

- Used `ratios=(0.7, 0.15, 0.15)` instead of `train_frac=` — `temporal_split` uses positional tuple, not keyword fractions
- `LOB_SCHEMA` is a `pyarrow.Schema` object (not a DataFrame); access fields via `.field(i)` not `.columns`
- Schema column order: `timestamp, mid_price, spread, bid_price_1..10, bid_size_1..10, ask_price_1..10, ask_size_1..10, trade_price, trade_size, trade_side`
- `LinearBaseline` and `DeepLOB` constructors use `n_levels` + `features_per_level` (not `in_features`)
- `CostModel.compute` signature: `(exec_price, exec_size, mid_price, spread, avg_daily_volume)`
- `validate_regime_conditioning` requires two dicts `(real_by_regime, synthetic_by_regime)`
- `CosineNoiseSchedule.alphas_cumprod` has shape `(num_timesteps,)` = 1000 (not 1001)
- MockEnv pattern: non-zero ask/bid values in observation so baseline `arrival_price` is non-degenerate
- `matplotlib.use('Agg')` set at cell level in each notebook for non-interactive kernel safety
