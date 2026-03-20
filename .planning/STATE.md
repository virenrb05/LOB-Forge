# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-19)

**Core value:** The three-component pipeline works end-to-end: transformer embeddings condition the diffusion model, which generates unlimited training environments for the RL agent that beats TWAP on real data.
**Current focus:** Phase 3 — Data Preprocessing

## Current Position

Phase: 3 of 10 (Data Preprocessing)
Plan: 03-01 complete (LOB derived features)
Status: In progress
Last activity: 2026-03-19 — Plan 03-01 executed (6 feature functions with VPIN)

Progress: ██░░░░░░░░ 20%

## Performance Metrics

**Velocity:**
- Total plans completed: 8
- Average duration: ~3.5 min
- Total execution time: ~28 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-scaffold | 2/2 | ~10 min | ~5 min |
| 02-data-ingestion | 3/3 | ~8 min | ~2.7 min |
| 03-data-preprocessing | 3/5 | ~10 min | ~3.3 min |

**Recent Trend:**
- Last 5 plans: 02-03, 02-02, 03-03, 03-02, 03-01
- Trend: Accelerating

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

### Pending Todos

None yet.

### Blockers/Concerns

- Bybit REST API returns 403 from some network locations (geo-restriction); does not affect code correctness

## Session Continuity

Last session: 2026-03-19
Stopped at: Plan 03-01 complete, continuing Phase 3
Resume file: .planning/phases/03-data-preprocessing/03-01-SUMMARY.md
