# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-19)

**Core value:** The three-component pipeline works end-to-end: transformer embeddings condition the diffusion model, which generates unlimited training environments for the RL agent that beats TWAP on real data.
**Current focus:** Phase 3 — Data Preprocessing

## Current Position

Phase: 3 of 10 (Data Preprocessing)
Plan: Not started
Status: Ready to plan
Last activity: 2026-03-19 — Phase 2 Data Ingestion verified and complete

Progress: ██░░░░░░░░ 20%

## Performance Metrics

**Velocity:**
- Total plans completed: 5
- Average duration: ~4 min
- Total execution time: ~18 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-scaffold | 2/2 | ~10 min | ~5 min |
| 02-data-ingestion | 3/3 | ~8 min | ~2.7 min |

**Recent Trend:**
- Last 5 plans: 01-01, 01-02, 02-01, 02-03, 02-02
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

### Pending Todos

None yet.

### Blockers/Concerns

- Bybit REST API returns 403 from some network locations (geo-restriction); does not affect code correctness

## Session Continuity

Last session: 2026-03-19
Stopped at: Phase 2 verified and complete, ready for Phase 3
Resume file: .planning/phases/02-data-ingestion/02-VERIFICATION.md
