---
plan: 12-02
phase: 12-coinbase-data-pipeline-run
status: complete
date: 2026-03-22
---

# 12-02 Summary: coinbase-pipeline-integration

## Outcome

CoinbaseDownloader wired into the full pipeline: CLI module created, configs/data.yaml updated to Coinbase as primary data source, and train_all.sh Stage 1 updated to use the Coinbase downloader instead of Bybit. Human checkpoint verified REST snapshot and WebSocket recording both produce valid LOB data.

## Tasks Completed

| Task | Commit | Status |
|------|--------|--------|
| Task 1: Create coinbase_downloader CLI module and update configs | 7a615a3 | done |
| Task 2: Update train_all.sh Stage 1 to use Coinbase downloader | 316b1e0 | done |
| Task 3: Human checkpoint (network verification) | — (approved) | done |

## Human Verification Results

- `fetch_snapshot()` returned shape `(1, 46)`, mid price ~$67,961, 2 NaNs (trade_price + trade_size only, as expected)
- `record_websocket()` via CLI: 60 rows in 30 seconds, 0 book NaNs, monotonic timestamps
- `configs/data.yaml`: `source=coinbase`, `symbol=BTC-USD` confirmed
- `scripts/train_all.sh` Stage 1 invokes `coinbase_downloader` at line 83

## Final Verification

- `python -m lob_forge.data.coinbase_downloader --help` — exits 0
- `configs/data.yaml` source=coinbase, symbol=BTC-USD — OK
- `grep coinbase_downloader scripts/train_all.sh` — Stage 1 confirmed
- `pytest tests/ -x -q` — 307 passed, 0 failures
- `ruff check lob_forge/ && black --check lob_forge/` — clean

## Key Decisions

- CLI module `lob_forge/data/coinbase_downloader.py` is a thin argparse wrapper over `CoinbaseDownloader.record_websocket()` — no business logic duplicated
- `RECORD_DURATION` env var in train_all.sh allows overriding the 300s default without editing the script (e.g. `RECORD_DURATION=60 bash scripts/train_all.sh`)
- Stage 1 failure is non-fatal: script logs `[WARN]` and continues with existing data if present — pipeline is resilient to transient network issues
