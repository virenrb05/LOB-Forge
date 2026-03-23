---
plan: 12-01
phase: 12-coinbase-data-pipeline-run
status: complete
date: 2026-03-22
---

# 12-01 Summary: coinbase-downloader-class

## Outcome

CoinbaseDownloader class implemented in `lob_forge/data/downloader.py` and exported from `lob_forge.data`. REST snapshot returns a valid (1, 46) DataFrame. WebSocket recorder implemented with correct Coinbase Advanced Trade protocol.

## Tasks Completed

| Task | Commit | Status |
|------|--------|--------|
| Task 1: Implement CoinbaseDownloader class | a9e68e5 | done |
| Task 2: Export CoinbaseDownloader from lob_forge.data | 52da60f | done |

## Key Decisions

- **REST API**: Used `https://api.exchange.coinbase.com/products/{symbol}/book?level=2` instead of the plan-specified `api.coinbase.com/api/v3/brokerage` endpoint — the v3 endpoint returned 401 (requires authentication); the Exchange API is fully public and returns identical JSON structure (`bids`, `asks`, `time` with same `[price_str, size_str, num_orders_int]` format).
- **WebSocket URL**: Kept `wss://advanced-trade-ws.coinbase.com` as specified in the plan; added `max_size=10 * 1024 * 1024` to handle large initial snapshots (default 1 MB limit caused `ConnectionClosedError: 1009 message too big`).
- **NaN count**: Plan verification comment says "3 NaNs (trade_price, trade_size; trade_side=0)" but `trade_side=0` is not NaN — actual NaN count is 2 (trade_price + trade_size). This is correct behaviour; `trade_side=0` is an integer sentinel.

## Verification Results

- `from lob_forge.data import CoinbaseDownloader` — OK
- `d.fetch_snapshot()` returns shape `(1, 46)` — OK
- `ruff check` — 0 errors
- `black --check` — 0 reformats
- `pytest tests/ -x -q` — 307 passed, 0 failures
