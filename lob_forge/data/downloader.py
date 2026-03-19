"""Download and cache raw limit order book data from supported exchanges."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from lob_forge.data.schema import (
    ALL_COLUMNS,
    ASK_PRICE_COLS,
    ASK_SIZE_COLS,
    BID_PRICE_COLS,
    BID_SIZE_COLS,
    MID_PRICE,
    SPREAD,
    TIMESTAMP,
    TRADE_PRICE,
    TRADE_SIDE,
    TRADE_SIZE,
    write_lob_parquet,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_MAX_RETRIES: int = 3
_BACKOFF_BASE: float = 1.0  # seconds


def _retry_get(url: str, params: dict | None = None) -> requests.Response:
    """GET with exponential backoff retry (up to 3 attempts)."""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, requests.HTTPError) as exc:
            if attempt == _MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Request failed after {_MAX_RETRIES} attempts: {url}"
                ) from exc
            wait = _BACKOFF_BASE * (2**attempt)
            logger.warning(
                "Request to %s failed (attempt %d/%d), retrying in %.1fs: %s",
                url,
                attempt + 1,
                _MAX_RETRIES,
                wait,
                exc,
            )
            time.sleep(wait)
    # unreachable
    raise RuntimeError("Unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# BybitDownloader
# ---------------------------------------------------------------------------


class BybitDownloader:
    """Fetch and record LOB data from Bybit exchange.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (default ``"BTCUSDT"``).
    depth : int
        Number of order book levels to keep (default 10).
    output_dir : str | Path
        Base directory for saved Parquet files (default ``"data/"``).
    """

    REST_BASE: str = "https://api.bybit.com/v5/market/orderbook"
    ARCHIVE_BASE: str = "https://public.bybit.com/trading/"
    WS_URL: str = "wss://stream.bybit.com/v5/public/linear"

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        depth: int = 10,
        output_dir: str | Path = "data/",
    ) -> None:
        self.symbol = symbol
        self.depth = depth
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Method 1: REST snapshot
    # ------------------------------------------------------------------

    def fetch_snapshot(self) -> pd.DataFrame:
        """Fetch current order book snapshot via REST API.

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame conforming to the unified LOB schema.
        """
        params = {
            "category": "linear",
            "symbol": self.symbol,
            "limit": self.depth * 5,  # request extra, keep top depth
        }
        resp = _retry_get(self.REST_BASE, params=params)
        data = resp.json()

        if data.get("retCode") != 0:
            raise RuntimeError(
                f"Bybit API error {data.get('retCode')}: {data.get('retMsg')}"
            )

        result = data["result"]
        bids = result["b"]  # [[price, size], ...]
        asks = result["a"]
        ts_ms = int(result["ts"])

        # Take top depth levels
        bids = bids[: self.depth]
        asks = asks[: self.depth]

        if len(bids) < self.depth or len(asks) < self.depth:
            raise ValueError(
                f"Insufficient book depth: got {len(bids)} bids, {len(asks)} asks, "
                f"need {self.depth}"
            )

        row: dict = {}
        # Timestamp in microseconds
        row[TIMESTAMP] = ts_ms * 1_000

        # Bids (sorted descending by price from API)
        for i in range(self.depth):
            row[BID_PRICE_COLS[i]] = float(bids[i][0])
            row[BID_SIZE_COLS[i]] = float(bids[i][1])

        # Asks (sorted ascending by price from API)
        for i in range(self.depth):
            row[ASK_PRICE_COLS[i]] = float(asks[i][0])
            row[ASK_SIZE_COLS[i]] = float(asks[i][1])

        # Derived fields
        row[MID_PRICE] = (row[BID_PRICE_COLS[0]] + row[ASK_PRICE_COLS[0]]) / 2
        row[SPREAD] = row[ASK_PRICE_COLS[0]] - row[BID_PRICE_COLS[0]]

        # Trade fields: not available from orderbook endpoint
        row[TRADE_PRICE] = np.nan
        row[TRADE_SIZE] = np.nan
        row[TRADE_SIDE] = 0

        df = pd.DataFrame([row], columns=ALL_COLUMNS)
        return df

    # ------------------------------------------------------------------
    # Method 2: Historical archive download
    # ------------------------------------------------------------------

    def download_historical(
        self,
        start_date: str,
        end_date: str,
    ) -> Path:
        """Download historical trade CSVs from Bybit public archive.

        These are TRADE files (not order books). Each trade is stored as a
        partial snapshot with trade_price/trade_size/trade_side populated and
        order book columns set to NaN.

        Parameters
        ----------
        start_date : str
            Start date in ``"YYYY-MM-DD"`` format.
        end_date : str
            End date in ``"YYYY-MM-DD"`` format (inclusive).

        Returns
        -------
        Path
            Path to the output Parquet file.
        """
        logger.warning(
            "Historical archives only provide trade data, not order books. "
            "Order book columns will be NaN. Use record_websocket() for "
            "full LOB snapshots."
        )

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        all_rows: list[dict] = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            url = f"{self.ARCHIVE_BASE}{self.symbol}/" f"{self.symbol}{date_str}.csv.gz"
            try:
                resp = _retry_get(url)
                lines = resp.text.strip().split("\n")
                # Bybit trade CSVs: timestamp,symbol,side,size,price,...
                for line in lines[1:]:  # skip header if present
                    parts = line.split(",")
                    if len(parts) < 5:
                        continue
                    ts_sec = float(parts[0])
                    side_str = parts[2].strip().strip('"')
                    size = float(parts[3])
                    price = float(parts[4])

                    row: dict = {}
                    row[TIMESTAMP] = int(ts_sec * 1_000_000)  # sec -> us
                    row[MID_PRICE] = np.nan
                    row[SPREAD] = np.nan
                    for col in BID_PRICE_COLS + BID_SIZE_COLS:
                        row[col] = np.nan
                    for col in ASK_PRICE_COLS + ASK_SIZE_COLS:
                        row[col] = np.nan
                    row[TRADE_PRICE] = price
                    row[TRADE_SIZE] = size
                    row[TRADE_SIDE] = 1 if side_str.lower() == "buy" else -1
                    all_rows.append(row)

                logger.info("Downloaded %s trades for %s", len(lines) - 1, date_str)
            except Exception:
                logger.warning("Failed to download trades for %s, skipping", date_str)

            current += timedelta(days=1)

        if not all_rows:
            raise RuntimeError(
                f"No trade data downloaded for {start_date} to {end_date}"
            )

        df = pd.DataFrame(all_rows, columns=ALL_COLUMNS)
        # Sort by timestamp
        df = df.sort_values(TIMESTAMP).reset_index(drop=True)

        out_path = (
            self.output_dir / f"{self.symbol}_trades_{start_date}_{end_date}.parquet"
        )
        # Historical trades have NaN in book columns, so we write raw Parquet
        # instead of write_lob_parquet which enforces non-NaN book columns.
        import pyarrow as pa
        import pyarrow.parquet as pq

        df[TIMESTAMP] = df[TIMESTAMP].astype(np.int64)
        df[TRADE_SIDE] = df[TRADE_SIDE].astype(np.int8)
        float_cols = [c for c in ALL_COLUMNS if c not in (TIMESTAMP, TRADE_SIDE)]
        df[float_cols] = df[float_cols].astype(np.float64)

        table = pa.Table.from_pandas(df, preserve_index=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_path)

        logger.info("Saved %d trade rows to %s", len(df), out_path)
        return out_path

    # ------------------------------------------------------------------
    # Method 3: WebSocket LOB recorder
    # ------------------------------------------------------------------

    def record_websocket(
        self,
        duration_seconds: int = 60,
        output_path: Path | None = None,
        flush_every: int = 1000,
    ) -> Path:
        """Record live LOB snapshots via WebSocket.

        Connects to Bybit linear WebSocket, subscribes to orderbook updates,
        and records each snapshot/delta as a row in the unified LOB schema.

        Parameters
        ----------
        duration_seconds : int
            How long to record (seconds).
        output_path : Path | None
            Explicit output file path. If ``None``, auto-generates based on
            symbol and current time.
        flush_every : int
            Flush buffer to Parquet every N rows (default 1000).

        Returns
        -------
        Path
            Path to the output Parquet file.
        """
        if output_path is None:
            now_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"{self.symbol}_{now_str}.parquet"
        output_path = Path(output_path)

        return asyncio.run(self._ws_record(duration_seconds, output_path, flush_every))

    async def _ws_record(
        self,
        duration_seconds: int,
        output_path: Path,
        flush_every: int,
    ) -> Path:
        """Internal async WebSocket recording loop."""
        import websockets

        # Book state: {price_str: size_float}
        bid_book: dict[str, float] = {}
        ask_book: dict[str, float] = {}
        buffer: list[dict] = []
        all_flushed_paths: list[Path] = []
        flush_count = 0

        end_time = time.monotonic() + duration_seconds

        async def _connect_and_record() -> None:
            nonlocal bid_book, ask_book, buffer, flush_count

            async for ws in websockets.connect(
                self.WS_URL,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ):
                try:
                    # Subscribe
                    sub_msg = json.dumps(
                        {
                            "op": "subscribe",
                            "args": [f"orderbook.{self.depth}.{self.symbol}"],
                        }
                    )
                    await ws.send(sub_msg)

                    while time.monotonic() < end_time:
                        try:
                            raw = await asyncio.wait_for(
                                ws.recv(), timeout=min(5.0, end_time - time.monotonic())
                            )
                        except asyncio.TimeoutError:
                            continue

                        msg = json.loads(raw)

                        # Skip subscription confirmations / pong
                        if "data" not in msg:
                            continue

                        msg_type = msg.get("type", "")
                        data = msg["data"]
                        ts_ms = int(data.get("ts", msg.get("ts", 0)))

                        if msg_type == "snapshot":
                            bid_book.clear()
                            ask_book.clear()
                            for price, size in data.get("b", []):
                                bid_book[price] = float(size)
                            for price, size in data.get("a", []):
                                ask_book[price] = float(size)

                        elif msg_type == "delta":
                            for price, size in data.get("b", []):
                                if float(size) == 0:
                                    bid_book.pop(price, None)
                                else:
                                    bid_book[price] = float(size)
                            for price, size in data.get("a", []):
                                if float(size) == 0:
                                    ask_book.pop(price, None)
                                else:
                                    ask_book[price] = float(size)
                        else:
                            continue

                        # Build snapshot row from current book state
                        row = self._book_to_row(bid_book, ask_book, ts_ms)
                        if row is not None:
                            buffer.append(row)

                        # Flush if buffer full
                        if len(buffer) >= flush_every:
                            self._flush_buffer(
                                buffer, output_path, flush_count, all_flushed_paths
                            )
                            flush_count += 1
                            buffer.clear()

                    # Duration reached, exit connection loop
                    break

                except websockets.ConnectionClosed:
                    logger.warning("WebSocket disconnected, reconnecting...")
                    continue

        await _connect_and_record()

        # Flush remaining buffer
        if buffer:
            self._flush_buffer(buffer, output_path, flush_count, all_flushed_paths)
            buffer.clear()

        # Merge all flushed parts into single output file
        if all_flushed_paths:
            dfs = [pd.read_parquet(p) for p in all_flushed_paths]
            merged = pd.concat(dfs, ignore_index=True)
            write_lob_parquet(merged, output_path)
            # Clean up temp files
            for p in all_flushed_paths:
                if p != output_path:
                    p.unlink(missing_ok=True)
        elif not output_path.exists():
            raise RuntimeError("No data recorded during WebSocket session")

        logger.info("Recorded %s to %s", self.symbol, output_path)
        return output_path

    def _book_to_row(
        self,
        bid_book: dict[str, float],
        ask_book: dict[str, float],
        ts_ms: int,
    ) -> dict | None:
        """Convert current book state to a schema-conforming row dict."""
        # Sort bids descending, asks ascending
        sorted_bids = sorted(bid_book.items(), key=lambda x: float(x[0]), reverse=True)
        sorted_asks = sorted(ask_book.items(), key=lambda x: float(x[0]))

        if len(sorted_bids) < self.depth or len(sorted_asks) < self.depth:
            return None

        row: dict = {}
        row[TIMESTAMP] = ts_ms * 1_000  # ms -> us

        for i in range(self.depth):
            row[BID_PRICE_COLS[i]] = float(sorted_bids[i][0])
            row[BID_SIZE_COLS[i]] = float(sorted_bids[i][1])
            row[ASK_PRICE_COLS[i]] = float(sorted_asks[i][0])
            row[ASK_SIZE_COLS[i]] = float(sorted_asks[i][1])

        row[MID_PRICE] = (row[BID_PRICE_COLS[0]] + row[ASK_PRICE_COLS[0]]) / 2
        row[SPREAD] = row[ASK_PRICE_COLS[0]] - row[BID_PRICE_COLS[0]]

        row[TRADE_PRICE] = np.nan
        row[TRADE_SIZE] = np.nan
        row[TRADE_SIDE] = 0

        return row

    def _flush_buffer(
        self,
        buffer: list[dict],
        output_path: Path,
        flush_count: int,
        flushed_paths: list[Path],
    ) -> None:
        """Write buffer to a temporary Parquet file."""
        df = pd.DataFrame(buffer, columns=ALL_COLUMNS)
        part_path = output_path.with_suffix(f".part{flush_count}.parquet")
        write_lob_parquet(df, part_path)
        flushed_paths.append(part_path)
        logger.debug("Flushed %d rows to %s", len(buffer), part_path)
