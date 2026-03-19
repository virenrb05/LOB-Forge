"""LOBSTER data adapter: reads NASDAQ equity LOB files and converts to unified Parquet schema."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

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

# Regex to extract ticker and date from LOBSTER filenames
# e.g. AAPL_2024-01-15_34200000_57600000_orderbook_10.csv
_LOBSTER_FILE_RE = re.compile(
    r"^(?P<ticker>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<start>\d+)_(?P<end>\d+)_(?P<kind>orderbook|message)_(?P<depth>\d+)\.csv$"
)


class LOBSTERAdapter:
    """Convert LOBSTER orderbook/message file pairs to unified LOB Parquet format.

    Parameters
    ----------
    depth : int
        Target book depth (number of levels). Source files with fewer levels
        are padded with NaN; files with more levels are truncated.
    output_dir : str | Path
        Default directory for output Parquet files.
    """

    def __init__(self, depth: int = 10, output_dir: str | Path = "data/") -> None:
        self.depth = depth
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert_file(
        self,
        orderbook_path: Path,
        message_path: Path,
        output_path: Path | None = None,
        date: str | None = None,
    ) -> Path:
        """Convert a single LOBSTER orderbook/message file pair to unified Parquet.

        Parameters
        ----------
        orderbook_path : Path
            Path to the LOBSTER orderbook CSV (no header).
        message_path : Path
            Path to the corresponding LOBSTER message CSV (no header).
        output_path : Path | None
            Destination Parquet path. Auto-generated if *None*.
        date : str | None
            Trading date as ``YYYY-MM-DD``. If *None*, extracted from filename.

        Returns
        -------
        Path
            Path to the written Parquet file.
        """
        orderbook_path = Path(orderbook_path)
        message_path = Path(message_path)

        # --- Resolve date --------------------------------------------------
        if date is None:
            date = self._extract_date(orderbook_path.name)
        date_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        midnight_us = int(date_dt.timestamp() * 1_000_000)

        # --- Read raw files ------------------------------------------------
        ob_df = pd.read_csv(orderbook_path, header=None)
        msg_df = pd.read_csv(message_path, header=None)

        if len(ob_df) != len(msg_df):
            raise ValueError(
                f"Row count mismatch: orderbook has {len(ob_df)} rows, "
                f"message has {len(msg_df)} rows"
            )

        n_rows = len(ob_df)
        source_depth = ob_df.shape[1] // 4

        if source_depth < self.depth:
            logger.warning(
                "Source depth %d < target depth %d; padding with NaN",
                source_depth,
                self.depth,
            )

        # --- Extract and remap book columns --------------------------------
        bid_prices: list[np.ndarray] = []
        bid_sizes: list[np.ndarray] = []
        ask_prices: list[np.ndarray] = []
        ask_sizes: list[np.ndarray] = []

        for i in range(min(source_depth, self.depth)):
            # LOBSTER ordering: ask_price, ask_size, bid_price, bid_size per level
            ask_prices.append(ob_df.iloc[:, i * 4].values / 10_000)
            ask_sizes.append(ob_df.iloc[:, i * 4 + 1].values.astype(np.float64))
            bid_prices.append(ob_df.iloc[:, i * 4 + 2].values / 10_000)
            bid_sizes.append(ob_df.iloc[:, i * 4 + 3].values.astype(np.float64))

        # Pad missing levels with NaN
        for _ in range(max(0, self.depth - source_depth)):
            nan_col = np.full(n_rows, np.nan)
            bid_prices.append(nan_col)
            bid_sizes.append(nan_col)
            ask_prices.append(nan_col)
            ask_sizes.append(nan_col)

        # --- Timestamps (seconds since midnight → unix microseconds) -------
        time_sec = msg_df.iloc[:, 0].values.astype(np.float64)
        timestamps = midnight_us + (time_sec * 1_000_000).astype(np.int64)

        # --- Derived columns -----------------------------------------------
        bid_p1 = bid_prices[0]
        ask_p1 = ask_prices[0]
        mid_price = (bid_p1 + ask_p1) / 2.0
        spread = ask_p1 - bid_p1

        # --- Trade columns from message file --------------------------------
        # Columns: time, event_type, order_id, size, price, direction
        event_type = msg_df.iloc[:, 1].values.astype(int)
        is_execution = np.isin(event_type, [4, 5])

        trade_price = np.where(is_execution, msg_df.iloc[:, 4].values / 10_000, np.nan)
        trade_size = np.where(
            is_execution, msg_df.iloc[:, 3].values.astype(np.float64), np.nan
        )
        # direction: -1=sell, 1=buy → map to trade_side (keep as int8, 0 for non-trades)
        raw_direction = msg_df.iloc[:, 5].values.astype(int)
        trade_side = np.where(is_execution, raw_direction, 0).astype(np.int8)

        # --- Construct unified DataFrame -----------------------------------
        data: dict[str, np.ndarray] = {
            TIMESTAMP: timestamps,
            MID_PRICE: mid_price,
            SPREAD: spread,
        }

        for i in range(self.depth):
            data[BID_PRICE_COLS[i]] = bid_prices[i]
        for i in range(self.depth):
            data[BID_SIZE_COLS[i]] = bid_sizes[i]
        for i in range(self.depth):
            data[ASK_PRICE_COLS[i]] = ask_prices[i]
        for i in range(self.depth):
            data[ASK_SIZE_COLS[i]] = ask_sizes[i]

        data[TRADE_PRICE] = trade_price
        data[TRADE_SIZE] = trade_size
        data[TRADE_SIDE] = trade_side

        df = pd.DataFrame(data, columns=ALL_COLUMNS)

        # --- Write Parquet --------------------------------------------------
        if output_path is None:
            stem = orderbook_path.stem.replace("_orderbook_", "_lob_")
            output_path = self.output_dir / f"{stem}.parquet"

        return write_lob_parquet(df, Path(output_path))

    def convert_directory(
        self,
        input_dir: Path,
        output_dir: Path | None = None,
    ) -> list[Path]:
        """Convert all LOBSTER file pairs in a directory.

        Parameters
        ----------
        input_dir : Path
            Directory containing LOBSTER CSV files.
        output_dir : Path | None
            Destination for Parquet files. Defaults to ``self.output_dir``.

        Returns
        -------
        list[Path]
            Paths to generated Parquet files.
        """
        input_dir = Path(input_dir)

        # Discover orderbook files and match with message files
        pairs = self._find_file_pairs(input_dir)
        if not pairs:
            logger.warning("No LOBSTER file pairs found in %s", input_dir)
            return []

        results: list[Path] = []
        for ob_path, msg_path, date_str in pairs:
            try:
                p = self.convert_file(ob_path, msg_path, date=date_str)
                results.append(p)
            except Exception:
                logger.exception("Failed to convert %s", ob_path.name)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_date(filename: str) -> str:
        """Extract the date string from a LOBSTER filename."""
        m = _LOBSTER_FILE_RE.match(filename)
        if m:
            return m.group("date")
        # Fallback: look for YYYY-MM-DD anywhere in filename
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
        if date_match:
            return date_match.group()
        raise ValueError(f"Cannot extract date from filename: {filename}")

    @staticmethod
    def _find_file_pairs(
        directory: Path,
    ) -> list[tuple[Path, Path, str]]:
        """Find matching orderbook/message file pairs in a directory.

        Returns list of (orderbook_path, message_path, date_str) tuples.
        """
        ob_files: dict[str, Path] = {}
        for f in sorted(directory.glob("*_orderbook_*.csv")):
            m = _LOBSTER_FILE_RE.match(f.name)
            if m:
                key = f"{m.group('ticker')}_{m.group('date')}_{m.group('start')}_{m.group('end')}_{m.group('depth')}"
                ob_files[key] = f

        pairs: list[tuple[Path, Path, str]] = []
        for _key, ob_path in ob_files.items():
            msg_name = ob_path.name.replace("_orderbook_", "_message_")
            msg_path = ob_path.parent / msg_name
            if msg_path.exists():
                m = _LOBSTER_FILE_RE.match(ob_path.name)
                assert m is not None
                pairs.append((ob_path, msg_path, m.group("date")))
            else:
                logger.warning("No matching message file for %s", ob_path.name)

        return pairs
