"""Unified LOB Parquet schema: column constants, dtypes, and read/write utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

NUM_LEVELS: int = 10

TIMESTAMP: str = "timestamp"
MID_PRICE: str = "mid_price"
SPREAD: str = "spread"

BID_PRICE_COLS: list[str] = [f"bid_price_{i}" for i in range(1, NUM_LEVELS + 1)]
BID_SIZE_COLS: list[str] = [f"bid_size_{i}" for i in range(1, NUM_LEVELS + 1)]
ASK_PRICE_COLS: list[str] = [f"ask_price_{i}" for i in range(1, NUM_LEVELS + 1)]
ASK_SIZE_COLS: list[str] = [f"ask_size_{i}" for i in range(1, NUM_LEVELS + 1)]

TRADE_PRICE: str = "trade_price"
TRADE_SIZE: str = "trade_size"
TRADE_SIDE: str = "trade_side"

ALL_COLUMNS: list[str] = (
    [TIMESTAMP, MID_PRICE, SPREAD]
    + BID_PRICE_COLS
    + BID_SIZE_COLS
    + ASK_PRICE_COLS
    + ASK_SIZE_COLS
    + [TRADE_PRICE, TRADE_SIZE, TRADE_SIDE]
)

# ---------------------------------------------------------------------------
# PyArrow schema with exact dtypes
# ---------------------------------------------------------------------------

LOB_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field(TIMESTAMP, pa.int64()),
        pa.field(MID_PRICE, pa.float64()),
        pa.field(SPREAD, pa.float64()),
    ]
    + [pa.field(c, pa.float64()) for c in BID_PRICE_COLS]
    + [pa.field(c, pa.float64()) for c in BID_SIZE_COLS]
    + [pa.field(c, pa.float64()) for c in ASK_PRICE_COLS]
    + [pa.field(c, pa.float64()) for c in ASK_SIZE_COLS]
    + [
        pa.field(TRADE_PRICE, pa.float64()),
        pa.field(TRADE_SIZE, pa.float64()),
        pa.field(TRADE_SIDE, pa.int8()),
    ]
)

# ---------------------------------------------------------------------------
# Read / write utilities
# ---------------------------------------------------------------------------


def write_lob_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write a LOB DataFrame to Parquet with schema enforcement.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all columns listed in :pydata:`ALL_COLUMNS`.
    path : Path
        Destination file path (parent directories are created automatically).

    Returns
    -------
    Path
        The written file path.
    """
    missing = set(ALL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    # Select and order columns, then cast to expected dtypes
    df = df[ALL_COLUMNS].copy()
    df[TIMESTAMP] = df[TIMESTAMP].astype(np.int64)
    df[TRADE_SIDE] = df[TRADE_SIDE].astype(np.int8)
    float_cols = [c for c in ALL_COLUMNS if c not in (TIMESTAMP, TRADE_SIDE)]
    df[float_cols] = df[float_cols].astype(np.float64)

    table = pa.Table.from_pandas(df, schema=LOB_SCHEMA, preserve_index=False)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def read_lob_parquet(path: Path) -> pd.DataFrame:
    """Read a LOB Parquet file with schema validation.

    Parameters
    ----------
    path : Path
        Parquet file written by :func:`write_lob_parquet`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching :pydata:`ALL_COLUMNS`.
    """
    table = pq.read_table(Path(path), schema=LOB_SCHEMA)
    return table.to_pandas()
