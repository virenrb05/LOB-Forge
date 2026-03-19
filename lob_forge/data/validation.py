"""Data integrity validation and quality metrics for LOB DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lob_forge.data.schema import (
    ASK_PRICE_COLS,
    ASK_SIZE_COLS,
    BID_PRICE_COLS,
    BID_SIZE_COLS,
    TIMESTAMP,
)

# Columns where NaN is never acceptable (LOB book fields)
_BOOK_PRICE_COLS = BID_PRICE_COLS + ASK_PRICE_COLS
_BOOK_SIZE_COLS = BID_SIZE_COLS + ASK_SIZE_COLS

# Threshold for "large gap" warnings (milliseconds)
_GAP_WARN_MS: int = 500


def validate_lob_dataframe(df: pd.DataFrame) -> list[str]:
    """Validate a LOB DataFrame for data integrity issues.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns are a superset of the unified LOB schema.

    Returns
    -------
    list[str]
        List of error/warning strings.  Empty list means data is valid.
    """
    issues: list[str] = []

    if df.empty:
        issues.append("ERROR: DataFrame is empty")
        return issues

    # (a) No NaN in price/size columns (trade columns may have NaN)
    for col in _BOOK_PRICE_COLS + _BOOK_SIZE_COLS:
        if col in df.columns:
            nan_count = int(df[col].isna().sum())
            if nan_count > 0:
                issues.append(f"ERROR: {nan_count} NaN values in {col}")

    # (b) All prices positive
    for col in _BOOK_PRICE_COLS:
        if col in df.columns:
            bad = int((df[col] <= 0).sum())
            if bad > 0:
                issues.append(f"ERROR: {bad} non-positive values in {col}")

    # (c) All sizes non-negative
    for col in _BOOK_SIZE_COLS:
        if col in df.columns:
            bad = int((df[col] < 0).sum())
            if bad > 0:
                issues.append(f"ERROR: {bad} negative values in {col}")

    # (d) No crossed books: bid_price_1 <= ask_price_1
    if BID_PRICE_COLS[0] in df.columns and ASK_PRICE_COLS[0] in df.columns:
        crossed = int((df[BID_PRICE_COLS[0]] > df[ASK_PRICE_COLS[0]]).sum())
        if crossed > 0:
            issues.append(f"ERROR: {crossed} crossed-book rows (bid_1 > ask_1)")

    # (e) Monotonically increasing timestamps
    if TIMESTAMP in df.columns and len(df) > 1:
        ts = df[TIMESTAMP].values
        non_mono = int((np.diff(ts) <= 0).sum())
        if non_mono > 0:
            issues.append(f"ERROR: {non_mono} non-monotonic timestamp transitions")

    # (f) Timestamp gaps > 500 ms as warnings
    if TIMESTAMP in df.columns and len(df) > 1:
        ts = df[TIMESTAMP].values
        diffs_ms = np.diff(ts) / 1_000  # microseconds -> milliseconds
        large_gaps = int((diffs_ms > _GAP_WARN_MS).sum())
        if large_gaps > 0:
            issues.append(
                f"WARNING: {large_gaps} timestamp gaps exceed {_GAP_WARN_MS}ms"
            )

    # (g) Bid prices descending: bid_price_1 >= bid_price_2 >= ... >= bid_price_10
    for i in range(len(BID_PRICE_COLS) - 1):
        c1, c2 = BID_PRICE_COLS[i], BID_PRICE_COLS[i + 1]
        if c1 in df.columns and c2 in df.columns:
            bad = int((df[c1] < df[c2]).sum())
            if bad > 0:
                issues.append(
                    f"ERROR: {bad} rows where {c1} < {c2} (bids not descending)"
                )

    # (h) Ask prices ascending: ask_price_1 <= ask_price_2 <= ... <= ask_price_10
    for i in range(len(ASK_PRICE_COLS) - 1):
        c1, c2 = ASK_PRICE_COLS[i], ASK_PRICE_COLS[i + 1]
        if c1 in df.columns and c2 in df.columns:
            bad = int((df[c1] > df[c2]).sum())
            if bad > 0:
                issues.append(
                    f"ERROR: {bad} rows where {c1} > {c2} (asks not ascending)"
                )

    return issues


def compute_quality_metrics(df: pd.DataFrame) -> dict:
    """Compute summary quality metrics for a LOB DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with unified LOB schema columns.

    Returns
    -------
    dict
        Keys: row_count, time_span_seconds, mean_gap_ms, max_gap_ms,
        gap_count_over_500ms, crossed_book_count, nan_count.
    """
    row_count = len(df)
    nan_count = int(df[_BOOK_PRICE_COLS + _BOOK_SIZE_COLS].isna().sum().sum())

    ts = df[TIMESTAMP].values if TIMESTAMP in df.columns else np.array([])

    if len(ts) > 1:
        diffs_us = np.diff(ts).astype(np.float64)
        diffs_ms = diffs_us / 1_000
        time_span_seconds = float((ts[-1] - ts[0]) / 1_000_000)
        mean_gap_ms = float(np.mean(diffs_ms))
        max_gap_ms = float(np.max(diffs_ms))
        gap_count_over_500ms = int((diffs_ms > _GAP_WARN_MS).sum())
    else:
        time_span_seconds = 0.0
        mean_gap_ms = 0.0
        max_gap_ms = 0.0
        gap_count_over_500ms = 0

    crossed_book_count = 0
    if BID_PRICE_COLS[0] in df.columns and ASK_PRICE_COLS[0] in df.columns:
        crossed_book_count = int((df[BID_PRICE_COLS[0]] > df[ASK_PRICE_COLS[0]]).sum())

    return {
        "crossed_book_count": crossed_book_count,
        "gap_count_over_500ms": gap_count_over_500ms,
        "max_gap_ms": max_gap_ms,
        "mean_gap_ms": mean_gap_ms,
        "nan_count": nan_count,
        "row_count": row_count,
        "time_span_seconds": time_span_seconds,
    }
