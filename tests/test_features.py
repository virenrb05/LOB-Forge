"""Unit tests for LOB derived feature computations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lob_forge.data.features import (
    compute_all_features,
    compute_depth_imbalance,
    compute_microprice,
    compute_mid_returns,
    compute_order_imbalance,
    compute_spread_bps,
    compute_vpin,
)
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
    TRADE_SIZE,
    TRADE_SIDE,
)


def _make_lob_df(n: int = 10, **overrides) -> pd.DataFrame:
    """Create a minimal valid LOB DataFrame for testing."""
    data = {TIMESTAMP: np.arange(n, dtype=np.int64) * 1_000_000}
    data[MID_PRICE] = np.linspace(100.0, 100.0 + (n - 1) * 0.01, n)
    data[SPREAD] = np.full(n, 0.02)

    for i, col in enumerate(BID_PRICE_COLS, 1):
        data[col] = data[MID_PRICE] - 0.01 * i
    for i, col in enumerate(BID_SIZE_COLS, 1):
        data[col] = np.full(n, 100.0 * i)
    for i, col in enumerate(ASK_PRICE_COLS, 1):
        data[col] = data[MID_PRICE] + 0.01 * i
    for i, col in enumerate(ASK_SIZE_COLS, 1):
        data[col] = np.full(n, 50.0 * i)

    data[TRADE_PRICE] = np.full(n, np.nan)
    data[TRADE_SIZE] = np.full(n, np.nan)
    data[TRADE_SIDE] = np.zeros(n, dtype=np.int8)

    data.update(overrides)
    return pd.DataFrame(data)[ALL_COLUMNS]


# ---- compute_mid_returns ----


class TestMidReturns:
    def test_basic_return(self):
        """Mid return_1 at row 1 = (101-100)/100 = 0.01."""
        df = _make_lob_df(5, **{MID_PRICE: np.array([100.0, 101.0, 102.0, 103.0, 104.0])})
        result = compute_mid_returns(df, lookbacks=[1])
        assert "mid_return_1" in result.columns
        assert np.isnan(result["mid_return_1"].iloc[0])
        np.testing.assert_almost_equal(result["mid_return_1"].iloc[1], 0.01)

    def test_multiple_lookbacks(self):
        df = _make_lob_df(10)
        result = compute_mid_returns(df, lookbacks=[1, 5])
        assert "mid_return_1" in result.columns
        assert "mid_return_5" in result.columns
        # First 5 rows of mid_return_5 should be NaN
        assert result["mid_return_5"].iloc[:5].isna().all()

    def test_default_lookbacks(self):
        df = _make_lob_df(60)
        result = compute_mid_returns(df)
        for k in [1, 5, 10, 50]:
            assert f"mid_return_{k}" in result.columns


# ---- compute_order_imbalance ----


class TestOrderImbalance:
    def test_balanced(self):
        """Equal bid/ask sizes -> imbalance = 0."""
        df = _make_lob_df(5)
        df[BID_SIZE_COLS[0]] = 100.0
        df[ASK_SIZE_COLS[0]] = 100.0
        result = compute_order_imbalance(df)
        np.testing.assert_almost_equal(result.iloc[0], 0.0)

    def test_known_value(self):
        """bid=100, ask=50 -> (100-50)/(100+50) = 0.3333."""
        df = _make_lob_df(5)
        df[BID_SIZE_COLS[0]] = 100.0
        df[ASK_SIZE_COLS[0]] = 50.0
        result = compute_order_imbalance(df)
        np.testing.assert_almost_equal(result.iloc[0], 1 / 3, decimal=4)

    def test_range(self):
        """Order imbalance in [-1, 1]."""
        df = _make_lob_df(5)
        df[BID_SIZE_COLS[0]] = 200.0
        df[ASK_SIZE_COLS[0]] = 1.0
        result = compute_order_imbalance(df)
        assert (result >= -1).all() and (result <= 1).all()


# ---- compute_microprice ----


class TestMicroprice:
    def test_equal_sizes_is_midprice(self):
        """When bid_size == ask_size, microprice == mid_price."""
        df = _make_lob_df(5)
        df[BID_SIZE_COLS[0]] = 100.0
        df[ASK_SIZE_COLS[0]] = 100.0
        result = compute_microprice(df)
        expected = (df[BID_PRICE_COLS[0]] + df[ASK_PRICE_COLS[0]]) / 2
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_known_value(self):
        """bid=100, ask=101, bid_size=200, ask_size=100 -> (100*100+101*200)/300 = 100.6667."""
        df = _make_lob_df(1)
        df[BID_PRICE_COLS[0]] = 100.0
        df[ASK_PRICE_COLS[0]] = 101.0
        df[BID_SIZE_COLS[0]] = 200.0
        df[ASK_SIZE_COLS[0]] = 100.0
        result = compute_microprice(df)
        expected = (100.0 * 100.0 + 101.0 * 200.0) / 300.0
        np.testing.assert_almost_equal(result.iloc[0], expected, decimal=4)


# ---- compute_depth_imbalance ----


class TestDepthImbalance:
    def test_column_count(self):
        df = _make_lob_df(5)
        result = compute_depth_imbalance(df)
        assert result.shape[1] == 10
        for i in range(1, 11):
            assert f"depth_imb_{i}" in result.columns

    def test_all_positive_when_bid_larger(self):
        """When all bid sizes > ask sizes, all imbalances are positive."""
        df = _make_lob_df(5)
        for col in BID_SIZE_COLS:
            df[col] = 200.0
        for col in ASK_SIZE_COLS:
            df[col] = 100.0
        result = compute_depth_imbalance(df)
        assert (result > 0).all().all()

    def test_range(self):
        df = _make_lob_df(5)
        result = compute_depth_imbalance(df)
        assert (result >= -1).all().all() and (result <= 1).all().all()


# ---- compute_spread_bps ----


class TestSpreadBps:
    def test_known_value(self):
        """spread=0.01, mid=100 -> 1.0 bps."""
        df = _make_lob_df(5, **{SPREAD: np.full(5, 0.01), MID_PRICE: np.full(5, 100.0)})
        result = compute_spread_bps(df)
        np.testing.assert_almost_equal(result.iloc[0], 1.0)

    def test_positive(self):
        df = _make_lob_df(5)
        result = compute_spread_bps(df)
        assert (result > 0).all()


# ---- compute_vpin ----


class TestVpin:
    def test_returns_series(self):
        """VPIN returns a pandas Series with correct length."""
        n = 200
        df = _make_lob_df(n)
        # Add some trades
        df[TRADE_PRICE] = df[MID_PRICE]
        df[TRADE_SIZE] = 10.0
        result = compute_vpin(df, n_buckets=10)
        assert isinstance(result, pd.Series)
        assert len(result) == n

    def test_range(self):
        """VPIN in [0, 1]."""
        n = 200
        df = _make_lob_df(n)
        df[TRADE_PRICE] = df[MID_PRICE]
        df[TRADE_SIZE] = 10.0
        result = compute_vpin(df, n_buckets=10)
        valid = result.dropna()
        if len(valid) > 0:
            assert (valid >= 0).all() and (valid <= 1).all()

    def test_nan_trades_forward_filled(self):
        """Rows without trades get forward-filled VPIN."""
        n = 200
        df = _make_lob_df(n)
        df[TRADE_PRICE] = df[MID_PRICE]
        df[TRADE_SIZE] = 10.0
        # Set last 10 rows to NaN trades
        df.loc[df.index[-10:], TRADE_SIZE] = np.nan
        df.loc[df.index[-10:], TRADE_PRICE] = np.nan
        result = compute_vpin(df, n_buckets=10)
        # Last 10 values should be forward-filled (not NaN if preceding was valid)
        if not result.iloc[:-10].isna().all():
            assert not result.iloc[-1:].isna().any()


# ---- compute_all_features ----


class TestComputeAllFeatures:
    def test_adds_18_columns(self):
        """compute_all_features adds 18 new columns."""
        n = 200
        df = _make_lob_df(n)
        df[TRADE_PRICE] = df[MID_PRICE]
        df[TRADE_SIZE] = 10.0
        original_cols = set(df.columns)
        result = compute_all_features(df)
        new_cols = set(result.columns) - original_cols
        assert len(new_cols) == 18

    def test_expected_columns_present(self):
        n = 200
        df = _make_lob_df(n)
        df[TRADE_PRICE] = df[MID_PRICE]
        df[TRADE_SIZE] = 10.0
        result = compute_all_features(df)
        expected = (
            [f"mid_return_{k}" for k in [1, 5, 10, 50]]
            + ["order_imbalance", "microprice"]
            + [f"depth_imb_{i}" for i in range(1, 11)]
            + ["spread_bps", "vpin"]
        )
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_preserves_original_columns(self):
        n = 200
        df = _make_lob_df(n)
        df[TRADE_PRICE] = df[MID_PRICE]
        df[TRADE_SIZE] = 10.0
        result = compute_all_features(df)
        for col in ALL_COLUMNS:
            assert col in result.columns
