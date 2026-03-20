"""Unit tests for LOB derived feature computations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lob_forge.data.features import (
    compute_all_features,
    compute_depth_imbalance,
    compute_microprice,
    compute_mid_returns,
    compute_mlofi,
    compute_ofi,
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
    TRADE_SIDE,
    TRADE_SIZE,
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
        df = _make_lob_df(
            5, **{MID_PRICE: np.array([100.0, 101.0, 102.0, 103.0, 104.0])}
        )
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


# ---- compute_ofi ----


class TestOfi:
    def test_unchanged_book_gives_zero_ofi(self):
        """When bid/ask prices and sizes don't change, OFI should be 0."""
        df = _make_lob_df(5)
        # Ensure constant prices and sizes (default helper already does this)
        result = compute_ofi(df)
        # Rows 1..4 should be 0 (unchanged book)
        np.testing.assert_array_almost_equal(result.iloc[1:].values, 0.0)

    def test_known_ofi_value(self):
        """Bid_size increases by 10 while bid_price stays same -> OFI = +10."""
        df = _make_lob_df(3)
        # Set constant bid price
        df[BID_PRICE_COLS[0]] = 100.0
        # Set constant ask price and ask size
        df[ASK_PRICE_COLS[0]] = 101.0
        df[ASK_SIZE_COLS[0]] = 50.0
        # Bid size increases: 100, 110, 120
        df[BID_SIZE_COLS[0]] = [100.0, 110.0, 120.0]
        result = compute_ofi(df)
        # Row 1: delta_bid=+10, bid_price unchanged so indicator=1
        #         delta_ask=0, ask_price unchanged so indicator=1
        #         OFI = 10*1 - 0*1 = 10
        np.testing.assert_almost_equal(result.iloc[1], 10.0)
        np.testing.assert_almost_equal(result.iloc[2], 10.0)

    def test_first_row_is_nan(self):
        """First row has no previous snapshot, so OFI must be NaN."""
        df = _make_lob_df(5)
        result = compute_ofi(df)
        assert np.isnan(result.iloc[0])

    def test_price_level_change_resets(self):
        """When bid_price drops below previous, bid_size delta should NOT contribute."""
        df = _make_lob_df(3)
        # Bid price drops from 100 to 99
        df[BID_PRICE_COLS[0]] = [100.0, 99.0, 99.0]
        df[BID_SIZE_COLS[0]] = [100.0, 200.0, 200.0]
        # Ask stays the same
        df[ASK_PRICE_COLS[0]] = 101.0
        df[ASK_SIZE_COLS[0]] = 50.0
        result = compute_ofi(df)
        # Row 1: bid_price dropped (99 < 100), indicator=0
        #         delta_bid=+100, but multiplied by 0 = 0
        #         delta_ask=0, ask_price unchanged indicator=1, 0*1 = 0
        #         OFI = 0 - 0 = 0
        np.testing.assert_almost_equal(result.iloc[1], 0.0)


# ---- compute_mlofi ----


class TestMlofi:
    def test_returns_series(self):
        """Output is a pd.Series named 'mlofi'."""
        df = _make_lob_df(5)
        result = compute_mlofi(df)
        assert isinstance(result, pd.Series)
        assert result.name == "mlofi"

    def test_decay_weighting(self):
        """Level 1 contributes more than level 2 (decay < 1)."""
        df = _make_lob_df(3)
        # Make level 1 and level 2 both have the same size change
        for i in range(2):
            df[BID_PRICE_COLS[i]] = 100.0
            df[ASK_PRICE_COLS[i]] = 101.0
            df[ASK_SIZE_COLS[i]] = 50.0
        df[BID_SIZE_COLS[0]] = [100.0, 110.0, 110.0]
        df[BID_SIZE_COLS[1]] = [100.0, 110.0, 110.0]
        # Level 1 only
        mlofi_1 = compute_mlofi(df, levels=1, decay=0.5)
        # Level 2 only contribution = mlofi(levels=2) - mlofi(levels=1)
        mlofi_2 = compute_mlofi(df, levels=2, decay=0.5)
        # Level 2 contribution should be decay * level1_contribution = 0.5 * level1
        level1_contrib = mlofi_1.iloc[1]
        level2_contrib = mlofi_2.iloc[1] - mlofi_1.iloc[1]
        np.testing.assert_almost_equal(level2_contrib, 0.5 * level1_contrib)

    def test_first_row_is_nan(self):
        """First row is NaN."""
        df = _make_lob_df(5)
        result = compute_mlofi(df)
        assert np.isnan(result.iloc[0])

    def test_single_level_equals_ofi(self):
        """compute_mlofi(df, levels=1, decay=0.5) should equal compute_ofi(df)."""
        df = _make_lob_df(5)
        # Make some variation
        df[BID_SIZE_COLS[0]] = [100.0, 110.0, 105.0, 120.0, 90.0]
        df[BID_PRICE_COLS[0]] = [100.0, 100.0, 99.0, 100.0, 100.0]
        ofi = compute_ofi(df)
        mlofi = compute_mlofi(df, levels=1, decay=0.5)
        # Both should be NaN at row 0, equal elsewhere
        assert np.isnan(ofi.iloc[0]) and np.isnan(mlofi.iloc[0])
        np.testing.assert_array_almost_equal(ofi.iloc[1:].values, mlofi.iloc[1:].values)


# ---- compute_all_features ----


class TestComputeAllFeatures:
    def test_adds_20_columns(self):
        """compute_all_features adds 20 new columns."""
        n = 200
        df = _make_lob_df(n)
        df[TRADE_PRICE] = df[MID_PRICE]
        df[TRADE_SIZE] = 10.0
        original_cols = set(df.columns)
        result = compute_all_features(df)
        new_cols = set(result.columns) - original_cols
        assert len(new_cols) == 20

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
            + ["spread_bps", "vpin", "ofi", "mlofi"]
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
