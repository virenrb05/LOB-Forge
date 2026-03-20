"""Tests for mid-price movement labeling with causality guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lob_forge.data.labels import compute_labels
from lob_forge.data.schema import MID_PRICE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mid_price_df(prices: list[float]) -> pd.DataFrame:
    """Create a minimal DataFrame with a mid_price column."""
    return pd.DataFrame({MID_PRICE: prices})


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


class TestMonotonicPrices:
    """Monotonically increasing / decreasing / flat prices."""

    def test_increasing_prices_labelled_up(self):
        """Steadily increasing prices should produce UP (2) labels."""
        prices = [100.0 + 0.01 * i for i in range(200)]
        df = _make_mid_price_df(prices)
        result = compute_labels(df, horizons=[10], threshold=0.00002)

        col = "label_h10"
        assert col in result.columns
        valid = result[col].dropna()
        assert (valid == 2.0).all(), f"Expected all UP, got:\n{valid.value_counts()}"

    def test_decreasing_prices_labelled_down(self):
        """Steadily decreasing prices should produce DOWN (0) labels."""
        prices = [200.0 - 0.01 * i for i in range(200)]
        df = _make_mid_price_df(prices)
        result = compute_labels(df, horizons=[10], threshold=0.00002)

        col = "label_h10"
        valid = result[col].dropna()
        assert (valid == 0.0).all(), f"Expected all DOWN, got:\n{valid.value_counts()}"

    def test_flat_prices_labelled_stationary(self):
        """Constant prices should produce STATIONARY (1) labels."""
        prices = [100.0] * 200
        df = _make_mid_price_df(prices)
        result = compute_labels(df, horizons=[10], threshold=0.00002)

        col = "label_h10"
        valid = result[col].dropna()
        assert (valid == 1.0).all(), f"Expected all STATIONARY, got:\n{valid.value_counts()}"


# ---------------------------------------------------------------------------
# Multi-horizon
# ---------------------------------------------------------------------------


class TestMultiHorizon:
    """Multiple horizons produce independent columns."""

    def test_multiple_horizons_produce_columns(self):
        horizons = [10, 20, 50, 100]
        df = _make_mid_price_df([100.0] * 300)
        result = compute_labels(df, horizons=horizons)

        for h in horizons:
            assert f"label_h{h}" in result.columns

    def test_each_horizon_has_correct_nan_tail(self):
        """Last h rows of label_h{h} must be NaN."""
        horizons = [10, 20, 50]
        n = 200
        df = _make_mid_price_df([100.0 + 0.001 * i for i in range(n)])
        result = compute_labels(df, horizons=horizons, threshold=0.00002)

        for h in horizons:
            col = f"label_h{h}"
            tail = result[col].iloc[-h:]
            assert tail.isna().all(), (
                f"label_h{h}: last {h} rows should be NaN, got {tail.tolist()}"
            )
            # Rows before that should NOT be NaN
            head = result[col].iloc[:-h]
            assert head.notna().all(), (
                f"label_h{h}: rows before last {h} should not be NaN"
            )


# ---------------------------------------------------------------------------
# Dtype
# ---------------------------------------------------------------------------


class TestLabelDtype:
    """Label columns should be float64 with values in {0.0, 1.0, 2.0, NaN}."""

    def test_dtype_is_float64(self):
        df = _make_mid_price_df([100.0 + 0.001 * i for i in range(200)])
        result = compute_labels(df, horizons=[10])
        assert result["label_h10"].dtype == np.float64

    def test_values_in_valid_set(self):
        df = _make_mid_price_df([100.0 + 0.001 * i for i in range(200)])
        result = compute_labels(df, horizons=[10])
        valid_values = result["label_h10"].dropna().unique()
        assert set(valid_values).issubset({0.0, 1.0, 2.0}), (
            f"Unexpected label values: {valid_values}"
        )


# ---------------------------------------------------------------------------
# Manual computation test
# ---------------------------------------------------------------------------


class TestManualComputation:
    """Verify label computation against hand-calculated values."""

    def test_known_trajectory(self):
        """Small hand-computed example with horizon=3, threshold=0.001."""
        # Prices: [10.0, 10.1, 10.2, 10.3, 10.0, 9.9, 9.8]
        # For horizon h=3:
        #   t=0: smoothed = mean(10.1, 10.2, 10.3) = 10.2
        #         return = (10.2 - 10.0) / 10.0 = 0.02 > 0.001 → UP (2)
        #   t=1: smoothed = mean(10.2, 10.3, 10.0) = 10.1667
        #         return = (10.1667 - 10.1) / 10.1 = 0.0066 > 0.001 → UP (2)
        #   t=2: smoothed = mean(10.3, 10.0, 9.9) = 10.0667
        #         return = (10.0667 - 10.2) / 10.2 = -0.01307 < -0.001 → DOWN (0)
        #   t=3: smoothed = mean(10.0, 9.9, 9.8) = 9.9
        #         return = (9.9 - 10.3) / 10.3 = -0.03883 < -0.001 → DOWN (0)
        #   t=4,5,6: NaN (last 3)
        prices = [10.0, 10.1, 10.2, 10.3, 10.0, 9.9, 9.8]
        df = _make_mid_price_df(prices)
        result = compute_labels(df, horizons=[3], threshold=0.001)

        col = "label_h3"
        expected = [2.0, 2.0, 0.0, 0.0, np.nan, np.nan, np.nan]
        actual = result[col].tolist()

        for i, (exp, act) in enumerate(zip(expected, actual)):
            if np.isnan(exp):
                assert np.isnan(act), f"Row {i}: expected NaN, got {act}"
            else:
                assert act == exp, f"Row {i}: expected {exp}, got {act}"


# ---------------------------------------------------------------------------
# Causality test
# ---------------------------------------------------------------------------


class TestCausality:
    """Labels must be strictly causal — no future data leakage."""

    def test_modifying_future_does_not_change_past_labels(self):
        """
        Compute labels, modify mid_price at row 50, recompute.
        Labels at rows 0..49 must NOT change.
        """
        n = 200
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.01)
        df_original = _make_mid_price_df(prices.tolist())

        horizons = [10, 20, 50]
        labels_before = compute_labels(df_original.copy(), horizons=horizons)

        # Modify price at row 50
        df_modified = df_original.copy()
        df_modified.loc[50, MID_PRICE] = 999.0  # drastic change

        labels_after = compute_labels(df_modified, horizons=horizons)

        for h in horizons:
            col = f"label_h{h}"
            # Rows 0..49 must be identical (no future leakage)
            before_slice = labels_before[col].iloc[:50]
            after_slice = labels_after[col].iloc[:50]

            # Handle NaN comparison correctly
            both_nan = before_slice.isna() & after_slice.isna()
            both_equal = before_slice == after_slice
            match = both_nan | both_equal

            assert match.all(), (
                f"Causality violated for {col}: labels at rows <50 changed "
                f"after modifying row 50"
            )

    def test_modifying_past_may_change_nearby_labels(self):
        """
        Modifying mid_price at row 50 CAN change labels at rows 50..50+max_h.
        This test just verifies the boundary behavior is correct.
        """
        n = 200
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.01)
        df_original = _make_mid_price_df(prices.tolist())

        max_h = 50
        labels_before = compute_labels(df_original.copy(), horizons=[max_h])
        col = f"label_h{max_h}"

        df_modified = df_original.copy()
        df_modified.loc[50, MID_PRICE] = 999.0

        labels_after = compute_labels(df_modified, horizons=[max_h])

        # Labels far enough after the modification point should remain the same
        # Rows beyond 50 + max_h are unaffected
        safe_start = 50 + max_h + 1
        before_safe = labels_before[col].iloc[safe_start:]
        after_safe = labels_after[col].iloc[safe_start:]

        both_nan = before_safe.isna() & after_safe.isna()
        both_equal = before_safe == after_safe
        match = both_nan | both_equal
        assert match.all(), (
            f"Labels beyond row {safe_start} should not change "
            f"after modifying row 50"
        )


# ---------------------------------------------------------------------------
# Input passes through
# ---------------------------------------------------------------------------


class TestInputPassThrough:
    """compute_labels should add columns to the input DataFrame."""

    def test_original_columns_preserved(self):
        df = _make_mid_price_df([100.0] * 50)
        df["extra_col"] = 42
        result = compute_labels(df, horizons=[10])
        assert "extra_col" in result.columns
        assert (result["extra_col"] == 42).all()

    def test_original_data_unchanged(self):
        prices = [100.0 + i for i in range(50)]
        df = _make_mid_price_df(prices)
        original_prices = df[MID_PRICE].copy()
        compute_labels(df, horizons=[10])
        pd.testing.assert_series_equal(df[MID_PRICE], original_prices)
