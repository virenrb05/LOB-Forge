"""Unit tests for rolling_zscore normalization in lob_forge.data.preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lob_forge.data.preprocessor import rolling_zscore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_df(
    n: int,
    cols: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Create a DataFrame with random feature columns."""
    if rng is None:
        rng = np.random.default_rng(42)
    if cols is None:
        cols = ["feat_a", "feat_b"]
    data = {c: rng.standard_normal(n) * 10 + 100 for c in cols}
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRollingZscoreZeroMean:
    """After the warm-up period the normalized values should have ~zero mean."""

    def test_zero_mean_after_warmup(self):
        rng = np.random.default_rng(42)
        n = 2000
        window = 100
        cols = ["feat_a", "feat_b"]
        df = _make_feature_df(n, cols, rng)

        normed, _ = rolling_zscore(df, cols, window=window)

        for col in cols:
            post_warmup = normed[col].iloc[window:]
            assert abs(post_warmup.mean()) < 0.15, (
                f"Mean of {col} after warm-up is {post_warmup.mean():.4f}, "
                "expected close to 0"
            )


class TestRollingZscoreUnitVariance:
    """After warm-up the rolling std of normalized values should be ~1."""

    def test_unit_variance_after_warmup(self):
        rng = np.random.default_rng(42)
        n = 2000
        window = 100
        cols = ["feat_a", "feat_b"]
        df = _make_feature_df(n, cols, rng)

        normed, _ = rolling_zscore(df, cols, window=window)

        for col in cols:
            post_warmup = normed[col].iloc[window:]
            std = post_warmup.std()
            assert 0.5 < std < 2.0, (
                f"Std of {col} after warm-up is {std:.4f}, expected close to 1"
            )


class TestRollingZscoreCausality:
    """Changing a future value must not affect past normalized values."""

    def test_causal_independence(self):
        rng = np.random.default_rng(42)
        n = 1000
        window = 100
        cols = ["feat_a"]
        df = _make_feature_df(n, cols, rng)

        normed_original, _ = rolling_zscore(df, cols, window=window)

        # Mutate a value at row 500
        df_mutated = df.copy()
        df_mutated.loc[500, "feat_a"] = 99999.0

        normed_mutated, _ = rolling_zscore(df_mutated, cols, window=window)

        # Rows 0..499 must be identical
        pd.testing.assert_frame_equal(
            normed_original[cols].iloc[:500],
            normed_mutated[cols].iloc[:500],
        )

        # Rows >= 500 may differ
        assert not normed_original["feat_a"].iloc[500:].equals(
            normed_mutated["feat_a"].iloc[500:]
        ), "Expected rows >=500 to differ after mutation"


class TestRollingZscoreStatsReturned:
    """The second return value must contain {col}_mean and {col}_std columns."""

    def test_stats_columns(self):
        cols = ["alpha", "beta", "gamma"]
        df = _make_feature_df(50, cols)

        _, stats = rolling_zscore(df, cols, window=10)

        for col in cols:
            assert f"{col}_mean" in stats.columns, f"Missing {col}_mean"
            assert f"{col}_std" in stats.columns, f"Missing {col}_std"

        # Stats should have the same number of rows
        assert len(stats) == len(df)

    def test_stats_values_finite(self):
        cols = ["x"]
        df = _make_feature_df(100, cols)
        _, stats = rolling_zscore(df, cols, window=20)

        assert np.all(np.isfinite(stats["x_mean"].values))
        assert np.all(np.isfinite(stats["x_std"].values))


class TestRollingZscoreConstantSeries:
    """All identical values -> normalized values should be 0.0."""

    def test_constant_gives_zero(self):
        n = 100
        df = pd.DataFrame({"const": np.full(n, 42.0)})

        normed, stats = rolling_zscore(df, ["const"], window=20)

        # x - mean = 0 for all rows, so normalized = 0
        np.testing.assert_array_equal(normed["const"].values, 0.0)

        # std should be replaced with 1.0
        assert (stats["const_std"] == 1.0).all()


class TestRollingZscoreSingleRow:
    """Single-row DataFrame must not crash."""

    def test_single_row_no_crash(self):
        df = pd.DataFrame({"val": [5.0]})

        normed, stats = rolling_zscore(df, ["val"], window=100)

        assert len(normed) == 1
        assert len(stats) == 1
        # With a single value, mean = 5.0, std -> NaN -> replaced with 1.0
        # So normalized = (5 - 5) / 1 = 0
        assert normed["val"].iloc[0] == 0.0


class TestRollingZscoreWindowParameter:
    """Different window sizes should produce different normalizations."""

    def test_different_windows_differ(self):
        rng = np.random.default_rng(42)
        n = 500
        cols = ["feat"]
        df = _make_feature_df(n, cols, rng)

        normed_small, _ = rolling_zscore(df, cols, window=10)
        normed_large, _ = rolling_zscore(df, cols, window=200)

        # After row 200 (both windows fully warmed), values should differ
        diff = (normed_small["feat"].iloc[200:] - normed_large["feat"].iloc[200:]).abs()
        assert diff.sum() > 0, "Expected different windows to produce different results"
