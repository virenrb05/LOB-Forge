"""Unit tests for stylized fact validation functions.

Uses deterministic synthetic data with known statistical properties
to verify each of the 6 stylized fact test functions.
"""

from __future__ import annotations

import unittest

import numpy as np

from lob_forge.evaluation.stylized_facts import (
    bid_ask_bounce_test,
    book_shape_test,
    market_impact_test,
    return_distribution_test,
    spread_cdf_test,
    volatility_clustering_test,
)


def _make_book(rng: np.random.Generator, n: int, spread: float = 1.0) -> np.ndarray:
    """Create a minimal (n, 40) book array with realistic structure.

    Layout: ask_price(0-9), ask_size(10-19), bid_price(20-29), bid_size(30-39).
    """
    book = np.zeros((n, 40), dtype=np.float64)
    mid = 100.0 + rng.standard_normal(n).cumsum() * 0.01
    for i in range(10):
        book[:, i] = mid + spread / 2 + i * 0.01  # ask prices
        book[:, 20 + i] = mid - spread / 2 - i * 0.01  # bid prices
        book[:, 10 + i] = rng.exponential(10.0, size=n)  # ask sizes
        book[:, 30 + i] = rng.exponential(10.0, size=n)  # bid sizes
    return book


# ---------------------------------------------------------------------------
# Test 1: Return distribution
# ---------------------------------------------------------------------------


class TestReturnDistribution(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(seed=42)
        self.n = 5000
        base = 100.0 + self.rng.standard_normal(self.n).cumsum() * 0.01
        self.real_mid = np.abs(base)  # ensure positive

    def test_identical_distributions(self):
        """Same data should give KS p-value near 1 and pass."""
        result = return_distribution_test(self.real_mid, self.real_mid)
        self.assertGreater(result["p_value"], 0.9)
        self.assertTrue(result["passed"])

    def test_different_distributions(self):
        """Normal vs uniform returns should fail."""
        # Uniform mid-price changes produce uniform returns
        uniform_mid = 100.0 + np.cumsum(self.rng.uniform(-1, 1, self.n)) * 0.001
        uniform_mid = np.abs(uniform_mid) + 1.0
        result = return_distribution_test(self.real_mid, uniform_mid)
        self.assertLess(result["p_value"], 0.05)
        self.assertFalse(result["passed"])

    def test_fat_tails(self):
        """Student-t returns should show excess kurtosis > 3."""
        t_mid = 100.0 + np.cumsum(self.rng.standard_t(df=3, size=self.n)) * 0.001
        t_mid = np.abs(t_mid) + 1.0
        result = return_distribution_test(t_mid, t_mid)
        self.assertGreater(result["real_kurtosis"], 3.0)


# ---------------------------------------------------------------------------
# Test 2: Volatility clustering
# ---------------------------------------------------------------------------


class TestVolatilityClustering(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(seed=42)
        self.n = 5000

    def test_clustered_returns(self):
        """GARCH-like alternating high/low vol blocks should show clustering."""
        # Alternate between high and low volatility blocks
        block_size = 100
        n_blocks = self.n // block_size
        returns = np.empty(self.n)
        for i in range(n_blocks):
            sigma = 0.01 if i % 2 == 0 else 0.1
            returns[i * block_size : (i + 1) * block_size] = (
                self.rng.standard_normal(block_size) * sigma
            )
        mid = 100.0 + np.cumsum(returns)
        mid = np.abs(mid) + 1.0

        result = volatility_clustering_test(mid, mid)
        self.assertGreater(result["mean_acf_real"], 0)
        self.assertTrue(result["passed"])

    def test_iid_returns(self):
        """IID normal returns should have near-zero ACF of absolute returns."""
        iid_returns = self.rng.standard_normal(self.n) * 0.01
        mid = 100.0 + np.cumsum(iid_returns)
        mid = np.abs(mid) + 1.0

        result = volatility_clustering_test(mid, mid)
        self.assertAlmostEqual(result["mean_acf_real"], 0.0, delta=0.05)


# ---------------------------------------------------------------------------
# Test 3: Bid-ask bounce
# ---------------------------------------------------------------------------


class TestBidAskBounce(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(seed=42)

    def test_negative_autocorrelation(self):
        """Alternating +1/-1 returns should show negative lag-1 ACF."""
        n = 2000
        returns = np.tile([1.0, -1.0], n // 2)
        # Add tiny noise to avoid perfect correlation edge cases
        returns += self.rng.standard_normal(n) * 0.01

        result = bid_ask_bounce_test(returns, returns)
        self.assertLess(result["real_lag1_acf"], 0)
        self.assertLess(result["synthetic_lag1_acf"], 0)
        self.assertTrue(result["passed"])

    def test_positive_autocorrelation(self):
        """Trending returns (positive autocorrelation) should fail."""
        n = 2000
        # Create trending series: returns that follow previous direction
        returns = np.empty(n)
        returns[0] = 1.0
        for i in range(1, n):
            returns[i] = returns[i - 1] + self.rng.standard_normal() * 0.1
        result = bid_ask_bounce_test(returns, returns)
        self.assertGreater(result["real_lag1_acf"], 0)
        self.assertFalse(result["passed"])


# ---------------------------------------------------------------------------
# Test 4: Spread CDF
# ---------------------------------------------------------------------------


class TestSpreadCdf(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(seed=42)
        self.n = 2000

    def test_matching_spreads(self):
        """Same book data should pass."""
        book = _make_book(self.rng, self.n, spread=1.0)
        result = spread_cdf_test(book, book)
        self.assertTrue(result["passed"])
        self.assertGreater(result["p_value"], 0.9)

    def test_different_spreads(self):
        """Books with different spread distributions should fail."""
        book_narrow = _make_book(self.rng, self.n, spread=0.5)
        book_wide = _make_book(self.rng, self.n, spread=5.0)
        result = spread_cdf_test(book_narrow, book_wide)
        self.assertFalse(result["passed"])
        self.assertLess(result["p_value"], 0.05)


# ---------------------------------------------------------------------------
# Test 5: Book shape
# ---------------------------------------------------------------------------


class TestBookShape(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(seed=42)
        self.n = 2000

    def test_matching_depth(self):
        """Same book data should pass."""
        book = _make_book(self.rng, self.n)
        result = book_shape_test(book, book)
        self.assertTrue(result["passed"])
        self.assertGreater(result["min_p_value"], 0.9)

    def test_different_depth(self):
        """Books with different size profiles should have low p-value."""
        book1 = _make_book(self.rng, self.n)
        # Create book with very different size distribution
        book2 = book1.copy()
        # Multiply all size columns by large factor
        book2[:, 10:20] *= 10.0  # ask sizes
        book2[:, 30:40] *= 10.0  # bid sizes
        result = book_shape_test(book1, book2)
        self.assertLess(result["min_p_value"], 0.05)


# ---------------------------------------------------------------------------
# Test 6: Market impact
# ---------------------------------------------------------------------------


class TestMarketImpact(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(seed=42)
        self.n = 5000

    def test_concave_impact(self):
        """Square-root impact data should give beta < 1."""
        book = _make_book(self.rng, self.n)
        # Create mid-price that follows sqrt of bid-size changes
        bid_sizes = book[:, 30:40]
        vol = np.sum(np.abs(np.diff(bid_sizes, axis=0)), axis=1)
        # Impact = sqrt(volume) + noise → concave (beta ~ 0.5)
        impact = np.sqrt(vol + 1e-12) * 0.001
        mid = 100.0 + np.cumsum(impact * self.rng.choice([-1, 1], size=len(impact)))
        mid = np.abs(mid) + 1.0
        # Prepend to match book length
        mid = np.concatenate([[100.0], mid])

        result = market_impact_test(book, mid, book, mid)
        self.assertLess(result["real_beta"], 1.0)
        self.assertTrue(result["passed"])

    def test_linear_impact(self):
        """Linear impact should give beta close to 1."""
        book = _make_book(self.rng, self.n)
        bid_sizes = book[:, 30:40]
        vol = np.sum(np.abs(np.diff(bid_sizes, axis=0)), axis=1)
        # Impact = volume (linear) → beta ≈ 1.0
        impact = vol * 0.0001
        mid = 100.0 + np.cumsum(impact * self.rng.choice([-1, 1], size=len(impact)))
        mid = np.abs(mid) + 1.0
        mid = np.concatenate([[100.0], mid])

        result = market_impact_test(book, mid, book, mid)
        self.assertAlmostEqual(result["real_beta"], 1.0, delta=0.3)


if __name__ == "__main__":
    unittest.main()
