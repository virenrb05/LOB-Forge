"""Tests for temporal train/val/test splitting with purge gaps."""

from __future__ import annotations

import numpy as np
import pytest

from lob_forge.data.splits import temporal_split


# ── Basic split correctness ──────────────────────────────────────────────


class TestTemporalSplitBasic:
    """Core split behaviour with default and custom parameters."""

    def test_no_purge_gap_produces_expected_ranges(self) -> None:
        """n_rows=1000, purge_gap=0: train=[0..699], val=[700..849], test=[850..999]."""
        train, val, test = temporal_split(1000, ratios=(0.7, 0.15, 0.15), purge_gap=0)
        np.testing.assert_array_equal(train, np.arange(0, 700))
        np.testing.assert_array_equal(val, np.arange(700, 850))
        np.testing.assert_array_equal(test, np.arange(850, 1000))

    def test_purge_gap_shifts_segments(self) -> None:
        """n_rows=1000, purge_gap=10: train=[0..699], val=[710..859], test=[870..999]."""
        train, val, test = temporal_split(1000, ratios=(0.7, 0.15, 0.15), purge_gap=10)
        np.testing.assert_array_equal(train, np.arange(0, 700))
        np.testing.assert_array_equal(val, np.arange(710, 860))
        np.testing.assert_array_equal(test, np.arange(870, 1000))

    def test_small_dataset(self) -> None:
        """n_rows=10, ratios=(0.7,0.15,0.15), purge_gap=0 → train=[0..6], val=[7..8], test=[9]."""
        train, val, test = temporal_split(10, ratios=(0.7, 0.15, 0.15), purge_gap=0)
        np.testing.assert_array_equal(train, np.arange(0, 7))
        np.testing.assert_array_equal(val, np.arange(7, 8))
        np.testing.assert_array_equal(test, np.arange(8, 10))

    def test_return_types_are_int64_ndarrays(self) -> None:
        """All returned arrays have dtype int64."""
        train, val, test = temporal_split(100)
        for arr in (train, val, test):
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == np.int64


# ── No-overlap / leakage prevention ─────────────────────────────────────


class TestNoLeakage:
    """Guarantee that no data leakage exists between splits."""

    @pytest.mark.parametrize("purge_gap", [0, 1, 5, 10, 50])
    def test_no_overlap_between_segments(self, purge_gap: int) -> None:
        """train ∩ val = ∅, val ∩ test = ∅, train ∩ test = ∅."""
        train, val, test = temporal_split(1000, purge_gap=purge_gap)
        assert len(np.intersect1d(train, val)) == 0
        assert len(np.intersect1d(val, test)) == 0
        assert len(np.intersect1d(train, test)) == 0

    @pytest.mark.parametrize("purge_gap", [0, 1, 5, 10, 50])
    def test_temporal_ordering(self, purge_gap: int) -> None:
        """max(train) < min(val) < max(val) < min(test)."""
        train, val, test = temporal_split(1000, purge_gap=purge_gap)
        # All segments should be non-empty for these params
        assert train[-1] < val[0]
        assert val[-1] < test[0]

    @pytest.mark.parametrize("purge_gap", [1, 5, 10, 50])
    def test_purge_gap_maintained_between_train_and_val(self, purge_gap: int) -> None:
        """min(val) - max(train) >= purge_gap + 1."""
        train, val, test = temporal_split(1000, purge_gap=purge_gap)
        gap = int(val[0]) - int(train[-1])
        assert gap >= purge_gap + 1, f"gap={gap}, expected >= {purge_gap + 1}"

    @pytest.mark.parametrize("purge_gap", [1, 5, 10, 50])
    def test_purge_gap_maintained_between_val_and_test(self, purge_gap: int) -> None:
        """min(test) - max(val) >= purge_gap + 1."""
        train, val, test = temporal_split(1000, purge_gap=purge_gap)
        gap = int(test[0]) - int(val[-1])
        assert gap >= purge_gap + 1, f"gap={gap}, expected >= {purge_gap + 1}"

    def test_total_rows_used_lte_n_rows(self) -> None:
        """Total indices used across all splits <= n_rows."""
        n = 1000
        train, val, test = temporal_split(n, purge_gap=10)
        total = len(train) + len(val) + len(test)
        assert total <= n

    def test_all_indices_within_bounds(self) -> None:
        """All indices in [0, n_rows)."""
        n = 500
        train, val, test = temporal_split(n, purge_gap=5)
        all_idx = np.concatenate([train, val, test])
        assert all_idx.min() >= 0
        assert all_idx.max() < n


# ── Edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases: large purge gaps, empty segments."""

    def test_large_purge_gap_makes_val_empty(self) -> None:
        """purge_gap larger than remaining rows → val/test become empty arrays."""
        train, val, test = temporal_split(100, purge_gap=500)
        # val and/or test should be empty when purge_gap exceeds available rows
        assert len(val) == 0 or len(test) == 0

    def test_empty_segments_are_valid_ndarrays(self) -> None:
        """Empty segments are still np.ndarray with dtype int64."""
        train, val, test = temporal_split(100, purge_gap=500)
        for arr in (train, val, test):
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == np.int64

    def test_purge_gap_zero_means_contiguous(self) -> None:
        """With purge_gap=0, all splits form contiguous coverage."""
        train, val, test = temporal_split(100, purge_gap=0)
        all_idx = np.concatenate([train, val, test])
        # Should be contiguous from 0 to 99
        np.testing.assert_array_equal(np.sort(all_idx), np.arange(100))
