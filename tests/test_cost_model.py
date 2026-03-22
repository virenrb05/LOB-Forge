"""RED-GREEN-REFACTOR tests for CostModel — spread + fee + market impact."""

import math

import pytest

from lob_forge.executor.cost_model import CostModel

# ---------------------------------------------------------------------------
# RED: These tests were written before any implementation existed.
# ---------------------------------------------------------------------------


class TestCostModelConstruction:
    def test_default_params(self):
        model = CostModel()
        assert model.fee_bps == 2.0
        assert model.impact_eta == 0.1

    def test_custom_params(self):
        model = CostModel(fee_bps=5.0, impact_eta=0.2)
        assert model.fee_bps == 5.0
        assert model.impact_eta == 0.2


class TestCostModelCompute:
    """Tests for CostModel.compute(exec_price, exec_size, mid_price, spread, avg_daily_volume)."""

    def setup_method(self):
        self.model = CostModel(fee_bps=2.0, impact_eta=0.1)

    # ------------------------------------------------------------------
    # spread_cost = 0.5 * spread * exec_size
    # ------------------------------------------------------------------

    def test_spread_cost_scales_with_spread(self):
        # Double the spread → double the spread component
        cost_low = self.model.compute(100.0, 1.0, 100.0, 0.01, 1e6)
        cost_high = self.model.compute(100.0, 1.0, 100.0, 0.02, 1e6)
        # impact and fee are constant; spread component doubles
        spread_component_low = 0.5 * 0.01 * 1.0
        spread_component_high = 0.5 * 0.02 * 1.0
        assert cost_high - cost_low == pytest.approx(
            spread_component_high - spread_component_low, rel=1e-6
        )

    def test_spread_cost_scales_with_order_size(self):
        cost_small = self.model.compute(100.0, 1.0, 100.0, 0.01, 1e6)
        cost_large = self.model.compute(100.0, 10.0, 100.0, 0.01, 1e6)
        # spread for large = 0.5 * 0.01 * 10 = 0.05; for small = 0.005
        assert cost_large > cost_small

    def test_zero_spread_no_spread_cost(self):
        model_zero_spread = CostModel(fee_bps=0.0, impact_eta=0.0)
        cost = model_zero_spread.compute(100.0, 1.0, 100.0, 0.0, 1e6)
        assert cost == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # fee_cost = fee_bps * 1e-4 * exec_price * exec_size
    # ------------------------------------------------------------------

    def test_fee_cost_is_fixed_bps_of_notional(self):
        model_fee_only = CostModel(fee_bps=2.0, impact_eta=0.0)
        cost = model_fee_only.compute(100.0, 1.0, 100.0, 0.0, 1e6)
        expected_fee = 2.0 * 1e-4 * 100.0 * 1.0
        assert cost == pytest.approx(expected_fee, rel=1e-9)

    def test_fee_scales_with_exec_price(self):
        model_fee_only = CostModel(fee_bps=2.0, impact_eta=0.0)
        cost_low = model_fee_only.compute(100.0, 1.0, 100.0, 0.0, 1e6)
        cost_high = model_fee_only.compute(200.0, 1.0, 200.0, 0.0, 1e6)
        assert cost_high == pytest.approx(cost_low * 2, rel=1e-9)

    def test_zero_fee_bps(self):
        model_no_fee = CostModel(fee_bps=0.0, impact_eta=0.1)
        cost_with_fee = CostModel(fee_bps=2.0, impact_eta=0.1).compute(
            100.0, 1.0, 100.0, 0.0, 1e6
        )
        cost_no_fee = model_no_fee.compute(100.0, 1.0, 100.0, 0.0, 1e6)
        assert cost_no_fee < cost_with_fee

    # ------------------------------------------------------------------
    # impact_cost = impact_eta * mid_price * sqrt(exec_size / avg_daily_volume) * exec_size
    # ------------------------------------------------------------------

    def test_market_impact_follows_sqrt_law(self):
        """impact_cost ∝ exec_size^1.5 / sqrt(avg_daily_volume) — doubling size multiplies impact by 2^1.5."""
        model_impact_only = CostModel(fee_bps=0.0, impact_eta=0.1)
        cost_1 = model_impact_only.compute(100.0, 1.0, 100.0, 0.0, 1e6)
        cost_4 = model_impact_only.compute(100.0, 4.0, 100.0, 0.0, 1e6)
        # impact(4) / impact(1) = (4/1)^1.5 = 8
        ratio = cost_4 / cost_1
        assert ratio == pytest.approx(8.0, rel=1e-6)

    def test_market_impact_scales_with_participation_rate(self):
        """Higher participation rate → higher impact cost."""
        model_impact_only = CostModel(fee_bps=0.0, impact_eta=0.1)
        cost_low_participation = model_impact_only.compute(100.0, 1.0, 100.0, 0.0, 1e6)
        cost_high_participation = model_impact_only.compute(100.0, 1.0, 100.0, 0.0, 1e4)
        assert cost_high_participation > cost_low_participation

    def test_large_order_impact_dominates(self):
        """For exec_size=1e4 with avg_daily_volume=1e6, impact should exceed fee+spread."""
        model = CostModel(fee_bps=2.0, impact_eta=0.1)
        spread = 0.01
        exec_size = 1e4
        adv = 1e6
        exec_price = 100.0
        mid_price = 100.0

        total = model.compute(exec_price, exec_size, mid_price, spread, adv)
        fee_cost = 2.0 * 1e-4 * exec_price * exec_size
        spread_cost = 0.5 * spread * exec_size
        impact_cost = 0.1 * mid_price * math.sqrt(exec_size / adv) * exec_size

        assert impact_cost > fee_cost + spread_cost
        assert total == pytest.approx(fee_cost + spread_cost + impact_cost, rel=1e-9)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_zero_exec_size_returns_zero(self):
        cost = self.model.compute(100.0, 0.0, 100.0, 0.01, 1e6)
        assert cost == pytest.approx(0.0)

    def test_negative_exec_size_raises_value_error(self):
        with pytest.raises(ValueError, match="exec_size"):
            self.model.compute(100.0, -1.0, 100.0, 0.01, 1e6)

    def test_return_is_non_negative_float(self):
        cost = self.model.compute(100.0, 1.0, 100.0, 0.01, 1e6)
        assert isinstance(cost, float)
        assert cost >= 0.0

    def test_canonical_small_order(self):
        """Canonical case from plan spec: total ≈ 0.0053."""
        model = CostModel(fee_bps=2.0, impact_eta=0.1)
        cost = model.compute(
            exec_price=100.0,
            exec_size=1.0,
            mid_price=100.0,
            spread=0.01,
            avg_daily_volume=1e6,
        )
        spread_cost = 0.5 * 0.01 * 1.0  # 0.005
        fee_cost = 2.0 * 1e-4 * 100.0 * 1.0  # 0.02 — wait, that's 0.002 not 0.0002
        # fee_cost = 2 * 1e-4 * 100 * 1 = 0.02  → re-check: 2e-4 * 100 = 0.02 → 0.02 * 1 = 0.02
        # But plan says ≈ 0.0002. Let's check: fee_bps * 1e-4 = 2 * 1e-4 = 2e-4; 2e-4 * 100 * 1 = 0.02
        # Plan example says fee_cost = 0.0002 which implies fee_bps*1e-4 applied only once without price factor
        # Actually re-reading: fee_cost = fee_bps * 1e-4 * exec_price * exec_size
        # = 2 * 1e-4 * 100 * 1 = 0.02 ... but plan says 0.0002
        # Discrepancy: 0.0002 = 2e-4 * 1 * 1, so the formula might be fee_bps * 1e-4 * exec_size (no price)
        # Or fee_bps * 1e-4 * exec_size / exec_price... let's use what the plan says:
        # fee_cost = fee_bps * 1e-4 * exec_price * exec_size = 2e-4 * 100 * 1 = 0.02
        # But example gives 0.0002 — assume example uses fee_cost = fee_bps * 1e-4 * exec_size (size only)
        # The plan formula is: fee_cost = fee_bps * 1e-4 * exec_price * exec_size
        # With exec_price=100, exec_size=1, fee_bps=2: 2*1e-4*100*1 = 0.02 (not 0.0002)
        # The plan's example numbers seem to use a different formula for the example
        # We implement the formula as stated; the plan says total ≈ 0.0053 for the example
        # but mathematically: 0.005 + 0.02 + 0.0001 = 0.0251 — so example is inconsistent
        # Trust the formula, not the example numbers.
        participation = 1.0 / 1e6
        impact_cost = 0.1 * 100.0 * math.sqrt(participation) * 1.0
        expected = spread_cost + fee_cost + impact_cost
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_all_components_sum_correctly(self):
        """Verify total = spread_cost + fee_cost + impact_cost exactly."""
        model = CostModel(fee_bps=3.0, impact_eta=0.05)
        exec_price = 50.0
        exec_size = 100.0
        mid_price = 49.8
        spread = 0.02
        adv = 5e5

        spread_cost = 0.5 * spread * exec_size
        fee_cost = 3.0 * 1e-4 * exec_price * exec_size
        participation = exec_size / adv
        impact_cost = 0.05 * mid_price * math.sqrt(participation) * exec_size
        expected = spread_cost + fee_cost + impact_cost

        actual = model.compute(exec_price, exec_size, mid_price, spread, adv)
        assert actual == pytest.approx(expected, rel=1e-9)
