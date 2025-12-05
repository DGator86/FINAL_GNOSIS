"""Tests for strategy analysis functionality (PoP, breakevens, profit zones)."""

import pytest
import numpy as np
from gnosis.utils.greeks_calculator import GreeksCalculator


@pytest.fixture
def calc():
    """Create a GreeksCalculator instance."""
    return GreeksCalculator()


class TestOptionPayoff:
    """Test individual option payoff calculations."""

    def test_long_call_itm(self, calc):
        """Long call ITM should be profitable."""
        # Bought call at $5, strike 100, stock at 110
        pnl = calc.calculate_option_payoff("call", 100, 5.0, 1, 110)
        assert pnl == 5.0  # (110 - 100) - 5 = 5

    def test_long_call_otm(self, calc):
        """Long call OTM should lose premium."""
        # Bought call at $5, strike 100, stock at 95
        pnl = calc.calculate_option_payoff("call", 100, 5.0, 1, 95)
        assert pnl == -5.0  # max(0, 95-100) - 5 = -5

    def test_short_call_otm(self, calc):
        """Short call OTM should keep premium."""
        # Sold call at $5, strike 100, stock at 95
        pnl = calc.calculate_option_payoff("call", 100, 5.0, -1, 95)
        assert pnl == 5.0  # Premium received, no intrinsic

    def test_long_put_itm(self, calc):
        """Long put ITM should be profitable."""
        # Bought put at $3, strike 100, stock at 90
        pnl = calc.calculate_option_payoff("put", 100, 3.0, 1, 90)
        assert pnl == 7.0  # (100 - 90) - 3 = 7

    def test_short_put_itm(self, calc):
        """Short put ITM should lose money."""
        # Sold put at $3, strike 100, stock at 90
        pnl = calc.calculate_option_payoff("put", 100, 3.0, -1, 90)
        assert pnl == -7.0  # 3 - 10 = -7


class TestStrategyPayoff:
    """Test multi-leg strategy payoff calculations."""

    def test_bull_call_spread(self, calc):
        """Bull call spread payoff calculation."""
        # Buy 100 call @ $5, sell 110 call @ $2
        legs = [
            {"type": "call", "strike": 100, "premium": 5.0, "quantity": 1},
            {"type": "call", "strike": 110, "premium": 2.0, "quantity": -1},
        ]

        # Stock at 105: long call +5, short call expires worthless
        pnl = calc.calculate_strategy_payoff(legs, 105)
        assert pnl == 2.0  # (5 - 5) + (2 - 0) = 2

        # Max profit at 110+
        pnl = calc.calculate_strategy_payoff(legs, 115)
        assert pnl == 7.0  # (15 - 5) + (2 - 5) = 7

        # Max loss below 100
        pnl = calc.calculate_strategy_payoff(legs, 95)
        assert pnl == -3.0  # Net debit = 5 - 2 = 3

    def test_iron_condor(self, calc):
        """Iron condor payoff calculation."""
        # Sell 95 put @ $1, buy 90 put @ $0.50
        # Sell 105 call @ $1, buy 110 call @ $0.50
        legs = [
            {"type": "put", "strike": 95, "premium": 1.0, "quantity": -1},
            {"type": "put", "strike": 90, "premium": 0.5, "quantity": 1},
            {"type": "call", "strike": 105, "premium": 1.0, "quantity": -1},
            {"type": "call", "strike": 110, "premium": 0.5, "quantity": 1},
        ]

        # Max profit when stock between 95-105
        pnl = calc.calculate_strategy_payoff(legs, 100)
        assert pnl == 1.0  # Net credit = (1 + 1) - (0.5 + 0.5) = 1


class TestProbabilityOfProfit:
    """Test probability of profit calculations."""

    def test_pop_returns_valid_probability(self, calc):
        """PoP should return probability between 0 and 1."""
        legs = [{"type": "call", "strike": 100, "premium": 5.0, "quantity": 1}]

        result = calc.calculate_probability_of_profit(
            legs=legs,
            spot_price=100,
            volatility=0.25,
            days_to_expiration=30,
            simulations=1000,
        )

        assert 0 <= result["probability_of_profit"] <= 1
        assert "expected_profit" in result
        assert "expected_loss" in result
        assert "average_pnl" in result

    def test_deep_itm_call_high_pop(self, calc):
        """Deep ITM call should have high PoP."""
        # Buy 80 call when stock is at 100
        legs = [{"type": "call", "strike": 80, "premium": 21.0, "quantity": 1}]

        np.random.seed(42)  # For reproducibility
        result = calc.calculate_probability_of_profit(
            legs=legs,
            spot_price=100,
            volatility=0.20,
            days_to_expiration=30,
            simulations=5000,
        )

        # Should have moderate to high PoP since already $20 ITM
        assert result["probability_of_profit"] > 0.4


class TestBreakevens:
    """Test breakeven point calculations."""

    def test_long_call_breakeven(self, calc):
        """Long call breakeven should be strike + premium."""
        legs = [{"type": "call", "strike": 100, "premium": 5.0, "quantity": 1}]
        breakevens = calc.find_breakevens(legs, spot_price=100)

        assert len(breakevens) == 1
        assert abs(breakevens[0] - 105.0) < 0.1

    def test_bull_call_spread_breakeven(self, calc):
        """Bull call spread should have one breakeven."""
        legs = [
            {"type": "call", "strike": 100, "premium": 5.0, "quantity": 1},
            {"type": "call", "strike": 110, "premium": 2.0, "quantity": -1},
        ]
        breakevens = calc.find_breakevens(legs, spot_price=100)

        # Breakeven = 100 + (5 - 2) = 103
        assert len(breakevens) == 1
        assert abs(breakevens[0] - 103.0) < 0.1


class TestProfitZones:
    """Test profit zone analysis."""

    def test_analyze_long_call(self, calc):
        """Long call profit zone analysis."""
        legs = [{"type": "call", "strike": 100, "premium": 5.0, "quantity": 1}]

        result = calc.analyze_profit_zones(legs, spot_price=100)

        assert "profit_zones" in result
        assert "max_profit" in result
        assert "max_loss" in result
        assert "breakevens" in result

        # Max loss should be -$500 (premium paid)
        assert result["max_loss"] == -500.0

    def test_iron_condor_profit_zones(self, calc):
        """Iron condor should have profit zone in the middle."""
        legs = [
            {"type": "put", "strike": 95, "premium": 1.5, "quantity": -1},
            {"type": "put", "strike": 90, "premium": 0.5, "quantity": 1},
            {"type": "call", "strike": 105, "premium": 1.5, "quantity": -1},
            {"type": "call", "strike": 110, "premium": 0.5, "quantity": 1},
        ]

        result = calc.analyze_profit_zones(legs, spot_price=100)

        # Should have profit zone between the short strikes
        assert len(result["profit_zones"]) >= 1

        # Should have two breakevens (one on each side)
        assert len(result["breakevens"]) == 2
