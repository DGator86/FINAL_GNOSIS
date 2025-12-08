import pytest

from execution import risk_utils


def test_option_notional_respects_multiplier():
    value = risk_utils.calculate_order_value("AAPL240119C00150000", 2, 1.5)
    assert value == 300.0


def test_position_limit_enforced():
    with pytest.raises(ValueError):
        risk_utils.assert_within_max("AAPL240119C00150000", order_value=1000, portfolio_value=10000, max_position_pct=0.02)
