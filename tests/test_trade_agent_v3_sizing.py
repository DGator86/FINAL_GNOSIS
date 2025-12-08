import pytest

pytest.importorskip("loguru")

from trade.trade_agent_v3 import TradeAgentV3


def test_minimum_share_size_when_budget_small_but_capital_healthy():
    agent = TradeAgentV3({"max_position_size_pct": 0.02, "base_position_size_pct": 0.02})

    quantity, reason = agent._calculate_equity_quantity(
        symbol="MRK",
        current_price=100.0,
        available_capital=30_000.0,
        position_size_pct=0.002,  # 0.2% => $60 budget, below price but capital can support 1 share
    )

    assert quantity >= 1


def test_zero_quantity_when_even_one_share_unaffordable():
    agent = TradeAgentV3({"max_position_size_pct": 0.02, "base_position_size_pct": 0.02})

    quantity, reason = agent._calculate_equity_quantity(
        symbol="GOOGL",
        current_price=150.0,
        available_capital=50.0,
        position_size_pct=0.5,
    )

    assert quantity == 0
    assert "insufficient capital" in reason
