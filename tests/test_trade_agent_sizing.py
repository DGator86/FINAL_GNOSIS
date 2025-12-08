from datetime import datetime

import pytest

pytest.importorskip("loguru")

from datetime import datetime

from trade.trade_agent_v3 import TradeAgentV3


def test_equity_position_size_uses_dollars_budget():
    agent = TradeAgentV3(
        config={
            "max_position_size_pct": 0.02,
            "base_position_size_pct": 0.02,
            "min_shares_per_trade": 1,
        }
    )

    shares, reason = agent._calculate_equity_quantity(
        symbol="TEST",
        current_price=150.0,
        available_capital=30000.0,
        position_size_pct=0.02,
    )

    assert shares >= 4  # $600 budget / $150 price


def test_strategy_generation_does_not_zero_out_normal_trade():
    agent = TradeAgentV3(
        config={
            "max_position_size_pct": 0.02,
            "base_position_size_pct": 0.02,
            "min_shares_per_trade": 1,
        }
    )

    class DummyDecision:
        symbol = "JPM"
        go_signal = True
        predicted_direction = "LONG"
        confidence = 1.0
        predicted_timeframe = "1Hour"
        reasoning = "test"
        risk_reward_ratio = 2.0
        timestamp = datetime.now()

    strategy = agent.generate_strategy(
        composer_decision=DummyDecision(),
        current_price=150.0,
        available_capital=30000.0,
        timestamp=datetime.now(),
    )

    assert strategy is not None
    assert strategy.quantity >= 4


def test_sizing_requires_min_dollars_or_logs_reason():
    agent = TradeAgentV3(
        config={
            "max_position_size_pct": 0.02,
            "base_position_size_pct": 0.02,
            "min_shares_per_trade": 1,
            "min_dollars_per_trade": 1000,
        }
    )

    qty, reason = agent._calculate_equity_quantity(
        symbol="TALL",
        current_price=10.0,
        available_capital=1000.0,
        position_size_pct=0.01,
    )

    assert qty == 0
    assert "min_dollars_per_trade" in reason
