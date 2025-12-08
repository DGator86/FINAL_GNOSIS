import datetime

import pytest

pytest.importorskip("loguru")

from loguru import logger

from trade.trade_agent_v3 import TradeAgentV3
from agents.composer.composer_agent_v2 import ComposerDecision


def _decision(symbol: str, direction: str = "LONG", confidence: float = 1.0):
    return ComposerDecision(
        timestamp=datetime.datetime.now(),
        symbol=symbol,
        go_signal=True,
        predicted_direction=direction,
        confidence=confidence,
        predicted_timeframe="1Hour",
        risk_reward_ratio=2.0,
        reasoning="test",
    )


def test_minimum_sizing_allows_single_share():
    agent = TradeAgentV3(
        {
            "base_position_size_pct": 0.02,
            "max_position_size_pct": 0.02,
            "min_shares_per_trade": 1,
            "min_dollars_per_trade": 0,
        }
    )

    qty = agent._calculate_equity_quantity(
        symbol="TEST",
        current_price=500,
        available_capital=30000,
        position_size_pct=0.02,
    )

    assert qty >= 1


def test_sizing_logs_reason_when_budget_too_small(capsys):
    agent = TradeAgentV3(
        {
            "base_position_size_pct": 0.02,
            "max_position_size_pct": 0.02,
            "min_shares_per_trade": 1,
            "min_dollars_per_trade": 0,
        }
    )

    sink_messages = []
    sink_id = logger.add(lambda msg: sink_messages.append(msg), level="WARNING")

    qty = agent._calculate_equity_quantity(
        symbol="EXPENSIVE",
        current_price=1200,
        available_capital=30000,
        position_size_pct=0.02,
    )

    logger.remove(sink_id)

    assert qty == 0
    assert any("Position sizing below minimum" in str(m) for m in sink_messages)


def test_generate_strategy_uses_minimum_shares():
    agent = TradeAgentV3(
        {
            "base_position_size_pct": 0.02,
            "max_position_size_pct": 0.02,
            "min_shares_per_trade": 1,
        }
    )

    strategy = agent.generate_strategy(
        composer_decision=_decision("ABC"),
        current_price=50.0,
        available_capital=30000.0,
        timestamp=datetime.datetime.now(),
    )

    assert strategy is not None
    assert strategy.quantity >= 1
