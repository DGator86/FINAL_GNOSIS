import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from loguru import logger
import traceback

from trade.trade_agent_v3 import TradeAgentV3, TradeStrategy
from agents.composer.composer_agent_v2 import ComposerDecision
from engines.inputs.options_chain_adapter import OptionContract
from gnosis.utils.option_utils import OptionUtils


async def test_options_generation():
    logger.info("Testing Options Strategy Generation...")

    # Mock Options Adapter
    mock_adapter = MagicMock()

    # Create mock chain
    now = datetime.now()
    expiration = now + timedelta(days=35)

    mock_chain = [
        OptionContract(
            symbol="AAPL230616C00150000",
            strike=150.0,
            expiration=expiration,
            option_type="call",
            bid=5.0,
            ask=5.10,
            last=5.05,
            volume=1000,
            open_interest=5000,
            delta=0.40,  # Target delta
            gamma=0.05,
            theta=-0.05,
            vega=0.10,
            rho=0.01,
        ),
        OptionContract(
            symbol="AAPL230616C00160000",
            strike=160.0,
            expiration=expiration,
            option_type="call",
            bid=2.0,
            ask=2.10,
            last=2.05,
            volume=500,
            open_interest=2000,
            delta=0.20,
            gamma=0.04,
            theta=-0.04,
            vega=0.08,
            rho=0.01,
        ),
    ]

    mock_adapter.get_chain.return_value = mock_chain

    # Initialize Agent
    config = {
        "prefer_options": True,
        "min_dte": 30,
        "max_dte": 45,
        "target_delta": 0.40,
        "base_position_size_pct": 0.05,
        "max_position_size_pct": 0.10,
    }

    agent = TradeAgentV3(config=config, options_adapter=mock_adapter)

    # Create Decision
    decision = ComposerDecision(
        symbol="AAPL",
        timestamp=now,
        go_signal=True,
        confidence=0.8,
        predicted_direction="LONG",
        predicted_timeframe="1Hour",
        risk_reward_ratio=2.0,
        reasoning="Test signal",
    )

    # Generate Strategy
    strategy = agent.generate_strategy(
        composer_decision=decision, current_price=145.0, available_capital=100000.0, timestamp=now
    )

    if strategy:
        logger.info(f"Strategy Generated: {strategy.asset_class}")
        logger.info(f"Symbol: {strategy.symbol}")
        logger.info(f"Option Symbol: {strategy.option_symbol}")
        logger.info(f"Strike: {strategy.strike}")
        logger.info(f"Expiration: {strategy.expiration}")
        logger.info(f"Quantity: {strategy.quantity}")

        assert strategy.asset_class == "option"
        assert strategy.option_symbol == "AAPL230616C00150000"
        assert strategy.quantity > 0

        logger.success("Options Strategy Generation Test Passed!")
    else:
        logger.error("Failed to generate strategy")

    # Test OCC Utils
    logger.info("Testing OptionUtils...")
    generated_symbol = OptionUtils.generate_occ_symbol("AAPL", datetime(2023, 6, 16), "call", 150.0)
    logger.info(f"Generated: {generated_symbol}")
    assert generated_symbol == "AAPL230616C00150000"

    parsed = OptionUtils.parse_occ_symbol("AAPL230616C00150000")
    logger.info(f"Parsed: {parsed}")
    assert parsed["symbol"] == "AAPL"
    assert parsed["strike"] == 150.0
    assert parsed["option_type"] == "call"

    logger.success("OptionUtils Test Passed!")


if __name__ == "__main__":
    try:
        asyncio.run(test_options_generation())
    except Exception as e:
        with open("error_log.txt", "w") as f:
            f.write(traceback.format_exc())
        print("Error occurred, check error_log.txt")
        # Re-raise to ensure non-zero exit code
        raise e
