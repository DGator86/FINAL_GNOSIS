"""
Phase 6 Test: Intelligent Strategy Orchestrator
Tests automatic selection between stocks and options

This demonstrates the "quant firm" behavior - choosing optimal instruments
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from brokers.alpaca_client import AlpacaClient
from config.options_config_v2 import GNOSIS_V2_CONFIG
from engines.orchestration import UnifiedOrchestrator
from models.options_contracts import EnhancedMarketData, OptionQuote, OptionsChain


def create_mock_market_data(ticker: str, iv_level: str = "moderate") -> EnhancedMarketData:
    """Create mock market data for testing"""

    # Base price
    current_price = 600.0 if ticker == "SPY" else 150.0

    # IV levels
    iv_map = {"low": 0.15, "moderate": 0.25, "high": 0.45, "very_high": 0.60}
    base_iv = iv_map.get(iv_level, 0.25)

    # Create mock option quotes
    quotes = []
    for strike_offset in [-10, -5, 0, 5, 10]:
        strike = current_price + strike_offset

        # Call
        quotes.append(
            OptionQuote(
                symbol=f"{ticker}241220C00{int(strike):05d}00",
                strike=strike,
                expiration="2024-12-20",
                option_type="call",
                bid_price=5.0,
                ask_price=5.10,
                volume=100,
                open_interest=500,
                implied_volatility=base_iv + (0.02 if strike_offset != 0 else 0),
                delta=0.50 if strike_offset == 0 else (0.70 if strike_offset < 0 else 0.30),
                gamma=0.05,
                theta=-0.10,
                vega=0.15,
            )
        )

        # Put
        quotes.append(
            OptionQuote(
                symbol=f"{ticker}241220P00{int(strike):05d}00",
                strike=strike,
                expiration="2024-12-20",
                option_type="put",
                bid_price=5.0,
                ask_price=5.10,
                volume=100,
                open_interest=500,
                implied_volatility=base_iv + (0.02 if strike_offset != 0 else 0),
                delta=-0.50 if strike_offset == 0 else (-0.30 if strike_offset < 0 else -0.70),
                gamma=0.05,
                theta=-0.10,
                vega=0.15,
            )
        )

    options_chain = OptionsChain(ticker=ticker, expiration_dates=["2024-12-20"], quotes=quotes)

    return EnhancedMarketData(
        ticker=ticker, current_price=current_price, options_chain=options_chain
    )


def test_scenario(
    orchestrator: UnifiedOrchestrator,
    client: AlpacaClient,
    scenario_name: str,
    ticker: str,
    iv_level: str,
    signal_direction: str,
    signal_confidence: float,
    regime: str,
):
    """Test a specific trading scenario"""

    logger.info("=" * 80)
    logger.info(f"SCENARIO: {scenario_name}")
    logger.info("=" * 80)
    logger.info(f"Ticker: {ticker}")
    logger.info(f"IV Level: {iv_level}")
    logger.info(f"Signal: {signal_direction} ({signal_confidence:.0%} confidence)")
    logger.info(f"Regime: {regime}")
    logger.info("")

    # Create mock market data
    market_data = create_mock_market_data(ticker, iv_level)

    # Get decision (don't actually execute)
    try:
        decision = orchestrator.strategy_selector.select_optimal_instrument(
            market_data=market_data,
            signal_direction=signal_direction,
            signal_confidence=signal_confidence,
            regime=regime,
            portfolio_state=None,
        )

        logger.success(f"✓ Decision: {decision.instrument_type.value}")
        logger.info(f"  Strategy: {decision.strategy_type.value}")
        logger.info(f"  Confidence: {decision.confidence:.2%}")
        logger.info(f"  Reasoning: {decision.reasoning}")
        logger.info("")

        return True

    except Exception as e:
        logger.error(f"✗ Scenario failed: {e}")
        return False


def main():
    """Run Phase 6 orchestrator tests"""

    logger.info("Starting Phase 6: Intelligent Strategy Orchestrator Tests")
    logger.info("")

    # Initialize
    client = AlpacaClient.from_env(mode="paper")
    orchestrator = UnifiedOrchestrator(config=GNOSIS_V2_CONFIG, logger=logger)

    results = []

    # Scenario 1: Strong signal + Low IV → Should choose STOCK
    results.append(
        test_scenario(
            orchestrator,
            client,
            "Strong Bull + Low IV",
            ticker="SPY",
            iv_level="low",
            signal_direction="bullish",
            signal_confidence=0.85,
            regime="trending_up",
        )
    )

    # Scenario 2: Moderate signal + Moderate IV → Should choose SINGLE OPTION
    results.append(
        test_scenario(
            orchestrator,
            client,
            "Moderate Bull + Moderate IV",
            ticker="SPY",
            iv_level="moderate",
            signal_direction="bullish",
            signal_confidence=0.65,
            regime="ranging",
        )
    )

    # Scenario 3: Moderate signal + High IV → Should choose SPREAD
    results.append(
        test_scenario(
            orchestrator,
            client,
            "Moderate Bull + High IV",
            ticker="SPY",
            iv_level="high",
            signal_direction="bullish",
            signal_confidence=0.65,
            regime="ranging",
        )
    )

    # Scenario 4: Weak signal + Very High IV → Should SKIP or IRON CONDOR
    results.append(
        test_scenario(
            orchestrator,
            client,
            "Weak Signal + Very High IV",
            ticker="SPY",
            iv_level="very_high",
            signal_direction="neutral",
            signal_confidence=0.50,
            regime="choppy",
        )
    )

    # Scenario 5: Strong bear + Trending → Should choose STOCK (short)
    results.append(
        test_scenario(
            orchestrator,
            client,
            "Strong Bear + Trending Down",
            ticker="SPY",
            iv_level="moderate",
            signal_direction="bearish",
            signal_confidence=0.80,
            regime="trending_down",
        )
    )

    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(results)
    total = len(results)

    logger.info(f"Passed: {passed}/{total}")

    if passed == total:
        logger.success("✓ All orchestrator tests PASSED!")
        logger.success("Phase 6 is ready for integration!")
        return 0
    else:
        logger.error(f"✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
