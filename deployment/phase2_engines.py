"""
Phase 2: Implement and test engine modules in isolation
"""

import logging
import os
import sys
from datetime import date, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.options_contracts import EnhancedMarketData

# Add project root to path
sys.path.append(os.getcwd())

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_market_data() -> "EnhancedMarketData":
    """Create sample data for testing"""
    from models.options_contracts import (
        EnhancedMarketData,
        MacroVolatilityData,
        OptionQuote,
        OptionsChain,
        VolatilityMetrics,
        VolatilityStructure,
    )

    # Create quotes
    quotes = []
    for strike in [440, 445, 450, 455, 460]:
        quotes.append(
            OptionQuote(
                symbol=f"SPY230616C{strike}",
                type="call",
                strike=float(strike),
                expiration=date(2023, 6, 16),
                bid=5.0,
                ask=5.10,
                iv=0.15,
                volume=1000,
                oi=5000,
                delta=0.50,
                gamma=0.05,
                vega=0.10,
                theta=-0.05,
            )
        )

    return EnhancedMarketData(
        ticker="SPY",
        timestamp=datetime.now(),
        spot_price=450.0,
        options_chain=OptionsChain(quotes=quotes),
        volatility_metrics=VolatilityMetrics(
            atm_iv=0.15, iv_rank=50.0, iv_percentile=50.0, hv_20=0.14, hv_60=0.16
        ),
        vol_structure=VolatilityStructure(
            front_month_iv=0.15, back_month_iv=0.16, put_skew_25d=0.02, call_skew_25d=-0.01
        ),
        macro_vol_data=MacroVolatilityData(
            vix=15.0, vvix=90.0, move_index=100.0, credit_spreads=1.5, dxy_volatility=0.05
        ),
    )


def test_volatility_intelligence() -> bool:
    """Test Volatility Intelligence Module logic"""
    print("\nTesting Volatility Intelligence Module...")

    try:
        from config.options_config_v2 import GNOSIS_V2_CONFIG
        from engines.hedge.volatility_intel_v2 import VolatilityIntelligenceModule

        # Mock dependencies
        def mock_vix_history() -> list[float]:
            return [10.0] * 50 + [20.0] * 50  # Simple history

        module = VolatilityIntelligenceModule(
            garch_model=None,
            correlation_engine=None,
            config=GNOSIS_V2_CONFIG["regime_config"],  # type: ignore
            vix_history_provider=mock_vix_history,
            logger=logger,
        )

        # Test Regime Classification
        # Case 1: Low VIX (R1/R2)
        regime, conf = module._classify_regime(
            vix=12.0, vix_percentile=15.0, vvix=80.0, term_slope=0.05
        )
        print(f"Regime Test 1 (Low VIX): {regime} (Expected R1)")
        assert regime == "R1"

        # Case 2: Crisis (R5)
        regime, conf = module._classify_regime(
            vix=40.0, vix_percentile=99.0, vvix=150.0, term_slope=-0.10
        )
        print(f"Regime Test 2 (Crisis): {regime} (Expected R5)")
        assert regime == "R5"

        # Test Full Processing
        market_data = create_mock_market_data()
        result = module.process_volatility_intelligence(market_data)

        print(
            f"Full Processing Result: {result['regime_classification']}, "
            f"Vol Edge: {result['vol_edge']:.4f}"
        )
        assert "regime_classification" in result
        assert "vol_edge" in result

        print("✓ Volatility Intelligence Tests Passed")
        return True

    except Exception as e:
        print(f"✗ Volatility Intelligence Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_options_execution() -> bool:
    """Test Options Execution Module logic"""
    print("\nTesting Options Execution Module...")

    try:
        from config.options_config_v2 import GNOSIS_V2_CONFIG
        from engines.liquidity.options_execution_v2 import OptionsExecutionModule

        module = OptionsExecutionModule(
            config=GNOSIS_V2_CONFIG["liquidity_config"],  # type: ignore
            logger=logger,
        )

        market_data = create_mock_market_data()

        # Test Assessment
        result = module.assess_execution_environment(market_data)

        print(f"Liquidity Tier: {result['liquidity_tier']}")
        print(f"Execution Cost: {result['execution_cost_bps']:.2f} bps")
        print(f"Feasibility: {result['execution_feasibility']}")

        # Our mock data is very liquid (1000 vol, 5000 OI, tight spread)
        # Should be Tier 1 or 2
        assert result["liquidity_tier"] in ["tier_1", "tier_2"]
        assert result["execution_cost_bps"] < 50.0

        print("✓ Options Execution Tests Passed")
        return True

    except Exception as e:
        print(f"✗ Options Execution Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    v_success = test_volatility_intelligence()
    e_success = test_options_execution()

    if v_success and e_success:
        print("\nAll Phase 2 Tests PASSED")
        sys.exit(0)
    else:
        print("\nPhase 2 Tests FAILED")
        sys.exit(1)
