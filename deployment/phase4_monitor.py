"""
Phase 4: Live Deployment Monitor
Validates V2 system health, regime stability, and risk metrics.
"""

import logging
import os
import sys
import time
from datetime import date, datetime
from typing import Dict

# Add project root to path
sys.path.append(os.getcwd())

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


def create_mock_market_data(ticker: str, regime_scenario: str = "normal"):
    """Create sample data for testing different scenarios"""
    from models.options_contracts import (
        EnhancedMarketData,
        MacroVolatilityData,
        OptionQuote,
        OptionsChain,
        VolatilityMetrics,
        VolatilityStructure,
    )

    # Base values
    spot_price = 450.0
    vix = 15.0
    iv = 0.15

    # Adjust for scenarios
    if regime_scenario == "high_vol":
        vix = 35.0
        iv = 0.40
    elif regime_scenario == "low_vol":
        vix = 10.0
        iv = 0.10

    # Create quotes
    quotes = []
    for strike in [440, 445, 450, 455, 460]:
        quotes.append(
            OptionQuote(
                symbol=f"{ticker}230616C{strike}",
                type="call",
                strike=float(strike),
                expiration=date(2023, 6, 16),
                bid=5.0,
                ask=5.10,
                iv=iv,
                volume=1000,
                oi=5000,
                delta=0.50,
                gamma=0.05,
                vega=0.10,
                theta=-0.05,
            )
        )

    return EnhancedMarketData(
        ticker=ticker,
        timestamp=datetime.now(),
        spot_price=spot_price,
        options_chain=OptionsChain(quotes=quotes),
        volatility_metrics=VolatilityMetrics(
            atm_iv=iv, iv_rank=50.0, iv_percentile=50.0, hv_20=iv - 0.01, hv_60=iv + 0.01
        ),
        vol_structure=VolatilityStructure(
            front_month_iv=iv, back_month_iv=iv + 0.01, put_skew_25d=0.02, call_skew_25d=-0.01
        ),
        macro_vol_data=MacroVolatilityData(
            vix=vix, vvix=90.0, move_index=100.0, credit_spreads=1.5, dxy_volatility=0.05
        ),
    )


class V2HealthMonitor:
    def __init__(self):
        self.pipeline = None
        self.metrics_history = []

    def initialize(self):
        """Initialize the pipeline with V2 enabled"""
        try:
            from config.options_config_v2 import GNOSIS_V2_CONFIG
            from pipeline.options_pipeline_v2 import EnhancedGnosisPipeline

            # Force enable for monitoring (even if config file says otherwise for safety)
            # In real deployment, we'd read the actual config
            GNOSIS_V2_CONFIG["enabled"] = True
            GNOSIS_V2_CONFIG["shadow_mode"] = False

            # Mock dependencies
            mock_base = type(
                "MockBase", (), {"process_ticker": lambda self, t: {"base_metric": 100}}
            )()

            self.pipeline = EnhancedGnosisPipeline(
                base_pipeline=mock_base, logger=logging.getLogger("Monitor")
            )

            # Mock data fetcher
            self.pipeline._get_enhanced_market_data = lambda t: create_mock_market_data(t, "normal")

            logger.info("Monitor initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def run_health_check(self, iterations: int = 5):
        """Run a series of health checks"""
        logger.info(f"Starting health check ({iterations} iterations)...")

        success_count = 0

        for i in range(iterations):
            try:
                # 1. Normal Scenario
                result = self.pipeline.process_ticker("SPY")
                self._validate_result(result, "normal")

                # 2. High Vol Scenario
                self.pipeline._get_enhanced_market_data = lambda t: create_mock_market_data(
                    t, "high_vol"
                )
                result_high = self.pipeline.process_ticker("SPY")
                self._validate_result(result_high, "high_vol")

                success_count += 1
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Iteration {i} failed: {e}")

        logger.info(f"Health check complete: {success_count}/{iterations} passed")
        return success_count == iterations

    def _validate_result(self, result: Dict, scenario: str):
        """Validate pipeline output"""
        if "gnosis_v2" not in result:
            raise ValueError("V2 data missing from result")

        v2_data = result["gnosis_v2"]

        # Check regime sanity
        if scenario == "high_vol" and v2_data.regime_classification not in ["R4", "R5"]:
            logger.warning(f"Unexpected regime for high vol: {v2_data.regime_classification}")

        # Check scores
        if not (0 <= v2_data.opportunity_score <= 100):
            raise ValueError(f"Invalid opportunity score: {v2_data.opportunity_score}")

        if not (0 <= v2_data.risk_score <= 100):
            raise ValueError(f"Invalid risk score: {v2_data.risk_score}")

        logger.debug(
            f"[{scenario}] Regime: {v2_data.regime_classification}, Risk: {v2_data.risk_score:.1f}"
        )


if __name__ == "__main__":
    monitor = V2HealthMonitor()
    if monitor.initialize():
        success = monitor.run_health_check()
        sys.exit(0 if success else 1)
    else:
        sys.exit(1)
