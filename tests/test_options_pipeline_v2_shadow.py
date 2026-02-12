import logging
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from models.options_contracts import (
    EnhancedMarketData,
    OptionsChain,
    VolatilityMetrics,
    VolatilityStructure,
    MacroVolatilityData,
)

# Import the pipeline to test
from pipeline.options_pipeline_v2 import EnhancedGnosisPipeline


class TestEnhancedGnosisPipeline(unittest.TestCase):
    def setUp(self):
        # Mock base pipeline
        self.mock_base_pipeline = MagicMock()
        self.mock_base_pipeline.process_ticker.return_value = {"base_result": "success"}

        # Mock logger
        self.logger = logging.getLogger("TestLogger")

        # Create dummy market data using proper Pydantic models
        self.dummy_market_data = EnhancedMarketData(
            ticker="AAPL",
            spot_price=150.0,
            timestamp=datetime.now(),
            options_chain=OptionsChain(quotes=[]),  # Empty quotes list is valid
            volatility_metrics=VolatilityMetrics(
                atm_iv=0.25,
                iv_rank=50.0,
                iv_percentile=50.0,
                hv_20=0.20,
                hv_60=0.22,
            ),
            vol_structure=VolatilityStructure(
                front_month_iv=0.25,
                back_month_iv=0.27,
                put_skew_25d=0.05,
                call_skew_25d=-0.02,
            ),
            macro_vol_data=MacroVolatilityData(
                vix=20.0,
                vvix=100.0,
                move_index=80.0,
                credit_spreads=3.5,
                dxy_volatility=0.05,
            ),
        )

    @patch("pipeline.options_pipeline_v2.GNOSIS_V2_CONFIG")
    def test_shadow_mode_integration(self, mock_config):
        # Configure shadow mode
        mock_config.__getitem__.side_effect = lambda key: {
            "enabled": False,
            "shadow_mode": True,
            "adaptive_regimes": True,
            "slippage_modeling": True,
            "regime_config": {},
            "liquidity_config": {},
        }.get(key)

        # Mock internal V2 modules to avoid complex initialization
        with (
            patch("pipeline.options_pipeline_v2.VolatilityIntelligenceModule") as MockVolIntel,
            patch("pipeline.options_pipeline_v2.OptionsExecutionModule") as MockExecModule,
        ):
            # Setup mock instances
            mock_vol_instance = MockVolIntel.return_value
            mock_vol_instance.process_volatility_intelligence.return_value = {
                "regime_classification": "R3",
                "vol_edge": 0.05,
                "regime_confidence": 0.8,
            }

            mock_exec_instance = MockExecModule.return_value
            mock_exec_instance.assess_execution_environment.return_value = {
                "liquidity_tier": "tier_1",
                "execution_cost_bps": 10.0,
            }

            # Initialize pipeline
            pipeline = EnhancedGnosisPipeline(
                base_pipeline=self.mock_base_pipeline, logger=self.logger
            )

            # Mock _get_enhanced_market_data to return dummy data
            pipeline._get_enhanced_market_data = MagicMock(return_value=self.dummy_market_data)

            # Run process_ticker
            result = pipeline.process_ticker("AAPL")

            # Verify base pipeline was called
            self.mock_base_pipeline.process_ticker.assert_called_with("AAPL")

            # Verify result contains base result (since enabled=False)
            self.assertEqual(result["base_result"], "success")

            # Verify shadow mode logging (we can check if V2 logic was executed internally)
            pipeline._get_enhanced_market_data.assert_called_with("AAPL")
            mock_vol_instance.process_volatility_intelligence.assert_called()

            # If we were in enabled mode, we would expect "gnosis_v2" in result
            # But in shadow mode + enabled=False, it returns base_result.
            # However, the pipeline logic:
            # if GNOSIS_V2_CONFIG["shadow_mode"]:
            #     self._log_shadow_mode_decisions(enhanced_result)
            #     if not GNOSIS_V2_CONFIG["enabled"]:
            #         return base_result

            # So we expect base_result only.
            self.assertNotIn("gnosis_v2", result)

    @patch("pipeline.options_pipeline_v2.GNOSIS_V2_CONFIG")
    def test_v2_enabled_integration(self, mock_config):
        # Configure enabled mode
        mock_config.__getitem__.side_effect = lambda key: {
            "enabled": True,
            "shadow_mode": False,
            "adaptive_regimes": True,
            "slippage_modeling": True,
            "regime_config": {},
            "liquidity_config": {},
        }.get(key)

        with (
            patch("pipeline.options_pipeline_v2.VolatilityIntelligenceModule") as MockVolIntel,
            patch("pipeline.options_pipeline_v2.OptionsExecutionModule") as MockExecModule,
        ):
            mock_vol_instance = MockVolIntel.return_value
            mock_vol_instance.process_volatility_intelligence.return_value = {
                "regime_classification": "R3",
                "vol_edge": 0.05,
                "regime_confidence": 0.8,
            }

            mock_exec_instance = MockExecModule.return_value
            mock_exec_instance.assess_execution_environment.return_value = {
                "liquidity_tier": "tier_1",
                "execution_cost_bps": 10.0,
            }

            pipeline = EnhancedGnosisPipeline(
                base_pipeline=self.mock_base_pipeline, logger=self.logger
            )
            pipeline._get_enhanced_market_data = MagicMock(return_value=self.dummy_market_data)

            result = pipeline.process_ticker("AAPL")

            # Verify result contains V2 data
            self.assertIn("gnosis_v2", result)
            self.assertEqual(result["gnosis_v2"].regime_classification, "R3")
            self.assertTrue(result["v2_enabled"])


if __name__ == "__main__":
    unittest.main()
