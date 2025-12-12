"""
Enhanced pipeline with V2 options intelligence.
Safely integrates with existing pipeline via feature flags.
"""

import logging
from typing import Any, Dict, Optional

from config.options_config_v2 import GNOSIS_V2_CONFIG
from models.options_contracts import EnhancedMarketData, OptionsIntelligenceOutput

# V2 Engine imports (only if enabled)
if GNOSIS_V2_CONFIG["enabled"] or GNOSIS_V2_CONFIG["shadow_mode"]:
    from engines.hedge.volatility_intel_v2 import VolatilityIntelligenceModule
    from engines.liquidity.options_execution_v2 import OptionsExecutionModule


class EnhancedGnosisPipeline:
    """
    Drop-in enhancement to existing GNOSIS pipeline.
    Preserves all existing functionality while adding V2 capabilities.

    Architecture Role:
    1. Orchestrates V2 Engines (Volatility, Liquidity)
    2. Standardizes outputs into OptionsIntelligenceOutput
    3. Feeds this intelligence to the respective Agents (Hedge, Liquidity)
    """

    def __init__(
        self,
        # Existing dependencies
        base_pipeline,
        # New V2 dependencies (injected)
        garch_model=None,
        correlation_engine=None,
        vix_history_provider=None,
        logger: Optional[logging.Logger] = None,
    ):
        self.base_pipeline = base_pipeline
        self.logger = logger or logging.getLogger(__name__)

        # Initialize V2 modules only if enabled or in shadow mode
        if GNOSIS_V2_CONFIG["enabled"] or GNOSIS_V2_CONFIG["shadow_mode"]:
            self.logger.info("Initializing GNOSIS V2.0 Options Intelligence")
            self._initialize_v2_modules(garch_model, correlation_engine, vix_history_provider)
        else:
            self.logger.info("V2 Options disabled - using base pipeline")
            self.vol_intel_v2 = None
            self.options_execution_v2 = None

    def _initialize_v2_modules(
        self, garch_model: Any, correlation_engine: Any, vix_history_provider: Any
    ) -> None:
        """Initialize V2 modules with proper dependency injection"""

        try:
            # Volatility Intelligence Module
            if GNOSIS_V2_CONFIG["adaptive_regimes"]:
                self.vol_intel_v2 = VolatilityIntelligenceModule(
                    garch_model=garch_model,
                    correlation_engine=correlation_engine,
                    config=GNOSIS_V2_CONFIG["regime_config"],  # type: ignore
                    vix_history_provider=vix_history_provider,
                    logger=self.logger,
                )
            else:
                self.vol_intel_v2 = None

            # Options Execution Module
            if GNOSIS_V2_CONFIG["slippage_modeling"]:
                self.options_execution_v2 = OptionsExecutionModule(
                    config=GNOSIS_V2_CONFIG["liquidity_config"],  # type: ignore
                    logger=self.logger,
                )
            else:
                self.options_execution_v2 = None

        except Exception as e:
            self.logger.error(f"V2 module initialization failed: {e}")
            # Graceful fallback to V1
            self.vol_intel_v2 = None
            self.options_execution_v2 = None

    def process_ticker(
        self, ticker: str, current_positions: Optional[list[Any]] = None
    ) -> Dict[str, Any]:
        """
        Main processing function with V2 enhancement.
        Falls back gracefully to V1 if V2 fails.
        """

        try:
            # Always run base pipeline first (V1 Logic)
            base_result = self.base_pipeline.process_ticker(ticker)

            # Enhance with V2 if enabled/shadow and modules are ready
            if (GNOSIS_V2_CONFIG["enabled"] or GNOSIS_V2_CONFIG["shadow_mode"]) and (
                self.vol_intel_v2 is not None
            ):
                v2_enhancement = self._process_v2_enhancement(ticker, current_positions)

                # Merge results
                enhanced_result = {**base_result, **v2_enhancement}

                # Log shadow mode decisions
                if GNOSIS_V2_CONFIG["shadow_mode"]:
                    self._log_shadow_mode_decisions(enhanced_result)
                    # Return base result in shadow mode to protect live trading
                    if not GNOSIS_V2_CONFIG["enabled"]:
                        return base_result

                return enhanced_result

            else:
                # V2 disabled or unavailable - return base result
                return base_result

        except Exception as e:
            self.logger.error(f"Pipeline processing failed for {ticker}: {e}")
            # Always fall back to base pipeline
            return self.base_pipeline.process_ticker(ticker)

    def _process_v2_enhancement(
        self, ticker: str, current_positions: Optional[list[Any]]
    ) -> Dict[str, Any]:
        """
        Process V2 enhancements with error handling.
        """

        # Get enhanced market data
        market_data = self._get_enhanced_market_data(ticker)

        if not market_data:
            return {}

        v2_results: dict[str, Any] = {}

        # Volatility Intelligence (Feeds Hedge Agent)
        if self.vol_intel_v2:
            vol_intel_output = self.vol_intel_v2.process_volatility_intelligence(
                market_data, current_positions
            )
            v2_results["volatility_intelligence"] = vol_intel_output

        # Options Execution Analysis (Feeds Liquidity Agent)
        if self.options_execution_v2:
            execution_output = self.options_execution_v2.assess_execution_environment(market_data)
            v2_results["execution_analysis"] = execution_output

        # Standardize and combine
        standardized_context = self._standardize_v2_outputs(v2_results)

        return {
            "gnosis_v2": standardized_context,
            "v2_enabled": GNOSIS_V2_CONFIG["enabled"],
            "shadow_mode": GNOSIS_V2_CONFIG["shadow_mode"],
        }

    def _get_enhanced_market_data(self, ticker: str) -> Optional[EnhancedMarketData]:
        """
        Retrieve and validate enhanced market data.
        Returns None if data is insufficient.
        """

        try:
            # This would integrate with your existing data providers
            # For now, we return None or mock data if needed, as the data fetcher isn't fully V2 yet
            # In a real implementation, this calls the Data Adapter
            return None

        except Exception as e:
            self.logger.error(f"Enhanced market data retrieval failed for {ticker}: {e}")
            return None

    def _standardize_v2_outputs(self, v2_results: dict[str, Any]) -> OptionsIntelligenceOutput:
        """
        Convert V2 engine outputs to standardized format.
        """

        vol_intel = v2_results.get("volatility_intelligence", {})
        execution = v2_results.get("execution_analysis", {})

        # Calculate unified opportunity and risk scores
        opportunity_score = self._calculate_opportunity_score(vol_intel)
        risk_score = self._calculate_risk_score(vol_intel, execution)

        return OptionsIntelligenceOutput(
            # Volatility Intelligence
            regime_classification=vol_intel.get("regime_classification", "R2"),
            regime_confidence=vol_intel.get("regime_confidence", 0.5),
            vol_edge=vol_intel.get("vol_edge", 0.0),
            macro_stress_score=vol_intel.get("macro_stress_score", 0.0),
            # Risk Context
            portfolio_vega_effective=vol_intel.get("portfolio_vega_effective", 0.0),
            portfolio_gamma_effective=vol_intel.get("portfolio_gamma_effective", 0.0),
            portfolio_delta_effective=vol_intel.get("portfolio_delta_effective", 0.0),
            vega_utilization=vol_intel.get("vega_utilization", 0.0),
            # Execution Context
            liquidity_tier=execution.get("liquidity_tier", "tier_3"),
            execution_cost_bps=execution.get("execution_cost_bps", 50.0),
            tradeable_strikes=execution.get("tradeable_strikes", []),
            execution_feasibility=execution.get("execution_feasibility", "fair"),
            # Market Intelligence (placeholder - would come from sentiment module)
            edge_confidence=0.5,
            flow_bias="neutral",
            dealer_positioning="neutral",
            # Unified Scores
            opportunity_score=opportunity_score,
            risk_score=risk_score,
        )

    def _calculate_opportunity_score(self, vol_intel: dict[str, Any]) -> float:
        """Calculate unified opportunity score 0-100"""
        vol_edge = vol_intel.get("vol_edge", 0.0)
        regime_conf = vol_intel.get("regime_confidence", 0.5)

        # Simple scoring: higher vol edge + higher confidence = higher opportunity
        edge_score = min(100, abs(vol_edge) * 200)  # ±50% vol edge = 100 points
        confidence_bonus = regime_conf * 20  # Up to 20 bonus points

        return min(100, edge_score + confidence_bonus)

    def _calculate_risk_score(self, vol_intel: dict[str, Any], execution: dict[str, Any]) -> float:
        """Calculate unified risk score 0-100"""
        vega_util = vol_intel.get("vega_utilization", 0.0)
        regime = vol_intel.get("regime_classification", "R2")
        exec_cost = execution.get("execution_cost_bps", 50.0)

        # Risk components
        portfolio_risk = vega_util * 50  # 50 points max for portfolio risk
        regime_risk = {"R1": 10, "R2": 20, "R3": 30, "R4": 40, "R5": 50}.get(regime, 20)
        execution_risk = min(30, exec_cost / 2)  # Execution cost component

        return min(100, portfolio_risk + regime_risk + execution_risk)

    def _log_shadow_mode_decisions(self, enhanced_result: dict[str, Any]) -> None:
        """Log V2 decisions for validation during shadow mode"""

        v2_data = enhanced_result.get("gnosis_v2")
        if not v2_data:
            return

        self.logger.info(
            f"SHADOW MODE - V2 Decision: "
            f"Regime={v2_data.regime_classification}, "
            f"Vol Edge={v2_data.vol_edge:.3f}, "
            f"Opportunity={v2_data.opportunity_score:.1f}, "
            f"Risk={v2_data.risk_score:.1f}"
        )

    def _fetch_options_data(self, ticker: str) -> Optional[dict[str, Any]]:
        """
        Placeholder for options data fetching.
        Implementation depends on your data providers.
        """
        # This would integrate with your existing data infrastructure
