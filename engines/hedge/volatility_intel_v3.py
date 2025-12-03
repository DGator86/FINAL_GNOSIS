"""Volatility Intelligence Module V3 - Enhanced with GEX calculations."""

import numpy as np
from typing import Dict, List, Optional, Protocol, Tuple
import logging
from datetime import datetime
from models.options_contracts import EnhancedMarketData, OptionsIntelligenceOutput


class GARCHModel(Protocol):
    """Protocol for GARCH model dependency"""

    def forecast_volatility(self, data: EnhancedMarketData) -> float: ...


class CorrelationEngine(Protocol):
    """Protocol for correlation calculations"""

    def calculate_effective_greeks(
        self, positions: List, reference_asset: str
    ) -> Dict[str, float]: ...


class VolatilityIntelligenceModuleV3:
    """
    Enhanced volatility analysis with GEX (Gamma Exposure) calculations.

    New V3 Features:
    - Market-wide GEX calculation
    - Charm flow detection for expiration impacts
    - Enhanced dealer positioning analysis
    """

    def __init__(
        self,
        garch_model: Optional[GARCHModel],
        correlation_engine: Optional[CorrelationEngine],
        config: Dict,
        vix_history_provider: callable,  # Returns List[float] of VIX history
        logger: Optional[logging.Logger] = None,
    ):
        self.garch_model = garch_model
        self.correlation_engine = correlation_engine
        self.config = config
        self.vix_history_provider = vix_history_provider
        self.logger = logger or logging.getLogger(__name__)

    def process_volatility_intelligence(
        self, market_data: EnhancedMarketData, current_positions: List = None
    ) -> Dict[str, float]:
        """
        Main processing function with V3 enhancements.
        Returns dict compatible with OptionsIntelligenceOutput.
        """

        try:
            # 1. Adaptive Regime Classification
            vix_percentile = self._calculate_vix_percentile(market_data.macro_vol_data.vix)
            regime, confidence = self._classify_regime(
                vix=market_data.macro_vol_data.vix,
                vix_percentile=vix_percentile,
                vvix=market_data.macro_vol_data.vvix,
                term_slope=market_data.vol_structure.term_slope,
            )

            # 2. Enhanced Vol Edge
            rv_forecast = self._bayesian_rv_forecast(market_data=market_data, regime=regime)
            vol_edge = self._calculate_vol_edge(
                current_iv=market_data.volatility_metrics.atm_iv, rv_forecast=rv_forecast
            )

            # 3. Macro Stress Score
            macro_stress = self._calculate_macro_stress_score(market_data.macro_vol_data)

            # 4. Correlation-Adjusted Greeks
            effective_greeks = self._calculate_portfolio_greeks(
                current_positions or [], market_data.ticker
            )

            # V3: Calculate market GEX
            market_gex = self._calculate_market_gex(market_data)

            # V3: Detect charm flows
            charm_flow_impact = self._detect_charm_flows(market_data)

            return {
                "regime_classification": regime,
                "regime_confidence": confidence,
                "vol_edge": vol_edge,
                "macro_stress_score": macro_stress,
                "vix_percentile": vix_percentile,
                "market_gex": market_gex,  # V3
                "charm_flow_impact": charm_flow_impact,  # V3
                **effective_greeks,
            }

        except Exception as e:
            self.logger.error(f"Volatility intelligence processing failed: {e}")
            return self._get_fallback_output()

    def _calculate_vix_percentile(self, current_vix: float) -> float:
        """Calculate VIX percentile over rolling window."""
        try:
            vix_history = self.vix_history_provider()
            if not vix_history or len(vix_history) < 100:
                self.logger.warning("Insufficient VIX history, using default percentile")
                return 50.0

            vix_array = np.array(vix_history)
            percentile = (np.sum(vix_array < current_vix) / len(vix_array)) * 100
            return float(percentile)

        except Exception as e:
            self.logger.error(f"VIX percentile calculation failed: {e}")
            return 50.0

    def _classify_regime(
        self, vix: float, vix_percentile: float, vvix: float, term_slope: float
    ) -> Tuple[str, float]:
        """Adaptive regime classification using percentile boundaries."""

        # Crisis mode detection first (backwardation)
        if term_slope < -0.05:  # Inverted term structure
            return "R5", 0.9

        # VVIX/VIX ratio for vol-of-vol assessment
        vvix_vix_ratio = vvix / vix if vix > 0 else 1.0

        # Percentile-based classification with confidence scoring
        if vix_percentile < 20:
            confidence = 1.0 - (vix_percentile / 20) * 0.3
            return "R1", confidence
        elif vix_percentile < 50:
            distance_from_mid = abs(vix_percentile - 35) / 15
            confidence = 1.0 - distance_from_mid * 0.2
            return "R2", confidence
        elif vix_percentile < 80:
            if vvix_vix_ratio < 1.1:
                confidence = 0.8 if vvix_vix_ratio > 1.05 else 0.9
                return "R3", confidence
            else:
                return "R4", 0.7
        else:
            if vvix_vix_ratio > 1.3:
                return "R5", 0.8
            else:
                return "R4", 0.6

    def _bayesian_rv_forecast(self, market_data: EnhancedMarketData, regime: str) -> float:
        """Enhanced RV forecasting with regime-dependent weights."""

        weights = self.config.get("bayesian_weights", {}).get(
            regime, {"garch": 0.5, "hv20": 0.4, "hv60": 0.1}
        )

        hv20 = market_data.volatility_metrics.hv_20
        hv60 = market_data.volatility_metrics.hv_60

        garch_forecast = hv20  # Default fallback
        if self.garch_model:
            try:
                garch_forecast = self.garch_model.forecast_volatility(market_data)
            except Exception as e:
                self.logger.warning(f"GARCH forecast failed: {e}, using HV20")

        rv_forecast = (
            weights["garch"] * garch_forecast + weights["hv20"] * hv20 + weights["hv60"] * hv60
        )

        return max(0.05, rv_forecast)

    def _calculate_vol_edge(self, current_iv: float, rv_forecast: float) -> float:
        """Vol edge calculation with proper error handling."""
        if rv_forecast <= 0:
            return 0.0

        return (current_iv - rv_forecast) / rv_forecast

    def _calculate_macro_stress_score(self, macro_data) -> float:
        """Macro stress calculation using pre-computed z-scores."""

        move_z = macro_data.move_z_score or 0.0
        credit_z = macro_data.credit_z_score or 0.0
        dxy_z = macro_data.dxy_z_score or 0.0

        stress_score = (
            0.35 * move_z
            + 0.30 * credit_z
            + 0.25 * dxy_z
            + 0.10 * 0  # Placeholder for commodity vol z-score
        )

        return stress_score

    def _calculate_portfolio_greeks(
        self, positions: List, reference_ticker: str
    ) -> Dict[str, float]:
        """Correlation-adjusted portfolio Greeks."""

        if not positions:
            return {
                "portfolio_vega_effective": 0.0,
                "portfolio_gamma_effective": 0.0,
                "portfolio_delta_effective": 0.0,
                "vega_utilization": 0.0,
            }

        try:
            if self.correlation_engine:
                effective_greeks = self.correlation_engine.calculate_effective_greeks(
                    positions, reference_ticker
                )
            else:
                effective_greeks = {
                    "vega_effective": 0.0,
                    "gamma_effective": 0.0,
                    "delta_effective": 0.0,
                }

            vega_limit = self.config.get("risk_limits", {}).get("max_vega", 500)
            vega_util = (
                abs(effective_greeks.get("vega_effective", 0)) / vega_limit
                if vega_limit > 0
                else 0.0
            )

            return {
                "portfolio_vega_effective": effective_greeks.get("vega_effective", 0),
                "portfolio_gamma_effective": effective_greeks.get("gamma_effective", 0),
                "portfolio_delta_effective": effective_greeks.get("delta_effective", 0),
                "vega_utilization": min(1.0, vega_util),
            }

        except Exception as e:
            self.logger.error(f"Greek calculation failed: {e}")
            return {
                "portfolio_vega_effective": 0.0,
                "portfolio_gamma_effective": 0.0,
                "portfolio_delta_effective": 0.0,
                "vega_utilization": 0.0,
            }

    def _calculate_market_gex(self, market_data: EnhancedMarketData) -> float:
        """
        V3: Calculate market-wide Gamma Exposure (GEX).

        GEX = Sum of (Gamma * Open Interest * Spot Price^2)

        Returns:
            Total market GEX
        """
        try:
            chain = market_data.options_chain.quotes
            spot_price = market_data.spot_price

            total_gex = 0.0

            for quote in chain:
                # GEX contribution = Gamma * OI * Spot^2 * 100 (contract multiplier)
                gex_contribution = quote.gamma * quote.open_interest * (spot_price**2) * 100
                total_gex += gex_contribution

            self.logger.debug(f"Market GEX for {market_data.ticker}: {total_gex:.2f}")
            return total_gex

        except Exception as e:
            self.logger.error(f"GEX calculation failed: {e}")
            return 0.0

    def _detect_charm_flows(self, market_data: EnhancedMarketData) -> float:
        """
        V3: Detect charm flows (theta decay impact on delta).

        Charm = dDelta/dTime
        High charm near expiration causes dealers to rehedge.

        Returns:
            Charm flow impact score
        """
        try:
            chain = market_data.options_chain.quotes

            total_charm = 0.0

            for quote in chain:
                # Charm impact = Theta * Delta * OI
                charm_impact = abs(quote.theta * quote.delta * quote.open_interest)
                total_charm += charm_impact

            # Normalize by total OI
            total_oi = sum(q.open_interest for q in chain)
            if total_oi > 0:
                normalized_charm = total_charm / total_oi
            else:
                normalized_charm = 0.0

            self.logger.debug(f"Charm flow impact for {market_data.ticker}: {normalized_charm:.4f}")
            return normalized_charm

        except Exception as e:
            self.logger.error(f"Charm flow detection failed: {e}")
            return 0.0

    def _get_fallback_output(self) -> Dict[str, float]:
        """Safe fallback values if processing fails"""
        return {
            "regime_classification": "R2",
            "regime_confidence": 0.5,
            "vol_edge": 0.0,
            "macro_stress_score": 0.0,
            "vix_percentile": 50.0,
            "market_gex": 0.0,
            "charm_flow_impact": 0.0,
            "portfolio_vega_effective": 0.0,
            "portfolio_gamma_effective": 0.0,
            "portfolio_delta_effective": 0.0,
            "vega_utilization": 0.0,
        }
