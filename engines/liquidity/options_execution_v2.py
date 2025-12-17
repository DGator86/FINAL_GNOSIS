"""
Options Execution Module - Addresses slippage model feedback
- Fixed slippage calculation (removed IV mixing with spreads)
- Proper liquidity tier classification
- Realistic execution cost modeling
"""

from typing import Any, Dict, List, Literal

import numpy as np

from models.options_contracts import EnhancedMarketData, OptionsChain


class OptionsExecutionModule:
    """
    Enhanced liquidity analysis with corrected slippage model.
    Addresses all feedback about microstructure modeling.
    """

    def __init__(self, config: Dict, logger=None):
        self.config = config
        self.logger = logger

    def assess_execution_environment(self, market_data: EnhancedMarketData) -> Dict[str, Any]:
        """
        Comprehensive execution analysis with proper slippage modeling.
        """

        options_chain = market_data.options_chain

        # 1. Spread Quality Analysis
        spread_analysis = self._analyze_spread_quality(options_chain)

        # 2. Liquidity Tier Classification
        liquidity_tier = self._classify_liquidity_tier(options_chain, spread_analysis)

        # 3. Fixed Slippage Model (addresses feedback issues)
        execution_cost_bps = self._calculate_realistic_slippage(
            spread_analysis, liquidity_tier, options_chain
        )

        # 4. Tradeable Strike Identification
        tradeable_strikes = self._identify_tradeable_strikes(options_chain, liquidity_tier)

        # 5. Overall Feasibility Assessment
        execution_feasibility = self._assess_execution_feasibility(
            liquidity_tier, execution_cost_bps
        )

        return {
            "liquidity_tier": liquidity_tier,
            "execution_cost_bps": execution_cost_bps,
            "tradeable_strikes": tradeable_strikes,
            "execution_feasibility": execution_feasibility,
            "spread_quality_score": spread_analysis["quality_score"],
        }

    def _analyze_spread_quality(self, options_chain: OptionsChain) -> Dict:
        """Analyze bid-ask spreads across the chain"""

        spreads = [quote.spread_pct for quote in options_chain.quotes]

        if not spreads:
            return {"median_spread_pct": 1.0, "avg_spread_pct": 1.0, "quality_score": 0.0}

        median_spread = float(np.median(spreads))
        avg_spread = float(np.mean(spreads))

        # Quality score: 1.0 = perfect (0% spreads), 0.0 = terrible (100% spreads)
        quality_score = max(0.0, 1.0 - avg_spread)

        return {
            "median_spread_pct": median_spread,
            "avg_spread_pct": avg_spread,
            "quality_score": quality_score,
        }

    def _classify_liquidity_tier(
        self, options_chain: OptionsChain, spread_analysis: Dict
    ) -> Literal["tier_1", "tier_2", "tier_3", "tier_4"]:
        """
        Classify liquidity based on spreads, volume, and open interest.
        """

        avg_spread = spread_analysis["avg_spread_pct"]
        total_volume = sum(quote.volume for quote in options_chain.quotes)
        total_oi = sum(quote.open_interest for quote in options_chain.quotes)

        # Tier thresholds from config
        tiers = self.config.get("liquidity_tiers", {})

        # Check tier 1 (best)
        if (
            avg_spread <= tiers.get("tier_1", {}).get("max_spread_pct", 0.02)
            and total_volume >= tiers.get("tier_1", {}).get("min_volume", 100)
            and total_oi >= tiers.get("tier_1", {}).get("min_oi", 500)
        ):
            return "tier_1"

        # Check tier 2
        elif (
            avg_spread <= tiers.get("tier_2", {}).get("max_spread_pct", 0.05)
            and total_volume >= tiers.get("tier_2", {}).get("min_volume", 50)
            and total_oi >= tiers.get("tier_2", {}).get("min_oi", 200)
        ):
            return "tier_2"

        # Check tier 3
        elif (
            avg_spread <= tiers.get("tier_3", {}).get("max_spread_pct", 0.10)
            and total_volume >= tiers.get("tier_3", {}).get("min_volume", 20)
            and total_oi >= tiers.get("tier_3", {}).get("min_oi", 50)
        ):
            return "tier_3"

        # Default to tier 4 (worst)
        else:
            return "tier_4"

    def _calculate_realistic_slippage(
        self, spread_analysis: Dict, liquidity_tier: str, options_chain: OptionsChain
    ) -> float:
        """
        Fixed slippage model addressing feedback issues.

        OLD (broken): mixed IV with spreads
        NEW (correct): pure microstructure approach
        """

        median_spread_pct = spread_analysis["median_spread_pct"]

        # Base slippage: 25-50% of spread, with floor at 1 bp
        # This is realistic - you typically pay 25-50% of the spread in slippage
        base_slippage_pct = max(
            0.5 * median_spread_pct,  # 50% of spread
            0.0001,  # 1 bp floor (0.01%)
        )

        # Liquidity tier multipliers (from config)
        tier_multipliers = self.config.get(
            "slippage_multipliers",
            {
                "tier_1": 0.15,  # Ultra liquid - minimal slippage
                "tier_2": 0.25,  # Good liquid - modest slippage
                "tier_3": 0.40,  # Fair liquid - higher slippage
                "tier_4": 0.60,  # Poor liquid - significant slippage
            },
        )

        multiplier = tier_multipliers.get(liquidity_tier, 0.50)

        # Final slippage in basis points
        slippage_bps = base_slippage_pct * multiplier * 10000

        return float(slippage_bps)

    def _identify_tradeable_strikes(
        self, options_chain: OptionsChain, liquidity_tier: str
    ) -> List[float]:
        """
        Filter strikes that meet minimum liquidity standards.
        """

        # Minimum thresholds by tier
        thresholds = self.config.get(
            "tradeable_thresholds",
            {
                "tier_1": {"min_volume": 10, "max_spread_pct": 0.05},
                "tier_2": {"min_volume": 5, "max_spread_pct": 0.10},
                "tier_3": {"min_volume": 2, "max_spread_pct": 0.20},
                "tier_4": {"min_volume": 1, "max_spread_pct": 0.50},
            },
        )

        tier_config = thresholds.get(liquidity_tier, thresholds["tier_4"])
        min_volume = tier_config["min_volume"]
        max_spread = tier_config["max_spread_pct"]

        # Filter tradeable strikes
        tradeable = []
        for quote in options_chain.quotes:
            if quote.volume >= min_volume and quote.spread_pct <= max_spread:
                tradeable.append(quote.strike)

        return sorted(list(set(tradeable)))  # Remove duplicates and sort

    def _assess_execution_feasibility(
        self, liquidity_tier: str, execution_cost_bps: float
    ) -> Literal["excellent", "good", "fair", "poor"]:
        """
        Overall execution quality assessment.
        """

        if liquidity_tier == "tier_1" and execution_cost_bps < 15:
            return "excellent"
        elif liquidity_tier in ["tier_1", "tier_2"] and execution_cost_bps < 30:
            return "good"
        elif liquidity_tier in ["tier_2", "tier_3"] and execution_cost_bps < 50:
            return "fair"
        else:
            return "poor"

    def execute_order(
        self,
        strategy_type: Literal["single_leg", "multi_leg"],
        legs: List[Dict[str, Any]],
        alpaca_client: Any,
        quantity: int = 1,
    ) -> Dict[str, Any]:
        """Execute option order with support for multi-leg strategies.

        Args:
            strategy_type: Type of strategy ('single_leg' or 'multi_leg')
            legs: List of option legs with symbol, side, ratio_qty
            alpaca_client: AlpacaClient instance
            quantity: Base quantity for the order

        Returns:
            Order execution result

        Example:
            # Single leg
            result = module.execute_order(
                strategy_type="single_leg",
                legs=[{"symbol": "AAPL230616C00150000", "side": "buy", "ratio_qty": 1}],
                alpaca_client=client,
                quantity=1
            )

            # Multi-leg (bull call spread)
            result = module.execute_order(
                strategy_type="multi_leg",
                legs=[
                    {"symbol": "AAPL230616C00150000", "side": "buy", "ratio_qty": 1},
                    {"symbol": "AAPL230616C00160000", "side": "sell", "ratio_qty": 1}
                ],
                alpaca_client=client,
                quantity=1
            )
        """
        if strategy_type == "multi_leg":
            # Use the new multi-leg order method
            if self.logger:
                self.logger.info(
                    f"Executing multi-leg order with {len(legs)} legs, quantity={quantity}"
                )

            result = alpaca_client.place_multi_leg_option_order(
                legs=legs,
                quantity=quantity,
                time_in_force="day",
                extended_hours=False,
            )

            if self.logger:
                self.logger.info(f"Multi-leg order placed: {result['id']}")

            return {
                "success": True,
                "order_id": result["id"],
                "order_type": "multi_leg",
                "legs": result["legs"],
                "status": result["status"],
            }

        elif strategy_type == "single_leg":
            # Single leg order
            if len(legs) != 1:
                raise ValueError("Single leg strategy requires exactly one leg")

            leg = legs[0]

            if self.logger:
                self.logger.info(
                    f"Executing single-leg order: {leg['symbol']}, "
                    f"side={leg['side']}, qty={quantity}"
                )

            result = alpaca_client.place_multi_leg_option_order(
                legs=legs,
                quantity=quantity,
                time_in_force="day",
                extended_hours=False,
            )

            if self.logger:
                self.logger.info(f"Single-leg order placed: {result['id']}")

            return {
                "success": True,
                "order_id": result["id"],
                "order_type": "single_leg",
                "symbol": leg["symbol"],
                "side": leg["side"],
                "status": result["status"],
            }

        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
