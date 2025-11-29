"""
Intelligent Strategy Selector
Automatically decides between stocks and options based on market conditions

This is the "brain" that makes Gnosis trade like a professional quant firm.
"""

from typing import Dict, List, Literal, Optional, Any
from dataclasses import dataclass
from enum import Enum

from models.options_contracts import EnhancedMarketData, OptionsChain


class InstrumentType(Enum):
    """Types of instruments we can trade"""

    STOCK = "stock"
    OPTION_SINGLE = "option_single"
    OPTION_SPREAD = "option_spread"
    SKIP = "skip"


class StrategyType(Enum):
    """Specific strategy types"""

    # Stock strategies
    LONG_STOCK = "long_stock"
    SHORT_STOCK = "short_stock"

    # Single option strategies
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_CALL = "short_call"  # Covered call
    SHORT_PUT = "short_put"  # Cash-secured put

    # Multi-leg strategies
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    LONG_STRADDLE = "long_straddle"
    IRON_CONDOR = "iron_condor"


@dataclass
class InstrumentDecision:
    """Decision on what instrument to trade"""

    instrument_type: InstrumentType
    strategy_type: StrategyType
    confidence: float  # 0.0 to 1.0
    reasoning: str

    # Strategy-specific details
    details: Dict[str, Any]  # Strike prices, expiration, etc.


class IntelligentStrategySelector:
    """
    Decides whether to trade stocks or options based on:
    - Volatility environment
    - Signal strength
    - Market regime
    - Execution quality
    - Portfolio constraints
    """

    def __init__(self, config: Dict, logger=None):
        self.config = config
        self.logger = logger

        # Thresholds from config
        self.low_iv_threshold = config.get("low_iv_threshold", 20)
        self.high_iv_threshold = config.get("high_iv_threshold", 40)
        self.spread_iv_rank_threshold = config.get("spread_iv_rank_threshold", 50)
        self.stock_confidence_threshold = config.get("stock_confidence_threshold", 0.70)
        self.option_confidence_threshold = config.get("option_confidence_threshold", 0.55)

    def select_optimal_instrument(
        self,
        market_data: EnhancedMarketData,
        signal_direction: Literal["bullish", "bearish", "neutral"],
        signal_confidence: float,
        regime: str,
        portfolio_state: Optional[Dict] = None,
    ) -> InstrumentDecision:
        """
        Main decision function - chooses optimal instrument

        Args:
            market_data: Enhanced market data with options chain
            signal_direction: Trading signal direction
            signal_confidence: Signal confidence (0.0 to 1.0)
            regime: Market regime ("trending_up", "trending_down", "ranging", etc.)
            portfolio_state: Current portfolio state (optional)

        Returns:
            InstrumentDecision with recommended instrument and strategy
        """

        # Extract key metrics
        iv_percentile = self._calculate_iv_percentile(market_data)
        execution_quality = self._assess_execution_quality(market_data)

        if self.logger:
            self.logger.info(f"Strategy Selection for {market_data.ticker}:")
            self.logger.info(f"  Signal: {signal_direction} ({signal_confidence:.2%})")
            self.logger.info(f"  IV Percentile: {iv_percentile:.1f}%")
            self.logger.info(f"  Regime: {regime}")

        # Decision logic
        decision = self._make_decision(
            signal_direction=signal_direction,
            signal_confidence=signal_confidence,
            iv_percentile=iv_percentile,
            regime=regime,
            execution_quality=execution_quality,
            market_data=market_data,
            portfolio_state=portfolio_state,
        )

        if self.logger:
            self.logger.info(f"  Decision: {decision.instrument_type.value}")
            self.logger.info(f"  Strategy: {decision.strategy_type.value}")
            self.logger.info(f"  Confidence: {decision.confidence:.2%}")
            self.logger.info(f"  Reasoning: {decision.reasoning}")

        return decision

    def _make_decision(
        self,
        signal_direction: str,
        signal_confidence: float,
        iv_percentile: float,
        regime: str,
        execution_quality: str,
        market_data: EnhancedMarketData,
        portfolio_state: Optional[Dict],
    ) -> InstrumentDecision:
        """Core decision logic"""

        # Rule 1: Poor execution quality → Skip or stocks only
        if execution_quality == "poor":
            if signal_confidence > 0.80:
                return InstrumentDecision(
                    instrument_type=InstrumentType.STOCK,
                    strategy_type=StrategyType.LONG_STOCK
                    if signal_direction == "bullish"
                    else StrategyType.SHORT_STOCK,
                    confidence=signal_confidence * 0.8,  # Reduce confidence due to poor execution
                    reasoning="Poor execution quality, but strong signal → Stock trade",
                    details={},
                )
            else:
                return InstrumentDecision(
                    instrument_type=InstrumentType.SKIP,
                    strategy_type=StrategyType.LONG_STOCK,  # Placeholder
                    confidence=0.0,
                    reasoning="Poor execution quality + weak signal → Skip",
                    details={},
                )

        # Rule 2: Very high confidence + low IV → Trade stocks (maximize upside)
        if (
            signal_confidence > self.stock_confidence_threshold
            and iv_percentile < self.low_iv_threshold
        ):
            return self._create_stock_decision(
                signal_direction,
                signal_confidence,
                "High confidence + Low IV → Stock for maximum upside",
            )

        # Rule 3: Very high IV → Prefer spreads (expensive options)
        if iv_percentile > self.high_iv_threshold:
            return self._create_spread_decision(
                signal_direction,
                signal_confidence,
                iv_percentile,
                "High IV → Spread to reduce cost",
            )

        # Rule 4: Moderate confidence + moderate IV → Single options
        if (
            signal_confidence > self.option_confidence_threshold
            and self.low_iv_threshold <= iv_percentile <= self.high_iv_threshold
        ):
            return self._create_single_option_decision(
                signal_direction,
                signal_confidence,
                "Moderate confidence + Moderate IV → Single option",
            )

        # Rule 5: Trending regime + decent confidence → Stocks
        if "trending" in regime.lower() and signal_confidence > 0.65:
            return self._create_stock_decision(
                signal_direction,
                signal_confidence,
                f"Trending regime ({regime}) → Stock for trend following",
            )

        # Rule 6: Ranging regime + high IV → Spreads or skip
        if "ranging" in regime.lower() or "choppy" in regime.lower():
            if iv_percentile > 35:
                return self._create_spread_decision(
                    signal_direction,
                    signal_confidence,
                    iv_percentile,
                    "Ranging market + High IV → Spread",
                )
            else:
                return InstrumentDecision(
                    instrument_type=InstrumentType.SKIP,
                    strategy_type=StrategyType.LONG_STOCK,
                    confidence=0.0,
                    reasoning="Ranging market + Low IV → Skip (unfavorable conditions)",
                    details={},
                )

        # Default: Moderate confidence → Single option
        if signal_confidence > 0.55:
            return self._create_single_option_decision(
                signal_direction, signal_confidence, "Default: Moderate signal → Single option"
            )

        # Fallback: Skip
        return InstrumentDecision(
            instrument_type=InstrumentType.SKIP,
            strategy_type=StrategyType.LONG_STOCK,
            confidence=0.0,
            reasoning=f"Low confidence ({signal_confidence:.2%}) → Skip",
            details={},
        )

    def _create_stock_decision(
        self, direction: str, confidence: float, reasoning: str
    ) -> InstrumentDecision:
        """Create stock trading decision"""
        strategy = StrategyType.LONG_STOCK if direction == "bullish" else StrategyType.SHORT_STOCK

        return InstrumentDecision(
            instrument_type=InstrumentType.STOCK,
            strategy_type=strategy,
            confidence=confidence,
            reasoning=reasoning,
            details={"direction": direction},
        )

    def _create_single_option_decision(
        self, direction: str, confidence: float, reasoning: str
    ) -> InstrumentDecision:
        """Create single option decision"""
        if direction == "bullish":
            strategy = StrategyType.LONG_CALL
        elif direction == "bearish":
            strategy = StrategyType.LONG_PUT
        else:  # neutral
            strategy = StrategyType.LONG_STRADDLE

        return InstrumentDecision(
            instrument_type=InstrumentType.OPTION_SINGLE,
            strategy_type=strategy,
            confidence=confidence,
            reasoning=reasoning,
            details={"direction": direction},
        )

    def _create_spread_decision(
        self, direction: str, confidence: float, iv_percentile: float, reasoning: str
    ) -> InstrumentDecision:
        """Create multi-leg spread decision"""
        if direction == "bullish":
            strategy = StrategyType.BULL_CALL_SPREAD
        elif direction == "bearish":
            strategy = StrategyType.BEAR_PUT_SPREAD
        else:  # neutral or low confidence
            if iv_percentile > 60:
                strategy = StrategyType.IRON_CONDOR  # Sell premium
            else:
                strategy = StrategyType.LONG_STRADDLE  # Buy volatility

        return InstrumentDecision(
            instrument_type=InstrumentType.OPTION_SPREAD,
            strategy_type=strategy,
            confidence=confidence,
            reasoning=reasoning,
            details={"direction": direction, "iv_percentile": iv_percentile},
        )

    def _calculate_iv_percentile(self, market_data: EnhancedMarketData) -> float:
        """Calculate IV percentile from market data"""
        if not market_data.options_chain or not market_data.options_chain.quotes:
            return 25.0  # Default to moderate IV

        # Get ATM option IV
        current_price = market_data.current_price
        atm_quotes = [
            q
            for q in market_data.options_chain.quotes
            if abs(q.strike - current_price) < current_price * 0.05  # Within 5% of ATM
        ]

        if not atm_quotes:
            return 25.0

        # Average IV of ATM options
        avg_iv = sum(q.implied_volatility for q in atm_quotes) / len(atm_quotes)

        # Convert to percentile (simplified - would use historical data in production)
        # For now, map IV to percentile heuristically
        if avg_iv < 0.15:
            return 10.0
        elif avg_iv < 0.25:
            return 30.0
        elif avg_iv < 0.35:
            return 50.0
        elif avg_iv < 0.50:
            return 70.0
        else:
            return 90.0

    def _assess_execution_quality(
        self, market_data: EnhancedMarketData
    ) -> Literal["excellent", "good", "fair", "poor"]:
        """Assess execution quality based on spreads and liquidity"""
        if not market_data.options_chain or not market_data.options_chain.quotes:
            return "good"  # Assume stocks have good execution

        # Check option spreads
        spreads = [q.spread_pct for q in market_data.options_chain.quotes]
        avg_spread = sum(spreads) / len(spreads) if spreads else 0.05

        # Check volume
        total_volume = sum(q.volume for q in market_data.options_chain.quotes)

        if avg_spread < 0.02 and total_volume > 100:
            return "excellent"
        elif avg_spread < 0.05 and total_volume > 50:
            return "good"
        elif avg_spread < 0.10 and total_volume > 20:
            return "fair"
        else:
            return "poor"
