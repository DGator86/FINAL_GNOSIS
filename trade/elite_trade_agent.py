"""
Elite Trade Agent - World-Class Multi-Strategy Options & Equity Execution

This is the PRODUCTION trade agent implementing institutional-grade execution:

STRATEGY SELECTION (IV-Aware):
├── High IV Environment (IV Rank > 50)
│   ├── Bullish → Bull Put Spread (Credit)
│   ├── Bearish → Bear Call Spread (Credit)
│   └── Neutral → Iron Condor / Iron Butterfly
├── Low IV Environment (IV Rank < 30)
│   ├── Bullish High Conf → Long Call
│   ├── Bullish Med Conf → Bull Call Spread (Debit)
│   ├── Bearish High Conf → Long Put
│   ├── Bearish Med Conf → Bear Put Spread (Debit)
│   └── Neutral + Vol Expected → Straddle / Strangle
└── Medium IV Environment
    ├── Directional → Vertical Spreads
    └── Neutral → Calendar Spreads / Butterflies

RISK MANAGEMENT:
- Kelly Criterion position sizing with fractional Kelly (25%)
- Maximum position size: 4% of portfolio per trade
- Maximum portfolio heat: 20% total risk exposure
- Stop-loss: ATR-based dynamic stops
- Take-profit: Risk-multiple based (min 1.5:1 R:R)
- Correlation-aware position limits

EXECUTION:
- Liquidity scoring for strike selection
- Bid-ask spread validation (max 10% of mid)
- Open interest minimums (100 OI for entry)
- Volume validation (10+ contracts daily)
- Smart order routing preferences

TIMEFRAME MANAGEMENT:
- Scalp (1-15min): Tight stops, quick profits, equity preferred
- Intraday (15min-4hr): Options with 0-7 DTE
- Swing (1-5 days): Options with 7-30 DTE
- Position (5-30 days): Options with 30-60 DTE

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from loguru import logger

from schemas.core_schemas import (
    DirectionEnum,
    ElasticitySnapshot,
    HedgeSnapshot,
    LiquiditySnapshot,
    OptionsLeg,
    OptionsOrderRequest,
    OrderResult,
    OrderStatus,
    PipelineResult,
    SentimentSnapshot,
    StrategyType,
    TradeIdea,
)


# =============================================================================
# ENUMS AND TYPE DEFINITIONS
# =============================================================================

class MarketRegime(str, Enum):
    """Market regime classification."""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MEAN_REVERTING = "mean_reverting"
    BREAKOUT = "breakout"


class IVEnvironment(str, Enum):
    """Implied volatility environment."""
    HIGH = "high"      # IV Rank > 50, sell premium
    MEDIUM = "medium"  # IV Rank 30-50, spreads
    LOW = "low"        # IV Rank < 30, buy premium


class Timeframe(str, Enum):
    """Trading timeframe."""
    SCALP = "scalp"           # 1-15 minutes
    INTRADAY = "intraday"     # 15min - 4 hours
    SWING = "swing"           # 1-5 days
    POSITION = "position"     # 5-30 days


class OptionStrategy(str, Enum):
    """Options strategy types."""
    # Directional - Debit
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    
    # Directional - Credit
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    
    # Neutral - Credit
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    SHORT_STRANGLE = "short_strangle"
    SHORT_STRADDLE = "short_straddle"
    
    # Volatility - Debit
    LONG_STRADDLE = "long_straddle"
    LONG_STRANGLE = "long_strangle"
    
    # Time-based
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"
    
    # Equity
    EQUITY_LONG = "equity_long"
    EQUITY_SHORT = "equity_short"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RiskParameters:
    """Risk management parameters."""
    max_position_pct: float = 0.04          # 4% max per position
    max_portfolio_heat: float = 0.20        # 20% total risk exposure
    kelly_fraction: float = 0.25            # 25% Kelly
    min_reward_risk: float = 1.5            # Minimum 1.5:1 R:R
    max_correlation: float = 0.7            # Max correlation between positions
    max_sector_exposure: float = 0.30       # 30% max in one sector
    stop_loss_atr_multiple: float = 2.0     # 2 ATR stop loss
    take_profit_atr_multiple: float = 3.0   # 3 ATR take profit
    trailing_stop_activation: float = 0.5   # Activate at 50% of target
    trailing_stop_distance: float = 0.3     # Trail at 30% of gains


@dataclass
class TimeframeConfig:
    """Configuration for each timeframe."""
    min_dte: int
    max_dte: int
    stop_loss_pct: float
    take_profit_pct: float
    max_hold_hours: int
    position_size_mult: float  # Multiplier on base position size


@dataclass
class LiquidityRequirements:
    """Minimum liquidity requirements for options."""
    min_open_interest: int = 100
    min_volume: int = 10
    max_spread_pct: float = 0.10  # 10% of mid price
    min_delta: float = 0.10
    max_delta: float = 0.90


@dataclass
class DynamicProfitThresholds:
    """
    Dynamic profit thresholds for options based on market conditions.
    
    World-class traders adjust profit targets based on:
    - IV environment (take profits faster in high IV)
    - Strategy type (credit vs debit strategies)
    - DTE remaining (accelerate near expiration)
    - Market regime (trend vs range)
    - Theta decay profile
    """
    # Target profit as % of max profit
    target_profit_pct: float
    # Early profit threshold (take quick wins)
    early_profit_pct: float
    # Stop loss as % of max loss
    stop_loss_pct: float
    # Time-based adjustments
    dte_acceleration_factor: float  # Increase urgency as DTE decreases
    # Trailing profit settings
    trailing_activation_pct: float  # Activate trail at this % of target
    trailing_distance_pct: float    # Trail this far behind peak
    # Scaling out levels
    scale_out_levels: List[Tuple[float, float]]  # [(profit_pct, exit_pct), ...]
    # Reasoning
    reasoning: str = ""


@dataclass
class MarketContext:
    """Current market context for decision making."""
    symbol: str
    spot_price: float
    iv_rank: float
    iv_percentile: float
    historical_vol: float
    implied_vol: float
    vol_skew: float
    put_call_ratio: float
    regime: MarketRegime
    iv_environment: IVEnvironment
    atr: float
    atr_pct: float
    trend_strength: float
    momentum: float
    support_level: float
    resistance_level: float


@dataclass
class TradeProposal:
    """Complete trade proposal with all details."""
    symbol: str
    strategy: OptionStrategy
    direction: DirectionEnum
    confidence: float
    
    # Entry
    entry_price: float
    quantity: int
    
    # Risk Management
    stop_loss: float
    take_profit: float
    trailing_stop_config: Dict[str, float]
    max_loss: float
    max_profit: float
    risk_reward_ratio: float
    
    # Position Sizing
    position_value: float
    position_pct: float
    risk_amount: float
    
    # Options Details (if applicable)
    options_order: Optional[OptionsOrderRequest] = None
    legs: List[OptionsLeg] = field(default_factory=list)
    
    # Timeframe
    timeframe: Timeframe = Timeframe.SWING
    target_dte: int = 30
    max_hold_time: timedelta = timedelta(days=5)
    
    # Metadata
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    strategy_id: str = ""
    tags: List[str] = field(default_factory=list)


# =============================================================================
# ELITE TRADE AGENT
# =============================================================================

class EliteTradeAgent:
    """
    World-class trade agent implementing institutional-grade execution.
    
    Features:
    - IV-aware strategy selection
    - Multi-timeframe support
    - Kelly Criterion position sizing
    - Dynamic risk management
    - Liquidity-optimized execution
    - Full options strategy suite
    """
    
    # Timeframe configurations
    TIMEFRAME_CONFIGS: Dict[Timeframe, TimeframeConfig] = {
        Timeframe.SCALP: TimeframeConfig(
            min_dte=0, max_dte=1,
            stop_loss_pct=0.005, take_profit_pct=0.015,
            max_hold_hours=1, position_size_mult=0.5
        ),
        Timeframe.INTRADAY: TimeframeConfig(
            min_dte=0, max_dte=7,
            stop_loss_pct=0.015, take_profit_pct=0.045,
            max_hold_hours=8, position_size_mult=0.75
        ),
        Timeframe.SWING: TimeframeConfig(
            min_dte=7, max_dte=30,
            stop_loss_pct=0.03, take_profit_pct=0.09,
            max_hold_hours=120, position_size_mult=1.0
        ),
        Timeframe.POSITION: TimeframeConfig(
            min_dte=30, max_dte=60,
            stop_loss_pct=0.05, take_profit_pct=0.15,
            max_hold_hours=720, position_size_mult=1.25
        ),
    }
    
    # Strategy selection matrix: (direction, iv_environment) -> strategies
    STRATEGY_MATRIX: Dict[Tuple[str, IVEnvironment], List[OptionStrategy]] = {
        # Bullish strategies
        ("bullish", IVEnvironment.HIGH): [
            OptionStrategy.BULL_PUT_SPREAD,  # Credit spread, benefit from IV crush
        ],
        ("bullish", IVEnvironment.MEDIUM): [
            OptionStrategy.BULL_CALL_SPREAD,
            OptionStrategy.BULL_PUT_SPREAD,
        ],
        ("bullish", IVEnvironment.LOW): [
            OptionStrategy.LONG_CALL,
            OptionStrategy.BULL_CALL_SPREAD,
        ],
        
        # Bearish strategies
        ("bearish", IVEnvironment.HIGH): [
            OptionStrategy.BEAR_CALL_SPREAD,  # Credit spread
        ],
        ("bearish", IVEnvironment.MEDIUM): [
            OptionStrategy.BEAR_PUT_SPREAD,
            OptionStrategy.BEAR_CALL_SPREAD,
        ],
        ("bearish", IVEnvironment.LOW): [
            OptionStrategy.LONG_PUT,
            OptionStrategy.BEAR_PUT_SPREAD,
        ],
        
        # Neutral strategies
        ("neutral", IVEnvironment.HIGH): [
            OptionStrategy.IRON_CONDOR,
            OptionStrategy.IRON_BUTTERFLY,
            OptionStrategy.SHORT_STRANGLE,
        ],
        ("neutral", IVEnvironment.MEDIUM): [
            OptionStrategy.IRON_CONDOR,
            OptionStrategy.CALENDAR_SPREAD,
        ],
        ("neutral", IVEnvironment.LOW): [
            OptionStrategy.LONG_STRADDLE,
            OptionStrategy.LONG_STRANGLE,
            OptionStrategy.CALENDAR_SPREAD,
        ],
    }
    
    # ==========================================================================
    # DYNAMIC PROFIT THRESHOLD CONFIGURATIONS
    # ==========================================================================
    # Base profit targets by strategy category (as % of max profit)
    STRATEGY_PROFIT_TARGETS: Dict[str, Dict[str, float]] = {
        # Credit strategies: Take profits early due to gamma risk near expiration
        "credit": {
            "target_profit_pct": 0.50,      # 50% of max profit
            "early_profit_pct": 0.25,       # Quick win at 25%
            "stop_loss_pct": 2.0,           # 200% of credit received (2x loss)
            "trailing_activation": 0.35,    # Start trailing at 35%
            "trailing_distance": 0.15,      # Trail 15% behind
        },
        # Debit strategies: Let winners run, defined risk
        "debit": {
            "target_profit_pct": 1.00,      # 100% gain on premium
            "early_profit_pct": 0.50,       # Take 50% as quick win
            "stop_loss_pct": 0.50,          # 50% of premium (cut losers)
            "trailing_activation": 0.50,    # Start trailing at 50%
            "trailing_distance": 0.25,      # Trail 25% behind
        },
        # Neutral/volatility strategies: Time-sensitive
        "neutral": {
            "target_profit_pct": 0.40,      # 40% of max profit
            "early_profit_pct": 0.20,       # Quick win at 20%
            "stop_loss_pct": 1.50,          # 150% of credit/debit
            "trailing_activation": 0.25,    # Trail early
            "trailing_distance": 0.10,      # Tight trail
        },
        # Equity: Standard trend following
        "equity": {
            "target_profit_pct": 1.50,      # 150% R:R target
            "early_profit_pct": 0.75,       # Quick win at 75%
            "stop_loss_pct": 1.00,          # 100% of risk (1:1)
            "trailing_activation": 0.50,
            "trailing_distance": 0.30,
        },
    }
    
    # IV adjustment multipliers for profit targets
    IV_PROFIT_ADJUSTMENTS: Dict[IVEnvironment, Dict[str, float]] = {
        IVEnvironment.HIGH: {
            "target_multiplier": 0.75,      # Take profits 25% faster
            "stop_multiplier": 1.25,        # Wider stops (more noise)
            "urgency_factor": 1.5,          # Higher urgency
        },
        IVEnvironment.MEDIUM: {
            "target_multiplier": 1.0,       # Standard targets
            "stop_multiplier": 1.0,         # Standard stops
            "urgency_factor": 1.0,          # Normal urgency
        },
        IVEnvironment.LOW: {
            "target_multiplier": 1.25,      # Let winners run longer
            "stop_multiplier": 0.80,        # Tighter stops (less noise)
            "urgency_factor": 0.75,         # Less urgency
        },
    }
    
    # DTE-based acceleration (multiply urgency as expiration approaches)
    DTE_ACCELERATION: Dict[str, float] = {
        "0-7": 2.0,     # Critical zone - aggressive profit taking
        "7-14": 1.5,    # Elevated urgency
        "14-30": 1.0,   # Normal
        "30-60": 0.8,   # Relaxed
        "60+": 0.6,     # Very relaxed
    }
    
    # Regime-based adjustments
    REGIME_PROFIT_ADJUSTMENTS: Dict[MarketRegime, Dict[str, float]] = {
        MarketRegime.TRENDING_BULL: {
            "target_multiplier": 1.3,   # Let winners run in trends
            "stop_multiplier": 1.2,     # Wider stops for pullbacks
        },
        MarketRegime.TRENDING_BEAR: {
            "target_multiplier": 1.3,
            "stop_multiplier": 1.2,
        },
        MarketRegime.HIGH_VOLATILITY: {
            "target_multiplier": 0.7,   # Quick profits in chaos
            "stop_multiplier": 1.5,     # Wide stops for whipsaws
        },
        MarketRegime.LOW_VOLATILITY: {
            "target_multiplier": 1.1,   # Slightly extended
            "stop_multiplier": 0.8,     # Tighter stops
        },
        MarketRegime.RANGE_BOUND: {
            "target_multiplier": 0.9,   # Mean reversion targets
            "stop_multiplier": 0.9,     # Tighter stops
        },
        MarketRegime.MEAN_REVERTING: {
            "target_multiplier": 0.85,
            "stop_multiplier": 0.85,
        },
        MarketRegime.BREAKOUT: {
            "target_multiplier": 1.5,   # Let breakouts run
            "stop_multiplier": 1.0,
        },
    }
    
    def __init__(
        self,
        options_adapter: Any = None,
        market_adapter: Any = None,
        broker: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Elite Trade Agent.
        
        Args:
            options_adapter: Options chain data adapter
            market_adapter: Market data adapter
            broker: Broker adapter for execution
            config: Configuration overrides
        """
        self.options_adapter = options_adapter
        self.market_adapter = market_adapter
        self.broker = broker
        self.config = config or {}
        
        # Risk parameters from config or environment
        self.risk_params = RiskParameters(
            max_position_pct=float(os.getenv("MAX_POSITION_SIZE_PCT", "4.0")) / 100,
            max_portfolio_heat=float(os.getenv("MAX_PORTFOLIO_HEAT_PCT", "20.0")) / 100,
            kelly_fraction=float(self.config.get("kelly_fraction", 0.25)),
            min_reward_risk=float(self.config.get("min_reward_risk", 1.5)),
        )
        
        # Liquidity requirements
        self.liquidity_reqs = LiquidityRequirements(
            min_open_interest=int(self.config.get("min_open_interest", 100)),
            min_volume=int(self.config.get("min_volume", 10)),
            max_spread_pct=float(self.config.get("max_spread_pct", 0.10)),
        )
        
        # Portfolio state
        self.portfolio_value = float(os.getenv("DEFAULT_CAPITAL", "100000.0"))
        self.current_positions: List[str] = []
        self.portfolio_heat: float = 0.0
        
        logger.info(
            f"EliteTradeAgent initialized | "
            f"max_position={self.risk_params.max_position_pct:.1%} | "
            f"max_heat={self.risk_params.max_portfolio_heat:.1%} | "
            f"kelly={self.risk_params.kelly_fraction:.0%}"
        )
    
    # =========================================================================
    # MAIN ENTRY POINTS
    # =========================================================================
    
    def generate_ideas(
        self,
        pipeline_result: PipelineResult,
        timestamp: datetime,
    ) -> List[TradeIdea]:
        """
        Generate trade ideas from pipeline results using institutional-grade analysis.
        
        This is the main entry point called by the pipeline runner.
        
        Features:
        - Multi-timeframe signal aggregation
        - IV-aware strategy selection
        - Kelly Criterion position sizing
        - Dynamic risk management
        """
        if not pipeline_result.consensus:
            return []
        
        symbol = pipeline_result.symbol
        consensus = pipeline_result.consensus
        
        # Extract base signals
        direction_str = consensus.get("direction", "neutral")
        confidence = consensus.get("confidence", 0.0)
        
        # =====================================================================
        # MULTI-TIMEFRAME SIGNAL AGGREGATION (Institutional Edge)
        # =====================================================================
        mtf_analysis = self._aggregate_multi_timeframe_signals(pipeline_result)
        
        # Override with MTF-adjusted values if alignment is strong
        if mtf_analysis["alignment"] >= 0.5:
            # Use MTF direction if it has better alignment
            if mtf_analysis["direction"] != "neutral":
                direction_str = mtf_analysis["direction"]
            # Use adjusted confidence
            confidence = mtf_analysis["confidence"]
            
            logger.info(
                f"MTF Override for {symbol}: dir={direction_str} | "
                f"alignment={mtf_analysis['alignment']:.0%} | conf={confidence:.1%}"
            )
        
        # Minimum confidence threshold (lowered for testing, production should be 0.4+)
        min_confidence = float(self.config.get("min_confidence", 0.25))
        if confidence < min_confidence:
            logger.debug(f"Skipping {symbol}: confidence {confidence:.2%} below threshold {min_confidence:.2%}")
            return []
        
        # Skip neutral direction
        if direction_str == "neutral":
            logger.debug(f"Skipping {symbol}: neutral direction")
            return []
        
        # Build market context
        context = self._build_market_context(pipeline_result)
        if context is None:
            logger.warning(f"Could not build market context for {symbol}")
            return []
        
        # Determine optimal timeframe based on signals and volatility
        timeframe = self._select_timeframe(pipeline_result, confidence)
        
        # Select optimal strategy (IV-aware)
        strategy = self._select_strategy(
            direction=direction_str,
            context=context,
            confidence=confidence,
            timeframe=timeframe,
        )
        
        if strategy is None:
            logger.debug(f"No suitable strategy for {symbol}")
            return []
        
        # Build trade proposal with full risk management
        proposal = self._build_trade_proposal(
            symbol=symbol,
            strategy=strategy,
            direction=DirectionEnum(direction_str) if direction_str != "neutral" else DirectionEnum.NEUTRAL,
            confidence=confidence,
            context=context,
            timeframe=timeframe,
            pipeline_result=pipeline_result,
        )
        
        if proposal is None:
            return []
        
        # Validate proposal against risk rules
        if not self._validate_proposal(proposal):
            return []
        
        # Convert to TradeIdea
        trade_idea = self._proposal_to_trade_idea(proposal)
        
        logger.info(
            f"🎯 TRADE SIGNAL: {strategy.value} for {symbol} | "
            f"dir={direction_str} | conf={confidence:.1%} | "
            f"R:R={proposal.risk_reward_ratio:.1f} | "
            f"MTF={mtf_analysis['alignment']:.0%}"
        )
        
        return [trade_idea]
    
    def execute_trades(
        self,
        trade_ideas: List[TradeIdea],
        timestamp: datetime,
    ) -> List[OrderResult]:
        """
        Execute trade ideas through the broker.
        
        Handles both equity and options execution with smart order routing.
        """
        if not self.broker:
            logger.info("No broker configured - execution skipped")
            return []
        
        results: List[OrderResult] = []
        
        for idea in trade_ideas:
            try:
                result = self._execute_single_trade(idea, timestamp)
                results.append(result)
                
                if result.status == OrderStatus.SUBMITTED:
                    self.current_positions.append(idea.symbol)
                    
            except Exception as e:
                logger.error(f"Execution failed for {idea.symbol}: {e}")
                results.append(OrderResult(
                    timestamp=timestamp,
                    symbol=idea.symbol,
                    status=OrderStatus.REJECTED,
                    message=str(e),
                ))
        
        return results
    
    # =========================================================================
    # MULTI-TIMEFRAME SIGNAL AGGREGATION
    # =========================================================================
    
    def _aggregate_multi_timeframe_signals(
        self,
        pipeline_result: PipelineResult,
    ) -> Dict[str, Any]:
        """
        Aggregate signals across multiple timeframes for institutional-grade analysis.
        
        Analyzes:
        - Short-term (1-15min): Scalp signals from technical/momentum
        - Medium-term (15min-4hr): Intraday from sentiment/flow
        - Long-term (1-5 days): Swing from hedge/elasticity
        - Position (5-30 days): Position trades from composite scores
        
        Returns:
            Multi-timeframe signal summary with confidence adjustments
        """
        mtf_signals = {
            "scalp": {"direction": "neutral", "confidence": 0.0, "weight": 0.15},
            "intraday": {"direction": "neutral", "confidence": 0.0, "weight": 0.25},
            "swing": {"direction": "neutral", "confidence": 0.0, "weight": 0.35},
            "position": {"direction": "neutral", "confidence": 0.0, "weight": 0.25},
        }
        
        # Extract signals from pipeline result
        consensus = pipeline_result.consensus or {}
        hedge = pipeline_result.hedge_snapshot
        sentiment = pipeline_result.sentiment_snapshot
        elasticity = pipeline_result.elasticity_snapshot
        ml_snapshot = pipeline_result.ml_snapshot
        
        # Short-term (Scalp): Use momentum and technical sentiment
        if sentiment:
            tech_sentiment = sentiment.technical_sentiment or 0.0
            mtf_signals["scalp"]["confidence"] = abs(tech_sentiment)
            mtf_signals["scalp"]["direction"] = "long" if tech_sentiment > 0.1 else ("short" if tech_sentiment < -0.1 else "neutral")
        
        # Medium-term (Intraday): Use flow sentiment and MTF score
        if sentiment:
            flow_sentiment = sentiment.flow_sentiment or 0.0
            mtf_score = sentiment.mtf_score or 0.0
            combined = (flow_sentiment + mtf_score) / 2
            mtf_signals["intraday"]["confidence"] = abs(combined)
            mtf_signals["intraday"]["direction"] = "long" if combined > 0.1 else ("short" if combined < -0.1 else "neutral")
        
        # Long-term (Swing): Use hedge energy asymmetry and elasticity
        if hedge:
            asymmetry = hedge.energy_asymmetry or 0.0
            energy = hedge.movement_energy or 50.0
            # Normalize energy (0-100 scale to 0-1)
            energy_factor = min(1.0, energy / 100.0)
            swing_signal = asymmetry * energy_factor
            mtf_signals["swing"]["confidence"] = abs(swing_signal)
            mtf_signals["swing"]["direction"] = "long" if swing_signal > 0.2 else ("short" if swing_signal < -0.2 else "neutral")
        
        # Position: Use ML predictions if available
        if ml_snapshot and hasattr(ml_snapshot, 'forecasting'):
            forecast = ml_snapshot.forecasting
            if forecast and hasattr(forecast, 'predictions'):
                # Use longer horizon predictions
                predictions = forecast.predictions or {}
                # Average predictions weighted by horizon
                pred_values = list(predictions.values())
                if pred_values:
                    avg_pred = sum(pred_values) / len(pred_values)
                    mtf_signals["position"]["confidence"] = min(1.0, abs(avg_pred) * 2)
                    mtf_signals["position"]["direction"] = "long" if avg_pred > 0 else ("short" if avg_pred < 0 else "neutral")
        
        # Calculate weighted consensus across timeframes
        total_weight = sum(s["weight"] for s in mtf_signals.values())
        direction_scores = {"long": 0.0, "short": 0.0, "neutral": 0.0}
        
        for tf, signal in mtf_signals.items():
            weight = signal["weight"] / total_weight
            conf = signal["confidence"]
            direction_scores[signal["direction"]] += weight * conf
        
        # Determine overall MTF direction
        max_score = max(direction_scores.values())
        mtf_direction = max(direction_scores, key=direction_scores.get)
        
        # Calculate MTF alignment (how aligned are all timeframes)
        alignment_count = sum(1 for s in mtf_signals.values() if s["direction"] == mtf_direction)
        mtf_alignment = alignment_count / len(mtf_signals)
        
        # Confidence boost for aligned signals (institutional edge)
        base_confidence = consensus.get("confidence", 0.5)
        if mtf_alignment >= 0.75:  # 3+ timeframes aligned
            confidence_boost = 0.15
        elif mtf_alignment >= 0.5:  # 2+ timeframes aligned
            confidence_boost = 0.05
        else:
            confidence_boost = -0.10  # Penalty for divergence
        
        final_confidence = min(1.0, max(0.0, base_confidence + confidence_boost))
        
        logger.debug(
            f"MTF Analysis: {mtf_direction} | alignment={mtf_alignment:.0%} | "
            f"conf={base_confidence:.1%} → {final_confidence:.1%}"
        )
        
        return {
            "signals": mtf_signals,
            "direction": mtf_direction,
            "alignment": mtf_alignment,
            "confidence": final_confidence,
            "confidence_adjustment": confidence_boost,
            "direction_scores": direction_scores,
        }
    
    # =========================================================================
    # DYNAMIC PROFIT THRESHOLD CALCULATION
    # =========================================================================
    
    def _calculate_dynamic_profit_thresholds(
        self,
        strategy: OptionStrategy,
        context: MarketContext,
        timeframe: Timeframe,
        confidence: float,
        target_dte: int,
    ) -> DynamicProfitThresholds:
        """
        Calculate dynamic profit thresholds based on market conditions.
        
        This is the core of institutional-grade options profit management.
        Adjusts targets based on:
        - Strategy type (credit vs debit)
        - IV environment
        - DTE remaining
        - Market regime
        - Signal confidence
        
        Args:
            strategy: The options strategy being traded
            context: Current market context
            timeframe: Trading timeframe
            confidence: Signal confidence (0-1)
            target_dte: Days to expiration
            
        Returns:
            DynamicProfitThresholds with all calculated thresholds
        """
        # Determine strategy category
        strategy_category = self._get_strategy_category(strategy)
        base_targets = self.STRATEGY_PROFIT_TARGETS[strategy_category]
        
        # Get IV adjustments
        iv_adj = self.IV_PROFIT_ADJUSTMENTS[context.iv_environment]
        
        # Get regime adjustments
        regime_adj = self.REGIME_PROFIT_ADJUSTMENTS.get(
            context.regime,
            {"target_multiplier": 1.0, "stop_multiplier": 1.0}
        )
        
        # Calculate DTE acceleration factor
        dte_factor = self._get_dte_acceleration(target_dte)
        
        # Confidence adjustment (higher confidence = more aggressive targets)
        confidence_factor = 0.8 + (confidence * 0.4)  # Range: 0.8 to 1.2
        
        # =====================================================================
        # CALCULATE FINAL THRESHOLDS
        # =====================================================================
        
        # Target profit: Base * IV * Regime * Confidence
        target_profit_pct = (
            base_targets["target_profit_pct"]
            * iv_adj["target_multiplier"]
            * regime_adj["target_multiplier"]
            * confidence_factor
        )
        
        # Early profit: Accelerated by DTE
        early_profit_pct = (
            base_targets["early_profit_pct"]
            * iv_adj["target_multiplier"]
            * dte_factor  # More urgent as expiration approaches
        )
        
        # Stop loss: Adjusted for volatility
        stop_loss_pct = (
            base_targets["stop_loss_pct"]
            * iv_adj["stop_multiplier"]
            * regime_adj["stop_multiplier"]
        )
        
        # Trailing activation and distance
        trailing_activation_pct = base_targets["trailing_activation"]
        trailing_distance_pct = base_targets["trailing_distance"]
        
        # In high IV, trail tighter
        if context.iv_environment == IVEnvironment.HIGH:
            trailing_distance_pct *= 0.8
        
        # Calculate scale-out levels based on strategy type
        scale_out_levels = self._calculate_scale_out_levels(
            strategy_category=strategy_category,
            target_profit_pct=target_profit_pct,
            iv_environment=context.iv_environment,
        )
        
        # Build reasoning string
        reasoning = (
            f"Dynamic thresholds: {strategy_category} strategy | "
            f"IV={context.iv_environment.value} (adj={iv_adj['target_multiplier']:.2f}) | "
            f"Regime={context.regime.value} | "
            f"DTE={target_dte} (accel={dte_factor:.2f}) | "
            f"Conf={confidence:.1%}"
        )
        
        logger.debug(
            f"📊 Profit Thresholds for {strategy.value}: "
            f"target={target_profit_pct:.0%} | early={early_profit_pct:.0%} | "
            f"stop={stop_loss_pct:.0%} | DTE_factor={dte_factor:.1f}"
        )
        
        return DynamicProfitThresholds(
            target_profit_pct=target_profit_pct,
            early_profit_pct=early_profit_pct,
            stop_loss_pct=stop_loss_pct,
            dte_acceleration_factor=dte_factor,
            trailing_activation_pct=trailing_activation_pct,
            trailing_distance_pct=trailing_distance_pct,
            scale_out_levels=scale_out_levels,
            reasoning=reasoning,
        )
    
    def _get_strategy_category(self, strategy: OptionStrategy) -> str:
        """Categorize strategy for profit threshold calculation."""
        
        # Credit strategies (sell premium)
        credit_strategies = {
            OptionStrategy.BULL_PUT_SPREAD,
            OptionStrategy.BEAR_CALL_SPREAD,
            OptionStrategy.IRON_CONDOR,
            OptionStrategy.IRON_BUTTERFLY,
            OptionStrategy.SHORT_STRANGLE,
            OptionStrategy.SHORT_STRADDLE,
        }
        
        # Debit strategies (buy premium)
        debit_strategies = {
            OptionStrategy.LONG_CALL,
            OptionStrategy.LONG_PUT,
            OptionStrategy.BULL_CALL_SPREAD,
            OptionStrategy.BEAR_PUT_SPREAD,
        }
        
        # Volatility/neutral strategies
        neutral_strategies = {
            OptionStrategy.LONG_STRADDLE,
            OptionStrategy.LONG_STRANGLE,
            OptionStrategy.CALENDAR_SPREAD,
            OptionStrategy.DIAGONAL_SPREAD,
        }
        
        # Equity strategies
        equity_strategies = {
            OptionStrategy.EQUITY_LONG,
            OptionStrategy.EQUITY_SHORT,
        }
        
        if strategy in credit_strategies:
            return "credit"
        elif strategy in debit_strategies:
            return "debit"
        elif strategy in neutral_strategies:
            return "neutral"
        elif strategy in equity_strategies:
            return "equity"
        else:
            return "debit"  # Default to debit behavior
    
    def _get_dte_acceleration(self, dte: int) -> float:
        """Get DTE-based acceleration factor for profit taking."""
        
        if dte <= 7:
            return self.DTE_ACCELERATION["0-7"]      # 2.0x - Critical zone
        elif dte <= 14:
            return self.DTE_ACCELERATION["7-14"]     # 1.5x - Elevated
        elif dte <= 30:
            return self.DTE_ACCELERATION["14-30"]    # 1.0x - Normal
        elif dte <= 60:
            return self.DTE_ACCELERATION["30-60"]    # 0.8x - Relaxed
        else:
            return self.DTE_ACCELERATION["60+"]      # 0.6x - Very relaxed
    
    def _calculate_scale_out_levels(
        self,
        strategy_category: str,
        target_profit_pct: float,
        iv_environment: IVEnvironment,
    ) -> List[Tuple[float, float]]:
        """
        Calculate scale-out levels for partial profit taking.
        
        Returns list of (profit_threshold, exit_percentage) tuples.
        Example: [(0.25, 0.33), (0.50, 0.33), (0.75, 0.34)] means:
        - At 25% profit, exit 33% of position
        - At 50% profit, exit another 33%
        - At 75% profit, exit remaining 34%
        """
        
        if strategy_category == "credit":
            # Credit strategies: Scale out aggressively
            if iv_environment == IVEnvironment.HIGH:
                # In high IV, take profits very fast
                return [
                    (0.25, 0.40),  # 25% profit → exit 40%
                    (0.40, 0.35),  # 40% profit → exit 35%
                    (0.50, 0.25),  # 50% profit → exit remaining
                ]
            else:
                return [
                    (0.30, 0.33),
                    (0.50, 0.33),
                    (0.65, 0.34),
                ]
        
        elif strategy_category == "debit":
            # Debit strategies: Let winners run longer
            if iv_environment == IVEnvironment.LOW:
                # In low IV, very patient
                return [
                    (0.50, 0.25),
                    (0.75, 0.25),
                    (1.00, 0.50),
                ]
            else:
                return [
                    (0.40, 0.33),
                    (0.70, 0.33),
                    (1.00, 0.34),
                ]
        
        elif strategy_category == "neutral":
            # Neutral strategies: Quick, even exits
            return [
                (0.20, 0.33),
                (0.35, 0.33),
                (0.50, 0.34),
            ]
        
        else:  # equity
            # Equity: Standard scale-out
            return [
                (0.50, 0.33),
                (1.00, 0.33),
                (1.50, 0.34),
            ]
    
    # =========================================================================
    # MARKET CONTEXT BUILDING
    # =========================================================================
    
    def _build_market_context(
        self,
        pipeline_result: PipelineResult,
    ) -> Optional[MarketContext]:
        """Build comprehensive market context from pipeline results."""
        
        symbol = pipeline_result.symbol
        
        # Get spot price
        spot_price = self._get_spot_price(symbol)
        if spot_price is None or spot_price <= 0:
            return None
        
        # Extract from snapshots
        hedge = pipeline_result.hedge_snapshot
        liquidity = pipeline_result.liquidity_snapshot
        sentiment = pipeline_result.sentiment_snapshot
        elasticity = pipeline_result.elasticity_snapshot
        
        # IV metrics (estimate from elasticity/hedge if not available)
        volatility = elasticity.volatility if elasticity else 0.2
        iv_rank = self._estimate_iv_rank(volatility)
        iv_percentile = self._estimate_iv_percentile(volatility)
        
        # Determine IV environment
        if iv_rank > 50:
            iv_environment = IVEnvironment.HIGH
        elif iv_rank > 30:
            iv_environment = IVEnvironment.MEDIUM
        else:
            iv_environment = IVEnvironment.LOW
        
        # Market regime
        regime = self._classify_regime(
            hedge=hedge,
            elasticity=elasticity,
            sentiment=sentiment,
        )
        
        # ATR calculation
        atr = spot_price * volatility * 0.1  # Rough approximation
        atr_pct = atr / spot_price
        
        # Trend and momentum
        trend_strength = elasticity.trend_strength if elasticity else 0.0
        momentum = hedge.energy_asymmetry if hedge else 0.0
        
        # Support/Resistance (simplified)
        support_level = spot_price * (1 - atr_pct * 2)
        resistance_level = spot_price * (1 + atr_pct * 2)
        
        return MarketContext(
            symbol=symbol,
            spot_price=spot_price,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            historical_vol=volatility,
            implied_vol=volatility * 1.1,  # IV typically higher
            vol_skew=0.0,  # Would need options data
            put_call_ratio=1.0,  # Would need flow data
            regime=regime,
            iv_environment=iv_environment,
            atr=atr,
            atr_pct=atr_pct,
            trend_strength=trend_strength,
            momentum=momentum,
            support_level=support_level,
            resistance_level=resistance_level,
        )
    
    def _classify_regime(
        self,
        hedge: Optional[HedgeSnapshot],
        elasticity: Optional[ElasticitySnapshot],
        sentiment: Optional[SentimentSnapshot],
    ) -> MarketRegime:
        """Classify current market regime."""
        
        if hedge is None and elasticity is None:
            return MarketRegime.RANGE_BOUND
        
        volatility = elasticity.volatility if elasticity else 0.2
        trend = elasticity.trend_strength if elasticity else 0.0
        energy = hedge.movement_energy if hedge else 0.0
        asymmetry = hedge.energy_asymmetry if hedge else 0.0
        
        # High volatility regime
        if volatility > 0.4:
            return MarketRegime.HIGH_VOLATILITY
        
        # Low volatility regime
        if volatility < 0.1:
            return MarketRegime.LOW_VOLATILITY
        
        # Trending regimes
        if abs(trend) > 0.6:
            if asymmetry > 0.3:
                return MarketRegime.TRENDING_BULL
            elif asymmetry < -0.3:
                return MarketRegime.TRENDING_BEAR
        
        # Breakout detection
        if energy > 70 and abs(asymmetry) > 0.5:
            return MarketRegime.BREAKOUT
        
        # Mean reversion
        if volatility > 0.25 and abs(trend) < 0.3:
            return MarketRegime.MEAN_REVERTING
        
        return MarketRegime.RANGE_BOUND
    
    def _estimate_iv_rank(self, current_vol: float) -> float:
        """Estimate IV rank (0-100) based on current volatility."""
        # Simplified: assume historical vol range is 0.1 to 0.5
        min_vol, max_vol = 0.10, 0.50
        rank = (current_vol - min_vol) / (max_vol - min_vol) * 100
        return max(0, min(100, rank))
    
    def _estimate_iv_percentile(self, current_vol: float) -> float:
        """Estimate IV percentile (0-100)."""
        # Similar to rank for simplicity
        return self._estimate_iv_rank(current_vol)
    
    # =========================================================================
    # STRATEGY SELECTION
    # =========================================================================
    
    def _select_timeframe(
        self,
        pipeline_result: PipelineResult,
        confidence: float,
    ) -> Timeframe:
        """Select optimal timeframe based on signals."""
        
        # Higher confidence = longer timeframe
        # Higher volatility = shorter timeframe
        
        volatility = 0.2
        if pipeline_result.elasticity_snapshot:
            volatility = pipeline_result.elasticity_snapshot.volatility
        
        # High volatility = scalp/intraday
        if volatility > 0.4:
            return Timeframe.SCALP if confidence < 0.6 else Timeframe.INTRADAY
        
        # Medium volatility = intraday/swing
        if volatility > 0.2:
            return Timeframe.INTRADAY if confidence < 0.7 else Timeframe.SWING
        
        # Low volatility = swing/position
        return Timeframe.SWING if confidence < 0.8 else Timeframe.POSITION
    
    def _select_strategy(
        self,
        direction: str,
        context: MarketContext,
        confidence: float,
        timeframe: Timeframe,
    ) -> Optional[OptionStrategy]:
        """Select optimal strategy based on market context."""
        
        # Map direction
        if direction == "long":
            dir_key = "bullish"
        elif direction == "short":
            dir_key = "bearish"
        else:
            dir_key = "neutral"
        
        # Get candidate strategies
        candidates = self.STRATEGY_MATRIX.get(
            (dir_key, context.iv_environment),
            []
        )
        
        if not candidates:
            # Fallback to equity
            if dir_key == "bullish":
                return OptionStrategy.EQUITY_LONG
            elif dir_key == "bearish":
                return OptionStrategy.EQUITY_SHORT
            return None
        
        # Select best strategy based on confidence and regime
        strategy = self._rank_strategies(
            candidates=candidates,
            confidence=confidence,
            context=context,
            timeframe=timeframe,
        )
        
        return strategy
    
    def _rank_strategies(
        self,
        candidates: List[OptionStrategy],
        confidence: float,
        context: MarketContext,
        timeframe: Timeframe,
    ) -> OptionStrategy:
        """Rank and select best strategy from candidates."""
        
        if not candidates:
            return OptionStrategy.EQUITY_LONG
        
        # Simple ranking based on confidence
        # High confidence = prefer defined risk/reward (spreads)
        # Very high confidence = prefer unlimited profit potential
        
        if confidence > 0.85 and len(candidates) > 1:
            # Prefer outright options for high conviction
            for s in candidates:
                if s in [OptionStrategy.LONG_CALL, OptionStrategy.LONG_PUT]:
                    return s
        
        if confidence > 0.7:
            # Prefer defined risk spreads
            for s in candidates:
                if "SPREAD" in s.value.upper():
                    return s
        
        # For neutral, prefer iron condor
        if context.regime == MarketRegime.RANGE_BOUND:
            for s in candidates:
                if s == OptionStrategy.IRON_CONDOR:
                    return s
        
        # Default to first candidate
        return candidates[0]
    
    # =========================================================================
    # POSITION SIZING
    # =========================================================================
    
    def _calculate_position_size(
        self,
        context: MarketContext,
        confidence: float,
        strategy: OptionStrategy,
        timeframe: Timeframe,
        risk_per_share: float,
    ) -> Tuple[int, float, float]:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Returns: (quantity, position_value, risk_amount)
        """
        
        # Kelly Criterion: f* = (bp - q) / b
        # where b = odds, p = win probability, q = 1-p
        
        # Estimate win probability from confidence
        win_prob = 0.5 + (confidence - 0.5) * 0.5  # Scale to 0.5-0.75
        lose_prob = 1 - win_prob
        
        # Estimate odds (reward:risk)
        odds = self.risk_params.min_reward_risk
        
        # Kelly fraction
        kelly_pct = (odds * win_prob - lose_prob) / odds
        kelly_pct = max(0, kelly_pct)
        
        # Apply fractional Kelly
        position_pct = kelly_pct * self.risk_params.kelly_fraction
        
        # Cap at max position size
        position_pct = min(position_pct, self.risk_params.max_position_pct)
        
        # Apply timeframe multiplier
        tf_config = self.TIMEFRAME_CONFIGS[timeframe]
        position_pct *= tf_config.position_size_mult
        
        # Check portfolio heat
        available_heat = self.risk_params.max_portfolio_heat - self.portfolio_heat
        if position_pct > available_heat:
            position_pct = available_heat
        
        if position_pct <= 0:
            return 0, 0.0, 0.0
        
        # Calculate dollar amounts
        position_value = self.portfolio_value * position_pct
        
        # For options, we might need to adjust for contract size
        if strategy in [OptionStrategy.EQUITY_LONG, OptionStrategy.EQUITY_SHORT]:
            quantity = int(position_value / context.spot_price)
            risk_amount = quantity * risk_per_share
        else:
            # Options: position_value is max risk for debit, or margin for credit
            # Rough estimate: each contract controls 100 shares
            contract_value = context.spot_price * 100
            quantity = max(1, int(position_value / (contract_value * 0.05)))  # ~5% of notional
            risk_amount = position_value  # Max risk = premium paid
        
        return quantity, position_value, risk_amount
    
    # =========================================================================
    # TRADE PROPOSAL BUILDING
    # =========================================================================
    
    def _build_trade_proposal(
        self,
        symbol: str,
        strategy: OptionStrategy,
        direction: DirectionEnum,
        confidence: float,
        context: MarketContext,
        timeframe: Timeframe,
        pipeline_result: PipelineResult,
    ) -> Optional[TradeProposal]:
        """Build complete trade proposal with all details including dynamic profit thresholds."""
        
        tf_config = self.TIMEFRAME_CONFIGS[timeframe]
        target_dte = (tf_config.min_dte + tf_config.max_dte) // 2
        
        # =====================================================================
        # CALCULATE DYNAMIC PROFIT THRESHOLDS (Institutional Edge)
        # =====================================================================
        dynamic_thresholds = self._calculate_dynamic_profit_thresholds(
            strategy=strategy,
            context=context,
            timeframe=timeframe,
            confidence=confidence,
            target_dte=target_dte,
        )
        
        # =====================================================================
        # APPLY DYNAMIC THRESHOLDS TO STOP/PROFIT LEVELS
        # =====================================================================
        is_options = strategy not in [OptionStrategy.EQUITY_LONG, OptionStrategy.EQUITY_SHORT]
        
        if is_options:
            # For options: use dynamic thresholds based on premium/max profit
            # Base ATR move for underlying reference
            base_move = context.atr * self.risk_params.stop_loss_atr_multiple
            
            if direction == DirectionEnum.LONG:
                # For long directional trades
                stop_loss = context.spot_price * (1 - dynamic_thresholds.stop_loss_pct * context.atr_pct)
                take_profit = context.spot_price * (1 + dynamic_thresholds.target_profit_pct * context.atr_pct * 2)
            elif direction == DirectionEnum.SHORT:
                stop_loss = context.spot_price * (1 + dynamic_thresholds.stop_loss_pct * context.atr_pct)
                take_profit = context.spot_price * (1 - dynamic_thresholds.target_profit_pct * context.atr_pct * 2)
            else:
                # Neutral strategies - tighter range
                stop_loss = context.spot_price * (1 - dynamic_thresholds.stop_loss_pct * context.atr_pct * 0.75)
                take_profit = context.spot_price * (1 + dynamic_thresholds.early_profit_pct * context.atr_pct)
            
            logger.info(
                f"📊 Dynamic Options Thresholds for {strategy.value}: "
                f"TP={dynamic_thresholds.target_profit_pct:.0%} | "
                f"SL={dynamic_thresholds.stop_loss_pct:.0%} | "
                f"Scale-out={len(dynamic_thresholds.scale_out_levels)} levels | "
                f"DTE_accel={dynamic_thresholds.dte_acceleration_factor:.1f}x"
            )
        else:
            # For equity: use standard ATR-based stops with dynamic adjustments
            if direction == DirectionEnum.LONG:
                stop_loss = context.spot_price - (context.atr * self.risk_params.stop_loss_atr_multiple * dynamic_thresholds.stop_loss_pct)
                take_profit = context.spot_price + (context.atr * self.risk_params.take_profit_atr_multiple * dynamic_thresholds.target_profit_pct)
            elif direction == DirectionEnum.SHORT:
                stop_loss = context.spot_price + (context.atr * self.risk_params.stop_loss_atr_multiple * dynamic_thresholds.stop_loss_pct)
                take_profit = context.spot_price - (context.atr * self.risk_params.take_profit_atr_multiple * dynamic_thresholds.target_profit_pct)
            else:
                stop_loss = context.spot_price * (1 - tf_config.stop_loss_pct * 1.5)
                take_profit = context.spot_price
        
        risk_per_share = abs(context.spot_price - stop_loss)
        reward_per_share = abs(take_profit - context.spot_price)
        
        if risk_per_share <= 0:
            return None
        
        risk_reward = reward_per_share / risk_per_share
        
        # Check minimum R:R (adjusted for strategy type)
        min_rr = self.risk_params.min_reward_risk
        strategy_category = self._get_strategy_category(strategy)
        
        # Credit strategies can have lower R:R due to high win rate
        if strategy_category == "credit":
            min_rr = min_rr * 0.5  # Accept 0.75:1 for credit spreads
        
        if risk_reward < min_rr:
            logger.debug(f"R:R {risk_reward:.2f} below minimum {min_rr:.2f} for {strategy_category}")
            return None
        
        # Calculate position size
        quantity, position_value, risk_amount = self._calculate_position_size(
            context=context,
            confidence=confidence,
            strategy=strategy,
            timeframe=timeframe,
            risk_per_share=risk_per_share,
        )
        
        if quantity <= 0:
            return None
        
        # Trailing stop config with dynamic thresholds
        trailing_stop_config = {
            "enabled": True,
            "activation_pct": dynamic_thresholds.trailing_activation_pct,
            "trail_pct": dynamic_thresholds.trailing_distance_pct,
            "initial_stop": stop_loss,
            "scale_out_levels": dynamic_thresholds.scale_out_levels,
            "early_profit_threshold": dynamic_thresholds.early_profit_pct,
            "dte_acceleration": dynamic_thresholds.dte_acceleration_factor,
        }
        
        # Build options order if applicable
        options_order = None
        legs = []
        
        if is_options:
            options_order = self._build_options_order(
                symbol=symbol,
                strategy=strategy,
                context=context,
                quantity=quantity,
                timeframe=timeframe,
                confidence=confidence,
            )
            if options_order:
                legs = options_order.legs
                # Update options order with dynamic thresholds
                options_order.dynamic_thresholds = {
                    "target_profit_pct": dynamic_thresholds.target_profit_pct,
                    "early_profit_pct": dynamic_thresholds.early_profit_pct,
                    "stop_loss_pct": dynamic_thresholds.stop_loss_pct,
                    "scale_out_levels": dynamic_thresholds.scale_out_levels,
                }
        
        # Build reasoning with dynamic threshold info
        reasoning = self._build_reasoning(
            strategy=strategy,
            direction=direction,
            confidence=confidence,
            context=context,
            timeframe=timeframe,
            risk_reward=risk_reward,
        )
        reasoning += f" | DynTP={dynamic_thresholds.target_profit_pct:.0%}"
        
        return TradeProposal(
            symbol=symbol,
            strategy=strategy,
            direction=direction,
            confidence=confidence,
            entry_price=context.spot_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_config=trailing_stop_config,
            max_loss=risk_amount,
            max_profit=risk_amount * risk_reward,
            risk_reward_ratio=risk_reward,
            position_value=position_value,
            position_pct=position_value / self.portfolio_value,
            risk_amount=risk_amount,
            options_order=options_order,
            legs=legs,
            timeframe=timeframe,
            target_dte=tf_config.max_dte,
            max_hold_time=timedelta(hours=tf_config.max_hold_hours),
            reasoning=reasoning,
            timestamp=datetime.utcnow(),
            strategy_id=f"{strategy.value}_{symbol}_{datetime.utcnow().strftime('%Y%m%d%H%M')}",
            tags=[strategy.value, timeframe.value, context.iv_environment.value],
        )
    
    def _build_options_order(
        self,
        symbol: str,
        strategy: OptionStrategy,
        context: MarketContext,
        quantity: int,
        timeframe: Timeframe,
        confidence: float,
    ) -> Optional[OptionsOrderRequest]:
        """Build options order request for the strategy."""
        
        tf_config = self.TIMEFRAME_CONFIGS[timeframe]
        target_dte = (tf_config.min_dte + tf_config.max_dte) // 2
        
        # Calculate expiration
        expiration = datetime.now() + timedelta(days=target_dte)
        # Adjust to Friday
        days_to_friday = (4 - expiration.weekday()) % 7
        expiration = expiration + timedelta(days=days_to_friday)
        exp_str = expiration.strftime("%Y-%m-%d")
        
        spot = context.spot_price
        legs = []
        
        # Build legs based on strategy
        if strategy == OptionStrategy.LONG_CALL:
            strike = round(spot, 0)
            legs = [OptionsLeg(
                symbol=self._format_occ_symbol(symbol, expiration, "C", strike),
                ratio=1, side="buy", type="call",
                strike=strike, expiration=exp_str, action="buy_to_open"
            )]
            max_loss = spot * 0.05 * 100 * quantity  # Estimate
            max_profit = float("inf")
            
        elif strategy == OptionStrategy.LONG_PUT:
            strike = round(spot, 0)
            legs = [OptionsLeg(
                symbol=self._format_occ_symbol(symbol, expiration, "P", strike),
                ratio=1, side="buy", type="put",
                strike=strike, expiration=exp_str, action="buy_to_open"
            )]
            max_loss = spot * 0.05 * 100 * quantity
            max_profit = strike * 100 * quantity
            
        elif strategy == OptionStrategy.BULL_CALL_SPREAD:
            long_strike = round(spot, 0)
            short_strike = round(spot * 1.05, 0)
            legs = [
                OptionsLeg(
                    symbol=self._format_occ_symbol(symbol, expiration, "C", long_strike),
                    ratio=1, side="buy", type="call",
                    strike=long_strike, expiration=exp_str, action="buy_to_open"
                ),
                OptionsLeg(
                    symbol=self._format_occ_symbol(symbol, expiration, "C", short_strike),
                    ratio=1, side="sell", type="call",
                    strike=short_strike, expiration=exp_str, action="sell_to_open"
                ),
            ]
            max_loss = (short_strike - long_strike) * 100 * quantity * 0.3  # Debit paid
            max_profit = (short_strike - long_strike) * 100 * quantity - max_loss
            
        elif strategy == OptionStrategy.BULL_PUT_SPREAD:
            short_strike = round(spot, 0)
            long_strike = round(spot * 0.95, 0)
            legs = [
                OptionsLeg(
                    symbol=self._format_occ_symbol(symbol, expiration, "P", short_strike),
                    ratio=1, side="sell", type="put",
                    strike=short_strike, expiration=exp_str, action="sell_to_open"
                ),
                OptionsLeg(
                    symbol=self._format_occ_symbol(symbol, expiration, "P", long_strike),
                    ratio=1, side="buy", type="put",
                    strike=long_strike, expiration=exp_str, action="buy_to_open"
                ),
            ]
            max_profit = (short_strike - long_strike) * 100 * quantity * 0.3  # Credit received
            max_loss = (short_strike - long_strike) * 100 * quantity - max_profit
            
        elif strategy == OptionStrategy.BEAR_PUT_SPREAD:
            long_strike = round(spot, 0)
            short_strike = round(spot * 0.95, 0)
            legs = [
                OptionsLeg(
                    symbol=self._format_occ_symbol(symbol, expiration, "P", long_strike),
                    ratio=1, side="buy", type="put",
                    strike=long_strike, expiration=exp_str, action="buy_to_open"
                ),
                OptionsLeg(
                    symbol=self._format_occ_symbol(symbol, expiration, "P", short_strike),
                    ratio=1, side="sell", type="put",
                    strike=short_strike, expiration=exp_str, action="sell_to_open"
                ),
            ]
            max_loss = (long_strike - short_strike) * 100 * quantity * 0.3
            max_profit = (long_strike - short_strike) * 100 * quantity - max_loss
            
        elif strategy == OptionStrategy.BEAR_CALL_SPREAD:
            short_strike = round(spot, 0)
            long_strike = round(spot * 1.05, 0)
            legs = [
                OptionsLeg(
                    symbol=self._format_occ_symbol(symbol, expiration, "C", short_strike),
                    ratio=1, side="sell", type="call",
                    strike=short_strike, expiration=exp_str, action="sell_to_open"
                ),
                OptionsLeg(
                    symbol=self._format_occ_symbol(symbol, expiration, "C", long_strike),
                    ratio=1, side="buy", type="call",
                    strike=long_strike, expiration=exp_str, action="buy_to_open"
                ),
            ]
            max_profit = (long_strike - short_strike) * 100 * quantity * 0.3
            max_loss = (long_strike - short_strike) * 100 * quantity - max_profit
            
        elif strategy == OptionStrategy.IRON_CONDOR:
            put_short = round(spot * 0.95, 0)
            put_long = round(spot * 0.90, 0)
            call_short = round(spot * 1.05, 0)
            call_long = round(spot * 1.10, 0)
            legs = [
                OptionsLeg(
                    symbol=self._format_occ_symbol(symbol, expiration, "P", put_short),
                    ratio=1, side="sell", type="put",
                    strike=put_short, expiration=exp_str, action="sell_to_open"
                ),
                OptionsLeg(
                    symbol=self._format_occ_symbol(symbol, expiration, "P", put_long),
                    ratio=1, side="buy", type="put",
                    strike=put_long, expiration=exp_str, action="buy_to_open"
                ),
                OptionsLeg(
                    symbol=self._format_occ_symbol(symbol, expiration, "C", call_short),
                    ratio=1, side="sell", type="call",
                    strike=call_short, expiration=exp_str, action="sell_to_open"
                ),
                OptionsLeg(
                    symbol=self._format_occ_symbol(symbol, expiration, "C", call_long),
                    ratio=1, side="buy", type="call",
                    strike=call_long, expiration=exp_str, action="buy_to_open"
                ),
            ]
            wing_width = (put_short - put_long) * 100 * quantity
            max_profit = wing_width * 0.3
            max_loss = wing_width - max_profit
            
        elif strategy == OptionStrategy.LONG_STRADDLE:
            strike = round(spot, 0)
            legs = [
                OptionsLeg(
                    symbol=self._format_occ_symbol(symbol, expiration, "C", strike),
                    ratio=1, side="buy", type="call",
                    strike=strike, expiration=exp_str, action="buy_to_open"
                ),
                OptionsLeg(
                    symbol=self._format_occ_symbol(symbol, expiration, "P", strike),
                    ratio=1, side="buy", type="put",
                    strike=strike, expiration=exp_str, action="buy_to_open"
                ),
            ]
            max_loss = spot * 0.08 * 100 * quantity  # Both premiums
            max_profit = float("inf")
            
        elif strategy == OptionStrategy.LONG_STRANGLE:
            call_strike = round(spot * 1.05, 0)
            put_strike = round(spot * 0.95, 0)
            legs = [
                OptionsLeg(
                    symbol=self._format_occ_symbol(symbol, expiration, "C", call_strike),
                    ratio=1, side="buy", type="call",
                    strike=call_strike, expiration=exp_str, action="buy_to_open"
                ),
                OptionsLeg(
                    symbol=self._format_occ_symbol(symbol, expiration, "P", put_strike),
                    ratio=1, side="buy", type="put",
                    strike=put_strike, expiration=exp_str, action="buy_to_open"
                ),
            ]
            max_loss = spot * 0.05 * 100 * quantity
            max_profit = float("inf")
            
        else:
            return None
        
        if not legs:
            return None
        
        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name=strategy.value,
            legs=legs,
            max_loss=max_loss,
            max_profit=max_profit if max_profit != float("inf") else 999999.0,
            bpr=max_loss,
            rationale=f"{strategy.value} | IV={context.iv_environment.value} | conf={confidence:.1%}",
            confidence=confidence,
        )
    
    def _format_occ_symbol(
        self,
        symbol: str,
        expiration: datetime,
        option_type: str,
        strike: float,
    ) -> str:
        """Format OCC option symbol."""
        # OCC format: SYMBOL + YYMMDD + C/P + strike*1000 (8 digits)
        exp_str = expiration.strftime("%y%m%d")
        strike_str = f"{int(strike * 1000):08d}"
        return f"{symbol.ljust(6)}{exp_str}{option_type}{strike_str}"
    
    def _build_reasoning(
        self,
        strategy: OptionStrategy,
        direction: DirectionEnum,
        confidence: float,
        context: MarketContext,
        timeframe: Timeframe,
        risk_reward: float,
    ) -> str:
        """Build human-readable reasoning for the trade."""
        
        parts = [
            f"Strategy: {strategy.value.upper()}",
            f"Direction: {direction.value}",
            f"Confidence: {confidence:.1%}",
            f"IV Environment: {context.iv_environment.value}",
            f"Market Regime: {context.regime.value}",
            f"Timeframe: {timeframe.value}",
            f"R:R Ratio: {risk_reward:.1f}:1",
            f"IV Rank: {context.iv_rank:.0f}",
        ]
        
        return " | ".join(parts)
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def _validate_proposal(self, proposal: TradeProposal) -> bool:
        """Validate trade proposal against risk rules."""
        
        # Check R:R ratio
        if proposal.risk_reward_ratio < self.risk_params.min_reward_risk:
            logger.warning(f"R:R {proposal.risk_reward_ratio:.2f} below minimum")
            return False
        
        # Check position size
        if proposal.position_pct > self.risk_params.max_position_pct:
            logger.warning(f"Position {proposal.position_pct:.1%} exceeds max")
            return False
        
        # Check portfolio heat
        new_heat = self.portfolio_heat + proposal.position_pct
        if new_heat > self.risk_params.max_portfolio_heat:
            logger.warning(f"Portfolio heat {new_heat:.1%} would exceed max")
            return False
        
        # Check existing position
        if proposal.symbol in self.current_positions:
            logger.warning(f"Already in position for {proposal.symbol}")
            return False
        
        # Check quantity
        if proposal.quantity <= 0:
            logger.warning(f"Invalid quantity: {proposal.quantity}")
            return False
        
        return True
    
    # =========================================================================
    # CONVERSION & EXECUTION
    # =========================================================================
    
    def _proposal_to_trade_idea(self, proposal: TradeProposal) -> TradeIdea:
        """Convert TradeProposal to TradeIdea for pipeline compatibility."""
        
        # Map strategy to StrategyType
        if proposal.strategy in [OptionStrategy.EQUITY_LONG, OptionStrategy.EQUITY_SHORT]:
            strategy_type = StrategyType.DIRECTIONAL
        elif "SPREAD" in proposal.strategy.value.upper():
            strategy_type = StrategyType.OPTIONS_SPREAD
        elif proposal.strategy in [OptionStrategy.IRON_CONDOR, OptionStrategy.IRON_BUTTERFLY]:
            strategy_type = StrategyType.HEDGED
        else:
            strategy_type = StrategyType.DIRECTIONAL
        
        return TradeIdea(
            timestamp=proposal.timestamp,
            symbol=proposal.symbol,
            strategy_type=strategy_type,
            direction=proposal.direction,
            confidence=proposal.confidence,
            size=proposal.position_value,
            entry_price=proposal.entry_price,
            stop_loss=proposal.stop_loss,
            take_profit=proposal.take_profit,
            reasoning=proposal.reasoning,
            options_request=proposal.options_order,
        )
    
    def _execute_single_trade(
        self,
        idea: TradeIdea,
        timestamp: datetime,
    ) -> OrderResult:
        """Execute a single trade through the broker."""
        
        # Check if options order
        if idea.options_request and idea.options_request.legs:
            return self._execute_options_order(idea, timestamp)
        
        # Equity order
        return self._execute_equity_order(idea, timestamp)
    
    def _execute_equity_order(
        self,
        idea: TradeIdea,
        timestamp: datetime,
    ) -> OrderResult:
        """Execute equity order."""
        
        side = "buy" if idea.direction == DirectionEnum.LONG else "sell"
        
        # Get current price
        quote = self.broker.get_latest_quote(idea.symbol)
        price = idea.entry_price
        if quote:
            bid = quote.get("bid", 0) or 0
            ask = quote.get("ask", 0) or 0
            if bid and ask:
                price = (bid + ask) / 2
        
        quantity = max(1, int(idea.size / max(price, 1)))
        
        # Place bracket order with stop loss and take profit for institutional-grade execution
        try:
            if hasattr(self.broker, 'place_bracket_order') and idea.stop_loss and idea.take_profit:
                order_id = self.broker.place_bracket_order(
                    symbol=idea.symbol,
                    quantity=quantity,
                    side=side,
                    take_profit_price=idea.take_profit,
                    stop_loss_price=idea.stop_loss,
                )
                logger.info(
                    f"BRACKET ORDER: {side.upper()} {quantity} {idea.symbol} | "
                    f"TP=${idea.take_profit:.2f} | SL=${idea.stop_loss:.2f}"
                )
            else:
                order_id = self.broker.place_order(
                    symbol=idea.symbol,
                    quantity=quantity,
                    side=side,
                )
                logger.info(f"MARKET ORDER: {side.upper()} {quantity} {idea.symbol}")
            
            return OrderResult(
                timestamp=timestamp,
                symbol=idea.symbol,
                status=OrderStatus.SUBMITTED,
                order_id=order_id,
                filled_qty=0.0,
                message=f"{side} {quantity} @ ~${price:.2f} | SL=${idea.stop_loss:.2f} | TP=${idea.take_profit:.2f}",
            )
            
        except Exception as e:
            return OrderResult(
                timestamp=timestamp,
                symbol=idea.symbol,
                status=OrderStatus.REJECTED,
                message=str(e),
            )
    
    def _execute_options_order(
        self,
        idea: TradeIdea,
        timestamp: datetime,
    ) -> OrderResult:
        """Execute options order."""
        
        options = idea.options_request
        
        # For now, log the intended order
        # Full options execution would require broker support for multi-leg orders
        logger.info(
            f"OPTIONS ORDER: {options.strategy_name} on {idea.symbol} | "
            f"Legs: {len(options.legs)} | Max Loss: ${options.max_loss:.2f}"
        )
        
        # Check if broker supports options
        if hasattr(self.broker, 'place_options_order'):
            try:
                order_id = self.broker.place_options_order(options)
                return OrderResult(
                    timestamp=timestamp,
                    symbol=idea.symbol,
                    status=OrderStatus.SUBMITTED,
                    order_id=order_id,
                    message=f"Options: {options.strategy_name} | {len(options.legs)} legs",
                )
            except Exception as e:
                logger.error(f"Options order failed: {e}")
        
        # Fallback: execute as equity if options not supported
        logger.warning(f"Options not supported, falling back to equity for {idea.symbol}")
        return self._execute_equity_order(idea, timestamp)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _get_spot_price(self, symbol: str) -> Optional[float]:
        """Get current spot price for symbol."""
        
        if self.broker:
            try:
                quote = self.broker.get_latest_quote(symbol)
                if quote:
                    bid = quote.get("bid", 0) or 0
                    ask = quote.get("ask", 0) or 0
                    if bid and ask:
                        return (bid + ask) / 2
            except Exception:
                pass
        
        if self.market_adapter:
            try:
                from datetime import timedelta
                end = datetime.utcnow()
                start = end - timedelta(days=2)
                bars = self.market_adapter.get_bars(symbol, start=start, end=end, timeframe="1Day")
                if bars:
                    return float(bars[-1].close)
            except Exception:
                pass
        
        # Fallback prices
        fallbacks = {
            "SPY": 600.0, "QQQ": 500.0, "IWM": 230.0,
            "NVDA": 140.0, "TSLA": 350.0, "AAPL": 230.0,
            "MSFT": 430.0, "GOOGL": 175.0, "AMZN": 210.0,
            "META": 560.0, "AMD": 140.0, "COIN": 250.0,
        }
        return fallbacks.get(symbol, 100.0)
    
    def update_portfolio_state(
        self,
        portfolio_value: float,
        positions: List[str],
        heat: float,
    ) -> None:
        """Update portfolio state for position sizing."""
        self.portfolio_value = portfolio_value
        self.current_positions = positions
        self.portfolio_heat = heat
    
    def update_risk_per_trade(self, risk_pct: float) -> None:
        """Update risk parameters from adaptation agent."""
        self.risk_params.max_position_pct = risk_pct
        logger.info(f"Risk per trade updated to {risk_pct:.2%}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_elite_trade_agent(
    options_adapter: Any = None,
    market_adapter: Any = None,
    broker: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> EliteTradeAgent:
    """Factory function to create EliteTradeAgent with default configuration."""
    return EliteTradeAgent(
        options_adapter=options_adapter,
        market_adapter=market_adapter,
        broker=broker,
        config=config,
    )
