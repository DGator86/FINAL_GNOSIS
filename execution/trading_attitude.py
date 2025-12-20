"""Trading Attitude System - Adaptive trading parameters based on market conditions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class TradingAttitude(str, Enum):
    """
    Trading attitude profiles based on volatility regime and timeframe.

    Each attitude represents a complete set of trading parameters optimized
    for specific market conditions.
    """
    SCALPER = "scalper"           # High vol, short timeframe (0-15 min)
    DAY_TRADER = "day_trader"     # Medium vol, intraday (15 min - 4 hr)
    SWING_TRADER = "swing_trader" # Low-medium vol, multi-day (1-5 days)
    POSITION_TRADER = "position_trader"  # Low vol, multi-week


class VolatilityRegime(str, Enum):
    """Market volatility regime classification."""
    VERY_LOW = "very_low"     # IV < 15% or VIX < 12
    LOW = "low"               # IV 15-20% or VIX 12-16
    MODERATE = "moderate"     # IV 20-30% or VIX 16-22
    HIGH = "high"             # IV 30-45% or VIX 22-30
    EXTREME = "extreme"       # IV > 45% or VIX > 30


@dataclass
class AttitudeProfile:
    """
    Complete trading parameter profile for an attitude.

    All parameters are calibrated for the specific trading style.
    """
    attitude: TradingAttitude

    # Position Sizing
    max_position_size_pct: float = 5.0       # Max % of portfolio per position
    risk_per_trade_pct: float = 1.0          # % of portfolio risked per trade
    max_positions: int = 5                    # Maximum concurrent positions

    # Entry Requirements
    min_confidence: float = 0.5               # Minimum signal confidence
    min_mtf_alignment: float = 0.4            # Minimum MTF alignment score
    require_mtf_agreement: bool = False       # Require all timeframes agree

    # Stop Loss & Targets
    initial_stop_pct: float = 1.0             # Initial stop loss %
    trailing_stop_pct: float = 0.5            # Trailing stop distance %
    target_1_pct: float = 0.5                 # First profit target %
    target_2_pct: float = 1.0                 # Second profit target %
    partial_exit_at_target_1: bool = True     # Take partial profits
    partial_exit_pct: float = 0.5             # % to exit at target 1

    # Time Management
    max_hold_minutes: int = 60                # Maximum position duration
    time_stop_enabled: bool = True            # Use time-based exits
    exit_before_close_minutes: int = 5        # Exit before market close

    # Trailing Stop Behavior
    trailing_activation_pct: float = 0.3      # Profit % to activate trailing
    breakeven_trigger_pct: float = 0.5        # Profit % to move to breakeven
    tighten_after_target_1: bool = True       # Tighten trail after T1
    tightened_trail_pct: float = 0.25         # Tightened trailing %

    # Options Selection
    preferred_dte_min: int = 0                # Minimum DTE for options
    preferred_dte_max: int = 7                # Maximum DTE for options
    prefer_spreads: bool = False              # Prefer defined risk spreads
    delta_range: Tuple[float, float] = (0.30, 0.50)  # Target delta range

    # Signal Filters
    volume_filter_enabled: bool = True        # Require volume confirmation
    volatility_filter_enabled: bool = True    # Check vol regime match
    liquidity_min_score: float = 0.3          # Minimum liquidity score

    # Aggressiveness Metrics
    aggressiveness: float = 0.5               # 0.0 = conservative, 1.0 = aggressive
    bias_amplification: float = 1.0           # Multiply directional bias
    confidence_threshold_scaling: float = 1.0  # Scale confidence requirements

    def describe(self) -> str:
        """Return human-readable description of the profile."""
        return (
            f"{self.attitude.value.upper()}: "
            f"Aggr={self.aggressiveness:.0%}, "
            f"Stop={self.initial_stop_pct:.1f}%, "
            f"Trail={self.trailing_stop_pct:.1f}%, "
            f"Hold≤{self.max_hold_minutes}min, "
            f"DTE={self.preferred_dte_min}-{self.preferred_dte_max}"
        )


# =============================================================================
# Pre-defined Attitude Profiles
# =============================================================================

SCALPER_PROFILE = AttitudeProfile(
    attitude=TradingAttitude.SCALPER,

    # Aggressive sizing for quick trades
    max_position_size_pct=3.0,
    risk_per_trade_pct=0.5,
    max_positions=3,

    # Higher confidence needed for scalps
    min_confidence=0.65,
    min_mtf_alignment=0.5,
    require_mtf_agreement=False,

    # Tight stops for scalping
    initial_stop_pct=0.3,
    trailing_stop_pct=0.15,
    target_1_pct=0.2,
    target_2_pct=0.4,
    partial_exit_at_target_1=True,
    partial_exit_pct=0.5,

    # Very short holds
    max_hold_minutes=15,
    time_stop_enabled=True,
    exit_before_close_minutes=15,

    # Quick trailing activation
    trailing_activation_pct=0.1,
    breakeven_trigger_pct=0.2,
    tighten_after_target_1=True,
    tightened_trail_pct=0.1,

    # 0DTE options preferred
    preferred_dte_min=0,
    preferred_dte_max=1,
    prefer_spreads=False,  # Single leg for gamma
    delta_range=(0.40, 0.60),  # Higher delta for scalps

    # Aggressive signals
    aggressiveness=0.9,
    bias_amplification=1.3,
    confidence_threshold_scaling=0.8,
)

DAY_TRADER_PROFILE = AttitudeProfile(
    attitude=TradingAttitude.DAY_TRADER,

    # Moderate sizing
    max_position_size_pct=5.0,
    risk_per_trade_pct=1.0,
    max_positions=5,

    # Standard confidence
    min_confidence=0.5,
    min_mtf_alignment=0.4,
    require_mtf_agreement=False,

    # Balanced stops
    initial_stop_pct=0.75,
    trailing_stop_pct=0.4,
    target_1_pct=0.5,
    target_2_pct=1.0,
    partial_exit_at_target_1=True,
    partial_exit_pct=0.5,

    # Intraday holds
    max_hold_minutes=120,
    time_stop_enabled=True,
    exit_before_close_minutes=10,

    # Standard trailing
    trailing_activation_pct=0.3,
    breakeven_trigger_pct=0.5,
    tighten_after_target_1=True,
    tightened_trail_pct=0.25,

    # Short-dated options
    preferred_dte_min=1,
    preferred_dte_max=7,
    prefer_spreads=False,
    delta_range=(0.35, 0.50),

    # Balanced aggression
    aggressiveness=0.6,
    bias_amplification=1.0,
    confidence_threshold_scaling=1.0,
)

SWING_TRADER_PROFILE = AttitudeProfile(
    attitude=TradingAttitude.SWING_TRADER,

    # Conservative sizing for multi-day
    max_position_size_pct=7.0,
    risk_per_trade_pct=1.5,
    max_positions=4,

    # Lower confidence OK for swings
    min_confidence=0.45,
    min_mtf_alignment=0.5,
    require_mtf_agreement=True,  # Want alignment for swings

    # Wider stops for multi-day
    initial_stop_pct=2.0,
    trailing_stop_pct=1.0,
    target_1_pct=1.5,
    target_2_pct=3.0,
    partial_exit_at_target_1=True,
    partial_exit_pct=0.5,

    # Multi-day holds
    max_hold_minutes=60 * 24 * 3,  # 3 days
    time_stop_enabled=False,
    exit_before_close_minutes=0,

    # Patient trailing
    trailing_activation_pct=0.75,
    breakeven_trigger_pct=1.0,
    tighten_after_target_1=True,
    tightened_trail_pct=0.5,

    # Longer-dated options
    preferred_dte_min=14,
    preferred_dte_max=45,
    prefer_spreads=True,  # Defined risk for overnights
    delta_range=(0.30, 0.45),

    # Conservative aggression
    aggressiveness=0.4,
    bias_amplification=0.8,
    confidence_threshold_scaling=1.2,
)

POSITION_TRADER_PROFILE = AttitudeProfile(
    attitude=TradingAttitude.POSITION_TRADER,

    # Larger positions for position trades
    max_position_size_pct=10.0,
    risk_per_trade_pct=2.0,
    max_positions=3,

    # Very selective
    min_confidence=0.6,
    min_mtf_alignment=0.7,
    require_mtf_agreement=True,

    # Wide stops for position trades
    initial_stop_pct=5.0,
    trailing_stop_pct=2.5,
    target_1_pct=5.0,
    target_2_pct=10.0,
    partial_exit_at_target_1=True,
    partial_exit_pct=0.33,

    # Multi-week holds
    max_hold_minutes=60 * 24 * 21,  # 3 weeks
    time_stop_enabled=False,
    exit_before_close_minutes=0,

    # Very patient trailing
    trailing_activation_pct=2.0,
    breakeven_trigger_pct=3.0,
    tighten_after_target_1=True,
    tightened_trail_pct=1.5,

    # LEAPS-style options
    preferred_dte_min=45,
    preferred_dte_max=180,
    prefer_spreads=True,
    delta_range=(0.50, 0.70),  # ITM for position trades

    # Very conservative
    aggressiveness=0.2,
    bias_amplification=0.6,
    confidence_threshold_scaling=1.5,
)

# Profile mapping
ATTITUDE_PROFILES: Dict[TradingAttitude, AttitudeProfile] = {
    TradingAttitude.SCALPER: SCALPER_PROFILE,
    TradingAttitude.DAY_TRADER: DAY_TRADER_PROFILE,
    TradingAttitude.SWING_TRADER: SWING_TRADER_PROFILE,
    TradingAttitude.POSITION_TRADER: POSITION_TRADER_PROFILE,
}


# =============================================================================
# Attitude Selector - Dynamic selection based on conditions
# =============================================================================

@dataclass
class MarketConditions:
    """Current market conditions for attitude selection."""
    volatility: float = 0.0            # Current volatility (annualized)
    volatility_regime: str = "moderate"  # Regime classification
    vix: Optional[float] = None        # VIX level if available
    iv_rank: Optional[float] = None    # IV percentile (0-100)
    trend_strength: float = 0.0        # -1 to +1
    liquidity_score: float = 0.5       # 0 to 1
    time_to_close_minutes: Optional[int] = None
    dominant_timeframe: str = ""       # From MTF analysis

    def get_volatility_regime(self) -> VolatilityRegime:
        """Classify volatility into regime."""
        if self.vix is not None:
            if self.vix < 12:
                return VolatilityRegime.VERY_LOW
            elif self.vix < 16:
                return VolatilityRegime.LOW
            elif self.vix < 22:
                return VolatilityRegime.MODERATE
            elif self.vix < 30:
                return VolatilityRegime.HIGH
            else:
                return VolatilityRegime.EXTREME

        # Fallback to IV-based classification
        if self.volatility < 0.15:
            return VolatilityRegime.VERY_LOW
        elif self.volatility < 0.20:
            return VolatilityRegime.LOW
        elif self.volatility < 0.30:
            return VolatilityRegime.MODERATE
        elif self.volatility < 0.45:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME


class AttitudeSelector:
    """
    Selects and adjusts trading attitude based on market conditions.

    The selector considers:
    1. Volatility regime (high vol = scalping, low vol = swing)
    2. Dominant timeframe from MTF analysis
    3. Time to market close
    4. Liquidity conditions
    """

    # Timeframe to base attitude mapping
    TIMEFRAME_ATTITUDE_MAP = {
        "1Min": TradingAttitude.SCALPER,
        "5Min": TradingAttitude.SCALPER,
        "15Min": TradingAttitude.DAY_TRADER,
        "30Min": TradingAttitude.DAY_TRADER,
        "1Hour": TradingAttitude.DAY_TRADER,
        "4Hour": TradingAttitude.SWING_TRADER,
        "1Day": TradingAttitude.SWING_TRADER,
        "1Week": TradingAttitude.POSITION_TRADER,
    }

    # Volatility regime modifiers
    VOL_REGIME_MODIFIERS = {
        VolatilityRegime.VERY_LOW: -1,   # Shift toward longer-term
        VolatilityRegime.LOW: 0,
        VolatilityRegime.MODERATE: 0,
        VolatilityRegime.HIGH: 1,        # Shift toward shorter-term
        VolatilityRegime.EXTREME: 2,     # Strong shift to scalping
    }

    ATTITUDE_ORDER = [
        TradingAttitude.POSITION_TRADER,
        TradingAttitude.SWING_TRADER,
        TradingAttitude.DAY_TRADER,
        TradingAttitude.SCALPER,
    ]

    def __init__(self, default_attitude: TradingAttitude = TradingAttitude.DAY_TRADER):
        self.default_attitude = default_attitude
        self.current_attitude: Optional[TradingAttitude] = None
        self.current_profile: Optional[AttitudeProfile] = None
        self.override_attitude: Optional[TradingAttitude] = None  # Manual override

        logger.info(f"AttitudeSelector initialized (default: {default_attitude.value})")

    def select_attitude(
        self,
        conditions: MarketConditions,
        user_preference: Optional[TradingAttitude] = None,
    ) -> AttitudeProfile:
        """
        Select the optimal trading attitude based on conditions.

        Args:
            conditions: Current market conditions
            user_preference: Optional user override

        Returns:
            AttitudeProfile with all trading parameters
        """
        # Check for manual override
        if self.override_attitude:
            attitude = self.override_attitude
            logger.debug(f"Using override attitude: {attitude.value}")
        elif user_preference:
            attitude = user_preference
            logger.debug(f"Using user preference: {attitude.value}")
        else:
            attitude = self._calculate_optimal_attitude(conditions)

        # Get base profile
        profile = self._get_adjusted_profile(attitude, conditions)

        # Update state
        self.current_attitude = attitude
        self.current_profile = profile

        logger.info(
            f"Attitude selected: {attitude.value} | "
            f"Vol regime: {conditions.get_volatility_regime().value} | "
            f"{profile.describe()}"
        )

        return profile

    def _calculate_optimal_attitude(self, conditions: MarketConditions) -> TradingAttitude:
        """Calculate optimal attitude from market conditions."""

        # Start with timeframe-based attitude
        base_attitude = self.TIMEFRAME_ATTITUDE_MAP.get(
            conditions.dominant_timeframe,
            self.default_attitude
        )

        # Get volatility modifier
        vol_regime = conditions.get_volatility_regime()
        vol_modifier = self.VOL_REGIME_MODIFIERS.get(vol_regime, 0)

        # Calculate final attitude index
        base_idx = self.ATTITUDE_ORDER.index(base_attitude)
        final_idx = base_idx + vol_modifier

        # Clamp to valid range
        final_idx = max(0, min(len(self.ATTITUDE_ORDER) - 1, final_idx))

        # Special cases
        if conditions.time_to_close_minutes is not None:
            if conditions.time_to_close_minutes < 30:
                # Near close - shift to scalping
                final_idx = max(final_idx, self.ATTITUDE_ORDER.index(TradingAttitude.SCALPER) - 1)

        if conditions.liquidity_score < 0.3:
            # Low liquidity - avoid scalping
            if final_idx == len(self.ATTITUDE_ORDER) - 1:
                final_idx -= 1

        return self.ATTITUDE_ORDER[final_idx]

    def _get_adjusted_profile(
        self,
        attitude: TradingAttitude,
        conditions: MarketConditions,
    ) -> AttitudeProfile:
        """
        Get profile with dynamic adjustments based on conditions.

        Adjusts stops, targets, and sizing based on current volatility.
        """
        base_profile = ATTITUDE_PROFILES[attitude]

        # Create a copy to modify
        profile = AttitudeProfile(
            attitude=base_profile.attitude,
            max_position_size_pct=base_profile.max_position_size_pct,
            risk_per_trade_pct=base_profile.risk_per_trade_pct,
            max_positions=base_profile.max_positions,
            min_confidence=base_profile.min_confidence,
            min_mtf_alignment=base_profile.min_mtf_alignment,
            require_mtf_agreement=base_profile.require_mtf_agreement,
            initial_stop_pct=base_profile.initial_stop_pct,
            trailing_stop_pct=base_profile.trailing_stop_pct,
            target_1_pct=base_profile.target_1_pct,
            target_2_pct=base_profile.target_2_pct,
            partial_exit_at_target_1=base_profile.partial_exit_at_target_1,
            partial_exit_pct=base_profile.partial_exit_pct,
            max_hold_minutes=base_profile.max_hold_minutes,
            time_stop_enabled=base_profile.time_stop_enabled,
            exit_before_close_minutes=base_profile.exit_before_close_minutes,
            trailing_activation_pct=base_profile.trailing_activation_pct,
            breakeven_trigger_pct=base_profile.breakeven_trigger_pct,
            tighten_after_target_1=base_profile.tighten_after_target_1,
            tightened_trail_pct=base_profile.tightened_trail_pct,
            preferred_dte_min=base_profile.preferred_dte_min,
            preferred_dte_max=base_profile.preferred_dte_max,
            prefer_spreads=base_profile.prefer_spreads,
            delta_range=base_profile.delta_range,
            volume_filter_enabled=base_profile.volume_filter_enabled,
            volatility_filter_enabled=base_profile.volatility_filter_enabled,
            liquidity_min_score=base_profile.liquidity_min_score,
            aggressiveness=base_profile.aggressiveness,
            bias_amplification=base_profile.bias_amplification,
            confidence_threshold_scaling=base_profile.confidence_threshold_scaling,
        )

        # Dynamic adjustments based on volatility
        vol_regime = conditions.get_volatility_regime()

        if vol_regime == VolatilityRegime.HIGH or vol_regime == VolatilityRegime.EXTREME:
            # High vol: Widen stops, reduce position size
            vol_factor = 1.5 if vol_regime == VolatilityRegime.HIGH else 2.0

            profile.initial_stop_pct *= vol_factor
            profile.trailing_stop_pct *= vol_factor
            profile.target_1_pct *= vol_factor
            profile.target_2_pct *= vol_factor
            profile.max_position_size_pct *= 0.7  # Reduce size
            profile.aggressiveness = min(1.0, profile.aggressiveness * 1.2)  # More aggressive

        elif vol_regime == VolatilityRegime.VERY_LOW:
            # Very low vol: Tighten stops, can increase size
            profile.initial_stop_pct *= 0.7
            profile.trailing_stop_pct *= 0.7
            profile.target_1_pct *= 0.7
            profile.target_2_pct *= 0.7
            profile.max_position_size_pct *= 1.2  # Can take larger positions
            profile.aggressiveness *= 0.8  # More conservative

        # Adjust for liquidity
        if conditions.liquidity_score < 0.5:
            profile.max_position_size_pct *= conditions.liquidity_score * 2
            profile.liquidity_min_score = max(0.4, profile.liquidity_min_score)

        # Adjust for time to close
        if conditions.time_to_close_minutes is not None:
            if conditions.time_to_close_minutes < 60:
                profile.max_hold_minutes = min(
                    profile.max_hold_minutes,
                    conditions.time_to_close_minutes - 5
                )

        return profile

    def set_override(self, attitude: Optional[TradingAttitude]) -> None:
        """Set manual attitude override."""
        self.override_attitude = attitude
        if attitude:
            logger.info(f"Attitude override set: {attitude.value}")
        else:
            logger.info("Attitude override cleared")

    def get_current_profile(self) -> Optional[AttitudeProfile]:
        """Get the currently active profile."""
        return self.current_profile

    def get_attitude_for_timeframe(self, timeframe: str) -> TradingAttitude:
        """Get the default attitude for a specific timeframe."""
        return self.TIMEFRAME_ATTITUDE_MAP.get(timeframe, self.default_attitude)


# =============================================================================
# Helper functions
# =============================================================================

def create_conditions_from_pipeline(
    pipeline_result: Any,
    time_to_close: Optional[int] = None,
) -> MarketConditions:
    """Create MarketConditions from a PipelineResult."""
    conditions = MarketConditions(time_to_close_minutes=time_to_close)

    # Extract from elasticity snapshot
    if hasattr(pipeline_result, 'elasticity_snapshot') and pipeline_result.elasticity_snapshot:
        snap = pipeline_result.elasticity_snapshot
        conditions.volatility = getattr(snap, 'volatility', 0.0)
        conditions.volatility_regime = getattr(snap, 'volatility_regime', 'moderate')
        conditions.trend_strength = getattr(snap, 'trend_strength', 0.0)

    # Extract from liquidity snapshot
    if hasattr(pipeline_result, 'liquidity_snapshot') and pipeline_result.liquidity_snapshot:
        snap = pipeline_result.liquidity_snapshot
        conditions.liquidity_score = getattr(snap, 'liquidity_score', 0.5)

    # Extract from MTF analysis
    if hasattr(pipeline_result, 'mtf_analysis') and pipeline_result.mtf_analysis:
        mtf = pipeline_result.mtf_analysis
        conditions.dominant_timeframe = getattr(mtf, 'dominant_timeframe', '')

    # Extract from consensus if available
    if hasattr(pipeline_result, 'consensus') and pipeline_result.consensus:
        consensus = pipeline_result.consensus
        if 'iv_rank' in consensus:
            conditions.iv_rank = consensus['iv_rank']
        if 'vix' in consensus:
            conditions.vix = consensus['vix']
        if 'strongest_timeframe' in consensus:
            conditions.dominant_timeframe = consensus['strongest_timeframe']

    return conditions


def get_profile_comparison() -> str:
    """Get a formatted comparison of all attitude profiles."""
    lines = ["=" * 80]
    lines.append("TRADING ATTITUDE PROFILES COMPARISON")
    lines.append("=" * 80)

    headers = [
        "Parameter", "SCALPER", "DAY_TRADER", "SWING_TRADER", "POSITION"
    ]

    rows = [
        ("Max Position %", "max_position_size_pct"),
        ("Risk per Trade %", "risk_per_trade_pct"),
        ("Max Positions", "max_positions"),
        ("Min Confidence", "min_confidence"),
        ("Initial Stop %", "initial_stop_pct"),
        ("Trail Stop %", "trailing_stop_pct"),
        ("Target 1 %", "target_1_pct"),
        ("Target 2 %", "target_2_pct"),
        ("Max Hold (min)", "max_hold_minutes"),
        ("DTE Range", None),  # Special handling
        ("Aggressiveness", "aggressiveness"),
    ]

    # Format header
    lines.append(f"{headers[0]:<20} | " + " | ".join(f"{h:>12}" for h in headers[1:]))
    lines.append("-" * 80)

    profiles = [
        SCALPER_PROFILE,
        DAY_TRADER_PROFILE,
        SWING_TRADER_PROFILE,
        POSITION_TRADER_PROFILE,
    ]

    for label, attr in rows:
        if attr is None:
            # DTE Range special handling
            values = [f"{p.preferred_dte_min}-{p.preferred_dte_max}" for p in profiles]
        else:
            values = []
            for p in profiles:
                val = getattr(p, attr)
                if isinstance(val, float):
                    values.append(f"{val:.1f}" if val < 10 else f"{val:.0f}")
                else:
                    values.append(str(val))

        lines.append(f"{label:<20} | " + " | ".join(f"{v:>12}" for v in values))

    lines.append("=" * 80)
    return "\n".join(lines)
