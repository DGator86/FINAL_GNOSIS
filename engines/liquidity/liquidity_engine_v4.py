"""Liquidity Engine v4 - Full Wyckoff Methodology Integration.

This engine implements the complete Wyckoff trading methodology:
- Four Phases of Price Action (Accumulation, Uptrend, Distribution, Downtrend)
- Volume Spread Analysis (VSA)
- Seven Logical Events
- Five Market Phases (A-E)
- Cause and Effect Analysis
- Spring/Upthrust Detection
- Structure Recognition

Author: Super Gnosis Elite Trading System
Version: 4.0.0 - Wyckoff Integration
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from engines.inputs.market_data_adapter import MarketDataAdapter
from engines.inputs.options_chain_adapter import OptionsChainAdapter
from schemas.core_schemas import LiquiditySnapshot


# =============================================================================
# WYCKOFF ENUMS AND DATA STRUCTURES
# =============================================================================

class WyckoffPhase(str, Enum):
    """Wyckoff market phases."""
    UNKNOWN = "unknown"
    PHASE_A = "phase_a"  # Stop of previous trend
    PHASE_B = "phase_b"  # Building the cause
    PHASE_C = "phase_c"  # Test (Spring/Upthrust)
    PHASE_D = "phase_d"  # Trend within range
    PHASE_E = "phase_e"  # Trend outside range (Effect)
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"


class WyckoffEvent(str, Enum):
    """Wyckoff logical events."""
    NONE = "none"
    PRELIMINARY_SUPPORT = "preliminary_support"  # PS
    PRELIMINARY_SUPPLY = "preliminary_supply"  # PSY
    SELLING_CLIMAX = "selling_climax"  # SC
    BUYING_CLIMAX = "buying_climax"  # BC
    AUTOMATIC_RALLY = "automatic_rally"  # AR
    AUTOMATIC_REACTION = "automatic_reaction"  # AR (distribution)
    SECONDARY_TEST = "secondary_test"  # ST
    SPRING = "spring"  # False breakdown in accumulation
    UPTHRUST = "upthrust"  # False breakout in distribution
    UPTHRUST_AFTER_DISTRIBUTION = "upthrust_after_distribution"  # UTAD
    SIGN_OF_STRENGTH = "sign_of_strength"  # SOS / Jump Across Creek
    SIGN_OF_WEAKNESS = "sign_of_weakness"  # SOW / Fall Through Ice
    LAST_POINT_OF_SUPPORT = "last_point_of_support"  # LPS
    LAST_POINT_OF_SUPPLY = "last_point_of_supply"  # LPSY
    BACKUP_TO_EDGE = "backup_to_edge"  # BU (confirmation)


class VSASignal(str, Enum):
    """Volume Spread Analysis signals."""
    NEUTRAL = "neutral"
    CONFIRMATION = "confirmation"  # Wide range + high volume
    DIVERGENCE_WARNING = "divergence_warning"  # Wide range + low volume
    ABSORPTION = "absorption"  # Narrow range + high volume
    NO_INTEREST = "no_interest"  # Narrow range + low volume
    NO_DEMAND = "no_demand"  # Narrow bullish + low volume
    NO_SUPPLY = "no_supply"  # Narrow bearish + low volume
    STOPPING_VOLUME = "stopping_volume"  # High volume rejection
    CLIMACTIC_ACTION = "climactic_action"  # Extreme volume + range


class MarketStructure(str, Enum):
    """Market structure types."""
    UNKNOWN = "unknown"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    RE_ACCUMULATION = "re_accumulation"
    RE_DISTRIBUTION = "re_distribution"
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGING = "ranging"


@dataclass
class OHLCV:
    """OHLCV bar with extended metrics."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def range(self) -> float:
        return self.high - self.low
    
    @property
    def body(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open
    
    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low


@dataclass
class RangeStructure:
    """Detected trading range structure."""
    upper_boundary: float
    lower_boundary: float
    creek_level: float  # Upper boundary for accumulation
    ice_level: float  # Lower boundary for distribution
    range_start: datetime
    range_bars: int
    touches_upper: int
    touches_lower: int
    structure_type: MarketStructure
    slope: float  # Positive = upsloping, negative = downsloping


@dataclass
class VSAAnalysis:
    """Volume Spread Analysis result."""
    signal: VSASignal
    range_type: str  # "wide", "narrow", "medium"
    volume_type: str  # "high", "low", "medium"
    effort_result_harmony: bool
    subsequent_shift: Optional[str] = None  # "bullish", "bearish", "neutral"
    reasoning: str = ""


@dataclass
class WyckoffEventDetection:
    """Detected Wyckoff event."""
    event: WyckoffEvent
    timestamp: datetime
    price: float
    volume: float
    confidence: float
    reasoning: str


@dataclass
class WyckoffState:
    """Current Wyckoff analysis state."""
    phase: WyckoffPhase
    structure: MarketStructure
    current_event: WyckoffEvent
    events_history: List[WyckoffEventDetection] = field(default_factory=list)
    range_structure: Optional[RangeStructure] = None
    vsa_analysis: Optional[VSAAnalysis] = None
    cause_duration: int = 0  # Bars in accumulation/distribution
    expected_effect_magnitude: float = 0.0
    composite_score: float = 0.0
    trading_bias: str = "neutral"  # "bullish", "bearish", "neutral"
    confidence: float = 0.0


@dataclass
class WyckoffSnapshot:
    """Complete Wyckoff analysis snapshot."""
    timestamp: datetime
    symbol: str
    state: WyckoffState
    vsa_signals: List[VSAAnalysis] = field(default_factory=list)
    detected_events: List[WyckoffEventDetection] = field(default_factory=list)
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    spring_detected: bool = False
    upthrust_detected: bool = False
    change_of_character: bool = False
    entry_signal: Optional[str] = None  # "long", "short", None
    entry_confidence: float = 0.0
    reasoning: str = ""


# =============================================================================
# VOLUME SPREAD ANALYZER
# =============================================================================

class VolumeSpreadAnalyzer:
    """Wyckoff Volume Spread Analysis implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.wide_range_mult = config.get("wide_range_multiplier", 1.5)
        self.narrow_range_mult = config.get("narrow_range_multiplier", 0.5)
        self.high_volume_mult = config.get("high_volume_multiplier", 1.5)
        self.low_volume_mult = config.get("low_volume_multiplier", 0.5)
        
    def analyze(self, bar: OHLCV, atr: float, avg_volume: float, 
                prev_bars: List[OHLCV] = None) -> VSAAnalysis:
        """Perform VSA on a single bar."""
        
        # Classify range
        if atr > 0:
            range_ratio = bar.range / atr
            if range_ratio > self.wide_range_mult:
                range_type = "wide"
            elif range_ratio < self.narrow_range_mult:
                range_type = "narrow"
            else:
                range_type = "medium"
        else:
            range_type = "medium"
        
        # Classify volume
        if avg_volume > 0:
            volume_ratio = bar.volume / avg_volume
            if volume_ratio > self.high_volume_mult:
                volume_type = "high"
            elif volume_ratio < self.low_volume_mult:
                volume_type = "low"
            else:
                volume_type = "medium"
        else:
            volume_type = "medium"
        
        # Analyze VSA combination
        signal, harmony, reasoning = self._analyze_combination(
            bar, range_type, volume_type, prev_bars
        )
        
        # Determine subsequent shift if we have previous bars
        subsequent_shift = None
        if prev_bars and len(prev_bars) >= 3:
            subsequent_shift = self._analyze_subsequent_shift(bar, prev_bars[-3:])
        
        return VSAAnalysis(
            signal=signal,
            range_type=range_type,
            volume_type=volume_type,
            effort_result_harmony=harmony,
            subsequent_shift=subsequent_shift,
            reasoning=reasoning
        )
    
    def _analyze_combination(self, bar: OHLCV, range_type: str, volume_type: str,
                             prev_bars: List[OHLCV] = None) -> Tuple[VSASignal, bool, str]:
        """Analyze VSA combination for signals."""
        
        # Wide range + high volume = Confirmation (Harmony)
        if range_type == "wide" and volume_type == "high":
            direction = "bullish" if bar.is_bullish else "bearish"
            return (VSASignal.CONFIRMATION, True, 
                    f"Strong {direction} move with volume confirmation")
        
        # Wide range + low volume = Divergence Warning
        if range_type == "wide" and volume_type == "low":
            return (VSASignal.DIVERGENCE_WARNING, False,
                    "Wide range without volume support - potential weakness")
        
        # Narrow range + high volume = Absorption
        if range_type == "narrow" and volume_type == "high":
            return (VSASignal.ABSORPTION, False,
                    "High volume absorbed in narrow range - potential reversal")
        
        # Narrow range + low volume = No Interest
        if range_type == "narrow" and volume_type == "low":
            # Check for specific no demand/supply
            if bar.is_bullish:
                return (VSASignal.NO_DEMAND, True,
                        "No demand - narrow bullish bar with low volume")
            else:
                return (VSASignal.NO_SUPPLY, True,
                        "No supply - narrow bearish bar with low volume")
        
        # Check for stopping volume (high volume with large wick)
        if volume_type == "high":
            wick_ratio = max(bar.upper_wick, bar.lower_wick) / bar.range if bar.range > 0 else 0
            if wick_ratio > 0.5:
                return (VSASignal.STOPPING_VOLUME, False,
                        "High volume rejection - potential stopping action")
        
        # Check for climactic action
        if volume_type == "high" and range_type == "wide":
            if prev_bars:
                avg_prev_vol = sum(b.volume for b in prev_bars) / len(prev_bars)
                if bar.volume > avg_prev_vol * 2.5:
                    return (VSASignal.CLIMACTIC_ACTION, False,
                            "Climactic volume - potential exhaustion")
        
        return (VSASignal.NEUTRAL, True, "Normal price action")
    
    def _analyze_subsequent_shift(self, current: OHLCV, prev_bars: List[OHLCV]) -> str:
        """Analyze if subsequent price confirms or diverges from prior bar."""
        if not prev_bars:
            return "neutral"
        
        # Look at the last significant bar
        last_bar = prev_bars[-1]
        
        # If last bar was bullish with volume, next bars should go up
        if last_bar.is_bullish and current.close > last_bar.close:
            return "bullish"
        elif last_bar.is_bearish and current.close < last_bar.close:
            return "bearish"
        elif last_bar.is_bullish and current.close < last_bar.close:
            return "divergent_bearish"
        elif last_bar.is_bearish and current.close > last_bar.close:
            return "divergent_bullish"
        
        return "neutral"
    
    def detect_no_demand(self, bars: List[OHLCV], avg_volume: float) -> bool:
        """Detect No Demand candle pattern."""
        if len(bars) < 3:
            return False
        
        current = bars[-1]
        
        # Narrow range bullish bar
        if not current.is_bullish:
            return False
        
        # Volume lower than previous 2 bars
        if current.volume >= bars[-2].volume or current.volume >= bars[-3].volume:
            return False
        
        # Volume significantly below average
        if current.volume > avg_volume * 0.7:
            return False
        
        return True
    
    def detect_no_supply(self, bars: List[OHLCV], avg_volume: float) -> bool:
        """Detect No Supply candle pattern."""
        if len(bars) < 3:
            return False
        
        current = bars[-1]
        
        # Narrow range bearish bar
        if not current.is_bearish:
            return False
        
        # Volume lower than previous 2 bars
        if current.volume >= bars[-2].volume or current.volume >= bars[-3].volume:
            return False
        
        # Volume significantly below average
        if current.volume > avg_volume * 0.7:
            return False
        
        return True


# =============================================================================
# WYCKOFF STRUCTURE DETECTOR
# =============================================================================

class WyckoffStructureDetector:
    """Detect Wyckoff accumulation/distribution structures."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_range_bars = config.get("min_range_bars", 20)
        self.touch_threshold_pct = config.get("touch_threshold_pct", 0.02)
        
    def detect_range(self, bars: List[OHLCV], lookback: int = 50) -> Optional[RangeStructure]:
        """Identify potential accumulation/distribution range."""
        if len(bars) < self.min_range_bars:
            return None
        
        analysis_bars = bars[-lookback:] if len(bars) > lookback else bars
        
        highs = [b.high for b in analysis_bars]
        lows = [b.low for b in analysis_bars]
        
        range_high = max(highs)
        range_low = min(lows)
        range_width = range_high - range_low
        
        if range_width <= 0:
            return None
        
        # Count touches of boundaries
        touch_zone = range_width * self.touch_threshold_pct
        touches_upper = sum(1 for h in highs if h > range_high - touch_zone)
        touches_lower = sum(1 for l in lows if l < range_low + touch_zone)
        
        # Need at least 2 touches on each side for valid range
        if touches_upper < 2 or touches_lower < 2:
            return None
        
        # Calculate slope (regression)
        closes = [b.close for b in analysis_bars]
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0] if len(closes) > 1 else 0.0
        
        # Determine structure type based on prior trend and slope
        structure_type = self._classify_structure(analysis_bars, slope)
        
        return RangeStructure(
            upper_boundary=range_high,
            lower_boundary=range_low,
            creek_level=range_high,  # Creek = resistance for accumulation
            ice_level=range_low,  # Ice = support for distribution
            range_start=analysis_bars[0].timestamp,
            range_bars=len(analysis_bars),
            touches_upper=touches_upper,
            touches_lower=touches_lower,
            structure_type=structure_type,
            slope=slope
        )
    
    def _classify_structure(self, bars: List[OHLCV], slope: float) -> MarketStructure:
        """Classify the type of market structure."""
        if len(bars) < 10:
            return MarketStructure.UNKNOWN
        
        # Check trend before range
        first_half = bars[:len(bars)//3]
        
        # Calculate trend direction
        first_avg = sum(b.close for b in first_half) / len(first_half)
        last_price = bars[-1].close
        mid_price = (bars[0].close + bars[-1].close) / 2
        
        # Was there a prior downtrend? -> Likely accumulation
        # Was there a prior uptrend? -> Likely distribution
        
        # Simple heuristic: compare first third with middle
        if first_avg > mid_price * 1.02:  # Price dropped into range
            if slope > 0.001:  # Upsloping
                return MarketStructure.ACCUMULATION
            elif slope < -0.001:  # Downsloping
                return MarketStructure.RE_DISTRIBUTION
            else:
                return MarketStructure.ACCUMULATION
        elif first_avg < mid_price * 0.98:  # Price rose into range
            if slope < -0.001:  # Downsloping
                return MarketStructure.DISTRIBUTION
            elif slope > 0.001:  # Upsloping
                return MarketStructure.RE_ACCUMULATION
            else:
                return MarketStructure.DISTRIBUTION
        
        return MarketStructure.RANGING
    
    def detect_change_of_character(self, bars: List[OHLCV], 
                                   range_struct: RangeStructure) -> bool:
        """Detect Change of Character (CHoCH) - transition from trending to ranging."""
        if len(bars) < 5:
            return False
        
        recent = bars[-5:]
        
        # Look for significant shift in behavior
        # CHoCH: Strong counter-trend move that establishes range boundary
        
        for i in range(1, len(recent)):
            prev = recent[i-1]
            curr = recent[i]
            
            # Strong reversal bar
            if curr.range > prev.range * 1.5:
                # Reversal at range boundary
                if curr.high >= range_struct.upper_boundary * 0.98:
                    if curr.close < curr.open:  # Bearish reversal at top
                        return True
                if curr.low <= range_struct.lower_boundary * 1.02:
                    if curr.close > curr.open:  # Bullish reversal at bottom
                        return True
        
        return False


# =============================================================================
# WYCKOFF EVENT DETECTOR
# =============================================================================

class WyckoffEventDetector:
    """Detect Wyckoff logical events."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.climax_volume_mult = config.get("climax_volume_multiplier", 2.0)
        self.climax_range_mult = config.get("climax_range_multiplier", 1.5)
        
    def detect_preliminary_stop(self, bars: List[OHLCV], avg_volume: float,
                                 avg_range: float, trend: str) -> Optional[WyckoffEventDetection]:
        """Detect Preliminary Support (PS) or Preliminary Supply (PSY)."""
        if len(bars) < 3:
            return None
        
        current = bars[-1]
        
        # Look for first sign of large trader entry against trend
        # High volume narrow range bar OR large tail with high volume
        
        volume_elevated = current.volume > avg_volume * 1.3
        
        # Check for absorption pattern (narrow range + high volume)
        narrow_range = current.range < avg_range * 0.7
        
        # Check for rejection tail
        if trend == "down":
            rejection_tail = current.lower_wick > current.body * 1.5
            if volume_elevated and (narrow_range or rejection_tail):
                return WyckoffEventDetection(
                    event=WyckoffEvent.PRELIMINARY_SUPPORT,
                    timestamp=current.timestamp,
                    price=current.low,
                    volume=current.volume,
                    confidence=0.6,
                    reasoning="First buying interest against downtrend"
                )
        elif trend == "up":
            rejection_tail = current.upper_wick > current.body * 1.5
            if volume_elevated and (narrow_range or rejection_tail):
                return WyckoffEventDetection(
                    event=WyckoffEvent.PRELIMINARY_SUPPLY,
                    timestamp=current.timestamp,
                    price=current.high,
                    volume=current.volume,
                    confidence=0.6,
                    reasoning="First selling interest against uptrend"
                )
        
        return None
    
    def detect_climax(self, bars: List[OHLCV], avg_volume: float,
                      avg_range: float, trend: str) -> Optional[WyckoffEventDetection]:
        """Detect Selling Climax (SC) or Buying Climax (BC)."""
        if len(bars) < 2:
            return None
        
        current = bars[-1]
        
        # Climax: Wide range + Climactic volume
        is_climax_volume = current.volume > avg_volume * self.climax_volume_mult
        is_wide_range = current.range > avg_range * self.climax_range_mult
        
        if not (is_climax_volume and is_wide_range):
            return None
        
        if trend == "down" and current.is_bearish:
            return WyckoffEventDetection(
                event=WyckoffEvent.SELLING_CLIMAX,
                timestamp=current.timestamp,
                price=current.low,
                volume=current.volume,
                confidence=0.75,
                reasoning=f"Climactic selling: volume {current.volume/avg_volume:.1f}x avg, range {current.range/avg_range:.1f}x avg"
            )
        elif trend == "up" and current.is_bullish:
            return WyckoffEventDetection(
                event=WyckoffEvent.BUYING_CLIMAX,
                timestamp=current.timestamp,
                price=current.high,
                volume=current.volume,
                confidence=0.75,
                reasoning=f"Climactic buying: volume {current.volume/avg_volume:.1f}x avg, range {current.range/avg_range:.1f}x avg"
            )
        
        return None
    
    def detect_automatic_reaction(self, bars: List[OHLCV], climax_event: WyckoffEventDetection,
                                   avg_volume: float) -> Optional[WyckoffEventDetection]:
        """Detect Automatic Rally (AR) or Automatic Reaction."""
        if len(bars) < 3 or climax_event is None:
            return None
        
        current = bars[-1]
        
        # AR should be a sharp move in opposite direction of climax
        if climax_event.event == WyckoffEvent.SELLING_CLIMAX:
            # Look for bullish move after selling climax
            if current.is_bullish and current.close > climax_event.price:
                move_pct = (current.close - climax_event.price) / climax_event.price
                if move_pct > 0.01:  # At least 1% move
                    return WyckoffEventDetection(
                        event=WyckoffEvent.AUTOMATIC_RALLY,
                        timestamp=current.timestamp,
                        price=current.high,
                        volume=current.volume,
                        confidence=0.7,
                        reasoning=f"Automatic rally: {move_pct*100:.1f}% off selling climax"
                    )
        
        elif climax_event.event == WyckoffEvent.BUYING_CLIMAX:
            # Look for bearish move after buying climax
            if current.is_bearish and current.close < climax_event.price:
                move_pct = (climax_event.price - current.close) / climax_event.price
                if move_pct > 0.01:
                    return WyckoffEventDetection(
                        event=WyckoffEvent.AUTOMATIC_REACTION,
                        timestamp=current.timestamp,
                        price=current.low,
                        volume=current.volume,
                        confidence=0.7,
                        reasoning=f"Automatic reaction: {move_pct*100:.1f}% off buying climax"
                    )
        
        return None
    
    def detect_secondary_test(self, bars: List[OHLCV], climax_price: float,
                               avg_volume: float, structure_type: MarketStructure) -> Optional[WyckoffEventDetection]:
        """Detect Secondary Test (ST)."""
        if len(bars) < 3:
            return None
        
        current = bars[-1]
        
        # ST should test climax level with LOWER volume
        is_lower_volume = current.volume < avg_volume * 0.8
        
        if structure_type == MarketStructure.ACCUMULATION:
            # Test of selling climax low
            at_climax_level = current.low <= climax_price * 1.02
            if at_climax_level and is_lower_volume:
                return WyckoffEventDetection(
                    event=WyckoffEvent.SECONDARY_TEST,
                    timestamp=current.timestamp,
                    price=current.low,
                    volume=current.volume,
                    confidence=0.65,
                    reasoning="Secondary test of climax low with declining volume"
                )
        
        elif structure_type == MarketStructure.DISTRIBUTION:
            # Test of buying climax high
            at_climax_level = current.high >= climax_price * 0.98
            if at_climax_level and is_lower_volume:
                return WyckoffEventDetection(
                    event=WyckoffEvent.SECONDARY_TEST,
                    timestamp=current.timestamp,
                    price=current.high,
                    volume=current.volume,
                    confidence=0.65,
                    reasoning="Secondary test of climax high with declining volume"
                )
        
        return None
    
    def detect_spring(self, bars: List[OHLCV], range_low: float,
                      avg_volume: float) -> Optional[WyckoffEventDetection]:
        """Detect Spring (false breakdown in accumulation)."""
        if len(bars) < 3:
            return None
        
        current = bars[-1]
        prev = bars[-2]
        
        # Spring characteristics:
        # 1. Price breaks below range support
        broke_support = current.low < range_low
        
        # 2. Low volume on breakdown (divergence)
        low_volume = current.volume < avg_volume * 0.8
        
        # 3. Quick rejection (closes back inside range or has lower wick)
        rejection = current.close >= range_low * 0.99 or current.lower_wick > current.body
        
        if broke_support and rejection:
            confidence = 0.8 if low_volume else 0.6
            return WyckoffEventDetection(
                event=WyckoffEvent.SPRING,
                timestamp=current.timestamp,
                price=current.low,
                volume=current.volume,
                confidence=confidence,
                reasoning=f"Spring: broke support at {range_low:.2f}, rejected with {'low' if low_volume else 'normal'} volume"
            )
        
        return None
    
    def detect_upthrust(self, bars: List[OHLCV], range_high: float,
                        avg_volume: float) -> Optional[WyckoffEventDetection]:
        """Detect Upthrust (false breakout in distribution)."""
        if len(bars) < 3:
            return None
        
        current = bars[-1]
        
        # Upthrust characteristics:
        # 1. Price breaks above range resistance
        broke_resistance = current.high > range_high
        
        # 2. Low volume on breakout (divergence)
        low_volume = current.volume < avg_volume * 0.8
        
        # 3. Quick rejection (closes back inside range or has upper wick)
        rejection = current.close <= range_high * 1.01 or current.upper_wick > current.body
        
        if broke_resistance and rejection:
            confidence = 0.8 if low_volume else 0.6
            event = WyckoffEvent.UPTHRUST_AFTER_DISTRIBUTION if current.volume > avg_volume else WyckoffEvent.UPTHRUST
            return WyckoffEventDetection(
                event=event,
                timestamp=current.timestamp,
                price=current.high,
                volume=current.volume,
                confidence=confidence,
                reasoning=f"Upthrust: broke resistance at {range_high:.2f}, rejected with {'low' if low_volume else 'normal'} volume"
            )
        
        return None
    
    def detect_sign_of_strength(self, bars: List[OHLCV], range_high: float,
                                 avg_volume: float) -> Optional[WyckoffEventDetection]:
        """Detect Sign of Strength (SOS) / Jump Across Creek."""
        if len(bars) < 3:
            return None
        
        current = bars[-1]
        
        # SOS characteristics:
        # 1. Strong bullish bar
        is_bullish = current.is_bullish and current.body > current.range * 0.6
        
        # 2. Breaks above range with HIGH volume
        broke_resistance = current.close > range_high
        high_volume = current.volume > avg_volume * 1.3
        
        if is_bullish and broke_resistance and high_volume:
            return WyckoffEventDetection(
                event=WyckoffEvent.SIGN_OF_STRENGTH,
                timestamp=current.timestamp,
                price=current.close,
                volume=current.volume,
                confidence=0.75,
                reasoning=f"Sign of Strength: broke above {range_high:.2f} with volume confirmation"
            )
        
        return None
    
    def detect_sign_of_weakness(self, bars: List[OHLCV], range_low: float,
                                 avg_volume: float) -> Optional[WyckoffEventDetection]:
        """Detect Sign of Weakness (SOW) / Fall Through Ice."""
        if len(bars) < 3:
            return None
        
        current = bars[-1]
        
        # SOW characteristics:
        # 1. Strong bearish bar
        is_bearish = current.is_bearish and current.body > current.range * 0.6
        
        # 2. Breaks below range with HIGH volume
        broke_support = current.close < range_low
        high_volume = current.volume > avg_volume * 1.3
        
        if is_bearish and broke_support and high_volume:
            return WyckoffEventDetection(
                event=WyckoffEvent.SIGN_OF_WEAKNESS,
                timestamp=current.timestamp,
                price=current.close,
                volume=current.volume,
                confidence=0.75,
                reasoning=f"Sign of Weakness: broke below {range_low:.2f} with volume confirmation"
            )
        
        return None
    
    def detect_last_point_of_support(self, bars: List[OHLCV], breakout_level: float,
                                      avg_volume: float) -> Optional[WyckoffEventDetection]:
        """Detect Last Point of Support (LPS) - confirmation after SOS."""
        if len(bars) < 3:
            return None
        
        current = bars[-1]
        
        # LPS: Pullback to broken resistance (now support) with low volume
        at_support = abs(current.low - breakout_level) / breakout_level < 0.01
        low_volume = current.volume < avg_volume * 0.7
        held_support = current.close >= breakout_level * 0.99
        
        if at_support and low_volume and held_support:
            return WyckoffEventDetection(
                event=WyckoffEvent.LAST_POINT_OF_SUPPORT,
                timestamp=current.timestamp,
                price=current.low,
                volume=current.volume,
                confidence=0.8,
                reasoning="Last Point of Support: successful retest with low volume"
            )
        
        return None
    
    def detect_last_point_of_supply(self, bars: List[OHLCV], breakdown_level: float,
                                     avg_volume: float) -> Optional[WyckoffEventDetection]:
        """Detect Last Point of Supply (LPSY) - confirmation after SOW."""
        if len(bars) < 3:
            return None
        
        current = bars[-1]
        
        # LPSY: Rally to broken support (now resistance) with low volume
        at_resistance = abs(current.high - breakdown_level) / breakdown_level < 0.01
        low_volume = current.volume < avg_volume * 0.7
        rejected = current.close <= breakdown_level * 1.01
        
        if at_resistance and low_volume and rejected:
            return WyckoffEventDetection(
                event=WyckoffEvent.LAST_POINT_OF_SUPPLY,
                timestamp=current.timestamp,
                price=current.high,
                volume=current.volume,
                confidence=0.8,
                reasoning="Last Point of Supply: failed rally with low volume"
            )
        
        return None


# =============================================================================
# WYCKOFF PHASE TRACKER
# =============================================================================

class WyckoffPhaseTracker:
    """Track Wyckoff phase progression."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_phase = WyckoffPhase.UNKNOWN
        self.events_in_phase: List[WyckoffEvent] = []
        self.phase_start_time: Optional[datetime] = None
        self.phase_start_price: Optional[float] = None
        
    def update(self, event: WyckoffEventDetection, 
               structure_type: MarketStructure) -> WyckoffPhase:
        """Update phase based on detected event."""
        
        event_type = event.event
        
        # Phase A: Stop of previous trend
        if event_type in [WyckoffEvent.PRELIMINARY_SUPPORT, WyckoffEvent.PRELIMINARY_SUPPLY,
                          WyckoffEvent.SELLING_CLIMAX, WyckoffEvent.BUYING_CLIMAX,
                          WyckoffEvent.AUTOMATIC_RALLY, WyckoffEvent.AUTOMATIC_REACTION]:
            if self.current_phase in [WyckoffPhase.UNKNOWN, WyckoffPhase.TRENDING_UP, 
                                       WyckoffPhase.TRENDING_DOWN, WyckoffPhase.PHASE_E]:
                self.current_phase = WyckoffPhase.PHASE_A
                self.events_in_phase = [event_type]
                self.phase_start_time = event.timestamp
                self.phase_start_price = event.price
        
        # Transition A -> B: After secondary test
        elif event_type == WyckoffEvent.SECONDARY_TEST:
            if self.current_phase == WyckoffPhase.PHASE_A:
                self.current_phase = WyckoffPhase.PHASE_B
                self.events_in_phase = [event_type]
        
        # Phase B continues with minor tests
        elif event_type in [WyckoffEvent.SECONDARY_TEST, WyckoffEvent.UPTHRUST]:
            if self.current_phase == WyckoffPhase.PHASE_B:
                self.events_in_phase.append(event_type)
        
        # Transition B -> C: Spring or UTAD
        elif event_type in [WyckoffEvent.SPRING, WyckoffEvent.UPTHRUST_AFTER_DISTRIBUTION]:
            if self.current_phase == WyckoffPhase.PHASE_B:
                self.current_phase = WyckoffPhase.PHASE_C
                self.events_in_phase = [event_type]
        
        # Transition C -> D: Sign of Strength/Weakness
        elif event_type in [WyckoffEvent.SIGN_OF_STRENGTH, WyckoffEvent.SIGN_OF_WEAKNESS]:
            if self.current_phase in [WyckoffPhase.PHASE_C, WyckoffPhase.PHASE_B]:
                self.current_phase = WyckoffPhase.PHASE_D
                self.events_in_phase = [event_type]
        
        # Transition D -> E: Last Point of Support/Supply
        elif event_type in [WyckoffEvent.LAST_POINT_OF_SUPPORT, WyckoffEvent.LAST_POINT_OF_SUPPLY]:
            if self.current_phase == WyckoffPhase.PHASE_D:
                self.current_phase = WyckoffPhase.PHASE_E
                self.events_in_phase = [event_type]
        
        return self.current_phase
    
    def get_expected_next_events(self) -> List[WyckoffEvent]:
        """Get expected next events based on current phase."""
        
        if self.current_phase == WyckoffPhase.PHASE_A:
            return [WyckoffEvent.SECONDARY_TEST]
        
        elif self.current_phase == WyckoffPhase.PHASE_B:
            return [WyckoffEvent.SPRING, WyckoffEvent.UPTHRUST_AFTER_DISTRIBUTION,
                    WyckoffEvent.SECONDARY_TEST]
        
        elif self.current_phase == WyckoffPhase.PHASE_C:
            return [WyckoffEvent.SIGN_OF_STRENGTH, WyckoffEvent.SIGN_OF_WEAKNESS]
        
        elif self.current_phase == WyckoffPhase.PHASE_D:
            return [WyckoffEvent.LAST_POINT_OF_SUPPORT, WyckoffEvent.LAST_POINT_OF_SUPPLY,
                    WyckoffEvent.BACKUP_TO_EDGE]
        
        elif self.current_phase == WyckoffPhase.PHASE_E:
            return [WyckoffEvent.PRELIMINARY_SUPPORT, WyckoffEvent.PRELIMINARY_SUPPLY]
        
        return []
    
    def get_trading_recommendation(self, structure_type: MarketStructure) -> Tuple[str, float]:
        """Get trading recommendation based on phase and structure."""
        
        # Phase C after Spring -> Best long entry
        if self.current_phase == WyckoffPhase.PHASE_C:
            if WyckoffEvent.SPRING in self.events_in_phase:
                return ("long", 0.9)
            if WyckoffEvent.UPTHRUST_AFTER_DISTRIBUTION in self.events_in_phase:
                return ("short", 0.9)
        
        # Phase D entries
        if self.current_phase == WyckoffPhase.PHASE_D:
            if structure_type == MarketStructure.ACCUMULATION:
                return ("long", 0.75)
            elif structure_type == MarketStructure.DISTRIBUTION:
                return ("short", 0.75)
        
        # Phase E trend following
        if self.current_phase == WyckoffPhase.PHASE_E:
            if structure_type == MarketStructure.ACCUMULATION:
                return ("long", 0.6)
            elif structure_type == MarketStructure.DISTRIBUTION:
                return ("short", 0.6)
        
        return ("neutral", 0.0)


# =============================================================================
# LIQUIDITY ENGINE V4 - MAIN CLASS
# =============================================================================

class LiquidityEngineV4:
    """
    Liquidity Engine v4 with Full Wyckoff Methodology Integration.
    
    Features:
    - Complete Volume Spread Analysis (VSA)
    - Seven Logical Events Detection
    - Five Phase Tracking
    - Accumulation/Distribution Structure Recognition
    - Spring/Upthrust Detection
    - Cause and Effect Analysis
    - Multi-timeframe support
    - Full V3 liquidity features (0DTE, gamma squeeze)
    """
    
    def __init__(
        self,
        market_adapter: MarketDataAdapter,
        options_adapter: OptionsChainAdapter,
        config: Dict[str, Any],
    ):
        """
        Initialize Liquidity Engine V4.
        
        Args:
            market_adapter: Market data provider
            options_adapter: Options chain data provider
            config: Engine configuration
        """
        self.market_adapter = market_adapter
        self.options_adapter = options_adapter
        self.config = config
        
        # Initialize Wyckoff components
        self.vsa_analyzer = VolumeSpreadAnalyzer(config.get("vsa", {}))
        self.structure_detector = WyckoffStructureDetector(config.get("structure", {}))
        self.event_detector = WyckoffEventDetector(config.get("events", {}))
        self.phase_tracker = WyckoffPhaseTracker(config.get("phases", {}))
        
        # State storage per symbol
        self._symbol_states: Dict[str, WyckoffState] = {}
        self._bar_history: Dict[str, deque] = {}
        self._max_history = config.get("max_bar_history", 200)
        
        logger.info("LiquidityEngineV4 (Wyckoff) initialized")
    
    def run(self, symbol: str, timestamp: datetime) -> LiquiditySnapshot:
        """
        Run full liquidity + Wyckoff analysis.
        
        Args:
            symbol: Trading symbol
            timestamp: Analysis timestamp
            
        Returns:
            LiquiditySnapshot with enhanced Wyckoff data
        """
        logger.debug(f"Running LiquidityEngineV4 for {symbol} at {timestamp}")
        
        try:
            # Get market data
            quote = self.market_adapter.get_quote(symbol)
            mid_price = (quote.bid + quote.ask) / 2
            spread_pct = ((quote.ask - quote.bid) / mid_price) * 100 if mid_price > 0 else 0.0
            
            # Get historical bars for Wyckoff analysis
            bars = self.market_adapter.get_bars(
                symbol, timestamp - timedelta(days=60), timestamp, timeframe="1Day"
            )
            
            if not bars:
                return self._create_empty_snapshot(symbol, timestamp)
            
            # Convert to OHLCV format
            ohlcv_bars = [
                OHLCV(
                    timestamp=bar.timestamp if hasattr(bar, 'timestamp') else timestamp,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=float(bar.volume)
                )
                for bar in bars
            ]
            
            # Update bar history
            if symbol not in self._bar_history:
                self._bar_history[symbol] = deque(maxlen=self._max_history)
            self._bar_history[symbol].extend(ohlcv_bars)
            
            # Calculate base metrics
            avg_volume = sum(b.volume for b in ohlcv_bars) / len(ohlcv_bars)
            avg_range = sum(b.range for b in ohlcv_bars) / len(ohlcv_bars)
            
            # === WYCKOFF ANALYSIS ===
            wyckoff_snapshot = self._run_wyckoff_analysis(
                symbol, timestamp, list(self._bar_history[symbol]), avg_volume, avg_range
            )
            
            # === V3 LIQUIDITY FEATURES ===
            depth = quote.bid_size + quote.ask_size
            zero_dte_depth = self._calculate_0dte_depth(symbol, timestamp)
            gamma_squeeze_risk = self._detect_gamma_squeeze_risk(symbol, timestamp, avg_volume)
            impact_cost = spread_pct * 0.5
            
            # Calculate base liquidity score
            volume_score = min(1.0, avg_volume / 10_000_000)
            spread_score = max(0.0, 1.0 - (spread_pct / 1.0))
            depth_boost = min(0.2, zero_dte_depth / 1000.0)
            
            base_liquidity = volume_score * 0.5 + spread_score * 0.3 + depth_boost * 0.1
            
            # Adjust liquidity score based on Wyckoff phase
            wyckoff_adjustment = self._calculate_wyckoff_liquidity_adjustment(wyckoff_snapshot)
            liquidity_score = base_liquidity * wyckoff_adjustment
            
            if gamma_squeeze_risk:
                liquidity_score *= 0.7
            
            # Create enhanced snapshot
            snapshot = LiquiditySnapshot(
                timestamp=timestamp,
                symbol=symbol,
                liquidity_score=liquidity_score,
                bid_ask_spread=spread_pct,
                volume=avg_volume,
                depth=depth,
                impact_cost=impact_cost,
            )
            
            # Store Wyckoff data in the state for access by other components
            self._symbol_states[symbol] = wyckoff_snapshot.state
            
            # Log Wyckoff findings
            self._log_wyckoff_analysis(symbol, wyckoff_snapshot)
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error in LiquidityEngineV4 for {symbol}: {e}")
            return self._create_empty_snapshot(symbol, timestamp)
    
    def _run_wyckoff_analysis(self, symbol: str, timestamp: datetime,
                              bars: List[OHLCV], avg_volume: float,
                              avg_range: float) -> WyckoffSnapshot:
        """Run complete Wyckoff analysis."""
        
        # Get or create state
        state = self._symbol_states.get(symbol, WyckoffState(
            phase=WyckoffPhase.UNKNOWN,
            structure=MarketStructure.UNKNOWN,
            current_event=WyckoffEvent.NONE
        ))
        
        detected_events: List[WyckoffEventDetection] = []
        vsa_signals: List[VSAAnalysis] = []
        
        # 1. Volume Spread Analysis on recent bars
        if len(bars) >= 5:
            for i in range(-5, 0):
                vsa = self.vsa_analyzer.analyze(
                    bars[i], avg_range, avg_volume, bars[i-3:i] if i > -len(bars)+3 else None
                )
                vsa_signals.append(vsa)
            state.vsa_analysis = vsa_signals[-1] if vsa_signals else None
        
        # 2. Structure Detection
        range_struct = self.structure_detector.detect_range(bars)
        state.range_structure = range_struct
        if range_struct:
            state.structure = range_struct.structure_type
            state.cause_duration = range_struct.range_bars
            # Calculate expected effect magnitude (cause → effect)
            range_width_pct = (range_struct.upper_boundary - range_struct.lower_boundary) / range_struct.lower_boundary
            state.expected_effect_magnitude = range_width_pct * (range_struct.range_bars / 50)  # Scale by duration
        
        # 3. Determine trend for event detection
        trend = self._determine_trend(bars)
        
        # 4. Detect Events
        # Preliminary Stop
        ps_event = self.event_detector.detect_preliminary_stop(bars, avg_volume, avg_range, trend)
        if ps_event:
            detected_events.append(ps_event)
            state.current_event = ps_event.event
        
        # Climax
        climax_event = self.event_detector.detect_climax(bars, avg_volume, avg_range, trend)
        if climax_event:
            detected_events.append(climax_event)
            state.current_event = climax_event.event
        
        # If we have a range structure, detect range-specific events
        if range_struct:
            # Spring (accumulation)
            if state.structure in [MarketStructure.ACCUMULATION, MarketStructure.RE_ACCUMULATION]:
                spring = self.event_detector.detect_spring(bars, range_struct.lower_boundary, avg_volume)
                if spring:
                    detected_events.append(spring)
                    state.current_event = spring.event
                
                sos = self.event_detector.detect_sign_of_strength(bars, range_struct.upper_boundary, avg_volume)
                if sos:
                    detected_events.append(sos)
                    state.current_event = sos.event
            
            # Upthrust (distribution)
            if state.structure in [MarketStructure.DISTRIBUTION, MarketStructure.RE_DISTRIBUTION]:
                upthrust = self.event_detector.detect_upthrust(bars, range_struct.upper_boundary, avg_volume)
                if upthrust:
                    detected_events.append(upthrust)
                    state.current_event = upthrust.event
                
                sow = self.event_detector.detect_sign_of_weakness(bars, range_struct.lower_boundary, avg_volume)
                if sow:
                    detected_events.append(sow)
                    state.current_event = sow.event
        
        # 5. Update Phase
        for event in detected_events:
            state.phase = self.phase_tracker.update(event, state.structure)
        
        # 6. Generate Trading Signals
        trading_bias, signal_confidence = self.phase_tracker.get_trading_recommendation(state.structure)
        state.trading_bias = trading_bias
        state.confidence = signal_confidence
        
        # 7. Calculate Composite Score
        state.composite_score = self._calculate_composite_score(state, vsa_signals)
        
        # 8. Detect change of character
        change_of_character = False
        if range_struct:
            change_of_character = self.structure_detector.detect_change_of_character(bars, range_struct)
        
        # 9. Determine entry signal
        entry_signal, entry_confidence = self._determine_entry_signal(state, detected_events)
        
        # 10. Calculate support/resistance levels
        support_levels, resistance_levels = self._calculate_sr_levels(bars, range_struct)
        
        # Build reasoning
        reasoning = self._build_reasoning(state, detected_events, vsa_signals)
        
        return WyckoffSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            state=state,
            vsa_signals=vsa_signals,
            detected_events=detected_events,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            spring_detected=any(e.event == WyckoffEvent.SPRING for e in detected_events),
            upthrust_detected=any(e.event in [WyckoffEvent.UPTHRUST, WyckoffEvent.UPTHRUST_AFTER_DISTRIBUTION] for e in detected_events),
            change_of_character=change_of_character,
            entry_signal=entry_signal,
            entry_confidence=entry_confidence,
            reasoning=reasoning
        )
    
    def _determine_trend(self, bars: List[OHLCV], lookback: int = 20) -> str:
        """Determine current trend direction."""
        if len(bars) < lookback:
            return "neutral"
        
        recent = bars[-lookback:]
        closes = [b.close for b in recent]
        
        # Simple linear regression slope
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        
        avg_price = sum(closes) / len(closes)
        slope_pct = slope / avg_price
        
        if slope_pct > 0.001:
            return "up"
        elif slope_pct < -0.001:
            return "down"
        return "neutral"
    
    def _calculate_composite_score(self, state: WyckoffState,
                                    vsa_signals: List[VSAAnalysis]) -> float:
        """Calculate composite Wyckoff score."""
        score = 0.0
        
        # Phase contribution (0-0.3)
        phase_scores = {
            WyckoffPhase.PHASE_C: 0.3,  # Best opportunity
            WyckoffPhase.PHASE_D: 0.25,
            WyckoffPhase.PHASE_E: 0.2,
            WyckoffPhase.PHASE_B: 0.15,
            WyckoffPhase.PHASE_A: 0.1,
        }
        score += phase_scores.get(state.phase, 0.0)
        
        # VSA contribution (0-0.3)
        if vsa_signals:
            bullish_signals = sum(1 for v in vsa_signals if v.signal in [VSASignal.NO_SUPPLY, VSASignal.CONFIRMATION] and v.subsequent_shift in ["bullish", "divergent_bullish"])
            bearish_signals = sum(1 for v in vsa_signals if v.signal in [VSASignal.NO_DEMAND, VSASignal.CONFIRMATION] and v.subsequent_shift in ["bearish", "divergent_bearish"])
            
            signal_strength = abs(bullish_signals - bearish_signals) / len(vsa_signals)
            score += signal_strength * 0.3
        
        # Event contribution (0-0.4)
        high_value_events = [WyckoffEvent.SPRING, WyckoffEvent.UPTHRUST_AFTER_DISTRIBUTION,
                            WyckoffEvent.SIGN_OF_STRENGTH, WyckoffEvent.SIGN_OF_WEAKNESS,
                            WyckoffEvent.LAST_POINT_OF_SUPPORT, WyckoffEvent.LAST_POINT_OF_SUPPLY]
        
        if state.current_event in high_value_events:
            score += 0.4
        elif state.current_event != WyckoffEvent.NONE:
            score += 0.2
        
        return min(1.0, score)
    
    def _determine_entry_signal(self, state: WyckoffState,
                                 events: List[WyckoffEventDetection]) -> Tuple[Optional[str], float]:
        """Determine if there's an entry signal."""
        
        # Best entries: Phase C after Spring/UTAD
        if state.phase == WyckoffPhase.PHASE_C:
            for event in events:
                if event.event == WyckoffEvent.SPRING:
                    return ("long", 0.85)
                if event.event == WyckoffEvent.UPTHRUST_AFTER_DISTRIBUTION:
                    return ("short", 0.85)
        
        # Phase D entries after SOS/SOW
        if state.phase == WyckoffPhase.PHASE_D:
            for event in events:
                if event.event == WyckoffEvent.SIGN_OF_STRENGTH:
                    return ("long", 0.75)
                if event.event == WyckoffEvent.SIGN_OF_WEAKNESS:
                    return ("short", 0.75)
                if event.event == WyckoffEvent.LAST_POINT_OF_SUPPORT:
                    return ("long", 0.8)
                if event.event == WyckoffEvent.LAST_POINT_OF_SUPPLY:
                    return ("short", 0.8)
        
        # Phase E trend following
        if state.phase == WyckoffPhase.PHASE_E:
            if state.structure == MarketStructure.ACCUMULATION:
                return ("long", 0.6)
            elif state.structure == MarketStructure.DISTRIBUTION:
                return ("short", 0.6)
        
        return (None, 0.0)
    
    def _calculate_sr_levels(self, bars: List[OHLCV],
                             range_struct: Optional[RangeStructure]) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels."""
        support = []
        resistance = []
        
        if range_struct:
            support.append(range_struct.lower_boundary)
            resistance.append(range_struct.upper_boundary)
            
            # Add creek and ice levels
            support.append(range_struct.ice_level)
            resistance.append(range_struct.creek_level)
        
        # Add recent swing highs/lows
        if len(bars) >= 20:
            recent = bars[-20:]
            
            # Find swing lows (support)
            for i in range(2, len(recent) - 2):
                if recent[i].low < recent[i-1].low and recent[i].low < recent[i-2].low and \
                   recent[i].low < recent[i+1].low and recent[i].low < recent[i+2].low:
                    support.append(recent[i].low)
            
            # Find swing highs (resistance)
            for i in range(2, len(recent) - 2):
                if recent[i].high > recent[i-1].high and recent[i].high > recent[i-2].high and \
                   recent[i].high > recent[i+1].high and recent[i].high > recent[i+2].high:
                    resistance.append(recent[i].high)
        
        return sorted(set(support)), sorted(set(resistance), reverse=True)
    
    def _calculate_wyckoff_liquidity_adjustment(self, snapshot: WyckoffSnapshot) -> float:
        """Adjust liquidity score based on Wyckoff analysis."""
        
        # Higher adjustment for favorable phases
        phase_mult = {
            WyckoffPhase.PHASE_C: 1.2,  # Best trading opportunity
            WyckoffPhase.PHASE_D: 1.15,
            WyckoffPhase.PHASE_E: 1.1,
            WyckoffPhase.PHASE_B: 1.0,
            WyckoffPhase.PHASE_A: 0.95,  # Wait for development
            WyckoffPhase.UNKNOWN: 1.0,
        }
        
        adjustment = phase_mult.get(snapshot.state.phase, 1.0)
        
        # Boost for spring/upthrust detection
        if snapshot.spring_detected or snapshot.upthrust_detected:
            adjustment *= 1.15
        
        # Boost for high-confidence entry signals
        if snapshot.entry_signal and snapshot.entry_confidence > 0.7:
            adjustment *= 1.1
        
        return adjustment
    
    def _build_reasoning(self, state: WyckoffState,
                         events: List[WyckoffEventDetection],
                         vsa_signals: List[VSAAnalysis]) -> str:
        """Build human-readable reasoning."""
        parts = []
        
        parts.append(f"Phase: {state.phase.value}")
        parts.append(f"Structure: {state.structure.value}")
        
        if state.current_event != WyckoffEvent.NONE:
            parts.append(f"Current Event: {state.current_event.value}")
        
        if events:
            recent_events = [e.event.value for e in events[-3:]]
            parts.append(f"Recent Events: {', '.join(recent_events)}")
        
        if vsa_signals:
            latest_vsa = vsa_signals[-1]
            parts.append(f"VSA: {latest_vsa.signal.value} ({latest_vsa.range_type} range, {latest_vsa.volume_type} volume)")
        
        if state.range_structure:
            parts.append(f"Range: {state.range_structure.lower_boundary:.2f} - {state.range_structure.upper_boundary:.2f}")
        
        parts.append(f"Bias: {state.trading_bias} (conf: {state.confidence:.0%})")
        
        return " | ".join(parts)
    
    def _log_wyckoff_analysis(self, symbol: str, snapshot: WyckoffSnapshot):
        """Log Wyckoff analysis results."""
        logger.info(
            f"Wyckoff {symbol}: Phase={snapshot.state.phase.value} | "
            f"Structure={snapshot.state.structure.value} | "
            f"Event={snapshot.state.current_event.value} | "
            f"Bias={snapshot.state.trading_bias} ({snapshot.state.confidence:.0%})"
        )
        
        if snapshot.entry_signal:
            logger.info(
                f"Wyckoff {symbol}: ENTRY SIGNAL -> {snapshot.entry_signal.upper()} "
                f"(confidence: {snapshot.entry_confidence:.0%})"
            )
        
        if snapshot.spring_detected:
            logger.warning(f"Wyckoff {symbol}: SPRING DETECTED - potential long entry")
        
        if snapshot.upthrust_detected:
            logger.warning(f"Wyckoff {symbol}: UPTHRUST DETECTED - potential short entry")
    
    def _calculate_0dte_depth(self, symbol: str, timestamp: datetime) -> float:
        """Calculate 0DTE options depth (from V3)."""
        try:
            chain = self.options_adapter.get_chain(symbol, timestamp)
            if not chain:
                return 0.0
            
            zero_dte_contracts = [
                c for c in chain if (c.expiration.date() - timestamp.date()).days == 0
            ]
            
            if not zero_dte_contracts:
                return 0.0
            
            total_oi = sum(c.open_interest for c in zero_dte_contracts)
            return float(total_oi)
        except Exception as e:
            logger.debug(f"Error calculating 0DTE depth for {symbol}: {e}")
            return 0.0
    
    def _detect_gamma_squeeze_risk(self, symbol: str, timestamp: datetime, 
                                    avg_volume: float) -> bool:
        """Detect gamma squeeze risk (from V3)."""
        try:
            bars = self.market_adapter.get_bars(
                symbol, timestamp - timedelta(days=1), timestamp, timeframe="1Day"
            )
            
            if not bars:
                return False
            
            current_volume = bars[-1].volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0.0
            
            if volume_ratio > 3.0:
                logger.info(f"Gamma squeeze risk detected for {symbol}: volume ratio {volume_ratio:.2f}")
                return True
            
            return False
        except Exception as e:
            logger.debug(f"Error detecting gamma squeeze for {symbol}: {e}")
            return False
    
    def _create_empty_snapshot(self, symbol: str, timestamp: datetime) -> LiquiditySnapshot:
        """Create empty snapshot for error cases."""
        return LiquiditySnapshot(
            timestamp=timestamp,
            symbol=symbol,
        )
    
    # === PUBLIC API FOR WYCKOFF DATA ===
    
    def get_wyckoff_state(self, symbol: str) -> Optional[WyckoffState]:
        """Get current Wyckoff state for a symbol."""
        return self._symbol_states.get(symbol)
    
    def get_trading_signal(self, symbol: str) -> Tuple[Optional[str], float]:
        """Get current trading signal from Wyckoff analysis."""
        state = self._symbol_states.get(symbol)
        if state:
            return (state.trading_bias if state.trading_bias != "neutral" else None, 
                    state.confidence)
        return (None, 0.0)
    
    def get_phase(self, symbol: str) -> WyckoffPhase:
        """Get current Wyckoff phase for a symbol."""
        state = self._symbol_states.get(symbol)
        return state.phase if state else WyckoffPhase.UNKNOWN
    
    def get_structure(self, symbol: str) -> MarketStructure:
        """Get current market structure for a symbol."""
        state = self._symbol_states.get(symbol)
        return state.structure if state else MarketStructure.UNKNOWN


__all__ = [
    "LiquidityEngineV4",
    "WyckoffPhase",
    "WyckoffEvent",
    "VSASignal",
    "MarketStructure",
    "WyckoffState",
    "WyckoffSnapshot",
    "VolumeSpreadAnalyzer",
    "WyckoffStructureDetector",
    "WyckoffEventDetector",
    "WyckoffPhaseTracker",
]
