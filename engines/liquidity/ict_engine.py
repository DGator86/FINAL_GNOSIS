"""ICT (Inner Circle Trader) Methodology Engine.

Implements core ICT trading concepts:
- Swing Points (Highs/Lows) identification
- Buy-side/Sell-side Liquidity levels
- Equal Highs/Lows and Old Highs/Lows
- Premium/Discount Zones
- OTE (Optimal Trade Entry) - Fibonacci 0.62-0.79 zone
- Fair Value Gaps (FVG) - BISI (Buy-side Imbalance, Sell-side Inefficiency)
                        - SIBI (Sell-side Imbalance, Buy-side Inefficiency)
- Volume Imbalances
- Order Blocks (High/Low Probability)
- Daily Bias determination
- Liquidity Sweeps detection

Author: Super Gnosis Elite Trading System
Version: 1.0.0 - ICT Methodology Integration
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# =============================================================================
# ICT ENUMS AND DATA STRUCTURES
# =============================================================================

class LiquidityType(str, Enum):
    """Type of liquidity level."""
    BUY_SIDE = "buy_side"  # Above swing highs (stop hunts for shorts)
    SELL_SIDE = "sell_side"  # Below swing lows (stop hunts for longs)


class SwingType(str, Enum):
    """Type of swing point."""
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"


class HighLowType(str, Enum):
    """Classification of highs and lows."""
    EQUAL_HIGHS = "equal_highs"  # Clustered swing highs
    EQUAL_LOWS = "equal_lows"  # Clustered swing lows
    OLD_HIGH = "old_high"  # Isolated swing high
    OLD_LOW = "old_low"  # Isolated swing low
    PREVIOUS_DAY_HIGH = "previous_day_high"
    PREVIOUS_DAY_LOW = "previous_day_low"
    PREVIOUS_WEEK_HIGH = "previous_week_high"
    PREVIOUS_WEEK_LOW = "previous_week_low"
    SESSION_HIGH = "session_high"
    SESSION_LOW = "session_low"


class FVGType(str, Enum):
    """Fair Value Gap type."""
    BISI = "bisi"  # Buy-side Imbalance, Sell-side Inefficiency (Bullish FVG)
    SIBI = "sibi"  # Sell-side Imbalance, Buy-side Inefficiency (Bearish FVG)


class FVGStatus(str, Enum):
    """Fair Value Gap status."""
    UNFILLED = "unfilled"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    INVERTED = "inverted"  # Price tested from other side


class OrderBlockType(str, Enum):
    """Order Block type."""
    BULLISH_HIGH_PROB = "bullish_high_probability"
    BEARISH_HIGH_PROB = "bearish_high_probability"
    BULLISH_LOW_PROB = "bullish_low_probability"
    BEARISH_LOW_PROB = "bearish_low_probability"


class DailyBias(str, Enum):
    """Daily directional bias."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class ZoneType(str, Enum):
    """Premium/Discount zone type."""
    PREMIUM = "premium"  # Upper half of range - look for shorts
    DISCOUNT = "discount"  # Lower half of range - look for longs
    EQUILIBRIUM = "equilibrium"  # 50% level


@dataclass
class OHLCV:
    """OHLCV bar with ICT-specific properties."""
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
    def body_top(self) -> float:
        return max(self.open, self.close)
    
    @property
    def body_bottom(self) -> float:
        return min(self.open, self.close)
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open
    
    @property
    def upper_wick(self) -> float:
        return self.high - self.body_top
    
    @property
    def lower_wick(self) -> float:
        return self.body_bottom - self.low
    
    @property
    def is_large_body(self) -> bool:
        """Check if body is large relative to range."""
        return self.body > self.range * 0.6 if self.range > 0 else False


@dataclass
class SwingPoint:
    """Identified swing point (high or low)."""
    type: SwingType
    price: float
    timestamp: datetime
    bar_index: int
    liquidity_type: LiquidityType
    is_swept: bool = False
    swept_timestamp: Optional[datetime] = None


@dataclass
class LiquidityLevel:
    """Liquidity level with classification."""
    type: LiquidityType
    price: float
    classification: HighLowType
    swing_points: List[SwingPoint] = field(default_factory=list)
    strength: float = 1.0  # Number of touches or clustered swings
    is_swept: bool = False
    swept_timestamp: Optional[datetime] = None


@dataclass
class PremiumDiscountZone:
    """Premium/Discount zone calculation."""
    range_high: float
    range_low: float
    equilibrium: float  # 50% level
    premium_start: float  # 50% level
    discount_end: float  # 50% level
    ote_high: float  # 0.62 level
    ote_low: float  # 0.79 level
    ote_midpoint: float  # 0.705 level (sweet spot)
    
    def get_zone(self, price: float) -> ZoneType:
        """Determine which zone a price is in."""
        if price >= self.equilibrium:
            return ZoneType.PREMIUM
        else:
            return ZoneType.DISCOUNT
    
    def is_in_ote(self, price: float) -> bool:
        """Check if price is in OTE zone."""
        return self.ote_low <= price <= self.ote_high


@dataclass
class FairValueGap:
    """Fair Value Gap (FVG) structure."""
    type: FVGType
    high: float  # Upper limit of gap
    low: float  # Lower limit of gap
    consequent_encroachment: float  # 50% of gap (midpoint)
    timestamp: datetime
    bar_index: int
    status: FVGStatus = FVGStatus.UNFILLED
    filled_percentage: float = 0.0
    
    @property
    def size(self) -> float:
        return self.high - self.low
    
    def update_fill_status(self, price_high: float, price_low: float):
        """Update fill status based on price interaction."""
        if self.type == FVGType.BISI:  # Bullish FVG
            if price_low <= self.low:
                self.status = FVGStatus.FILLED
                self.filled_percentage = 100.0
            elif price_low <= self.consequent_encroachment:
                self.status = FVGStatus.PARTIALLY_FILLED
                self.filled_percentage = ((self.high - price_low) / self.size) * 100
            elif price_high < self.low:  # Price went through from above
                self.status = FVGStatus.INVERTED
        else:  # SIBI - Bearish FVG
            if price_high >= self.high:
                self.status = FVGStatus.FILLED
                self.filled_percentage = 100.0
            elif price_high >= self.consequent_encroachment:
                self.status = FVGStatus.PARTIALLY_FILLED
                self.filled_percentage = ((price_high - self.low) / self.size) * 100
            elif price_low > self.high:  # Price went through from below
                self.status = FVGStatus.INVERTED


@dataclass
class VolumeImbalance:
    """Volume Imbalance structure."""
    is_bullish: bool
    gap_high: float
    gap_low: float
    timestamp: datetime
    bar_index: int


@dataclass
class OrderBlock:
    """Order Block structure."""
    type: OrderBlockType
    high: float
    low: float
    open_price: float  # Key level for entry
    mean_threshold: float  # 50% of the order block
    timestamp: datetime
    bar_index: int
    is_mitigated: bool = False
    mitigation_timestamp: Optional[datetime] = None
    
    @property
    def is_bullish(self) -> bool:
        return self.type in [OrderBlockType.BULLISH_HIGH_PROB, OrderBlockType.BULLISH_LOW_PROB]
    
    @property
    def is_high_probability(self) -> bool:
        return self.type in [OrderBlockType.BULLISH_HIGH_PROB, OrderBlockType.BEARISH_HIGH_PROB]


@dataclass
class DailyBiasResult:
    """Daily bias calculation result."""
    bias: DailyBias
    previous_day_high: float
    previous_day_low: float
    current_close: float
    broke_high: bool
    closed_above_high: bool
    broke_low: bool
    closed_below_low: bool
    confidence: float
    reasoning: str


@dataclass
class LiquiditySweep:
    """Detected liquidity sweep event."""
    liquidity_level: LiquidityLevel
    sweep_price: float
    sweep_timestamp: datetime
    bar_index: int
    failed_to_hold: bool  # Did price fail to continue in sweep direction?
    reversal_detected: bool


@dataclass
class ICTSnapshot:
    """Complete ICT analysis snapshot."""
    timestamp: datetime
    symbol: str
    
    # Swing Points
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    
    # Liquidity Levels
    buy_side_liquidity: List[LiquidityLevel] = field(default_factory=list)
    sell_side_liquidity: List[LiquidityLevel] = field(default_factory=list)
    
    # Premium/Discount
    current_range: Optional[PremiumDiscountZone] = None
    current_zone: Optional[ZoneType] = None
    in_ote: bool = False
    
    # Fair Value Gaps
    bullish_fvgs: List[FairValueGap] = field(default_factory=list)
    bearish_fvgs: List[FairValueGap] = field(default_factory=list)
    nearest_fvg: Optional[FairValueGap] = None
    
    # Volume Imbalances
    volume_imbalances: List[VolumeImbalance] = field(default_factory=list)
    
    # Order Blocks
    bullish_order_blocks: List[OrderBlock] = field(default_factory=list)
    bearish_order_blocks: List[OrderBlock] = field(default_factory=list)
    
    # Daily Bias
    daily_bias: Optional[DailyBiasResult] = None
    
    # Liquidity Sweeps
    recent_sweeps: List[LiquiditySweep] = field(default_factory=list)
    
    # Trading Signal
    entry_signal: Optional[str] = None  # "long", "short", None
    entry_confidence: float = 0.0
    entry_reasoning: str = ""


# =============================================================================
# SWING POINT DETECTOR
# =============================================================================

class SwingPointDetector:
    """Detect swing highs and lows using 3-candle pattern."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.lookback = self.config.get("lookback", 50)
        self.cluster_threshold_pct = self.config.get("cluster_threshold_pct", 0.005)
    
    def detect_swing_points(self, bars: List[OHLCV]) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """Detect all swing highs and lows in the bar series.
        
        A swing high has a lower high to the left AND a lower high to the right.
        A swing low has a higher low to the left AND a higher low to the right.
        """
        swing_highs = []
        swing_lows = []
        
        if len(bars) < 3:
            return swing_highs, swing_lows
        
        for i in range(1, len(bars) - 1):
            prev_bar = bars[i - 1]
            curr_bar = bars[i]
            next_bar = bars[i + 1]
            
            # Check for swing high
            if prev_bar.high < curr_bar.high and next_bar.high < curr_bar.high:
                swing_highs.append(SwingPoint(
                    type=SwingType.SWING_HIGH,
                    price=curr_bar.high,
                    timestamp=curr_bar.timestamp,
                    bar_index=i,
                    liquidity_type=LiquidityType.BUY_SIDE
                ))
            
            # Check for swing low
            if prev_bar.low > curr_bar.low and next_bar.low > curr_bar.low:
                swing_lows.append(SwingPoint(
                    type=SwingType.SWING_LOW,
                    price=curr_bar.low,
                    timestamp=curr_bar.timestamp,
                    bar_index=i,
                    liquidity_type=LiquidityType.SELL_SIDE
                ))
        
        return swing_highs, swing_lows
    
    def classify_highs_lows(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        current_price: float
    ) -> Tuple[List[LiquidityLevel], List[LiquidityLevel]]:
        """Classify swing points into equal/old highs and lows."""
        buy_side = []
        sell_side = []
        
        # Classify swing highs (buy-side liquidity)
        if swing_highs:
            clustered_highs = self._find_clusters(swing_highs, self.cluster_threshold_pct)
            
            for cluster in clustered_highs:
                if len(cluster) >= 2:
                    # Equal highs
                    avg_price = sum(sp.price for sp in cluster) / len(cluster)
                    buy_side.append(LiquidityLevel(
                        type=LiquidityType.BUY_SIDE,
                        price=avg_price,
                        classification=HighLowType.EQUAL_HIGHS,
                        swing_points=cluster,
                        strength=len(cluster)
                    ))
                else:
                    # Old high (isolated)
                    buy_side.append(LiquidityLevel(
                        type=LiquidityType.BUY_SIDE,
                        price=cluster[0].price,
                        classification=HighLowType.OLD_HIGH,
                        swing_points=cluster,
                        strength=1.0
                    ))
        
        # Classify swing lows (sell-side liquidity)
        if swing_lows:
            clustered_lows = self._find_clusters(swing_lows, self.cluster_threshold_pct)
            
            for cluster in clustered_lows:
                if len(cluster) >= 2:
                    # Equal lows
                    avg_price = sum(sp.price for sp in cluster) / len(cluster)
                    sell_side.append(LiquidityLevel(
                        type=LiquidityType.SELL_SIDE,
                        price=avg_price,
                        classification=HighLowType.EQUAL_LOWS,
                        swing_points=cluster,
                        strength=len(cluster)
                    ))
                else:
                    # Old low (isolated)
                    sell_side.append(LiquidityLevel(
                        type=LiquidityType.SELL_SIDE,
                        price=cluster[0].price,
                        classification=HighLowType.OLD_LOW,
                        swing_points=cluster,
                        strength=1.0
                    ))
        
        return buy_side, sell_side
    
    def _find_clusters(
        self,
        swing_points: List[SwingPoint],
        threshold_pct: float
    ) -> List[List[SwingPoint]]:
        """Group swing points into clusters based on price proximity."""
        if not swing_points:
            return []
        
        # Sort by price
        sorted_points = sorted(swing_points, key=lambda x: x.price)
        clusters = []
        current_cluster = [sorted_points[0]]
        
        for i in range(1, len(sorted_points)):
            prev_price = sorted_points[i - 1].price
            curr_price = sorted_points[i].price
            
            # Check if within threshold
            if abs(curr_price - prev_price) / prev_price <= threshold_pct:
                current_cluster.append(sorted_points[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_points[i]]
        
        clusters.append(current_cluster)
        return clusters


# =============================================================================
# PREMIUM/DISCOUNT & OTE CALCULATOR
# =============================================================================

class PremiumDiscountCalculator:
    """Calculate premium/discount zones and OTE levels."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.ote_high_fib = self.config.get("ote_high_fib", 0.62)
        self.ote_low_fib = self.config.get("ote_low_fib", 0.79)
    
    def calculate_zones(
        self,
        swing_high: float,
        swing_low: float,
        is_upward_range: bool = True
    ) -> PremiumDiscountZone:
        """Calculate premium/discount zones from a swing range.
        
        Args:
            swing_high: Range high
            swing_low: Range low
            is_upward_range: True if range goes from low to high (bullish move)
        """
        range_size = swing_high - swing_low
        equilibrium = swing_low + (range_size * 0.5)
        
        if is_upward_range:
            # For bullish move: look for longs in discount
            # OTE for long entry: 0.62 to 0.79 retracement from high
            ote_high = swing_high - (range_size * self.ote_high_fib)
            ote_low = swing_high - (range_size * self.ote_low_fib)
        else:
            # For bearish move: look for shorts in premium
            # OTE for short entry: 0.62 to 0.79 retracement from low
            ote_low = swing_low + (range_size * self.ote_high_fib)
            ote_high = swing_low + (range_size * self.ote_low_fib)
        
        ote_midpoint = (ote_high + ote_low) / 2
        
        return PremiumDiscountZone(
            range_high=swing_high,
            range_low=swing_low,
            equilibrium=equilibrium,
            premium_start=equilibrium,
            discount_end=equilibrium,
            ote_high=ote_high,
            ote_low=ote_low,
            ote_midpoint=ote_midpoint
        )


# =============================================================================
# FAIR VALUE GAP DETECTOR
# =============================================================================

class FairValueGapDetector:
    """Detect Fair Value Gaps (FVGs) in price action."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_gap_size_pct = self.config.get("min_gap_size_pct", 0.001)  # Minimum 0.1%
    
    def detect_fvgs(self, bars: List[OHLCV]) -> Tuple[List[FairValueGap], List[FairValueGap]]:
        """Detect all FVGs in the bar series.
        
        BISI (Bullish FVG): Gap between bar1's high and bar3's low
        SIBI (Bearish FVG): Gap between bar1's low and bar3's high
        """
        bullish_fvgs = []
        bearish_fvgs = []
        
        if len(bars) < 3:
            return bullish_fvgs, bearish_fvgs
        
        for i in range(2, len(bars)):
            bar1 = bars[i - 2]
            bar2 = bars[i - 1]  # Middle candle (creates the gap)
            bar3 = bars[i]
            
            # Check for Bullish FVG (BISI)
            # Upper shadow of bar1 doesn't overlap with lower shadow of bar3
            if bar1.high < bar3.low:
                gap_size = bar3.low - bar1.high
                mid_price = (bar1.high + bar3.low) / 2
                
                if gap_size / mid_price >= self.min_gap_size_pct:
                    bullish_fvgs.append(FairValueGap(
                        type=FVGType.BISI,
                        high=bar3.low,
                        low=bar1.high,
                        consequent_encroachment=(bar1.high + bar3.low) / 2,
                        timestamp=bar2.timestamp,
                        bar_index=i - 1
                    ))
            
            # Check for Bearish FVG (SIBI)
            # Lower shadow of bar1 doesn't overlap with upper shadow of bar3
            if bar1.low > bar3.high:
                gap_size = bar1.low - bar3.high
                mid_price = (bar1.low + bar3.high) / 2
                
                if gap_size / mid_price >= self.min_gap_size_pct:
                    bearish_fvgs.append(FairValueGap(
                        type=FVGType.SIBI,
                        high=bar1.low,
                        low=bar3.high,
                        consequent_encroachment=(bar1.low + bar3.high) / 2,
                        timestamp=bar2.timestamp,
                        bar_index=i - 1
                    ))
        
        return bullish_fvgs, bearish_fvgs
    
    def detect_volume_imbalances(self, bars: List[OHLCV]) -> List[VolumeImbalance]:
        """Detect volume imbalances (gap between close and open of adjacent candles)."""
        imbalances = []
        
        if len(bars) < 2:
            return imbalances
        
        for i in range(1, len(bars)):
            prev_bar = bars[i - 1]
            curr_bar = bars[i]
            
            # Bullish volume imbalance: gap between prev close and curr open
            if curr_bar.open > prev_bar.close:
                # Check if there's trading activity (shadows overlap)
                if prev_bar.high >= curr_bar.low:  # Shadows overlap = volume imbalance
                    imbalances.append(VolumeImbalance(
                        is_bullish=True,
                        gap_high=curr_bar.open,
                        gap_low=prev_bar.close,
                        timestamp=curr_bar.timestamp,
                        bar_index=i
                    ))
            
            # Bearish volume imbalance: gap between prev close and curr open
            elif curr_bar.open < prev_bar.close:
                if prev_bar.low <= curr_bar.high:  # Shadows overlap = volume imbalance
                    imbalances.append(VolumeImbalance(
                        is_bullish=False,
                        gap_high=prev_bar.close,
                        gap_low=curr_bar.open,
                        timestamp=curr_bar.timestamp,
                        bar_index=i
                    ))
        
        return imbalances
    
    def update_fvg_status(
        self,
        fvgs: List[FairValueGap],
        current_high: float,
        current_low: float
    ) -> List[FairValueGap]:
        """Update FVG fill status based on current price."""
        for fvg in fvgs:
            if fvg.status not in [FVGStatus.FILLED, FVGStatus.INVERTED]:
                fvg.update_fill_status(current_high, current_low)
        return fvgs


# =============================================================================
# ORDER BLOCK DETECTOR
# =============================================================================

class OrderBlockDetector:
    """Detect Order Blocks in price action."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.lookback = self.config.get("lookback", 20)
    
    def detect_order_blocks(
        self,
        bars: List[OHLCV],
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> Tuple[List[OrderBlock], List[OrderBlock]]:
        """Detect high and low probability order blocks."""
        bullish_obs = []
        bearish_obs = []
        
        if len(bars) < 5:
            return bullish_obs, bearish_obs
        
        # High probability order blocks
        hp_bullish, hp_bearish = self._detect_high_prob_order_blocks(bars, swing_highs, swing_lows)
        bullish_obs.extend(hp_bullish)
        bearish_obs.extend(hp_bearish)
        
        # Low probability order blocks
        lp_bullish, lp_bearish = self._detect_low_prob_order_blocks(bars)
        bullish_obs.extend(lp_bullish)
        bearish_obs.extend(lp_bearish)
        
        return bullish_obs, bearish_obs
    
    def _detect_high_prob_order_blocks(
        self,
        bars: List[OHLCV],
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> Tuple[List[OrderBlock], List[OrderBlock]]:
        """Detect high probability order blocks.
        
        Bullish: Large bearish candle sweeps sell-side liquidity, then breaks structure up
        Bearish: Large bullish candle sweeps buy-side liquidity, then breaks structure down
        """
        bullish_obs = []
        bearish_obs = []
        
        # Look for structure breaks
        for i in range(3, len(bars)):
            curr_bar = bars[i]
            
            # Find recent swing lows for sell-side liquidity
            recent_lows = [sl for sl in swing_lows if sl.bar_index < i and sl.bar_index >= i - self.lookback]
            recent_highs = [sh for sh in swing_highs if sh.bar_index < i and sh.bar_index >= i - self.lookback]
            
            # Bullish OB: Look for bearish candle that swept sell-side before bullish move
            for sl in recent_lows:
                # Find the candle that swept the low
                for j in range(sl.bar_index + 1, min(i, sl.bar_index + 5)):
                    if j >= len(bars):
                        continue
                    sweep_bar = bars[j]
                    
                    # Check if this bar swept the low
                    if sweep_bar.low < sl.price and sweep_bar.is_bearish and sweep_bar.is_large_body:
                        # Check if price then broke a recent high (structure break)
                        for sh in recent_highs:
                            if sh.bar_index < j and curr_bar.close > sh.price:
                                bullish_obs.append(OrderBlock(
                                    type=OrderBlockType.BULLISH_HIGH_PROB,
                                    high=sweep_bar.high,
                                    low=sweep_bar.low,
                                    open_price=sweep_bar.open,
                                    mean_threshold=(sweep_bar.open + sweep_bar.close) / 2,
                                    timestamp=sweep_bar.timestamp,
                                    bar_index=j
                                ))
                                break
            
            # Bearish OB: Look for bullish candle that swept buy-side before bearish move
            for sh in recent_highs:
                # Find the candle that swept the high
                for j in range(sh.bar_index + 1, min(i, sh.bar_index + 5)):
                    if j >= len(bars):
                        continue
                    sweep_bar = bars[j]
                    
                    # Check if this bar swept the high
                    if sweep_bar.high > sh.price and sweep_bar.is_bullish and sweep_bar.is_large_body:
                        # Check if price then broke a recent low (structure break)
                        for sl in recent_lows:
                            if sl.bar_index < j and curr_bar.close < sl.price:
                                bearish_obs.append(OrderBlock(
                                    type=OrderBlockType.BEARISH_HIGH_PROB,
                                    high=sweep_bar.high,
                                    low=sweep_bar.low,
                                    open_price=sweep_bar.open,
                                    mean_threshold=(sweep_bar.open + sweep_bar.close) / 2,
                                    timestamp=sweep_bar.timestamp,
                                    bar_index=j
                                ))
                                break
        
        return bullish_obs, bearish_obs
    
    def _detect_low_prob_order_blocks(
        self,
        bars: List[OHLCV]
    ) -> Tuple[List[OrderBlock], List[OrderBlock]]:
        """Detect low probability order blocks.
        
        Small body candles in the middle of a directional move.
        """
        bullish_obs = []
        bearish_obs = []
        
        if len(bars) < 5:
            return bullish_obs, bearish_obs
        
        for i in range(2, len(bars) - 2):
            curr_bar = bars[i]
            prev_bars = bars[i-2:i]
            next_bars = bars[i+1:i+3]
            
            # Check if current bar is small body
            avg_range = sum(b.range for b in bars[max(0, i-10):i]) / min(10, i) if i > 0 else curr_bar.range
            is_small_body = curr_bar.body < avg_range * 0.4
            
            if not is_small_body:
                continue
            
            # Bullish low prob OB: Small bearish candle in bullish move
            bullish_context = sum(1 for b in prev_bars + next_bars if b.is_bullish)
            if curr_bar.is_bearish and bullish_context >= 3:
                bullish_obs.append(OrderBlock(
                    type=OrderBlockType.BULLISH_LOW_PROB,
                    high=curr_bar.high,
                    low=curr_bar.low,
                    open_price=curr_bar.open,  # Space between high and open
                    mean_threshold=(curr_bar.high + curr_bar.open) / 2,
                    timestamp=curr_bar.timestamp,
                    bar_index=i
                ))
            
            # Bearish low prob OB: Small bullish candle in bearish move
            bearish_context = sum(1 for b in prev_bars + next_bars if b.is_bearish)
            if curr_bar.is_bullish and bearish_context >= 3:
                bearish_obs.append(OrderBlock(
                    type=OrderBlockType.BEARISH_LOW_PROB,
                    high=curr_bar.high,
                    low=curr_bar.low,
                    open_price=curr_bar.open,  # Space between low and open
                    mean_threshold=(curr_bar.low + curr_bar.open) / 2,
                    timestamp=curr_bar.timestamp,
                    bar_index=i
                ))
        
        return bullish_obs, bearish_obs


# =============================================================================
# DAILY BIAS CALCULATOR
# =============================================================================

class DailyBiasCalculator:
    """Calculate daily directional bias."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def calculate_bias(
        self,
        previous_day_high: float,
        previous_day_low: float,
        current_high: float,
        current_low: float,
        current_close: float
    ) -> DailyBiasResult:
        """Calculate daily bias based on ICT methodology.
        
        Bullish if:
        - Broke and closed above previous day high
        - Broke previous day low but failed to close below it
        
        Bearish if:
        - Broke and closed below previous day low
        - Broke previous day high but failed to close above it
        """
        broke_high = current_high > previous_day_high
        closed_above_high = current_close > previous_day_high
        broke_low = current_low < previous_day_low
        closed_below_low = current_close < previous_day_low
        
        reasoning_parts = []
        confidence = 0.5
        
        # Determine bias
        if closed_above_high:
            bias = DailyBias.BULLISH
            confidence = 0.8
            reasoning_parts.append("Closed above previous day high")
        elif closed_below_low:
            bias = DailyBias.BEARISH
            confidence = 0.8
            reasoning_parts.append("Closed below previous day low")
        elif broke_low and not closed_below_low:
            bias = DailyBias.BULLISH
            confidence = 0.7
            reasoning_parts.append("Failed to close below previous day low (potential reversal)")
        elif broke_high and not closed_above_high:
            bias = DailyBias.BEARISH
            confidence = 0.7
            reasoning_parts.append("Failed to close above previous day high (potential reversal)")
        else:
            bias = DailyBias.NEUTRAL
            confidence = 0.5
            reasoning_parts.append("No clear bias signal")
        
        if broke_high:
            reasoning_parts.append(f"Broke high at {previous_day_high:.2f}")
        if broke_low:
            reasoning_parts.append(f"Broke low at {previous_day_low:.2f}")
        
        return DailyBiasResult(
            bias=bias,
            previous_day_high=previous_day_high,
            previous_day_low=previous_day_low,
            current_close=current_close,
            broke_high=broke_high,
            closed_above_high=closed_above_high,
            broke_low=broke_low,
            closed_below_low=closed_below_low,
            confidence=confidence,
            reasoning=" | ".join(reasoning_parts)
        )


# =============================================================================
# LIQUIDITY SWEEP DETECTOR
# =============================================================================

class LiquiditySweepDetector:
    """Detect liquidity sweeps."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.reversal_bars = self.config.get("reversal_bars", 3)
    
    def detect_sweeps(
        self,
        bars: List[OHLCV],
        liquidity_levels: List[LiquidityLevel]
    ) -> List[LiquiditySweep]:
        """Detect liquidity sweeps and potential reversals."""
        sweeps = []
        
        if len(bars) < 2:
            return sweeps
        
        current_bar = bars[-1]
        prev_bars = bars[-self.reversal_bars - 1:-1] if len(bars) > self.reversal_bars else bars[:-1]
        
        for level in liquidity_levels:
            if level.is_swept:
                continue
            
            is_swept = False
            sweep_price = 0.0
            
            if level.type == LiquidityType.BUY_SIDE:
                # Check if we swept above the level
                if current_bar.high > level.price:
                    is_swept = True
                    sweep_price = current_bar.high
            else:  # SELL_SIDE
                # Check if we swept below the level
                if current_bar.low < level.price:
                    is_swept = True
                    sweep_price = current_bar.low
            
            if is_swept:
                # Check for reversal (failed to hold beyond sweep)
                failed_to_hold = False
                reversal_detected = False
                
                if level.type == LiquidityType.BUY_SIDE:
                    # Swept buy-side but closed below = potential reversal
                    failed_to_hold = current_bar.close < level.price
                    reversal_detected = failed_to_hold and current_bar.is_bearish
                else:
                    # Swept sell-side but closed above = potential reversal
                    failed_to_hold = current_bar.close > level.price
                    reversal_detected = failed_to_hold and current_bar.is_bullish
                
                sweeps.append(LiquiditySweep(
                    liquidity_level=level,
                    sweep_price=sweep_price,
                    sweep_timestamp=current_bar.timestamp,
                    bar_index=len(bars) - 1,
                    failed_to_hold=failed_to_hold,
                    reversal_detected=reversal_detected
                ))
                
                # Mark level as swept
                level.is_swept = True
                level.swept_timestamp = current_bar.timestamp
        
        return sweeps


# =============================================================================
# ICT ENGINE - MAIN CLASS
# =============================================================================

class ICTEngine:
    """
    ICT (Inner Circle Trader) Methodology Engine.
    
    Integrates all ICT concepts:
    - Swing Points & Liquidity Levels
    - Premium/Discount Zones & OTE
    - Fair Value Gaps (FVG)
    - Order Blocks
    - Daily Bias
    - Liquidity Sweeps
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize ICT Engine.
        
        Args:
            config: Engine configuration
        """
        self.config = config or {}
        
        # Initialize components
        self.swing_detector = SwingPointDetector(config.get("swing", {}))
        self.pd_calculator = PremiumDiscountCalculator(config.get("premium_discount", {}))
        self.fvg_detector = FairValueGapDetector(config.get("fvg", {}))
        self.ob_detector = OrderBlockDetector(config.get("order_blocks", {}))
        self.bias_calculator = DailyBiasCalculator(config.get("daily_bias", {}))
        self.sweep_detector = LiquiditySweepDetector(config.get("sweeps", {}))
        
        # State storage
        self._symbol_states: Dict[str, ICTSnapshot] = {}
        self._bar_history: Dict[str, deque] = {}
        self._daily_bars: Dict[str, List[OHLCV]] = {}
        self._max_history = config.get("max_bar_history", 200)
        
        logger.info("ICTEngine initialized")
    
    def analyze(
        self,
        symbol: str,
        bars: List[OHLCV],
        daily_bars: Optional[List[OHLCV]] = None,
        timestamp: Optional[datetime] = None
    ) -> ICTSnapshot:
        """Run complete ICT analysis.
        
        Args:
            symbol: Trading symbol
            bars: OHLCV bars for analysis (intraday timeframe)
            daily_bars: Daily bars for bias calculation (optional)
            timestamp: Analysis timestamp
            
        Returns:
            ICTSnapshot with complete analysis
        """
        if not bars:
            return self._create_empty_snapshot(symbol, timestamp or datetime.now())
        
        timestamp = timestamp or bars[-1].timestamp
        current_price = bars[-1].close
        current_high = bars[-1].high
        current_low = bars[-1].low
        
        # Update history
        if symbol not in self._bar_history:
            self._bar_history[symbol] = deque(maxlen=self._max_history)
        self._bar_history[symbol].extend(bars)
        history = list(self._bar_history[symbol])
        
        # 1. Detect Swing Points
        swing_highs, swing_lows = self.swing_detector.detect_swing_points(history)
        
        # 2. Classify Liquidity Levels
        buy_side_liq, sell_side_liq = self.swing_detector.classify_highs_lows(
            swing_highs, swing_lows, current_price
        )
        
        # 3. Calculate Premium/Discount Zones
        current_range = None
        current_zone = None
        in_ote = False
        
        if swing_highs and swing_lows:
            recent_high = max(sh.price for sh in swing_highs[-5:]) if swing_highs else None
            recent_low = min(sl.price for sl in swing_lows[-5:]) if swing_lows else None
            
            if recent_high and recent_low and recent_high > recent_low:
                # Determine if upward or downward range
                is_upward = swing_lows[-1].bar_index < swing_highs[-1].bar_index
                current_range = self.pd_calculator.calculate_zones(
                    recent_high, recent_low, is_upward
                )
                current_zone = current_range.get_zone(current_price)
                in_ote = current_range.is_in_ote(current_price)
        
        # 4. Detect Fair Value Gaps
        bullish_fvgs, bearish_fvgs = self.fvg_detector.detect_fvgs(history)
        
        # Update FVG status
        bullish_fvgs = self.fvg_detector.update_fvg_status(bullish_fvgs, current_high, current_low)
        bearish_fvgs = self.fvg_detector.update_fvg_status(bearish_fvgs, current_high, current_low)
        
        # Find nearest unfilled FVG
        nearest_fvg = self._find_nearest_fvg(bullish_fvgs + bearish_fvgs, current_price)
        
        # 5. Detect Volume Imbalances
        volume_imbalances = self.fvg_detector.detect_volume_imbalances(history)
        
        # 6. Detect Order Blocks
        bullish_obs, bearish_obs = self.ob_detector.detect_order_blocks(
            history, swing_highs, swing_lows
        )
        
        # 7. Calculate Daily Bias
        daily_bias = None
        if daily_bars and len(daily_bars) >= 2:
            prev_day = daily_bars[-2]
            curr_day = daily_bars[-1]
            daily_bias = self.bias_calculator.calculate_bias(
                prev_day.high, prev_day.low,
                curr_day.high, curr_day.low, curr_day.close
            )
        
        # 8. Detect Liquidity Sweeps
        all_liquidity = buy_side_liq + sell_side_liq
        recent_sweeps = self.sweep_detector.detect_sweeps(history, all_liquidity)
        
        # 9. Generate Entry Signal
        entry_signal, entry_confidence, entry_reasoning = self._generate_entry_signal(
            current_price=current_price,
            current_zone=current_zone,
            in_ote=in_ote,
            daily_bias=daily_bias,
            bullish_fvgs=bullish_fvgs,
            bearish_fvgs=bearish_fvgs,
            bullish_obs=bullish_obs,
            bearish_obs=bearish_obs,
            recent_sweeps=recent_sweeps
        )
        
        # Build snapshot
        snapshot = ICTSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            buy_side_liquidity=buy_side_liq,
            sell_side_liquidity=sell_side_liq,
            current_range=current_range,
            current_zone=current_zone,
            in_ote=in_ote,
            bullish_fvgs=bullish_fvgs,
            bearish_fvgs=bearish_fvgs,
            nearest_fvg=nearest_fvg,
            volume_imbalances=volume_imbalances,
            bullish_order_blocks=bullish_obs,
            bearish_order_blocks=bearish_obs,
            daily_bias=daily_bias,
            recent_sweeps=recent_sweeps,
            entry_signal=entry_signal,
            entry_confidence=entry_confidence,
            entry_reasoning=entry_reasoning
        )
        
        # Store state
        self._symbol_states[symbol] = snapshot
        
        # Log analysis
        self._log_analysis(symbol, snapshot)
        
        return snapshot
    
    def _find_nearest_fvg(
        self,
        fvgs: List[FairValueGap],
        current_price: float
    ) -> Optional[FairValueGap]:
        """Find the nearest unfilled FVG to current price."""
        unfilled = [f for f in fvgs if f.status == FVGStatus.UNFILLED]
        if not unfilled:
            return None
        
        return min(unfilled, key=lambda f: min(
            abs(current_price - f.high),
            abs(current_price - f.low)
        ))
    
    def _generate_entry_signal(
        self,
        current_price: float,
        current_zone: Optional[ZoneType],
        in_ote: bool,
        daily_bias: Optional[DailyBiasResult],
        bullish_fvgs: List[FairValueGap],
        bearish_fvgs: List[FairValueGap],
        bullish_obs: List[OrderBlock],
        bearish_obs: List[OrderBlock],
        recent_sweeps: List[LiquiditySweep]
    ) -> Tuple[Optional[str], float, str]:
        """Generate entry signal based on ICT confluence."""
        
        signal = None
        confidence = 0.0
        reasoning_parts = []
        
        # Score factors
        long_score = 0.0
        short_score = 0.0
        
        # 1. Daily Bias (25%)
        if daily_bias:
            if daily_bias.bias == DailyBias.BULLISH:
                long_score += 0.25 * daily_bias.confidence
                reasoning_parts.append(f"Bullish daily bias ({daily_bias.confidence:.0%})")
            elif daily_bias.bias == DailyBias.BEARISH:
                short_score += 0.25 * daily_bias.confidence
                reasoning_parts.append(f"Bearish daily bias ({daily_bias.confidence:.0%})")
        
        # 2. Premium/Discount Zone (20%)
        if current_zone == ZoneType.DISCOUNT:
            long_score += 0.20
            reasoning_parts.append("Price in discount zone")
        elif current_zone == ZoneType.PREMIUM:
            short_score += 0.20
            reasoning_parts.append("Price in premium zone")
        
        # 3. OTE Zone (15%)
        if in_ote:
            if current_zone == ZoneType.DISCOUNT:
                long_score += 0.15
                reasoning_parts.append("In OTE for long")
            elif current_zone == ZoneType.PREMIUM:
                short_score += 0.15
                reasoning_parts.append("In OTE for short")
        
        # 4. FVG proximity (20%)
        unfilled_bullish = [f for f in bullish_fvgs if f.status == FVGStatus.UNFILLED]
        unfilled_bearish = [f for f in bearish_fvgs if f.status == FVGStatus.UNFILLED]
        
        for fvg in unfilled_bullish:
            if fvg.low <= current_price <= fvg.high:
                long_score += 0.20
                reasoning_parts.append(f"Price in bullish FVG zone")
                break
        
        for fvg in unfilled_bearish:
            if fvg.low <= current_price <= fvg.high:
                short_score += 0.20
                reasoning_parts.append(f"Price in bearish FVG zone")
                break
        
        # 5. Order Block proximity (15%)
        for ob in bullish_obs:
            if not ob.is_mitigated and ob.low <= current_price <= ob.high:
                long_score += 0.15
                reasoning_parts.append(f"Price at bullish order block")
                break
        
        for ob in bearish_obs:
            if not ob.is_mitigated and ob.low <= current_price <= ob.high:
                short_score += 0.15
                reasoning_parts.append(f"Price at bearish order block")
                break
        
        # 6. Liquidity Sweep reversal (20%)
        for sweep in recent_sweeps:
            if sweep.reversal_detected:
                if sweep.liquidity_level.type == LiquidityType.SELL_SIDE:
                    long_score += 0.20
                    reasoning_parts.append("Sell-side sweep with reversal")
                else:
                    short_score += 0.20
                    reasoning_parts.append("Buy-side sweep with reversal")
        
        # Determine signal
        if long_score > short_score and long_score >= 0.4:
            signal = "long"
            confidence = min(1.0, long_score)
        elif short_score > long_score and short_score >= 0.4:
            signal = "short"
            confidence = min(1.0, short_score)
        
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "No clear signal"
        
        return signal, confidence, reasoning
    
    def _create_empty_snapshot(self, symbol: str, timestamp: datetime) -> ICTSnapshot:
        """Create empty snapshot."""
        return ICTSnapshot(timestamp=timestamp, symbol=symbol)
    
    def _log_analysis(self, symbol: str, snapshot: ICTSnapshot):
        """Log ICT analysis results."""
        logger.debug(
            f"ICT {symbol}: Zone={snapshot.current_zone.value if snapshot.current_zone else 'N/A'} | "
            f"OTE={snapshot.in_ote} | "
            f"Bias={snapshot.daily_bias.bias.value if snapshot.daily_bias else 'N/A'} | "
            f"FVGs={len(snapshot.bullish_fvgs)}B/{len(snapshot.bearish_fvgs)}S | "
            f"OBs={len(snapshot.bullish_order_blocks)}B/{len(snapshot.bearish_order_blocks)}S"
        )
        
        if snapshot.entry_signal:
            logger.info(
                f"ICT {symbol}: ENTRY -> {snapshot.entry_signal.upper()} "
                f"({snapshot.entry_confidence:.0%}) - {snapshot.entry_reasoning}"
            )
    
    # === PUBLIC API ===
    
    def get_state(self, symbol: str) -> Optional[ICTSnapshot]:
        """Get current ICT state for a symbol."""
        return self._symbol_states.get(symbol)
    
    def get_signal(self, symbol: str) -> Tuple[Optional[str], float]:
        """Get current trading signal."""
        state = self._symbol_states.get(symbol)
        if state:
            return state.entry_signal, state.entry_confidence
        return None, 0.0
    
    def get_daily_bias(self, symbol: str) -> Optional[DailyBias]:
        """Get daily bias for a symbol."""
        state = self._symbol_states.get(symbol)
        if state and state.daily_bias:
            return state.daily_bias.bias
        return None
    
    def get_nearest_fvg(self, symbol: str) -> Optional[FairValueGap]:
        """Get nearest unfilled FVG."""
        state = self._symbol_states.get(symbol)
        return state.nearest_fvg if state else None


__all__ = [
    # Main Engine
    "ICTEngine",
    # Data Structures
    "OHLCV",
    "SwingPoint",
    "LiquidityLevel",
    "PremiumDiscountZone",
    "FairValueGap",
    "VolumeImbalance",
    "OrderBlock",
    "DailyBiasResult",
    "LiquiditySweep",
    "ICTSnapshot",
    # Enums
    "LiquidityType",
    "SwingType",
    "HighLowType",
    "FVGType",
    "FVGStatus",
    "OrderBlockType",
    "DailyBias",
    "ZoneType",
    # Components
    "SwingPointDetector",
    "PremiumDiscountCalculator",
    "FairValueGapDetector",
    "OrderBlockDetector",
    "DailyBiasCalculator",
    "LiquiditySweepDetector",
]
