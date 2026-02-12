"""
Liquidity Concepts Engine for Gnosis Trading System.

Implements advanced liquidity concepts based on smart money trading methodology:

Core Concepts:
1. Latent Liquidity - Stop orders clustered above/below highs/lows
2. Liquidity Pools - Major and minor clusters of latent liquidity
3. Liquidity Voids - Areas of shallow market depth (price travels easily)
4. Strong/Weak Highs/Lows - Based on Break of Structure (BOS)
5. Fractal Market Structure - Smooth vs rough price movements
6. Liquidity Inducement - Patterns used by smart money to trap retail

Key Insights from Smart Money Methodology:
- "Liquidity is not a destination, it's a means" - Price follows VALUE, uses liquidity as fuel
- Latent liquidity emerges when brokers transform stop orders into market orders
- Major highs/lows create deeper liquidity pools than minor ones
- Rough market structure creates more minor liquidity pools
- Fresh supply/demand zones with internal liquidity pools are more likely to hold

Version: 1.0.0
Integration: Works with Wyckoff, ICT, Order Flow, and Supply/Demand methodologies

Author: Super Gnosis Elite Trading System
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS - Liquidity Concepts
# ============================================================================

class LiquidityPoolType(Enum):
    """Type and strength of liquidity pool."""
    MAJOR = auto()           # Above/below major swing high/low
    MINOR = auto()           # Above/below minor swing high/low
    CLUSTERED = auto()       # Multiple swings at same level (stronger)
    EQUAL_HIGHS = auto()     # EQH - Strong buy-side liquidity
    EQUAL_LOWS = auto()      # EQL - Strong sell-side liquidity


class LiquidityPoolSide(Enum):
    """Which side the liquidity sits."""
    BUY_SIDE = auto()        # Above highs - buy stops (short SL + long breakout)
    SELL_SIDE = auto()       # Below lows - sell stops (long SL + short breakout)


class SwingStrength(Enum):
    """Strength classification of swing points."""
    STRONG = auto()          # Swing that initiated BOS movement
    WEAK = auto()            # Swing that was broken by BOS
    INTERMEDIATE = auto()    # Between strong and weak


class MarketStructureType(Enum):
    """Classification of market structure smoothness."""
    SMOOTH = auto()          # Low fractal dimension - clean movements
    ROUGH = auto()           # High fractal dimension - choppy/jagged
    MIXED = auto()           # Combination of both


class BOSType(Enum):
    """Break of Structure type."""
    BULLISH_BOS = auto()     # Broke previous high
    BEARISH_BOS = auto()     # Broke previous low
    CHANGE_OF_CHARACTER = auto()  # First BOS against trend (CHoCH)


class LiquidityInducementType(Enum):
    """Type of liquidity inducement pattern."""
    STOP_HUNT = auto()       # Quick sweep and reversal
    FALSE_BREAKOUT = auto()  # Extended move beyond liquidity that fails
    LIQUIDITY_SWEEP = auto() # Sweep of liquidity pool with reversal
    INDUCEMENT_TRAP = auto() # Minor pool swept to attract positions before major move


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SwingPointExtended:
    """Extended swing point with strength classification."""
    index: int
    price: float
    timestamp: datetime
    is_high: bool
    strength: SwingStrength = SwingStrength.INTERMEDIATE
    is_swept: bool = False
    swept_timestamp: Optional[datetime] = None
    bos_created: bool = False  # Did this swing create a BOS?


@dataclass
class LiquidityPool:
    """A pool of latent liquidity above/below price levels."""
    pool_type: LiquidityPoolType
    side: LiquidityPoolSide
    price_level: float
    depth: float  # Estimated depth (number of swings * importance)
    swings: List[SwingPointExtended] = field(default_factory=list)
    
    # Formation details
    formation_time: datetime = field(default_factory=datetime.now)
    num_touches: int = 1
    price_range: float = 0.0  # Height of the pool area
    
    # Status
    is_swept: bool = False
    swept_timestamp: Optional[datetime] = None
    swept_price: float = 0.0
    reversal_after_sweep: bool = False
    
    def get_pool_strength(self) -> float:
        """Calculate pool strength (0-1)."""
        base_strength = 0.5
        
        # Type multiplier
        if self.pool_type == LiquidityPoolType.CLUSTERED:
            base_strength += 0.25
        elif self.pool_type == LiquidityPoolType.MAJOR:
            base_strength += 0.15
        elif self.pool_type in [LiquidityPoolType.EQUAL_HIGHS, LiquidityPoolType.EQUAL_LOWS]:
            base_strength += 0.20
        
        # Touches multiplier
        base_strength += min(self.num_touches * 0.05, 0.15)
        
        # Depth multiplier
        base_strength += min(self.depth * 0.02, 0.10)
        
        return min(base_strength, 1.0)


@dataclass
class LiquidityVoid:
    """An area of shallow market depth - price travels easily."""
    price_high: float
    price_low: float
    timestamp: datetime
    bar_start_index: int
    bar_end_index: int
    
    # Characteristics
    volume_ratio: float = 0.0  # Volume relative to average
    bars_to_traverse: int = 1  # How quickly price moved through
    
    @property
    def size(self) -> float:
        return self.price_high - self.price_low
    
    @property
    def midpoint(self) -> float:
        return (self.price_high + self.price_low) / 2


@dataclass
class BreakOfStructure:
    """A break of structure event."""
    bos_type: BOSType
    broken_swing: SwingPointExtended
    breaking_bar_index: int
    breaking_price: float
    timestamp: datetime
    
    # The swing that initiated the move
    initiating_swing: Optional[SwingPointExtended] = None
    
    # Created strong low/high
    created_strong_swing: Optional[SwingPointExtended] = None


@dataclass
class FractalStructureAnalysis:
    """Analysis of market structure smoothness."""
    structure_type: MarketStructureType
    fractal_dimension: float  # 1.0 = smooth, 2.0 = very rough
    
    # Swing counts
    major_swings: int = 0
    minor_swings: int = 0
    swing_ratio: float = 0.0  # minor/major ratio
    
    # Internal liquidity
    internal_pools: List[LiquidityPool] = field(default_factory=list)
    internal_voids: List[LiquidityVoid] = field(default_factory=list)
    
    # Trading implication
    zone_likely_to_hold: bool = False
    reasoning: str = ""


@dataclass
class LiquidityInducement:
    """Detected liquidity inducement pattern."""
    inducement_type: LiquidityInducementType
    pool_swept: LiquidityPool
    sweep_price: float
    sweep_timestamp: datetime
    
    # Reversal detection
    reversal_detected: bool = False
    reversal_price: float = 0.0
    reversal_strength: float = 0.0  # 0-1
    
    # Trading signal
    signal_direction: str = "neutral"  # "long", "short", "neutral"
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class LiquidityConceptsState:
    """Complete liquidity concepts analysis state."""
    timestamp: datetime
    symbol: str
    current_price: float
    
    # Swing Analysis
    all_swings: List[SwingPointExtended] = field(default_factory=list)
    strong_highs: List[SwingPointExtended] = field(default_factory=list)
    strong_lows: List[SwingPointExtended] = field(default_factory=list)
    weak_highs: List[SwingPointExtended] = field(default_factory=list)
    weak_lows: List[SwingPointExtended] = field(default_factory=list)
    
    # Liquidity Pools
    buy_side_pools: List[LiquidityPool] = field(default_factory=list)
    sell_side_pools: List[LiquidityPool] = field(default_factory=list)
    nearest_buy_side_pool: Optional[LiquidityPool] = None
    nearest_sell_side_pool: Optional[LiquidityPool] = None
    
    # Liquidity Voids
    voids: List[LiquidityVoid] = field(default_factory=list)
    nearest_void_above: Optional[LiquidityVoid] = None
    nearest_void_below: Optional[LiquidityVoid] = None
    
    # Break of Structure
    recent_bos: List[BreakOfStructure] = field(default_factory=list)
    last_bos: Optional[BreakOfStructure] = None
    trend_direction: str = "neutral"  # "bullish", "bearish", "neutral"
    
    # Fractal Structure
    range_structure: Optional[FractalStructureAnalysis] = None
    pullback_structure: Optional[FractalStructureAnalysis] = None
    
    # Inducements
    recent_inducements: List[LiquidityInducement] = field(default_factory=list)
    active_inducement: Optional[LiquidityInducement] = None
    
    # Trading Signal
    bias: str = "neutral"
    bias_confidence: float = 0.0
    bias_reasoning: str = ""


# ============================================================================
# SWING ANALYZER - Extended with Strong/Weak Classification
# ============================================================================

class ExtendedSwingAnalyzer:
    """
    Analyzes swing points with strong/weak classification based on BOS.
    
    Rules:
    - When price breaks a previous high (bullish BOS), the low that 
      initiated the move becomes a STRONG low
    - When price breaks a previous low (bearish BOS), the high that
      initiated the move becomes a STRONG high
    - Swings that get broken are WEAK
    """
    
    def __init__(self, lookback: int = 3, min_swing_distance: int = 2):
        """
        Initialize analyzer.
        
        Args:
            lookback: Bars on each side to confirm swing
            min_swing_distance: Minimum bars between swings
        """
        self.lookback = lookback
        self.min_swing_distance = min_swing_distance
    
    def detect_swings(
        self,
        bars: List[Dict[str, Any]],
    ) -> List[SwingPointExtended]:
        """Detect all swing points."""
        if len(bars) < self.lookback * 2 + 1:
            return []
        
        swings = []
        
        for i in range(self.lookback, len(bars) - self.lookback):
            bar = bars[i]
            high = bar.get('high', bar.get('close', 0))
            low = bar.get('low', bar.get('close', 0))
            timestamp = bar.get('timestamp', datetime.now())
            
            # Check for swing high
            is_swing_high = all(
                high > bars[i - j].get('high', bars[i - j].get('close', 0)) and
                high > bars[i + j].get('high', bars[i + j].get('close', 0))
                for j in range(1, self.lookback + 1)
            )
            
            if is_swing_high:
                swings.append(SwingPointExtended(
                    index=i,
                    price=high,
                    timestamp=timestamp,
                    is_high=True,
                    strength=SwingStrength.INTERMEDIATE,
                ))
            
            # Check for swing low
            is_swing_low = all(
                low < bars[i - j].get('low', bars[i - j].get('close', 0)) and
                low < bars[i + j].get('low', bars[i + j].get('close', 0))
                for j in range(1, self.lookback + 1)
            )
            
            if is_swing_low:
                swings.append(SwingPointExtended(
                    index=i,
                    price=low,
                    timestamp=timestamp,
                    is_high=False,
                    strength=SwingStrength.INTERMEDIATE,
                ))
        
        return swings
    
    def classify_swing_strength(
        self,
        swings: List[SwingPointExtended],
        bars: List[Dict[str, Any]],
    ) -> Tuple[List[SwingPointExtended], List[BreakOfStructure]]:
        """
        Classify swings as strong or weak based on BOS.
        
        Returns:
            Tuple of (classified swings, BOS events)
        """
        if len(swings) < 2:
            return swings, []
        
        bos_events = []
        highs = [s for s in swings if s.is_high]
        lows = [s for s in swings if not s.is_high]
        
        # Sort by index
        highs.sort(key=lambda s: s.index)
        lows.sort(key=lambda s: s.index)
        
        # Process BOS for highs
        for i in range(1, len(highs)):
            current_high = highs[i]
            prev_high = highs[i - 1]
            
            # Check if current high breaks previous
            if current_high.price > prev_high.price:
                # Bullish BOS - prev high becomes weak
                prev_high.strength = SwingStrength.WEAK
                current_high.bos_created = True
                
                # Find the low that initiated this move
                initiating_low = self._find_initiating_swing(
                    prev_high.index, current_high.index, lows
                )
                
                if initiating_low:
                    initiating_low.strength = SwingStrength.STRONG
                    
                    bos_events.append(BreakOfStructure(
                        bos_type=BOSType.BULLISH_BOS,
                        broken_swing=prev_high,
                        breaking_bar_index=current_high.index,
                        breaking_price=current_high.price,
                        timestamp=current_high.timestamp,
                        initiating_swing=initiating_low,
                        created_strong_swing=initiating_low,
                    ))
        
        # Process BOS for lows
        for i in range(1, len(lows)):
            current_low = lows[i]
            prev_low = lows[i - 1]
            
            # Check if current low breaks previous
            if current_low.price < prev_low.price:
                # Bearish BOS - prev low becomes weak
                prev_low.strength = SwingStrength.WEAK
                current_low.bos_created = True
                
                # Find the high that initiated this move
                initiating_high = self._find_initiating_swing(
                    prev_low.index, current_low.index, highs
                )
                
                if initiating_high:
                    initiating_high.strength = SwingStrength.STRONG
                    
                    bos_events.append(BreakOfStructure(
                        bos_type=BOSType.BEARISH_BOS,
                        broken_swing=prev_low,
                        breaking_bar_index=current_low.index,
                        breaking_price=current_low.price,
                        timestamp=current_low.timestamp,
                        initiating_swing=initiating_high,
                        created_strong_swing=initiating_high,
                    ))
        
        return swings, bos_events
    
    def _find_initiating_swing(
        self,
        start_index: int,
        end_index: int,
        opposite_swings: List[SwingPointExtended],
    ) -> Optional[SwingPointExtended]:
        """Find the swing that initiated a BOS move."""
        candidates = [
            s for s in opposite_swings
            if start_index < s.index < end_index
        ]
        
        if not candidates:
            return None
        
        # Return the extreme swing in the range
        if candidates[0].is_high:
            return max(candidates, key=lambda s: s.price)
        else:
            return min(candidates, key=lambda s: s.price)


# ============================================================================
# LIQUIDITY POOL DETECTOR
# ============================================================================

class LiquidityPoolDetector:
    """
    Detects liquidity pools above highs and below lows.
    
    Key concepts:
    - Major pools form above/below major swings
    - Minor pools form above/below minor swings
    - Equal highs/lows create clustered pools (stronger)
    - Pool depth is proportional to perceived importance
    """
    
    def __init__(
        self,
        cluster_threshold_pct: float = 0.003,
        min_pool_swings: int = 1,
    ):
        """
        Initialize detector.
        
        Args:
            cluster_threshold_pct: % distance to consider swings clustered
            min_pool_swings: Minimum swings to form a pool
        """
        self.cluster_threshold_pct = cluster_threshold_pct
        self.min_pool_swings = min_pool_swings
    
    def detect_pools(
        self,
        swings: List[SwingPointExtended],
        current_price: float,
    ) -> Tuple[List[LiquidityPool], List[LiquidityPool]]:
        """
        Detect buy-side and sell-side liquidity pools.
        
        Args:
            swings: List of classified swing points
            current_price: Current market price
            
        Returns:
            Tuple of (buy_side_pools, sell_side_pools)
        """
        highs = [s for s in swings if s.is_high and not s.is_swept]
        lows = [s for s in swings if not s.is_high and not s.is_swept]
        
        buy_side_pools = self._create_pools_from_swings(
            highs, LiquidityPoolSide.BUY_SIDE, current_price
        )
        
        sell_side_pools = self._create_pools_from_swings(
            lows, LiquidityPoolSide.SELL_SIDE, current_price
        )
        
        return buy_side_pools, sell_side_pools
    
    def _create_pools_from_swings(
        self,
        swings: List[SwingPointExtended],
        side: LiquidityPoolSide,
        current_price: float,
    ) -> List[LiquidityPool]:
        """Create liquidity pools from swing points."""
        if not swings:
            return []
        
        pools = []
        processed = set()
        
        # Sort by price
        sorted_swings = sorted(swings, key=lambda s: s.price, reverse=(side == LiquidityPoolSide.BUY_SIDE))
        
        for swing in sorted_swings:
            if swing.index in processed:
                continue
            
            # Find clustered swings
            cluster = [swing]
            threshold = swing.price * self.cluster_threshold_pct
            
            for other in sorted_swings:
                if other.index == swing.index or other.index in processed:
                    continue
                if abs(other.price - swing.price) <= threshold:
                    cluster.append(other)
                    processed.add(other.index)
            
            processed.add(swing.index)
            
            # Determine pool type
            if len(cluster) >= 2:
                if side == LiquidityPoolSide.BUY_SIDE:
                    pool_type = LiquidityPoolType.EQUAL_HIGHS
                else:
                    pool_type = LiquidityPoolType.EQUAL_LOWS
            elif any(s.strength == SwingStrength.STRONG for s in cluster):
                pool_type = LiquidityPoolType.MAJOR
            else:
                pool_type = LiquidityPoolType.MINOR
            
            # Calculate pool price level
            if side == LiquidityPoolSide.BUY_SIDE:
                price_level = max(s.price for s in cluster)
            else:
                price_level = min(s.price for s in cluster)
            
            # Calculate depth based on swing importance
            depth = sum(
                2.0 if s.strength == SwingStrength.STRONG else 1.0
                for s in cluster
            )
            
            pool = LiquidityPool(
                pool_type=pool_type,
                side=side,
                price_level=price_level,
                depth=depth,
                swings=cluster,
                formation_time=min(s.timestamp for s in cluster),
                num_touches=len(cluster),
                price_range=max(s.price for s in cluster) - min(s.price for s in cluster),
            )
            
            pools.append(pool)
        
        return pools


# ============================================================================
# LIQUIDITY VOID DETECTOR
# ============================================================================

class LiquidityVoidDetector:
    """
    Detects liquidity voids - areas where price traveled quickly.
    
    Voids indicate:
    - Shallow market depth
    - Low liquidity environment
    - Price can return to fill these areas
    """
    
    def __init__(
        self,
        min_void_size_pct: float = 0.005,
        max_bars_to_traverse: int = 3,
        volume_threshold: float = 0.5,
    ):
        """
        Initialize detector.
        
        Args:
            min_void_size_pct: Minimum void size as % of price
            max_bars_to_traverse: Max bars for quick traversal
            volume_threshold: Volume ratio threshold (vs average)
        """
        self.min_void_size_pct = min_void_size_pct
        self.max_bars_to_traverse = max_bars_to_traverse
        self.volume_threshold = volume_threshold
    
    def detect_voids(
        self,
        bars: List[Dict[str, Any]],
    ) -> List[LiquidityVoid]:
        """
        Detect liquidity voids in price action.
        
        Args:
            bars: OHLCV bar data
            
        Returns:
            List of detected voids
        """
        if len(bars) < 5:
            return []
        
        voids = []
        avg_range = np.mean([
            b.get('high', 0) - b.get('low', 0) for b in bars
        ])
        avg_volume = np.mean([b.get('volume', 1) for b in bars]) or 1
        
        i = 1
        while i < len(bars) - 1:
            bar = bars[i]
            prev_bar = bars[i - 1]
            
            # Check for gap up (bullish void)
            gap_up = bar.get('low', 0) - prev_bar.get('high', 0)
            if gap_up > avg_range * 0.5:
                void = LiquidityVoid(
                    price_high=bar.get('low', 0),
                    price_low=prev_bar.get('high', 0),
                    timestamp=bar.get('timestamp', datetime.now()),
                    bar_start_index=i - 1,
                    bar_end_index=i,
                    volume_ratio=bar.get('volume', 0) / avg_volume,
                    bars_to_traverse=1,
                )
                voids.append(void)
            
            # Check for gap down (bearish void)
            gap_down = prev_bar.get('low', 0) - bar.get('high', 0)
            if gap_down > avg_range * 0.5:
                void = LiquidityVoid(
                    price_high=prev_bar.get('low', 0),
                    price_low=bar.get('high', 0),
                    timestamp=bar.get('timestamp', datetime.now()),
                    bar_start_index=i - 1,
                    bar_end_index=i,
                    volume_ratio=bar.get('volume', 0) / avg_volume,
                    bars_to_traverse=1,
                )
                voids.append(void)
            
            # Check for large range bars with low volume (potential void)
            bar_range = bar.get('high', 0) - bar.get('low', 0)
            bar_volume_ratio = bar.get('volume', 0) / avg_volume
            
            if bar_range > avg_range * 2 and bar_volume_ratio < self.volume_threshold:
                # Large move with low volume = void
                void = LiquidityVoid(
                    price_high=bar.get('high', 0),
                    price_low=bar.get('low', 0),
                    timestamp=bar.get('timestamp', datetime.now()),
                    bar_start_index=i,
                    bar_end_index=i,
                    volume_ratio=bar_volume_ratio,
                    bars_to_traverse=1,
                )
                voids.append(void)
            
            i += 1
        
        return voids


# ============================================================================
# FRACTAL STRUCTURE ANALYZER
# ============================================================================

class FractalStructureAnalyzer:
    """
    Analyzes fractal dimension of market structure.
    
    Key insight:
    - Smooth structure = few minor liquidity pools = zone less likely to hold
    - Rough structure = many minor liquidity pools = zone more likely to hold
    
    This helps determine if smart money can enter without breaking the zone.
    """
    
    def __init__(
        self,
        smoothness_threshold: float = 0.3,
    ):
        """
        Initialize analyzer.
        
        Args:
            smoothness_threshold: Ratio below which structure is "smooth"
        """
        self.smoothness_threshold = smoothness_threshold
    
    def analyze_structure(
        self,
        bars: List[Dict[str, Any]],
        start_index: int,
        end_index: int,
        swings: List[SwingPointExtended],
    ) -> FractalStructureAnalysis:
        """
        Analyze the fractal structure of a price range.
        
        Args:
            bars: OHLCV data
            start_index: Start of range
            end_index: End of range
            swings: All swing points
            
        Returns:
            FractalStructureAnalysis result
        """
        if end_index <= start_index or end_index >= len(bars):
            return FractalStructureAnalysis(
                structure_type=MarketStructureType.MIXED,
                fractal_dimension=1.5,
            )
        
        # Get swings within range
        range_swings = [
            s for s in swings
            if start_index <= s.index <= end_index
        ]
        
        # Classify major vs minor swings
        major_swings = [
            s for s in range_swings
            if s.strength in [SwingStrength.STRONG, SwingStrength.INTERMEDIATE]
        ]
        minor_swings = [
            s for s in range_swings
            if s.strength == SwingStrength.WEAK
        ]
        
        # Calculate fractal dimension proxy
        total_bars = end_index - start_index
        total_swings = len(range_swings)
        
        if total_bars == 0:
            swing_ratio = 0
        else:
            swing_ratio = total_swings / total_bars
        
        # Estimate fractal dimension (1.0 = smooth, 2.0 = rough)
        fractal_dimension = 1.0 + min(swing_ratio * 2, 1.0)
        
        # Classify structure type
        if fractal_dimension < 1.3:
            structure_type = MarketStructureType.SMOOTH
        elif fractal_dimension > 1.7:
            structure_type = MarketStructureType.ROUGH
        else:
            structure_type = MarketStructureType.MIXED
        
        # Create internal liquidity pools
        pool_detector = LiquidityPoolDetector()
        internal_highs = [s for s in range_swings if s.is_high]
        internal_lows = [s for s in range_swings if not s.is_high]
        
        avg_price = np.mean([bars[i].get('close', 0) for i in range(start_index, end_index + 1)])
        
        internal_pools = []
        if internal_highs:
            buy_pools, _ = pool_detector.detect_pools(internal_highs, avg_price)
            internal_pools.extend(buy_pools)
        if internal_lows:
            _, sell_pools = pool_detector.detect_pools(internal_lows, avg_price)
            internal_pools.extend(sell_pools)
        
        # Detect internal voids
        void_detector = LiquidityVoidDetector()
        internal_voids = void_detector.detect_voids(bars[start_index:end_index + 1])
        
        # Determine if zone likely to hold
        zone_likely_to_hold = len(internal_pools) >= 2 or structure_type == MarketStructureType.ROUGH
        
        reasoning = f"Structure: {structure_type.name}, "
        reasoning += f"Fractal dim: {fractal_dimension:.2f}, "
        reasoning += f"Internal pools: {len(internal_pools)}, "
        reasoning += f"Voids: {len(internal_voids)}"
        
        return FractalStructureAnalysis(
            structure_type=structure_type,
            fractal_dimension=fractal_dimension,
            major_swings=len(major_swings),
            minor_swings=len(minor_swings),
            swing_ratio=swing_ratio,
            internal_pools=internal_pools,
            internal_voids=internal_voids,
            zone_likely_to_hold=zone_likely_to_hold,
            reasoning=reasoning,
        )


# ============================================================================
# LIQUIDITY INDUCEMENT DETECTOR
# ============================================================================

class LiquidityInducementDetector:
    """
    Detects liquidity inducement patterns.
    
    These are traps set by smart money:
    - Stop Hunt: Quick sweep and reversal
    - False Breakout: Extended move that fails
    - Liquidity Sweep: Major pool taken with reversal
    - Inducement Trap: Minor pool swept to attract positions
    """
    
    def __init__(
        self,
        reversal_threshold: float = 0.5,
        min_reversal_bars: int = 2,
    ):
        """
        Initialize detector.
        
        Args:
            reversal_threshold: Min retracement to confirm reversal
            min_reversal_bars: Bars after sweep to check reversal
        """
        self.reversal_threshold = reversal_threshold
        self.min_reversal_bars = min_reversal_bars
    
    def detect_inducements(
        self,
        bars: List[Dict[str, Any]],
        pools: List[LiquidityPool],
        current_index: int,
    ) -> List[LiquidityInducement]:
        """
        Detect liquidity inducement patterns.
        
        Args:
            bars: OHLCV data
            pools: All liquidity pools
            current_index: Current bar index
            
        Returns:
            List of detected inducements
        """
        inducements = []
        
        for pool in pools:
            if pool.is_swept:
                continue
            
            # Check if pool was recently swept
            sweep_result = self._check_pool_sweep(bars, pool, current_index)
            
            if sweep_result:
                sweep_bar_index, sweep_price = sweep_result
                
                # Check for reversal
                reversal = self._check_reversal(
                    bars, pool, sweep_bar_index, sweep_price, current_index
                )
                
                if reversal['detected']:
                    # Classify inducement type
                    if reversal['strength'] > 0.7:
                        ind_type = LiquidityInducementType.STOP_HUNT
                    elif reversal['bars_held'] > 3:
                        ind_type = LiquidityInducementType.FALSE_BREAKOUT
                    elif pool.pool_type == LiquidityPoolType.MAJOR:
                        ind_type = LiquidityInducementType.LIQUIDITY_SWEEP
                    else:
                        ind_type = LiquidityInducementType.INDUCEMENT_TRAP
                    
                    # Determine signal direction
                    if pool.side == LiquidityPoolSide.BUY_SIDE:
                        signal_direction = "short"
                    else:
                        signal_direction = "long"
                    
                    inducement = LiquidityInducement(
                        inducement_type=ind_type,
                        pool_swept=pool,
                        sweep_price=sweep_price,
                        sweep_timestamp=bars[sweep_bar_index].get('timestamp', datetime.now()),
                        reversal_detected=True,
                        reversal_price=reversal['price'],
                        reversal_strength=reversal['strength'],
                        signal_direction=signal_direction,
                        confidence=reversal['strength'] * pool.get_pool_strength(),
                        reasoning=f"{ind_type.name} at {pool.pool_type.name} pool",
                    )
                    
                    # Mark pool as swept
                    pool.is_swept = True
                    pool.swept_price = sweep_price
                    pool.reversal_after_sweep = True
                    
                    inducements.append(inducement)
        
        return inducements
    
    def _check_pool_sweep(
        self,
        bars: List[Dict[str, Any]],
        pool: LiquidityPool,
        current_index: int,
    ) -> Optional[Tuple[int, float]]:
        """Check if pool was swept in recent bars."""
        lookback = min(20, current_index)
        
        for i in range(current_index - lookback, current_index):
            if i < 0:
                continue
            
            bar = bars[i]
            
            if pool.side == LiquidityPoolSide.BUY_SIDE:
                # Check if price went above pool level
                if bar.get('high', 0) > pool.price_level:
                    return (i, bar.get('high', 0))
            else:
                # Check if price went below pool level
                if bar.get('low', float('inf')) < pool.price_level:
                    return (i, bar.get('low', 0))
        
        return None
    
    def _check_reversal(
        self,
        bars: List[Dict[str, Any]],
        pool: LiquidityPool,
        sweep_index: int,
        sweep_price: float,
        current_index: int,
    ) -> Dict[str, Any]:
        """Check for reversal after sweep."""
        result = {
            'detected': False,
            'price': 0,
            'strength': 0,
            'bars_held': 0,
        }
        
        if current_index - sweep_index < self.min_reversal_bars:
            return result
        
        # Check bars after sweep
        for i in range(sweep_index + 1, min(sweep_index + 10, current_index)):
            bar = bars[i]
            
            if pool.side == LiquidityPoolSide.BUY_SIDE:
                # Looking for reversal down
                if bar.get('close', float('inf')) < pool.price_level:
                    retracement = (sweep_price - bar.get('close', 0)) / (sweep_price - pool.price_level)
                    if retracement >= self.reversal_threshold:
                        result['detected'] = True
                        result['price'] = bar.get('close', 0)
                        result['strength'] = min(retracement, 1.0)
                        result['bars_held'] = i - sweep_index
                        return result
            else:
                # Looking for reversal up
                if bar.get('close', 0) > pool.price_level:
                    retracement = (bar.get('close', 0) - sweep_price) / (pool.price_level - sweep_price)
                    if retracement >= self.reversal_threshold:
                        result['detected'] = True
                        result['price'] = bar.get('close', 0)
                        result['strength'] = min(retracement, 1.0)
                        result['bars_held'] = i - sweep_index
                        return result
        
        return result


# ============================================================================
# MAIN LIQUIDITY CONCEPTS ENGINE
# ============================================================================

class LiquidityConceptsEngine:
    """
    Comprehensive Liquidity Concepts Analysis Engine.
    
    Implements smart money liquidity analysis:
    - Latent liquidity pools (buy-side/sell-side)
    - Strong/weak swing classification
    - Liquidity voids
    - Fractal market structure
    - Liquidity inducement detection
    
    Key Trading Insights:
    1. Price follows VALUE, not liquidity - liquidity is the fuel
    2. Major pools create deeper liquidity than minor pools
    3. Rough structure = more internal pools = zone more likely to hold
    4. Inducements are traps - trade in opposite direction after reversal
    """
    
    def __init__(
        self,
        swing_lookback: int = 3,
        cluster_threshold_pct: float = 0.003,
        min_void_size_pct: float = 0.005,
    ):
        """
        Initialize the Liquidity Concepts Engine.
        
        Args:
            swing_lookback: Bars to confirm swing points
            cluster_threshold_pct: % to consider swings clustered
            min_void_size_pct: Minimum void size as % of price
        """
        self.swing_analyzer = ExtendedSwingAnalyzer(lookback=swing_lookback)
        self.pool_detector = LiquidityPoolDetector(
            cluster_threshold_pct=cluster_threshold_pct
        )
        self.void_detector = LiquidityVoidDetector(
            min_void_size_pct=min_void_size_pct
        )
        self.fractal_analyzer = FractalStructureAnalyzer()
        self.inducement_detector = LiquidityInducementDetector()
        
        # State tracking
        self._states: Dict[str, LiquidityConceptsState] = {}
        
        logger.info(
            f"LiquidityConceptsEngine initialized | "
            f"swing_lookback={swing_lookback}, cluster_pct={cluster_threshold_pct:.1%}"
        )
    
    def analyze(
        self,
        symbol: str,
        bars: List[Dict[str, Any]],
        current_price: Optional[float] = None,
    ) -> LiquidityConceptsState:
        """
        Perform complete liquidity concepts analysis.
        
        Args:
            symbol: Trading symbol
            bars: OHLCV bar data
            current_price: Current market price
            
        Returns:
            LiquidityConceptsState with all analysis results
        """
        if not bars:
            return LiquidityConceptsState(
                timestamp=datetime.now(),
                symbol=symbol,
                current_price=current_price or 0,
            )
        
        timestamp = datetime.now()
        price = current_price or bars[-1].get('close', 0)
        current_index = len(bars) - 1
        
        # 1. Detect and classify swings
        swings = self.swing_analyzer.detect_swings(bars)
        swings, bos_events = self.swing_analyzer.classify_swing_strength(swings, bars)
        
        # Separate strong/weak
        strong_highs = [s for s in swings if s.is_high and s.strength == SwingStrength.STRONG]
        strong_lows = [s for s in swings if not s.is_high and s.strength == SwingStrength.STRONG]
        weak_highs = [s for s in swings if s.is_high and s.strength == SwingStrength.WEAK]
        weak_lows = [s for s in swings if not s.is_high and s.strength == SwingStrength.WEAK]
        
        # 2. Detect liquidity pools
        buy_side_pools, sell_side_pools = self.pool_detector.detect_pools(swings, price)
        
        # Find nearest pools
        nearest_buy = None
        nearest_sell = None
        
        above_price_pools = [p for p in buy_side_pools if p.price_level > price]
        if above_price_pools:
            nearest_buy = min(above_price_pools, key=lambda p: p.price_level - price)
        
        below_price_pools = [p for p in sell_side_pools if p.price_level < price]
        if below_price_pools:
            nearest_sell = max(below_price_pools, key=lambda p: p.price_level)
        
        # 3. Detect voids
        voids = self.void_detector.detect_voids(bars)
        
        # Find nearest voids
        nearest_void_above = None
        nearest_void_below = None
        
        voids_above = [v for v in voids if v.price_low > price]
        if voids_above:
            nearest_void_above = min(voids_above, key=lambda v: v.price_low - price)
        
        voids_below = [v for v in voids if v.price_high < price]
        if voids_below:
            nearest_void_below = max(voids_below, key=lambda v: v.price_high)
        
        # 4. Determine trend from BOS
        trend = "neutral"
        if bos_events:
            recent_bos = sorted(bos_events, key=lambda b: b.timestamp, reverse=True)[:3]
            bullish_count = sum(1 for b in recent_bos if b.bos_type == BOSType.BULLISH_BOS)
            bearish_count = sum(1 for b in recent_bos if b.bos_type == BOSType.BEARISH_BOS)
            
            if bullish_count > bearish_count:
                trend = "bullish"
            elif bearish_count > bullish_count:
                trend = "bearish"
        
        # 5. Detect inducements
        all_pools = buy_side_pools + sell_side_pools
        inducements = self.inducement_detector.detect_inducements(
            bars, all_pools, current_index
        )
        
        # 6. Calculate bias
        bias, bias_conf, bias_reason = self._calculate_bias(
            price, nearest_buy, nearest_sell, trend, inducements
        )
        
        # Build state
        state = LiquidityConceptsState(
            timestamp=timestamp,
            symbol=symbol,
            current_price=price,
            all_swings=swings,
            strong_highs=strong_highs,
            strong_lows=strong_lows,
            weak_highs=weak_highs,
            weak_lows=weak_lows,
            buy_side_pools=buy_side_pools,
            sell_side_pools=sell_side_pools,
            nearest_buy_side_pool=nearest_buy,
            nearest_sell_side_pool=nearest_sell,
            voids=voids,
            nearest_void_above=nearest_void_above,
            nearest_void_below=nearest_void_below,
            recent_bos=bos_events,
            last_bos=bos_events[-1] if bos_events else None,
            trend_direction=trend,
            recent_inducements=inducements,
            active_inducement=inducements[-1] if inducements else None,
            bias=bias,
            bias_confidence=bias_conf,
            bias_reasoning=bias_reason,
        )
        
        self._states[symbol] = state
        return state
    
    def analyze_zone_structure(
        self,
        symbol: str,
        bars: List[Dict[str, Any]],
        zone_start_index: int,
        zone_end_index: int,
    ) -> FractalStructureAnalysis:
        """
        Analyze the fractal structure of a supply/demand zone.
        
        Use this to determine if a zone is likely to hold based on
        internal liquidity pools.
        """
        state = self._states.get(symbol)
        swings = state.all_swings if state else []
        
        return self.fractal_analyzer.analyze_structure(
            bars, zone_start_index, zone_end_index, swings
        )
    
    def get_bias(self, symbol: str) -> Tuple[str, float, str]:
        """Get current liquidity-based bias."""
        state = self._states.get(symbol)
        if not state:
            return ('neutral', 0.0, 'No liquidity data')
        
        return (state.bias, state.bias_confidence, state.bias_reasoning)
    
    def get_nearest_pools(self, symbol: str) -> Dict[str, Any]:
        """Get nearest liquidity pools."""
        state = self._states.get(symbol)
        if not state:
            return {}
        
        return {
            'buy_side': {
                'price': state.nearest_buy_side_pool.price_level if state.nearest_buy_side_pool else None,
                'type': state.nearest_buy_side_pool.pool_type.name if state.nearest_buy_side_pool else None,
                'strength': state.nearest_buy_side_pool.get_pool_strength() if state.nearest_buy_side_pool else 0,
            },
            'sell_side': {
                'price': state.nearest_sell_side_pool.price_level if state.nearest_sell_side_pool else None,
                'type': state.nearest_sell_side_pool.pool_type.name if state.nearest_sell_side_pool else None,
                'strength': state.nearest_sell_side_pool.get_pool_strength() if state.nearest_sell_side_pool else 0,
            },
        }
    
    def get_state(self, symbol: str) -> Optional[LiquidityConceptsState]:
        """Get current state for a symbol."""
        return self._states.get(symbol)
    
    def _calculate_bias(
        self,
        price: float,
        nearest_buy: Optional[LiquidityPool],
        nearest_sell: Optional[LiquidityPool],
        trend: str,
        inducements: List[LiquidityInducement],
    ) -> Tuple[str, float, str]:
        """Calculate trading bias based on liquidity concepts."""
        
        # Check for active inducement (highest priority)
        if inducements:
            latest = inducements[-1]
            if latest.reversal_detected:
                return (
                    latest.signal_direction,
                    latest.confidence,
                    f"Liquidity inducement: {latest.inducement_type.name}"
                )
        
        # Calculate pool proximity bias
        buy_dist = (nearest_buy.price_level - price) / price if nearest_buy else float('inf')
        sell_dist = (price - nearest_sell.price_level) / price if nearest_sell else float('inf')
        
        # Closer to sell-side liquidity = smart money may sweep before going up
        if sell_dist < buy_dist * 0.5:
            if trend == "bullish":
                return ("bullish", 0.65, "Near sell-side liquidity in bullish trend - potential sweep before continuation")
            elif trend == "bearish":
                return ("bearish", 0.60, "Near sell-side liquidity in bearish trend - likely to sweep")
        
        # Closer to buy-side liquidity = smart money may sweep before going down
        elif buy_dist < sell_dist * 0.5:
            if trend == "bearish":
                return ("bearish", 0.65, "Near buy-side liquidity in bearish trend - potential sweep before continuation")
            elif trend == "bullish":
                return ("bullish", 0.60, "Near buy-side liquidity in bullish trend - likely to sweep")
        
        # Default to trend
        if trend != "neutral":
            return (trend, 0.50, f"Following {trend} trend based on BOS")
        
        return ("neutral", 0.30, "No clear liquidity bias")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_liquidity_concepts_engine(
    swing_lookback: int = 3,
    cluster_threshold_pct: float = 0.003,
    min_void_size_pct: float = 0.005,
) -> LiquidityConceptsEngine:
    """
    Factory function to create configured LiquidityConceptsEngine.
    
    Args:
        swing_lookback: Bars to confirm swing points (default 3)
        cluster_threshold_pct: % to consider swings clustered
        min_void_size_pct: Minimum void size as % of price
        
    Returns:
        Configured LiquidityConceptsEngine instance
    """
    return LiquidityConceptsEngine(
        swing_lookback=swing_lookback,
        cluster_threshold_pct=cluster_threshold_pct,
        min_void_size_pct=min_void_size_pct,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Engine
    'LiquidityConceptsEngine',
    'create_liquidity_concepts_engine',
    
    # Analyzers
    'ExtendedSwingAnalyzer',
    'LiquidityPoolDetector',
    'LiquidityVoidDetector',
    'FractalStructureAnalyzer',
    'LiquidityInducementDetector',
    
    # Enums
    'LiquidityPoolType',
    'LiquidityPoolSide',
    'SwingStrength',
    'MarketStructureType',
    'BOSType',
    'LiquidityInducementType',
    
    # Data Structures
    'SwingPointExtended',
    'LiquidityPool',
    'LiquidityVoid',
    'BreakOfStructure',
    'FractalStructureAnalysis',
    'LiquidityInducement',
    'LiquidityConceptsState',
]
