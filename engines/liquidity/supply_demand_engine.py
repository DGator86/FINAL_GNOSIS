"""
Supply and Demand Engine for Gnosis Trading System.

Implements the Supply and Demand trading methodology based on:
- Law of Supply and Demand (Economics)
- Market Equilibrium Theory
- Price Action Analysis

Key Concepts:
1. Demand Zones - Areas where buyers caused significant price increases
2. Supply Zones - Areas where sellers caused significant price decreases
3. Zone Strength - Validated by higher highs (demand) or lower lows (supply)
4. Zone Boundaries - Determined by volatility shifts after extremes

Version: 1.0.0
Integration: Works alongside Wyckoff, ICT, and Order Flow methodologies

Core Principles:
- Demand relates to BUYERS (seeking/demanding products)
- Supply relates to SELLERS (providing/supplying products)
- Law of Demand: Higher price = lower quantity demanded
- Law of Supply: Higher price = higher quantity supplied
- Market Equilibrium: Price seeks the point where supply = demand
- Supply/Demand Shifts: Changes in perception cause curve shifts

Zone Formation Rules:
- Demand Zone: Low between two highs (second high MUST be higher than first)
- Supply Zone: High between two lows (second low MUST be lower than first)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS - Supply and Demand Concepts
# ============================================================================

class ZoneType(Enum):
    """Type of supply/demand zone."""
    DEMAND = auto()    # Buying zone - low between two highs
    SUPPLY = auto()    # Selling zone - high between two lows


class ZoneStrength(Enum):
    """Strength of the zone based on validation."""
    STRONG = auto()      # Properly validated (HH for demand, LL for supply)
    MODERATE = auto()    # Partially validated
    WEAK = auto()        # Not validated - use with caution
    BROKEN = auto()      # Zone has been violated


class ZoneStatus(Enum):
    """Current status of the zone."""
    FRESH = auto()       # Never tested - highest probability
    TESTED = auto()      # Tested once - still valid
    RETESTED = auto()    # Tested multiple times - weakening
    BROKEN = auto()      # Price broke through - invalidated


class MarketEquilibrium(Enum):
    """Market equilibrium state."""
    BALANCED = auto()           # Price at equilibrium
    SEEKING_HIGHER = auto()     # Demand shift - seeking higher equilibrium
    SEEKING_LOWER = auto()      # Supply shift - seeking lower equilibrium


class ShiftType(Enum):
    """Type of supply/demand shift."""
    DEMAND_INCREASE = auto()    # More buyers willing to buy
    DEMAND_DECREASE = auto()    # Fewer buyers willing to buy
    SUPPLY_INCREASE = auto()    # More sellers willing to sell
    SUPPLY_DECREASE = auto()    # Fewer sellers willing to sell


class EntrySignal(Enum):
    """Entry signal types from S/D zones."""
    DEMAND_ZONE_TOUCH = auto()      # Price touching demand zone
    SUPPLY_ZONE_TOUCH = auto()      # Price touching supply zone
    DEMAND_ZONE_BOUNCE = auto()     # Price bouncing from demand zone
    SUPPLY_ZONE_BOUNCE = auto()     # Price bouncing from supply zone
    ZONE_BREAK = auto()             # Zone broken - potential reversal
    NO_SIGNAL = auto()              # No actionable signal


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SwingPoint:
    """A swing high or swing low point."""
    index: int
    price: float
    timestamp: datetime
    is_high: bool       # True = swing high, False = swing low
    confirmed: bool = False  # Whether it's been confirmed by subsequent price action


@dataclass
class ZoneBoundary:
    """Boundary of a supply/demand zone."""
    upper: float        # Upper boundary price
    lower: float        # Lower boundary price
    
    @property
    def midpoint(self) -> float:
        """Get the midpoint of the zone."""
        return (self.upper + self.lower) / 2
    
    @property
    def height(self) -> float:
        """Get the height of the zone."""
        return self.upper - self.lower


@dataclass
class SupplyDemandZone:
    """A supply or demand zone."""
    zone_type: ZoneType
    boundary: ZoneBoundary
    formation_time: datetime
    strength: ZoneStrength = ZoneStrength.MODERATE
    status: ZoneStatus = ZoneStatus.FRESH
    
    # Formation details
    origin_price: float = 0.0           # The extreme price that created the zone
    volatility_shift_price: float = 0.0  # Where volatility increased (other boundary)
    
    # Validation
    first_swing_price: float = 0.0      # First high/low before the zone
    second_swing_price: float = 0.0     # Second high/low after the zone
    momentum_confirmed: bool = False     # Whether momentum confirmed the zone
    
    # Testing history
    test_count: int = 0
    last_test_time: Optional[datetime] = None
    
    # Risk management levels
    stop_loss: float = 0.0
    take_profit_1: float = 0.0    # 1:1 R:R
    take_profit_2: float = 0.0    # 1:2 R:R
    take_profit_3: float = 0.0    # 1:3 R:R
    take_profit_4: float = 0.0    # 1:4 R:R
    
    def contains_price(self, price: float) -> bool:
        """Check if price is within the zone."""
        return self.boundary.lower <= price <= self.boundary.upper
    
    def distance_to_price(self, price: float) -> float:
        """Get distance from price to nearest zone boundary."""
        if price > self.boundary.upper:
            return price - self.boundary.upper
        elif price < self.boundary.lower:
            return self.boundary.lower - price
        return 0.0  # Price is inside zone
    
    def distance_percent(self, price: float) -> float:
        """Get distance as percentage of price."""
        dist = self.distance_to_price(price)
        return (dist / price) * 100 if price > 0 else 0.0


@dataclass
class SupplyDemandState:
    """Complete supply/demand analysis state."""
    timestamp: datetime
    symbol: str
    current_price: float
    
    # Detected zones
    demand_zones: List[SupplyDemandZone] = field(default_factory=list)
    supply_zones: List[SupplyDemandZone] = field(default_factory=list)
    
    # Active zones (nearest to current price)
    nearest_demand: Optional[SupplyDemandZone] = None
    nearest_supply: Optional[SupplyDemandZone] = None
    
    # Market state
    equilibrium_state: MarketEquilibrium = MarketEquilibrium.BALANCED
    recent_shift: Optional[ShiftType] = None
    
    # Current signals
    entry_signal: EntrySignal = EntrySignal.NO_SIGNAL
    signal_confidence: float = 0.0
    
    # Swing points for reference
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)


@dataclass
class ZoneEntry:
    """Entry signal from supply/demand zone."""
    direction: str                  # 'long' or 'short'
    confidence: float               # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit_1: float           # 1:1 R:R
    take_profit_2: float           # 1:2 R:R  
    take_profit_3: float           # 1:3 R:R
    take_profit_4: float           # 1:4 R:R
    zone: SupplyDemandZone
    signal_type: EntrySignal
    reasoning: str = ""
    risk_reward: float = 3.0       # Default 1:3 R:R


# ============================================================================
# SWING POINT DETECTOR
# ============================================================================

class SwingPointDetector:
    """
    Detects swing highs and swing lows in price data.
    
    A swing high is a high that is higher than the highs on both sides.
    A swing low is a low that is lower than the lows on both sides.
    
    Uses a configurable lookback period to define swing points.
    """
    
    def __init__(self, lookback: int = 3):
        """
        Initialize detector.
        
        Args:
            lookback: Number of bars on each side to confirm swing (default 3)
        """
        self.lookback = lookback
    
    def detect_swings(
        self,
        bars: List[Dict[str, Any]],
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Detect swing highs and lows in price data.
        
        Args:
            bars: List of OHLCV bars
            
        Returns:
            Tuple of (swing_highs, swing_lows)
        """
        if len(bars) < self.lookback * 2 + 1:
            return [], []
        
        swing_highs = []
        swing_lows = []
        
        for i in range(self.lookback, len(bars) - self.lookback):
            bar = bars[i]
            high = bar.get('high', bar.get('close', 0))
            low = bar.get('low', bar.get('close', 0))
            timestamp = bar.get('timestamp', datetime.now())
            
            # Check for swing high
            is_swing_high = True
            for j in range(1, self.lookback + 1):
                left_high = bars[i - j].get('high', bars[i - j].get('close', 0))
                right_high = bars[i + j].get('high', bars[i + j].get('close', 0))
                if high <= left_high or high <= right_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append(SwingPoint(
                    index=i,
                    price=high,
                    timestamp=timestamp,
                    is_high=True,
                    confirmed=True,
                ))
            
            # Check for swing low
            is_swing_low = True
            for j in range(1, self.lookback + 1):
                left_low = bars[i - j].get('low', bars[i - j].get('close', 0))
                right_low = bars[i + j].get('low', bars[i + j].get('close', 0))
                if low >= left_low or low >= right_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append(SwingPoint(
                    index=i,
                    price=low,
                    timestamp=timestamp,
                    is_high=False,
                    confirmed=True,
                ))
        
        return swing_highs, swing_lows


# ============================================================================
# ZONE BOUNDARY CALCULATOR
# ============================================================================

class ZoneBoundaryCalculator:
    """
    Calculates the boundaries of supply/demand zones.
    
    For Demand Zones:
    - Lower boundary: Absolute low between the two highs
    - Upper boundary: Point where volatility shifted (buyers decided to push up)
    
    For Supply Zones:
    - Upper boundary: Absolute high between the two lows
    - Lower boundary: Point where volatility shifted (sellers decided to push down)
    """
    
    def __init__(
        self,
        volatility_multiplier: float = 1.5,
        min_zone_height_pct: float = 0.1,
        max_zone_height_pct: float = 3.0,
    ):
        """
        Initialize calculator.
        
        Args:
            volatility_multiplier: Multiplier for detecting volatility shift
            min_zone_height_pct: Minimum zone height as % of price
            max_zone_height_pct: Maximum zone height as % of price
        """
        self.volatility_multiplier = volatility_multiplier
        self.min_zone_height_pct = min_zone_height_pct
        self.max_zone_height_pct = max_zone_height_pct
    
    def calculate_demand_zone(
        self,
        bars: List[Dict[str, Any]],
        low_index: int,
        first_high_index: int,
        second_high_index: int,
    ) -> Optional[ZoneBoundary]:
        """
        Calculate demand zone boundaries.
        
        Args:
            bars: Price bars
            low_index: Index of the low between highs
            first_high_index: Index of first high
            second_high_index: Index of second high
            
        Returns:
            ZoneBoundary or None
        """
        if low_index >= len(bars):
            return None
        
        # Lower boundary is the absolute low
        low_bar = bars[low_index]
        lower_boundary = low_bar.get('low', low_bar.get('close', 0))
        
        # Find upper boundary based on volatility shift
        upper_boundary = self._find_volatility_shift_up(bars, low_index)
        
        if upper_boundary is None or upper_boundary <= lower_boundary:
            # Fallback: use the open of the bar after the low
            if low_index + 1 < len(bars):
                next_bar = bars[low_index + 1]
                upper_boundary = next_bar.get('open', next_bar.get('close', lower_boundary * 1.005))
            else:
                upper_boundary = lower_boundary * 1.005  # Small default zone
        
        # Validate zone height
        zone_height_pct = ((upper_boundary - lower_boundary) / lower_boundary) * 100
        if zone_height_pct < self.min_zone_height_pct:
            upper_boundary = lower_boundary * (1 + self.min_zone_height_pct / 100)
        elif zone_height_pct > self.max_zone_height_pct:
            upper_boundary = lower_boundary * (1 + self.max_zone_height_pct / 100)
        
        return ZoneBoundary(upper=upper_boundary, lower=lower_boundary)
    
    def calculate_supply_zone(
        self,
        bars: List[Dict[str, Any]],
        high_index: int,
        first_low_index: int,
        second_low_index: int,
    ) -> Optional[ZoneBoundary]:
        """
        Calculate supply zone boundaries.
        
        Args:
            bars: Price bars
            high_index: Index of the high between lows
            first_low_index: Index of first low
            second_low_index: Index of second low
            
        Returns:
            ZoneBoundary or None
        """
        if high_index >= len(bars):
            return None
        
        # Upper boundary is the absolute high
        high_bar = bars[high_index]
        upper_boundary = high_bar.get('high', high_bar.get('close', 0))
        
        # Find lower boundary based on volatility shift
        lower_boundary = self._find_volatility_shift_down(bars, high_index)
        
        if lower_boundary is None or lower_boundary >= upper_boundary:
            # Fallback: use the open of the bar after the high
            if high_index + 1 < len(bars):
                next_bar = bars[high_index + 1]
                lower_boundary = next_bar.get('open', next_bar.get('close', upper_boundary * 0.995))
            else:
                lower_boundary = upper_boundary * 0.995  # Small default zone
        
        # Validate zone height
        zone_height_pct = ((upper_boundary - lower_boundary) / upper_boundary) * 100
        if zone_height_pct < self.min_zone_height_pct:
            lower_boundary = upper_boundary * (1 - self.min_zone_height_pct / 100)
        elif zone_height_pct > self.max_zone_height_pct:
            lower_boundary = upper_boundary * (1 - self.max_zone_height_pct / 100)
        
        return ZoneBoundary(upper=upper_boundary, lower=lower_boundary)
    
    def _find_volatility_shift_up(
        self,
        bars: List[Dict[str, Any]],
        start_index: int,
    ) -> Optional[float]:
        """
        Find where volatility shifted upward after a low.
        
        Returns:
            Price level of volatility shift
        """
        if start_index >= len(bars) - 1:
            return None
        
        # Calculate average range before the low
        lookback = min(5, start_index)
        if lookback < 2:
            return None
        
        avg_range = sum(
            bars[i].get('high', 0) - bars[i].get('low', 0)
            for i in range(start_index - lookback, start_index)
        ) / lookback
        
        # Find the first bar after low with range > avg * multiplier
        for i in range(start_index + 1, min(start_index + 10, len(bars))):
            bar = bars[i]
            bar_range = bar.get('high', 0) - bar.get('low', 0)
            
            if bar_range > avg_range * self.volatility_multiplier:
                # This bar shows volatility shift - use its open as upper boundary
                return bar.get('open', bar.get('low', 0))
        
        # Fallback: use the open of the bar after low
        if start_index + 1 < len(bars):
            return bars[start_index + 1].get('open', bars[start_index + 1].get('close', 0))
        
        return None
    
    def _find_volatility_shift_down(
        self,
        bars: List[Dict[str, Any]],
        start_index: int,
    ) -> Optional[float]:
        """
        Find where volatility shifted downward after a high.
        
        Returns:
            Price level of volatility shift
        """
        if start_index >= len(bars) - 1:
            return None
        
        # Calculate average range before the high
        lookback = min(5, start_index)
        if lookback < 2:
            return None
        
        avg_range = sum(
            bars[i].get('high', 0) - bars[i].get('low', 0)
            for i in range(start_index - lookback, start_index)
        ) / lookback
        
        # Find the first bar after high with range > avg * multiplier
        for i in range(start_index + 1, min(start_index + 10, len(bars))):
            bar = bars[i]
            bar_range = bar.get('high', 0) - bar.get('low', 0)
            
            if bar_range > avg_range * self.volatility_multiplier:
                # This bar shows volatility shift - use its open as lower boundary
                return bar.get('open', bar.get('high', 0))
        
        # Fallback: use the open of the bar after high
        if start_index + 1 < len(bars):
            return bars[start_index + 1].get('open', bars[start_index + 1].get('close', 0))
        
        return None


# ============================================================================
# SUPPLY DEMAND ZONE DETECTOR
# ============================================================================

class SupplyDemandZoneDetector:
    """
    Detects supply and demand zones from price data.
    
    Demand Zone Rules:
    - Occurs at the LOW between TWO HIGHS
    - Second high MUST be HIGHER than first high (momentum confirmation)
    - Zone is stronger when there's a clear volatility shift
    
    Supply Zone Rules:
    - Occurs at the HIGH between TWO LOWS
    - Second low MUST be LOWER than first low (momentum confirmation)
    - Zone is stronger when there's a clear volatility shift
    """
    
    def __init__(
        self,
        swing_lookback: int = 3,
        min_swing_distance: int = 3,
        max_zones: int = 10,
        volatility_multiplier: float = 1.5,
    ):
        """
        Initialize detector.
        
        Args:
            swing_lookback: Bars to confirm swing points
            min_swing_distance: Minimum bars between swings
            max_zones: Maximum zones to track per type
            volatility_multiplier: For boundary calculation
        """
        self.swing_detector = SwingPointDetector(lookback=swing_lookback)
        self.boundary_calculator = ZoneBoundaryCalculator(
            volatility_multiplier=volatility_multiplier
        )
        self.min_swing_distance = min_swing_distance
        self.max_zones = max_zones
    
    def detect_zones(
        self,
        bars: List[Dict[str, Any]],
    ) -> Tuple[List[SupplyDemandZone], List[SupplyDemandZone]]:
        """
        Detect supply and demand zones.
        
        Args:
            bars: List of OHLCV bars
            
        Returns:
            Tuple of (demand_zones, supply_zones)
        """
        # Detect swing points
        swing_highs, swing_lows = self.swing_detector.detect_swings(bars)
        
        # Detect demand zones (low between two highs)
        demand_zones = self._detect_demand_zones(bars, swing_highs, swing_lows)
        
        # Detect supply zones (high between two lows)
        supply_zones = self._detect_supply_zones(bars, swing_highs, swing_lows)
        
        return demand_zones, supply_zones
    
    def _detect_demand_zones(
        self,
        bars: List[Dict[str, Any]],
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
    ) -> List[SupplyDemandZone]:
        """
        Detect demand zones - lows between two highs where second high > first high.
        """
        zones = []
        
        # Need at least 2 swing highs
        if len(swing_highs) < 2:
            return zones
        
        # Iterate through pairs of swing highs
        for i in range(len(swing_highs) - 1):
            first_high = swing_highs[i]
            second_high = swing_highs[i + 1]
            
            # Check if second high is higher (momentum confirmation)
            if second_high.price <= first_high.price:
                continue  # Not a valid demand zone setup
            
            # Check minimum distance
            if second_high.index - first_high.index < self.min_swing_distance:
                continue
            
            # Find the lowest low between these two highs
            low_between = self._find_lowest_between(
                bars, 
                swing_lows, 
                first_high.index, 
                second_high.index
            )
            
            if low_between is None:
                continue
            
            # Calculate zone boundaries
            boundary = self.boundary_calculator.calculate_demand_zone(
                bars,
                low_between.index,
                first_high.index,
                second_high.index,
            )
            
            if boundary is None:
                continue
            
            # Create zone
            zone = SupplyDemandZone(
                zone_type=ZoneType.DEMAND,
                boundary=boundary,
                formation_time=low_between.timestamp,
                origin_price=low_between.price,
                volatility_shift_price=boundary.upper,
                first_swing_price=first_high.price,
                second_swing_price=second_high.price,
                momentum_confirmed=True,  # We already checked HH
                strength=ZoneStrength.STRONG,
                status=ZoneStatus.FRESH,
            )
            
            # Calculate risk management levels
            self._calculate_risk_levels(zone, is_long=True)
            
            zones.append(zone)
        
        # Sort by formation time and limit
        zones.sort(key=lambda z: z.formation_time, reverse=True)
        return zones[:self.max_zones]
    
    def _detect_supply_zones(
        self,
        bars: List[Dict[str, Any]],
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
    ) -> List[SupplyDemandZone]:
        """
        Detect supply zones - highs between two lows where second low < first low.
        """
        zones = []
        
        # Need at least 2 swing lows
        if len(swing_lows) < 2:
            return zones
        
        # Iterate through pairs of swing lows
        for i in range(len(swing_lows) - 1):
            first_low = swing_lows[i]
            second_low = swing_lows[i + 1]
            
            # Check if second low is lower (momentum confirmation)
            if second_low.price >= first_low.price:
                continue  # Not a valid supply zone setup
            
            # Check minimum distance
            if second_low.index - first_low.index < self.min_swing_distance:
                continue
            
            # Find the highest high between these two lows
            high_between = self._find_highest_between(
                bars,
                swing_highs,
                first_low.index,
                second_low.index
            )
            
            if high_between is None:
                continue
            
            # Calculate zone boundaries
            boundary = self.boundary_calculator.calculate_supply_zone(
                bars,
                high_between.index,
                first_low.index,
                second_low.index,
            )
            
            if boundary is None:
                continue
            
            # Create zone
            zone = SupplyDemandZone(
                zone_type=ZoneType.SUPPLY,
                boundary=boundary,
                formation_time=high_between.timestamp,
                origin_price=high_between.price,
                volatility_shift_price=boundary.lower,
                first_swing_price=first_low.price,
                second_swing_price=second_low.price,
                momentum_confirmed=True,  # We already checked LL
                strength=ZoneStrength.STRONG,
                status=ZoneStatus.FRESH,
            )
            
            # Calculate risk management levels
            self._calculate_risk_levels(zone, is_long=False)
            
            zones.append(zone)
        
        # Sort by formation time and limit
        zones.sort(key=lambda z: z.formation_time, reverse=True)
        return zones[:self.max_zones]
    
    def _find_lowest_between(
        self,
        bars: List[Dict[str, Any]],
        swing_lows: List[SwingPoint],
        start_index: int,
        end_index: int,
    ) -> Optional[SwingPoint]:
        """Find the lowest swing low between two indices."""
        candidates = [
            sl for sl in swing_lows
            if start_index < sl.index < end_index
        ]
        
        if candidates:
            return min(candidates, key=lambda sl: sl.price)
        
        # Fallback: find lowest bar
        lowest_price = float('inf')
        lowest_index = -1
        lowest_time = datetime.now()
        
        for i in range(start_index + 1, end_index):
            if i >= len(bars):
                break
            bar = bars[i]
            low = bar.get('low', bar.get('close', float('inf')))
            if low < lowest_price:
                lowest_price = low
                lowest_index = i
                lowest_time = bar.get('timestamp', datetime.now())
        
        if lowest_index >= 0:
            return SwingPoint(
                index=lowest_index,
                price=lowest_price,
                timestamp=lowest_time,
                is_high=False,
                confirmed=False,
            )
        
        return None
    
    def _find_highest_between(
        self,
        bars: List[Dict[str, Any]],
        swing_highs: List[SwingPoint],
        start_index: int,
        end_index: int,
    ) -> Optional[SwingPoint]:
        """Find the highest swing high between two indices."""
        candidates = [
            sh for sh in swing_highs
            if start_index < sh.index < end_index
        ]
        
        if candidates:
            return max(candidates, key=lambda sh: sh.price)
        
        # Fallback: find highest bar
        highest_price = float('-inf')
        highest_index = -1
        highest_time = datetime.now()
        
        for i in range(start_index + 1, end_index):
            if i >= len(bars):
                break
            bar = bars[i]
            high = bar.get('high', bar.get('close', float('-inf')))
            if high > highest_price:
                highest_price = high
                highest_index = i
                highest_time = bar.get('timestamp', datetime.now())
        
        if highest_index >= 0:
            return SwingPoint(
                index=highest_index,
                price=highest_price,
                timestamp=highest_time,
                is_high=True,
                confirmed=False,
            )
        
        return None
    
    def _calculate_risk_levels(
        self,
        zone: SupplyDemandZone,
        is_long: bool,
    ) -> None:
        """Calculate stop loss and take profit levels."""
        if is_long:
            # Long trade from demand zone
            zone.stop_loss = zone.boundary.lower  # Stop below zone
            risk = zone.boundary.upper - zone.stop_loss
            zone.take_profit_1 = zone.boundary.upper + risk * 1.0
            zone.take_profit_2 = zone.boundary.upper + risk * 2.0
            zone.take_profit_3 = zone.boundary.upper + risk * 3.0
            zone.take_profit_4 = zone.boundary.upper + risk * 4.0
        else:
            # Short trade from supply zone
            zone.stop_loss = zone.boundary.upper  # Stop above zone
            risk = zone.stop_loss - zone.boundary.lower
            zone.take_profit_1 = zone.boundary.lower - risk * 1.0
            zone.take_profit_2 = zone.boundary.lower - risk * 2.0
            zone.take_profit_3 = zone.boundary.lower - risk * 3.0
            zone.take_profit_4 = zone.boundary.lower - risk * 4.0


# ============================================================================
# ZONE STATUS TRACKER
# ============================================================================

class ZoneStatusTracker:
    """
    Tracks the status of supply/demand zones over time.
    
    Zones can be:
    - Fresh: Never tested
    - Tested: Touched once but held
    - Retested: Touched multiple times
    - Broken: Price passed through the zone
    """
    
    def update_zone_status(
        self,
        zone: SupplyDemandZone,
        current_price: float,
        previous_price: float,
    ) -> None:
        """
        Update zone status based on price action.
        
        Args:
            zone: The zone to update
            current_price: Current market price
            previous_price: Previous bar's close
        """
        # Check if price entered the zone
        price_entered_zone = (
            zone.contains_price(current_price) and 
            not zone.contains_price(previous_price)
        )
        
        # Check if zone was broken
        if zone.zone_type == ZoneType.DEMAND:
            # Demand zone broken if price closes below lower boundary
            if current_price < zone.boundary.lower:
                zone.status = ZoneStatus.BROKEN
                zone.strength = ZoneStrength.BROKEN
                return
        else:  # Supply zone
            # Supply zone broken if price closes above upper boundary
            if current_price > zone.boundary.upper:
                zone.status = ZoneStatus.BROKEN
                zone.strength = ZoneStrength.BROKEN
                return
        
        # Update test count if price entered zone
        if price_entered_zone:
            zone.test_count += 1
            zone.last_test_time = datetime.now()
            
            if zone.test_count == 1:
                zone.status = ZoneStatus.TESTED
            elif zone.test_count >= 2:
                zone.status = ZoneStatus.RETESTED
                # Zones weaken with multiple tests
                if zone.strength == ZoneStrength.STRONG:
                    zone.strength = ZoneStrength.MODERATE
                elif zone.strength == ZoneStrength.MODERATE:
                    zone.strength = ZoneStrength.WEAK
    
    def get_zone_reliability(self, zone: SupplyDemandZone) -> float:
        """
        Get reliability score for a zone (0.0 to 1.0).
        
        Fresh, strong zones are most reliable.
        """
        base_score = 0.5
        
        # Strength bonus
        if zone.strength == ZoneStrength.STRONG:
            base_score += 0.3
        elif zone.strength == ZoneStrength.MODERATE:
            base_score += 0.15
        elif zone.strength == ZoneStrength.WEAK:
            base_score += 0.0
        elif zone.strength == ZoneStrength.BROKEN:
            return 0.0
        
        # Status bonus
        if zone.status == ZoneStatus.FRESH:
            base_score += 0.2
        elif zone.status == ZoneStatus.TESTED:
            base_score += 0.1
        elif zone.status == ZoneStatus.RETESTED:
            base_score -= 0.1
        elif zone.status == ZoneStatus.BROKEN:
            return 0.0
        
        # Momentum confirmation bonus
        if zone.momentum_confirmed:
            base_score += 0.1
        
        return min(max(base_score, 0.0), 1.0)


# ============================================================================
# MAIN SUPPLY DEMAND ENGINE
# ============================================================================

class SupplyDemandEngine:
    """
    Comprehensive Supply and Demand Analysis Engine.
    
    Implements the complete supply/demand methodology:
    - Zone detection based on swing highs/lows
    - Zone strength validation (higher high / lower low)
    - Zone boundary calculation (volatility shifts)
    - Zone status tracking (fresh, tested, broken)
    - Entry signal generation with risk management
    
    Integration points:
    - Works with LiquidityEngineV4 (Wyckoff)
    - Works with ICTEngine
    - Works with OrderFlowEngine
    - Provides signals for LiquidityAgentV5
    """
    
    def __init__(
        self,
        swing_lookback: int = 3,
        min_swing_distance: int = 3,
        max_zones: int = 10,
        volatility_multiplier: float = 1.5,
        default_risk_reward: float = 3.0,
    ):
        """
        Initialize the Supply/Demand Engine.
        
        Args:
            swing_lookback: Bars to confirm swing points
            min_swing_distance: Minimum bars between swings
            max_zones: Maximum zones to track per type
            volatility_multiplier: For boundary calculation
            default_risk_reward: Default R:R ratio for targets
        """
        self.zone_detector = SupplyDemandZoneDetector(
            swing_lookback=swing_lookback,
            min_swing_distance=min_swing_distance,
            max_zones=max_zones,
            volatility_multiplier=volatility_multiplier,
        )
        self.status_tracker = ZoneStatusTracker()
        self.default_risk_reward = default_risk_reward
        
        # State tracking per symbol
        self._states: Dict[str, SupplyDemandState] = {}
        self._previous_prices: Dict[str, float] = {}
        
        logger.info(
            f"SupplyDemandEngine initialized | "
            f"swing_lookback={swing_lookback}, max_zones={max_zones}"
        )
    
    def analyze(
        self,
        symbol: str,
        bars: List[Dict[str, Any]],
        current_price: Optional[float] = None,
    ) -> SupplyDemandState:
        """
        Perform complete supply/demand analysis.
        
        Args:
            symbol: Trading symbol
            bars: OHLCV bar data
            current_price: Current market price
            
        Returns:
            SupplyDemandState with all analysis results
        """
        if not bars:
            return SupplyDemandState(
                timestamp=datetime.now(),
                symbol=symbol,
                current_price=current_price or 0,
            )
        
        timestamp = datetime.now()
        price = current_price or bars[-1].get('close', 0)
        previous_price = self._previous_prices.get(symbol, price)
        
        # Detect zones
        demand_zones, supply_zones = self.zone_detector.detect_zones(bars)
        
        # Update zone statuses
        for zone in demand_zones + supply_zones:
            self.status_tracker.update_zone_status(zone, price, previous_price)
        
        # Filter out broken zones
        demand_zones = [z for z in demand_zones if z.status != ZoneStatus.BROKEN]
        supply_zones = [z for z in supply_zones if z.status != ZoneStatus.BROKEN]
        
        # Find nearest zones
        nearest_demand = self._find_nearest_zone(demand_zones, price, below=True)
        nearest_supply = self._find_nearest_zone(supply_zones, price, below=False)
        
        # Determine market equilibrium state
        equilibrium_state = self._determine_equilibrium(price, nearest_demand, nearest_supply)
        
        # Detect entry signals
        entry_signal, signal_confidence = self._detect_entry_signal(
            price, previous_price, nearest_demand, nearest_supply
        )
        
        # Get swing points for reference
        swing_highs, swing_lows = self.zone_detector.swing_detector.detect_swings(bars)
        
        # Build state
        state = SupplyDemandState(
            timestamp=timestamp,
            symbol=symbol,
            current_price=price,
            demand_zones=demand_zones,
            supply_zones=supply_zones,
            nearest_demand=nearest_demand,
            nearest_supply=nearest_supply,
            equilibrium_state=equilibrium_state,
            entry_signal=entry_signal,
            signal_confidence=signal_confidence,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
        )
        
        # Store state
        self._states[symbol] = state
        self._previous_prices[symbol] = price
        
        return state
    
    def get_entry_signals(
        self,
        symbol: str,
        state: Optional[SupplyDemandState] = None,
    ) -> List[ZoneEntry]:
        """
        Generate entry signals from supply/demand zones.
        
        Args:
            symbol: Trading symbol
            state: Optional pre-computed state
            
        Returns:
            List of ZoneEntry signals
        """
        state = state or self._states.get(symbol)
        if not state:
            return []
        
        entries = []
        price = state.current_price
        
        # Check demand zone entries (long)
        if state.nearest_demand and state.nearest_demand.status != ZoneStatus.BROKEN:
            zone = state.nearest_demand
            
            # Price in or near demand zone
            if zone.contains_price(price) or zone.distance_percent(price) < 0.5:
                reliability = self.status_tracker.get_zone_reliability(zone)
                
                entry = ZoneEntry(
                    direction='long',
                    confidence=reliability,
                    entry_price=zone.boundary.upper,  # Enter at upper boundary
                    stop_loss=zone.stop_loss,
                    take_profit_1=zone.take_profit_1,
                    take_profit_2=zone.take_profit_2,
                    take_profit_3=zone.take_profit_3,
                    take_profit_4=zone.take_profit_4,
                    zone=zone,
                    signal_type=EntrySignal.DEMAND_ZONE_TOUCH,
                    reasoning=f"Price in demand zone ({zone.status.name}, {zone.strength.name})",
                    risk_reward=self.default_risk_reward,
                )
                entries.append(entry)
        
        # Check supply zone entries (short)
        if state.nearest_supply and state.nearest_supply.status != ZoneStatus.BROKEN:
            zone = state.nearest_supply
            
            # Price in or near supply zone
            if zone.contains_price(price) or zone.distance_percent(price) < 0.5:
                reliability = self.status_tracker.get_zone_reliability(zone)
                
                entry = ZoneEntry(
                    direction='short',
                    confidence=reliability,
                    entry_price=zone.boundary.lower,  # Enter at lower boundary
                    stop_loss=zone.stop_loss,
                    take_profit_1=zone.take_profit_1,
                    take_profit_2=zone.take_profit_2,
                    take_profit_3=zone.take_profit_3,
                    take_profit_4=zone.take_profit_4,
                    zone=zone,
                    signal_type=EntrySignal.SUPPLY_ZONE_TOUCH,
                    reasoning=f"Price in supply zone ({zone.status.name}, {zone.strength.name})",
                    risk_reward=self.default_risk_reward,
                )
                entries.append(entry)
        
        return entries
    
    def get_bias(self, symbol: str) -> Tuple[str, float, str]:
        """
        Get current supply/demand bias.
        
        Returns:
            Tuple of (direction, confidence, reasoning)
        """
        state = self._states.get(symbol)
        if not state:
            return ('neutral', 0.0, 'No supply/demand data')
        
        price = state.current_price
        demand = state.nearest_demand
        supply = state.nearest_supply
        
        # Calculate distances
        demand_dist = demand.distance_percent(price) if demand else float('inf')
        supply_dist = supply.distance_percent(price) if supply else float('inf')
        
        # Determine bias based on proximity and equilibrium
        if state.equilibrium_state == MarketEquilibrium.SEEKING_HIGHER:
            return ('bullish', 0.65, 'Market seeking higher equilibrium')
        elif state.equilibrium_state == MarketEquilibrium.SEEKING_LOWER:
            return ('bearish', 0.65, 'Market seeking lower equilibrium')
        
        # Closer zone has stronger pull
        if demand_dist < supply_dist and demand:
            reliability = self.status_tracker.get_zone_reliability(demand)
            return ('bullish', reliability * 0.8, f'Near demand zone ({demand.status.name})')
        elif supply_dist < demand_dist and supply:
            reliability = self.status_tracker.get_zone_reliability(supply)
            return ('bearish', reliability * 0.8, f'Near supply zone ({supply.status.name})')
        
        return ('neutral', 0.3, 'Between supply and demand zones')
    
    def get_key_levels(self, symbol: str) -> Dict[str, Any]:
        """Get key supply/demand levels for a symbol."""
        state = self._states.get(symbol)
        if not state:
            return {}
        
        levels = {
            'demand_zones': [],
            'supply_zones': [],
            'nearest_demand': None,
            'nearest_supply': None,
        }
        
        for zone in state.demand_zones:
            levels['demand_zones'].append({
                'upper': zone.boundary.upper,
                'lower': zone.boundary.lower,
                'strength': zone.strength.name,
                'status': zone.status.name,
                'reliability': self.status_tracker.get_zone_reliability(zone),
            })
        
        for zone in state.supply_zones:
            levels['supply_zones'].append({
                'upper': zone.boundary.upper,
                'lower': zone.boundary.lower,
                'strength': zone.strength.name,
                'status': zone.status.name,
                'reliability': self.status_tracker.get_zone_reliability(zone),
            })
        
        if state.nearest_demand:
            levels['nearest_demand'] = {
                'upper': state.nearest_demand.boundary.upper,
                'lower': state.nearest_demand.boundary.lower,
                'distance_pct': state.nearest_demand.distance_percent(state.current_price),
            }
        
        if state.nearest_supply:
            levels['nearest_supply'] = {
                'upper': state.nearest_supply.boundary.upper,
                'lower': state.nearest_supply.boundary.lower,
                'distance_pct': state.nearest_supply.distance_percent(state.current_price),
            }
        
        return levels
    
    def get_state(self, symbol: str) -> Optional[SupplyDemandState]:
        """Get current state for a symbol."""
        return self._states.get(symbol)
    
    def _find_nearest_zone(
        self,
        zones: List[SupplyDemandZone],
        price: float,
        below: bool = True,
    ) -> Optional[SupplyDemandZone]:
        """
        Find the nearest zone to current price.
        
        Args:
            zones: List of zones
            price: Current price
            below: If True, find zone below price (demand); if False, above (supply)
            
        Returns:
            Nearest zone or None
        """
        if not zones:
            return None
        
        if below:
            # For demand zones, find highest zone that is below or contains price
            candidates = [
                z for z in zones
                if z.boundary.upper <= price or z.contains_price(price)
            ]
            if candidates:
                return max(candidates, key=lambda z: z.boundary.upper)
        else:
            # For supply zones, find lowest zone that is above or contains price
            candidates = [
                z for z in zones
                if z.boundary.lower >= price or z.contains_price(price)
            ]
            if candidates:
                return min(candidates, key=lambda z: z.boundary.lower)
        
        # Fallback: closest zone
        return min(zones, key=lambda z: z.distance_to_price(price))
    
    def _determine_equilibrium(
        self,
        price: float,
        demand: Optional[SupplyDemandZone],
        supply: Optional[SupplyDemandZone],
    ) -> MarketEquilibrium:
        """Determine current market equilibrium state."""
        if not demand and not supply:
            return MarketEquilibrium.BALANCED
        
        # Check if price is in a zone
        if demand and demand.contains_price(price):
            return MarketEquilibrium.SEEKING_HIGHER
        if supply and supply.contains_price(price):
            return MarketEquilibrium.SEEKING_LOWER
        
        # Check proximity
        demand_dist = demand.distance_percent(price) if demand else float('inf')
        supply_dist = supply.distance_percent(price) if supply else float('inf')
        
        if demand_dist < 1.0:  # Within 1% of demand
            return MarketEquilibrium.SEEKING_HIGHER
        if supply_dist < 1.0:  # Within 1% of supply
            return MarketEquilibrium.SEEKING_LOWER
        
        return MarketEquilibrium.BALANCED
    
    def _detect_entry_signal(
        self,
        price: float,
        previous_price: float,
        demand: Optional[SupplyDemandZone],
        supply: Optional[SupplyDemandZone],
    ) -> Tuple[EntrySignal, float]:
        """
        Detect entry signals based on price action at zones.
        
        Returns:
            Tuple of (signal_type, confidence)
        """
        # Check demand zone
        if demand:
            in_zone_now = demand.contains_price(price)
            was_in_zone = demand.contains_price(previous_price)
            
            # Just entered demand zone
            if in_zone_now and not was_in_zone:
                confidence = self.status_tracker.get_zone_reliability(demand)
                return EntrySignal.DEMAND_ZONE_TOUCH, confidence
            
            # Bouncing from demand zone
            if was_in_zone and price > demand.boundary.upper:
                confidence = self.status_tracker.get_zone_reliability(demand) * 1.1
                return EntrySignal.DEMAND_ZONE_BOUNCE, min(confidence, 0.95)
        
        # Check supply zone
        if supply:
            in_zone_now = supply.contains_price(price)
            was_in_zone = supply.contains_price(previous_price)
            
            # Just entered supply zone
            if in_zone_now and not was_in_zone:
                confidence = self.status_tracker.get_zone_reliability(supply)
                return EntrySignal.SUPPLY_ZONE_TOUCH, confidence
            
            # Bouncing from supply zone
            if was_in_zone and price < supply.boundary.lower:
                confidence = self.status_tracker.get_zone_reliability(supply) * 1.1
                return EntrySignal.SUPPLY_ZONE_BOUNCE, min(confidence, 0.95)
        
        return EntrySignal.NO_SIGNAL, 0.0


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_supply_demand_engine(
    swing_lookback: int = 3,
    min_swing_distance: int = 3,
    max_zones: int = 10,
    volatility_multiplier: float = 1.5,
    default_risk_reward: float = 3.0,
) -> SupplyDemandEngine:
    """
    Factory function to create configured SupplyDemandEngine.
    
    Args:
        swing_lookback: Bars to confirm swing points (default 3)
        min_swing_distance: Minimum bars between swings
        max_zones: Maximum zones to track per type
        volatility_multiplier: For boundary calculation
        default_risk_reward: Default R:R ratio
        
    Returns:
        Configured SupplyDemandEngine instance
    """
    return SupplyDemandEngine(
        swing_lookback=swing_lookback,
        min_swing_distance=min_swing_distance,
        max_zones=max_zones,
        volatility_multiplier=volatility_multiplier,
        default_risk_reward=default_risk_reward,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Engine
    'SupplyDemandEngine',
    'create_supply_demand_engine',
    
    # Detectors
    'SwingPointDetector',
    'SupplyDemandZoneDetector',
    'ZoneBoundaryCalculator',
    'ZoneStatusTracker',
    
    # Enums
    'ZoneType',
    'ZoneStrength',
    'ZoneStatus',
    'MarketEquilibrium',
    'ShiftType',
    'EntrySignal',
    
    # Data Structures
    'SwingPoint',
    'ZoneBoundary',
    'SupplyDemandZone',
    'SupplyDemandState',
    'ZoneEntry',
]
