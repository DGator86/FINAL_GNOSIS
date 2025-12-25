"""
Order Flow Engine for Gnosis Trading System.

Implements comprehensive order flow analysis based on:
- Auction Market Theory
- Market Microstructure
- Footprint Charts
- Cumulative Volume Delta (CVD)
- Volume Profile Analysis

Version: 1.0.0
Integration: Works alongside Wyckoff (V4) and ICT methodologies

Key Components:
1. Footprint Analysis - Bid/Ask aggression, imbalance, absorption
2. CVD Analysis - Cumulative delta for exhaustion/divergence
3. Volume Profile - POC, Value Area, price acceptance/rejection
4. Market Microstructure - Order types, liquidity, market depth
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS - Order Flow Concepts
# ============================================================================

class OrderFlowSignal(Enum):
    """Order flow signal types."""
    ABSORPTION = auto()          # Large orders absorbed without price movement
    INITIATION = auto()          # Aggressive orders moving price
    EXHAUSTION = auto()          # Buying/selling pressure weakening
    DIVERGENCE = auto()          # Price vs delta divergence
    IMBALANCE = auto()           # Significant bid/ask imbalance
    DELTA_FLIP = auto()          # Delta changes direction
    POC_TEST = auto()            # Price testing Point of Control
    VALUE_AREA_BREAK = auto()    # Breaking out of value area
    HVN_REJECTION = auto()       # Rejection at high volume node
    LVN_ACCELERATION = auto()    # Acceleration through low volume node
    STACKED_IMBALANCE = auto()   # Multiple consecutive imbalances
    UNFINISHED_AUCTION = auto()  # Single prints indicating incomplete auction


class DeltaType(Enum):
    """Delta calculation types."""
    POSITIVE = auto()    # More buying than selling
    NEGATIVE = auto()    # More selling than buying
    NEUTRAL = auto()     # Balanced


class VolumeNodeType(Enum):
    """Volume node types."""
    HVN = auto()    # High Volume Node - support/resistance
    LVN = auto()    # Low Volume Node - fast price movement
    POC = auto()    # Point of Control - fairest price


class AuctionState(Enum):
    """Market auction state."""
    BALANCED = auto()        # Trading within value area
    INITIATING_LONG = auto() # Breaking out upward
    INITIATING_SHORT = auto()# Breaking out downward
    RESPONSIVE_LONG = auto() # Responsive buying at lows
    RESPONSIVE_SHORT = auto()# Responsive selling at highs
    ROTATIONAL = auto()      # Price rotating without direction


class FootprintPattern(Enum):
    """Footprint chart patterns."""
    ABSORPTION_AT_HIGH = auto()    # Buying absorbed at highs (bearish)
    ABSORPTION_AT_LOW = auto()     # Selling absorbed at lows (bullish)
    INITIATIVE_BUYING = auto()     # Aggressive buying lifting offers
    INITIATIVE_SELLING = auto()    # Aggressive selling hitting bids
    STOP_HUNT_COMPLETE = auto()    # Stops hit, reversal likely
    BREAKOUT_CONFIRMATION = auto() # Volume confirms breakout
    FAILED_AUCTION = auto()        # Price rejected, auction incomplete


class ImbalanceType(Enum):
    """Bid/Ask imbalance types."""
    BID_IMBALANCE = auto()    # More buying pressure (200%+ vs ask)
    ASK_IMBALANCE = auto()    # More selling pressure (200%+ vs bid)
    DIAGONAL_BID = auto()     # Stacked bid imbalances (very bullish)
    DIAGONAL_ASK = auto()     # Stacked ask imbalances (very bearish)
    BALANCED = auto()         # No significant imbalance


class MarketParticipant(Enum):
    """Types of market participants."""
    SPECULATOR = auto()      # Directional betting
    ARBITRAGER = auto()      # Price discrepancy exploitation
    HEDGER = auto()          # Risk management
    MARKET_MAKER = auto()    # Providing liquidity
    INSTITUTIONAL = auto()   # Large position management
    RETAIL = auto()          # Small individual traders


class OrderType(Enum):
    """Order types in market microstructure."""
    MARKET = auto()          # Immediate execution at best price
    LIMIT = auto()           # Execute at specific price or better
    STOP = auto()            # Becomes market order when triggered
    STOP_LIMIT = auto()      # Becomes limit order when triggered
    ICEBERG = auto()         # Large order shown in small pieces
    HIDDEN = auto()          # Non-displayed order


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FootprintCell:
    """Single cell in footprint chart."""
    price: float
    bid_volume: float            # Volume hitting bids (selling)
    ask_volume: float            # Volume hitting asks (buying)
    delta: float = 0.0           # ask_volume - bid_volume
    imbalance_ratio: float = 0.0 # Higher ratio = stronger imbalance
    imbalance_type: ImbalanceType = ImbalanceType.BALANCED
    
    def __post_init__(self):
        self.delta = self.ask_volume - self.bid_volume
        total = self.bid_volume + self.ask_volume
        if total > 0:
            # Calculate imbalance ratio
            if self.bid_volume > 0 and self.ask_volume > 0:
                self.imbalance_ratio = max(
                    self.ask_volume / self.bid_volume,
                    self.bid_volume / self.ask_volume
                )
                # 200% threshold for imbalance (2:1 ratio)
                if self.ask_volume >= self.bid_volume * 2:
                    self.imbalance_type = ImbalanceType.BID_IMBALANCE
                elif self.bid_volume >= self.ask_volume * 2:
                    self.imbalance_type = ImbalanceType.ASK_IMBALANCE
            elif self.ask_volume > 0:
                self.imbalance_type = ImbalanceType.BID_IMBALANCE
                self.imbalance_ratio = float('inf')
            elif self.bid_volume > 0:
                self.imbalance_type = ImbalanceType.ASK_IMBALANCE
                self.imbalance_ratio = float('inf')


@dataclass
class FootprintBar:
    """Complete footprint bar (candle with order flow data)."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    cells: List[FootprintCell] = field(default_factory=list)
    total_volume: float = 0.0
    total_delta: float = 0.0
    max_delta: float = 0.0
    min_delta: float = 0.0
    poc_price: float = 0.0       # Price with highest volume
    value_area_high: float = 0.0
    value_area_low: float = 0.0
    patterns: List[FootprintPattern] = field(default_factory=list)
    
    def calculate_metrics(self) -> None:
        """Calculate bar metrics from cells."""
        if not self.cells:
            return
            
        # Calculate totals
        self.total_volume = sum(c.bid_volume + c.ask_volume for c in self.cells)
        self.total_delta = sum(c.delta for c in self.cells)
        
        # Find max/min delta cells
        deltas = [c.delta for c in self.cells]
        self.max_delta = max(deltas) if deltas else 0
        self.min_delta = min(deltas) if deltas else 0
        
        # Find POC (highest volume price)
        if self.cells:
            poc_cell = max(self.cells, key=lambda c: c.bid_volume + c.ask_volume)
            self.poc_price = poc_cell.price
            
        # Calculate Value Area (70% of volume)
        self._calculate_value_area()
    
    def _calculate_value_area(self) -> None:
        """Calculate value area containing 70% of volume."""
        if not self.cells or self.total_volume == 0:
            return
            
        target_volume = self.total_volume * 0.70
        
        # Sort cells by price
        sorted_cells = sorted(self.cells, key=lambda c: c.price)
        
        # Start from POC and expand outward
        poc_idx = next(
            (i for i, c in enumerate(sorted_cells) if c.price == self.poc_price),
            len(sorted_cells) // 2
        )
        
        included = {poc_idx}
        current_volume = sorted_cells[poc_idx].bid_volume + sorted_cells[poc_idx].ask_volume
        
        lower_idx = poc_idx - 1
        upper_idx = poc_idx + 1
        
        while current_volume < target_volume:
            lower_vol = (
                sorted_cells[lower_idx].bid_volume + sorted_cells[lower_idx].ask_volume
                if lower_idx >= 0 else 0
            )
            upper_vol = (
                sorted_cells[upper_idx].bid_volume + sorted_cells[upper_idx].ask_volume
                if upper_idx < len(sorted_cells) else 0
            )
            
            if lower_vol >= upper_vol and lower_idx >= 0:
                included.add(lower_idx)
                current_volume += lower_vol
                lower_idx -= 1
            elif upper_idx < len(sorted_cells):
                included.add(upper_idx)
                current_volume += upper_vol
                upper_idx += 1
            else:
                break
        
        included_cells = [sorted_cells[i] for i in sorted(included)]
        if included_cells:
            self.value_area_low = min(c.price for c in included_cells)
            self.value_area_high = max(c.price for c in included_cells)


@dataclass
class CVDPoint:
    """Cumulative Volume Delta data point."""
    timestamp: datetime
    price: float
    delta: float              # Single bar delta
    cumulative_delta: float   # Running total
    delta_type: DeltaType = DeltaType.NEUTRAL
    divergence: bool = False  # Price/delta divergence
    
    def __post_init__(self):
        if self.delta > 0:
            self.delta_type = DeltaType.POSITIVE
        elif self.delta < 0:
            self.delta_type = DeltaType.NEGATIVE


@dataclass
class VolumeNode:
    """Volume profile node."""
    price: float
    volume: float
    node_type: VolumeNodeType
    is_poc: bool = False
    is_value_area: bool = False


@dataclass
class VolumeProfile:
    """Complete volume profile for a session/range."""
    start_time: datetime
    end_time: datetime
    nodes: List[VolumeNode] = field(default_factory=list)
    poc: float = 0.0             # Point of Control price
    value_area_high: float = 0.0
    value_area_low: float = 0.0
    total_volume: float = 0.0
    developing: bool = True      # Still building
    
    def calculate(self) -> None:
        """Calculate profile metrics."""
        if not self.nodes:
            return
            
        # Find POC
        self.total_volume = sum(n.volume for n in self.nodes)
        if self.nodes:
            poc_node = max(self.nodes, key=lambda n: n.volume)
            self.poc = poc_node.price
            poc_node.is_poc = True
            poc_node.node_type = VolumeNodeType.POC
        
        # Calculate value area (70% of volume)
        self._calculate_value_area()
        
        # Classify nodes as HVN or LVN
        self._classify_nodes()
    
    def _calculate_value_area(self) -> None:
        """Calculate 70% value area."""
        if self.total_volume == 0:
            return
            
        target = self.total_volume * 0.70
        sorted_nodes = sorted(self.nodes, key=lambda n: n.price)
        
        # Find POC index
        poc_idx = next(
            (i for i, n in enumerate(sorted_nodes) if n.price == self.poc),
            len(sorted_nodes) // 2
        )
        
        included = {poc_idx}
        current = sorted_nodes[poc_idx].volume
        lower, upper = poc_idx - 1, poc_idx + 1
        
        while current < target and (lower >= 0 or upper < len(sorted_nodes)):
            lower_vol = sorted_nodes[lower].volume if lower >= 0 else 0
            upper_vol = sorted_nodes[upper].volume if upper < len(sorted_nodes) else 0
            
            if lower_vol >= upper_vol and lower >= 0:
                included.add(lower)
                current += lower_vol
                lower -= 1
            elif upper < len(sorted_nodes):
                included.add(upper)
                current += upper_vol
                upper += 1
            else:
                break
        
        va_nodes = [sorted_nodes[i] for i in included]
        for n in va_nodes:
            n.is_value_area = True
        
        if va_nodes:
            self.value_area_low = min(n.price for n in va_nodes)
            self.value_area_high = max(n.price for n in va_nodes)
    
    def _classify_nodes(self) -> None:
        """Classify nodes as HVN or LVN based on volume."""
        if not self.nodes or self.total_volume == 0:
            return
            
        avg_volume = self.total_volume / len(self.nodes)
        
        for node in self.nodes:
            if node.node_type == VolumeNodeType.POC:
                continue
            if node.volume > avg_volume * 1.5:
                node.node_type = VolumeNodeType.HVN
            elif node.volume < avg_volume * 0.5:
                node.node_type = VolumeNodeType.LVN


@dataclass
class OrderFlowState:
    """Complete order flow state."""
    timestamp: datetime
    symbol: str
    current_price: float
    
    # Footprint data
    footprint_bars: List[FootprintBar] = field(default_factory=list)
    current_footprint: Optional[FootprintBar] = None
    
    # CVD data
    cvd_series: List[CVDPoint] = field(default_factory=list)
    current_cvd: float = 0.0
    cvd_ma: float = 0.0
    
    # Volume Profile
    session_profile: Optional[VolumeProfile] = None
    composite_profile: Optional[VolumeProfile] = None
    
    # Auction state
    auction_state: AuctionState = AuctionState.BALANCED
    
    # Detected signals
    signals: List[OrderFlowSignal] = field(default_factory=list)
    patterns: List[FootprintPattern] = field(default_factory=list)
    
    # Confidence metrics
    signal_strength: float = 0.0
    confluence_score: float = 0.0


@dataclass
class OrderFlowEntry:
    """Order flow based entry signal."""
    direction: str  # 'long' or 'short'
    confidence: float
    entry_price: float
    stop_price: float
    target_price: float
    signal_type: OrderFlowSignal
    supporting_signals: List[OrderFlowSignal] = field(default_factory=list)
    footprint_pattern: Optional[FootprintPattern] = None
    reasoning: str = ""


# ============================================================================
# FOOTPRINT ANALYZER
# ============================================================================

class FootprintAnalyzer:
    """
    Analyzes footprint chart data for order flow signals.
    
    Key patterns:
    - Absorption: Large orders absorbed without price movement
    - Initiative: Aggressive orders moving price
    - Imbalance: Significant bid/ask ratio differences
    - Stacked Imbalance: Multiple consecutive imbalances
    """
    
    def __init__(
        self,
        imbalance_threshold: float = 2.0,    # 200% for imbalance
        stacked_min_count: int = 3,           # Min for stacked imbalance
        absorption_volume_mult: float = 2.0,  # Volume multiplier for absorption
    ):
        self.imbalance_threshold = imbalance_threshold
        self.stacked_min_count = stacked_min_count
        self.absorption_volume_mult = absorption_volume_mult
    
    def create_footprint_bar(
        self,
        timestamp: datetime,
        ohlc: Dict[str, float],
        price_levels: List[Dict[str, float]],
    ) -> FootprintBar:
        """
        Create footprint bar from price level data.
        
        Args:
            timestamp: Bar timestamp
            ohlc: Dict with 'open', 'high', 'low', 'close'
            price_levels: List of dicts with 'price', 'bid_volume', 'ask_volume'
        
        Returns:
            FootprintBar with calculated metrics
        """
        cells = [
            FootprintCell(
                price=pl['price'],
                bid_volume=pl.get('bid_volume', 0),
                ask_volume=pl.get('ask_volume', 0),
            )
            for pl in price_levels
        ]
        
        bar = FootprintBar(
            timestamp=timestamp,
            open=ohlc.get('open', 0),
            high=ohlc.get('high', 0),
            low=ohlc.get('low', 0),
            close=ohlc.get('close', 0),
            cells=cells,
        )
        bar.calculate_metrics()
        
        # Detect patterns
        bar.patterns = self.detect_patterns(bar)
        
        return bar
    
    def analyze_imbalances(
        self,
        bar: FootprintBar,
    ) -> Tuple[List[FootprintCell], ImbalanceType]:
        """
        Find imbalance cells and determine overall imbalance type.
        
        Returns:
            Tuple of (imbalanced cells, overall imbalance type)
        """
        imbalanced = [
            c for c in bar.cells
            if c.imbalance_type != ImbalanceType.BALANCED
        ]
        
        # Check for stacked (diagonal) imbalances
        if len(imbalanced) >= self.stacked_min_count:
            # Sort by price
            sorted_imb = sorted(imbalanced, key=lambda c: c.price)
            
            # Check for consecutive bid imbalances (bullish)
            bid_stack = sum(
                1 for c in sorted_imb
                if c.imbalance_type == ImbalanceType.BID_IMBALANCE
            )
            if bid_stack >= self.stacked_min_count:
                return imbalanced, ImbalanceType.DIAGONAL_BID
            
            # Check for consecutive ask imbalances (bearish)
            ask_stack = sum(
                1 for c in sorted_imb
                if c.imbalance_type == ImbalanceType.ASK_IMBALANCE
            )
            if ask_stack >= self.stacked_min_count:
                return imbalanced, ImbalanceType.DIAGONAL_ASK
        
        # Determine overall
        bid_count = sum(
            1 for c in imbalanced
            if c.imbalance_type == ImbalanceType.BID_IMBALANCE
        )
        ask_count = len(imbalanced) - bid_count
        
        if bid_count > ask_count:
            return imbalanced, ImbalanceType.BID_IMBALANCE
        elif ask_count > bid_count:
            return imbalanced, ImbalanceType.ASK_IMBALANCE
        return imbalanced, ImbalanceType.BALANCED
    
    def detect_absorption(
        self,
        bar: FootprintBar,
        previous_bars: List[FootprintBar],
    ) -> Optional[FootprintPattern]:
        """
        Detect absorption patterns.
        
        Absorption occurs when large volume is traded but price doesn't move.
        This indicates large orders being absorbed by the market.
        
        Returns:
            FootprintPattern if absorption detected, None otherwise
        """
        if not previous_bars or len(previous_bars) < 2:
            return None
        
        avg_volume = sum(b.total_volume for b in previous_bars[-5:]) / min(5, len(previous_bars))
        price_range = bar.high - bar.low
        avg_range = sum(b.high - b.low for b in previous_bars[-5:]) / min(5, len(previous_bars))
        
        # High volume but small range = absorption
        is_high_volume = bar.total_volume > avg_volume * self.absorption_volume_mult
        is_small_range = price_range < avg_range * 0.5
        
        if is_high_volume and is_small_range:
            # Determine absorption location
            if bar.close >= (bar.high + bar.low) / 2:
                # Absorption at highs (bearish - buying being absorbed)
                if bar.total_delta > 0:
                    return FootprintPattern.ABSORPTION_AT_HIGH
            else:
                # Absorption at lows (bullish - selling being absorbed)
                if bar.total_delta < 0:
                    return FootprintPattern.ABSORPTION_AT_LOW
        
        return None
    
    def detect_initiative(
        self,
        bar: FootprintBar,
        previous_bars: List[FootprintBar],
    ) -> Optional[FootprintPattern]:
        """
        Detect initiative buying/selling.
        
        Initiative occurs when aggressive orders move price significantly
        with strong delta confirmation.
        """
        if not previous_bars:
            return None
        
        avg_range = sum(b.high - b.low for b in previous_bars[-5:]) / min(5, len(previous_bars))
        price_range = bar.high - bar.low
        
        # Large range with strong delta = initiative
        is_large_range = price_range > avg_range * 1.5
        
        if is_large_range:
            # Strong positive delta = initiative buying
            if bar.total_delta > 0 and bar.close > bar.open:
                return FootprintPattern.INITIATIVE_BUYING
            # Strong negative delta = initiative selling
            elif bar.total_delta < 0 and bar.close < bar.open:
                return FootprintPattern.INITIATIVE_SELLING
        
        return None
    
    def detect_patterns(
        self,
        bar: FootprintBar,
        previous_bars: Optional[List[FootprintBar]] = None,
    ) -> List[FootprintPattern]:
        """Detect all footprint patterns in a bar."""
        patterns = []
        previous_bars = previous_bars or []
        
        # Check for absorption
        absorption = self.detect_absorption(bar, previous_bars)
        if absorption:
            patterns.append(absorption)
        
        # Check for initiative
        initiative = self.detect_initiative(bar, previous_bars)
        if initiative:
            patterns.append(initiative)
        
        # Check for stacked imbalances
        _, imb_type = self.analyze_imbalances(bar)
        if imb_type == ImbalanceType.DIAGONAL_BID:
            patterns.append(FootprintPattern.BREAKOUT_CONFIRMATION)
        elif imb_type == ImbalanceType.DIAGONAL_ASK:
            patterns.append(FootprintPattern.FAILED_AUCTION)
        
        return patterns
    
    def get_delta_divergence(
        self,
        bars: List[FootprintBar],
        lookback: int = 5,
    ) -> Optional[OrderFlowSignal]:
        """
        Detect price/delta divergence.
        
        Bullish divergence: Price makes lower low, delta makes higher low
        Bearish divergence: Price makes higher high, delta makes lower high
        """
        if len(bars) < lookback:
            return None
        
        recent = bars[-lookback:]
        
        # Find price and delta extremes
        price_lows = [(i, b.low) for i, b in enumerate(recent)]
        price_highs = [(i, b.high) for i, b in enumerate(recent)]
        delta_values = [(i, b.total_delta) for i, b in enumerate(recent)]
        
        # Check for bullish divergence (lower price low, higher delta low)
        min_price_idx = min(price_lows, key=lambda x: x[1])[0]
        
        # Find if there's an earlier low that was higher
        for i, low in price_lows[:min_price_idx]:
            if low > recent[min_price_idx].low:
                # Price made lower low
                early_delta = recent[i].total_delta
                late_delta = recent[min_price_idx].total_delta
                if late_delta > early_delta:
                    # Delta made higher low = bullish divergence
                    return OrderFlowSignal.DIVERGENCE
        
        # Check for bearish divergence
        max_price_idx = max(price_highs, key=lambda x: x[1])[0]
        
        for i, high in price_highs[:max_price_idx]:
            if high < recent[max_price_idx].high:
                # Price made higher high
                early_delta = recent[i].total_delta
                late_delta = recent[max_price_idx].total_delta
                if late_delta < early_delta:
                    # Delta made lower high = bearish divergence
                    return OrderFlowSignal.DIVERGENCE
        
        return None


# ============================================================================
# CVD ANALYZER (Cumulative Volume Delta)
# ============================================================================

class CVDAnalyzer:
    """
    Analyzes Cumulative Volume Delta for exhaustion and divergence.
    
    CVD shows the running total of delta (buying - selling volume).
    Key signals:
    - Exhaustion: CVD flattening while price continues
    - Divergence: CVD and price moving in opposite directions
    - Delta flip: CVD changes from positive to negative or vice versa
    """
    
    def __init__(
        self,
        smoothing_period: int = 14,
        divergence_threshold: float = 0.3,
    ):
        self.smoothing_period = smoothing_period
        self.divergence_threshold = divergence_threshold
        self._cvd_history: List[CVDPoint] = []
    
    def calculate_cvd(
        self,
        footprint_bars: List[FootprintBar],
    ) -> List[CVDPoint]:
        """
        Calculate CVD from footprint bars.
        
        Returns:
            List of CVDPoint with cumulative delta values
        """
        if not footprint_bars:
            return []
        
        cvd_points = []
        cumulative = 0.0
        
        for bar in footprint_bars:
            cumulative += bar.total_delta
            
            point = CVDPoint(
                timestamp=bar.timestamp,
                price=bar.close,
                delta=bar.total_delta,
                cumulative_delta=cumulative,
            )
            cvd_points.append(point)
        
        # Detect divergences
        self._detect_divergences(cvd_points)
        
        self._cvd_history = cvd_points
        return cvd_points
    
    def _detect_divergences(self, points: List[CVDPoint]) -> None:
        """Mark divergence points in CVD series."""
        if len(points) < 3:
            return
        
        for i in range(2, len(points)):
            # Check price direction
            price_up = points[i].price > points[i-1].price
            price_down = points[i].price < points[i-1].price
            
            # Check CVD direction
            cvd_up = points[i].cumulative_delta > points[i-1].cumulative_delta
            cvd_down = points[i].cumulative_delta < points[i-1].cumulative_delta
            
            # Divergence when price and CVD move opposite
            if (price_up and cvd_down) or (price_down and cvd_up):
                points[i].divergence = True
    
    def detect_exhaustion(
        self,
        cvd_points: Optional[List[CVDPoint]] = None,
        lookback: int = 10,
    ) -> Optional[OrderFlowSignal]:
        """
        Detect exhaustion patterns in CVD.
        
        Exhaustion occurs when:
        - Price continues making new highs/lows
        - But CVD flattens or reverses
        """
        points = cvd_points or self._cvd_history
        if len(points) < lookback:
            return None
        
        recent = points[-lookback:]
        
        # Calculate price and CVD changes
        price_change = (recent[-1].price - recent[0].price) / recent[0].price
        cvd_change = recent[-1].cumulative_delta - recent[0].cumulative_delta
        
        # Normalize CVD change
        cvd_range = max(p.cumulative_delta for p in recent) - min(p.cumulative_delta for p in recent)
        normalized_cvd_change = cvd_change / cvd_range if cvd_range > 0 else 0
        
        # Check for exhaustion (price moving, CVD not following)
        if abs(price_change) > 0.01:  # Significant price move
            if abs(normalized_cvd_change) < self.divergence_threshold:
                return OrderFlowSignal.EXHAUSTION
        
        return None
    
    def detect_delta_flip(
        self,
        cvd_points: Optional[List[CVDPoint]] = None,
    ) -> Optional[OrderFlowSignal]:
        """
        Detect when CVD crosses from positive to negative or vice versa.
        """
        points = cvd_points or self._cvd_history
        if len(points) < 2:
            return None
        
        prev = points[-2].cumulative_delta
        curr = points[-1].cumulative_delta
        
        if (prev > 0 and curr < 0) or (prev < 0 and curr > 0):
            return OrderFlowSignal.DELTA_FLIP
        
        return None
    
    def get_current_bias(
        self,
        cvd_points: Optional[List[CVDPoint]] = None,
    ) -> Tuple[str, float]:
        """
        Get current bias based on CVD.
        
        Returns:
            Tuple of (direction, confidence)
        """
        points = cvd_points or self._cvd_history
        if not points:
            return ('neutral', 0.0)
        
        # Use recent points for bias
        recent = points[-min(20, len(points)):]
        
        # Calculate slope of CVD
        if len(recent) >= 2:
            cvd_slope = (recent[-1].cumulative_delta - recent[0].cumulative_delta) / len(recent)
            
            # Normalize
            max_cvd = max(abs(p.cumulative_delta) for p in recent) or 1
            normalized_slope = cvd_slope / max_cvd
            
            if normalized_slope > 0.05:
                return ('bullish', min(abs(normalized_slope) * 5, 1.0))
            elif normalized_slope < -0.05:
                return ('bearish', min(abs(normalized_slope) * 5, 1.0))
        
        return ('neutral', 0.0)


# ============================================================================
# VOLUME PROFILE ANALYZER
# ============================================================================

class VolumeProfileAnalyzer:
    """
    Analyzes volume profile for trading levels and market structure.
    
    Key concepts:
    - POC (Point of Control): Price with highest volume - "fair value"
    - Value Area: Range containing 70% of volume
    - HVN (High Volume Node): Strong support/resistance
    - LVN (Low Volume Node): Fast price movement zones
    """
    
    def __init__(
        self,
        value_area_percent: float = 0.70,
        hvn_threshold: float = 1.5,  # Multiplier above average
        lvn_threshold: float = 0.5,   # Multiplier below average
        price_resolution: float = 0.01,  # Price bucket size
    ):
        self.value_area_percent = value_area_percent
        self.hvn_threshold = hvn_threshold
        self.lvn_threshold = lvn_threshold
        self.price_resolution = price_resolution
    
    def build_profile(
        self,
        bars: List[Dict[str, Any]],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> VolumeProfile:
        """
        Build volume profile from bar data.
        
        Args:
            bars: List of dicts with 'high', 'low', 'close', 'volume', 'timestamp'
            start_time: Profile start time
            end_time: Profile end time
        
        Returns:
            VolumeProfile with calculated metrics
        """
        if not bars:
            return VolumeProfile(
                start_time=start_time or datetime.now(),
                end_time=end_time or datetime.now(),
            )
        
        # Distribute volume across price levels
        volume_by_price: Dict[float, float] = {}
        
        for bar in bars:
            high = bar.get('high', bar.get('close', 0))
            low = bar.get('low', bar.get('close', 0))
            volume = bar.get('volume', 0)
            
            # Distribute volume evenly across price range
            price_steps = int((high - low) / self.price_resolution) + 1
            vol_per_step = volume / max(price_steps, 1)
            
            for i in range(price_steps):
                price = round(low + i * self.price_resolution, 2)
                volume_by_price[price] = volume_by_price.get(price, 0) + vol_per_step
        
        # Create nodes
        nodes = [
            VolumeNode(
                price=price,
                volume=vol,
                node_type=VolumeNodeType.HVN,  # Will be reclassified
            )
            for price, vol in volume_by_price.items()
        ]
        
        profile = VolumeProfile(
            start_time=start_time or (
                min(b['timestamp'] for b in bars if 'timestamp' in b)
                if any('timestamp' in b for b in bars)
                else datetime.now()
            ),
            end_time=end_time or (
                max(b['timestamp'] for b in bars if 'timestamp' in b)
                if any('timestamp' in b for b in bars)
                else datetime.now()
            ),
            nodes=nodes,
            developing=False,
        )
        profile.calculate()
        
        return profile
    
    def analyze_price_position(
        self,
        current_price: float,
        profile: VolumeProfile,
    ) -> Tuple[AuctionState, List[OrderFlowSignal]]:
        """
        Analyze current price position relative to volume profile.
        
        Returns:
            Tuple of (auction state, detected signals)
        """
        signals = []
        
        if not profile.nodes:
            return AuctionState.BALANCED, signals
        
        # Check position relative to value area
        in_value_area = profile.value_area_low <= current_price <= profile.value_area_high
        above_value = current_price > profile.value_area_high
        below_value = current_price < profile.value_area_low
        
        # Check distance from POC
        poc_distance = abs(current_price - profile.poc) / profile.poc
        
        # Determine auction state
        if in_value_area:
            state = AuctionState.BALANCED
            if poc_distance < 0.005:  # Within 0.5% of POC
                signals.append(OrderFlowSignal.POC_TEST)
        elif above_value:
            state = AuctionState.INITIATING_LONG
            signals.append(OrderFlowSignal.VALUE_AREA_BREAK)
        else:
            state = AuctionState.INITIATING_SHORT
            signals.append(OrderFlowSignal.VALUE_AREA_BREAK)
        
        # Check for HVN/LVN interactions
        for node in profile.nodes:
            price_at_node = abs(current_price - node.price) / current_price < 0.003
            
            if price_at_node:
                if node.node_type == VolumeNodeType.HVN:
                    signals.append(OrderFlowSignal.HVN_REJECTION)
                elif node.node_type == VolumeNodeType.LVN:
                    signals.append(OrderFlowSignal.LVN_ACCELERATION)
        
        return state, signals
    
    def find_key_levels(
        self,
        profile: VolumeProfile,
        num_levels: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Find key support/resistance levels from profile.
        
        Returns:
            Dict with 'support' and 'resistance' price lists
        """
        if not profile.nodes:
            return {'support': [], 'resistance': []}
        
        # Sort HVN nodes by volume
        hvn_nodes = [
            n for n in profile.nodes
            if n.node_type == VolumeNodeType.HVN or n.is_poc
        ]
        hvn_nodes.sort(key=lambda n: n.volume, reverse=True)
        
        # Split into support/resistance based on current price
        current = profile.poc  # Use POC as reference
        
        support = [
            n.price for n in hvn_nodes
            if n.price < current
        ][:num_levels]
        
        resistance = [
            n.price for n in hvn_nodes
            if n.price > current
        ][:num_levels]
        
        return {
            'support': sorted(support, reverse=True),
            'resistance': sorted(resistance),
            'poc': profile.poc,
            'value_area': (profile.value_area_low, profile.value_area_high),
        }


# ============================================================================
# MAIN ORDER FLOW ENGINE
# ============================================================================

class OrderFlowEngine:
    """
    Comprehensive Order Flow Analysis Engine.
    
    Combines:
    - Footprint analysis (bid/ask aggression, imbalance)
    - CVD analysis (cumulative delta, exhaustion)
    - Volume profile (POC, value area, HVN/LVN)
    
    Integration points:
    - Works with LiquidityEngineV4 (Wyckoff)
    - Works with ICTEngine
    - Provides signals for LiquidityAgentV4
    """
    
    def __init__(
        self,
        footprint_config: Optional[Dict] = None,
        cvd_config: Optional[Dict] = None,
        profile_config: Optional[Dict] = None,
    ):
        # Initialize analyzers
        self.footprint = FootprintAnalyzer(**(footprint_config or {}))
        self.cvd = CVDAnalyzer(**(cvd_config or {}))
        self.profile = VolumeProfileAnalyzer(**(profile_config or {}))
        
        # State tracking
        self._footprint_bars: Dict[str, List[FootprintBar]] = {}
        self._cvd_data: Dict[str, List[CVDPoint]] = {}
        self._profiles: Dict[str, VolumeProfile] = {}
        self._states: Dict[str, OrderFlowState] = {}
        
        logger.info("OrderFlowEngine initialized with Footprint, CVD, and Volume Profile analyzers")
    
    def analyze(
        self,
        symbol: str,
        bars: List[Dict[str, Any]],
        price_levels: Optional[List[List[Dict[str, float]]]] = None,
        current_price: Optional[float] = None,
    ) -> OrderFlowState:
        """
        Perform complete order flow analysis.
        
        Args:
            symbol: Trading symbol
            bars: OHLCV bar data
            price_levels: Optional footprint price level data per bar
            current_price: Current market price
        
        Returns:
            OrderFlowState with all analysis results
        """
        if not bars:
            return OrderFlowState(
                timestamp=datetime.now(),
                symbol=symbol,
                current_price=current_price or 0,
            )
        
        timestamp = datetime.now()
        
        # Build footprint bars if we have price level data
        footprint_bars = []
        if price_levels and len(price_levels) == len(bars):
            for bar, levels in zip(bars, price_levels):
                fp_bar = self.footprint.create_footprint_bar(
                    timestamp=bar.get('timestamp', timestamp),
                    ohlc={
                        'open': bar.get('open', 0),
                        'high': bar.get('high', 0),
                        'low': bar.get('low', 0),
                        'close': bar.get('close', 0),
                    },
                    price_levels=levels,
                )
                footprint_bars.append(fp_bar)
        else:
            # Create synthetic footprint data from OHLCV
            footprint_bars = self._create_synthetic_footprint(bars)
        
        self._footprint_bars[symbol] = footprint_bars
        
        # Calculate CVD
        cvd_points = self.cvd.calculate_cvd(footprint_bars)
        self._cvd_data[symbol] = cvd_points
        
        # Build volume profile
        profile = self.profile.build_profile(bars)
        self._profiles[symbol] = profile
        
        # Determine current price
        price = current_price or bars[-1].get('close', 0)
        
        # Analyze price position
        auction_state, profile_signals = self.profile.analyze_price_position(price, profile)
        
        # Collect all signals
        signals = list(profile_signals)
        patterns = []
        
        # Add footprint patterns
        for fp_bar in footprint_bars[-5:]:  # Last 5 bars
            patterns.extend(fp_bar.patterns)
        
        # Check for CVD signals
        cvd_exhaustion = self.cvd.detect_exhaustion()
        if cvd_exhaustion:
            signals.append(cvd_exhaustion)
        
        cvd_flip = self.cvd.detect_delta_flip()
        if cvd_flip:
            signals.append(cvd_flip)
        
        # Check for divergence
        fp_divergence = self.footprint.get_delta_divergence(footprint_bars)
        if fp_divergence:
            signals.append(fp_divergence)
        
        # Calculate signal strength and confluence
        signal_strength = self._calculate_signal_strength(signals, patterns)
        confluence_score = self._calculate_confluence(signals, patterns, auction_state)
        
        # Build state
        state = OrderFlowState(
            timestamp=timestamp,
            symbol=symbol,
            current_price=price,
            footprint_bars=footprint_bars,
            current_footprint=footprint_bars[-1] if footprint_bars else None,
            cvd_series=cvd_points,
            current_cvd=cvd_points[-1].cumulative_delta if cvd_points else 0,
            cvd_ma=sum(p.cumulative_delta for p in cvd_points[-14:]) / min(14, len(cvd_points)) if cvd_points else 0,
            session_profile=profile,
            auction_state=auction_state,
            signals=signals,
            patterns=patterns,
            signal_strength=signal_strength,
            confluence_score=confluence_score,
        )
        
        self._states[symbol] = state
        return state
    
    def _create_synthetic_footprint(
        self,
        bars: List[Dict[str, Any]],
    ) -> List[FootprintBar]:
        """Create synthetic footprint data from OHLCV bars."""
        footprint_bars = []
        
        for bar in bars:
            open_price = bar.get('open', 0)
            close_price = bar.get('close', 0)
            high = bar.get('high', 0)
            low = bar.get('low', 0)
            volume = bar.get('volume', 0)
            
            # Estimate delta from candle direction
            is_bullish = close_price >= open_price
            
            # Distribute volume (simple model)
            if is_bullish:
                ask_vol = volume * 0.6  # More buying
                bid_vol = volume * 0.4
            else:
                ask_vol = volume * 0.4
                bid_vol = volume * 0.6  # More selling
            
            # Create cells for high, POC (mid), and low
            mid = (high + low) / 2
            cells = [
                FootprintCell(price=low, bid_volume=bid_vol * 0.3, ask_volume=ask_vol * 0.2),
                FootprintCell(price=mid, bid_volume=bid_vol * 0.4, ask_volume=ask_vol * 0.4),
                FootprintCell(price=high, bid_volume=bid_vol * 0.3, ask_volume=ask_vol * 0.4),
            ]
            
            fp_bar = FootprintBar(
                timestamp=bar.get('timestamp', datetime.now()),
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                cells=cells,
            )
            fp_bar.calculate_metrics()
            footprint_bars.append(fp_bar)
        
        return footprint_bars
    
    def _calculate_signal_strength(
        self,
        signals: List[OrderFlowSignal],
        patterns: List[FootprintPattern],
    ) -> float:
        """Calculate overall signal strength (0-1)."""
        if not signals and not patterns:
            return 0.0
        
        # Weight signals
        signal_weights = {
            OrderFlowSignal.ABSORPTION: 0.15,
            OrderFlowSignal.INITIATION: 0.12,
            OrderFlowSignal.EXHAUSTION: 0.15,
            OrderFlowSignal.DIVERGENCE: 0.18,
            OrderFlowSignal.IMBALANCE: 0.10,
            OrderFlowSignal.DELTA_FLIP: 0.12,
            OrderFlowSignal.POC_TEST: 0.08,
            OrderFlowSignal.VALUE_AREA_BREAK: 0.10,
            OrderFlowSignal.HVN_REJECTION: 0.10,
            OrderFlowSignal.LVN_ACCELERATION: 0.08,
            OrderFlowSignal.STACKED_IMBALANCE: 0.15,
            OrderFlowSignal.UNFINISHED_AUCTION: 0.10,
        }
        
        pattern_weights = {
            FootprintPattern.ABSORPTION_AT_HIGH: 0.12,
            FootprintPattern.ABSORPTION_AT_LOW: 0.12,
            FootprintPattern.INITIATIVE_BUYING: 0.10,
            FootprintPattern.INITIATIVE_SELLING: 0.10,
            FootprintPattern.STOP_HUNT_COMPLETE: 0.15,
            FootprintPattern.BREAKOUT_CONFIRMATION: 0.12,
            FootprintPattern.FAILED_AUCTION: 0.12,
        }
        
        total = 0.0
        for sig in signals:
            total += signal_weights.get(sig, 0.05)
        for pat in patterns:
            total += pattern_weights.get(pat, 0.05)
        
        return min(total, 1.0)
    
    def _calculate_confluence(
        self,
        signals: List[OrderFlowSignal],
        patterns: List[FootprintPattern],
        auction_state: AuctionState,
    ) -> float:
        """Calculate confluence score based on signal alignment."""
        if not signals and not patterns:
            return 0.0
        
        # Categorize signals by direction
        bullish_signals = {
            OrderFlowSignal.ABSORPTION,  # At lows
            OrderFlowSignal.INITIATION,   # Buying
            OrderFlowSignal.VALUE_AREA_BREAK,  # If breaking up
        }
        
        bearish_signals = {
            OrderFlowSignal.EXHAUSTION,
            OrderFlowSignal.DIVERGENCE,
        }
        
        bullish_patterns = {
            FootprintPattern.ABSORPTION_AT_LOW,
            FootprintPattern.INITIATIVE_BUYING,
            FootprintPattern.BREAKOUT_CONFIRMATION,
        }
        
        bearish_patterns = {
            FootprintPattern.ABSORPTION_AT_HIGH,
            FootprintPattern.INITIATIVE_SELLING,
            FootprintPattern.FAILED_AUCTION,
        }
        
        # Count aligned signals
        bullish_count = sum(1 for s in signals if s in bullish_signals)
        bullish_count += sum(1 for p in patterns if p in bullish_patterns)
        
        bearish_count = sum(1 for s in signals if s in bearish_signals)
        bearish_count += sum(1 for p in patterns if p in bearish_patterns)
        
        # Add auction state
        if auction_state == AuctionState.INITIATING_LONG:
            bullish_count += 1
        elif auction_state == AuctionState.INITIATING_SHORT:
            bearish_count += 1
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        # Confluence is higher when signals align
        alignment = max(bullish_count, bearish_count) / total
        return alignment
    
    def get_entry_signals(
        self,
        symbol: str,
        state: Optional[OrderFlowState] = None,
    ) -> List[OrderFlowEntry]:
        """
        Generate entry signals from order flow analysis.
        
        Returns:
            List of OrderFlowEntry signals
        """
        state = state or self._states.get(symbol)
        if not state or not state.current_footprint:
            return []
        
        entries = []
        current = state.current_price
        profile = state.session_profile
        
        # Get key levels
        levels = {}
        if profile:
            levels = self.profile.find_key_levels(profile)
        
        # Check for absorption entry
        if FootprintPattern.ABSORPTION_AT_LOW in state.patterns:
            # Bullish absorption setup
            stop = min(levels.get('support', [current * 0.98])[:1] or [current * 0.98])
            target = levels.get('poc', current * 1.02)
            
            entries.append(OrderFlowEntry(
                direction='long',
                confidence=0.70,
                entry_price=current,
                stop_price=stop,
                target_price=target,
                signal_type=OrderFlowSignal.ABSORPTION,
                supporting_signals=[s for s in state.signals if s != OrderFlowSignal.ABSORPTION],
                footprint_pattern=FootprintPattern.ABSORPTION_AT_LOW,
                reasoning="Selling absorbed at lows, likely reversal",
            ))
        
        elif FootprintPattern.ABSORPTION_AT_HIGH in state.patterns:
            # Bearish absorption setup
            stop = max(levels.get('resistance', [current * 1.02])[:1] or [current * 1.02])
            target = levels.get('poc', current * 0.98)
            
            entries.append(OrderFlowEntry(
                direction='short',
                confidence=0.70,
                entry_price=current,
                stop_price=stop,
                target_price=target,
                signal_type=OrderFlowSignal.ABSORPTION,
                supporting_signals=[s for s in state.signals if s != OrderFlowSignal.ABSORPTION],
                footprint_pattern=FootprintPattern.ABSORPTION_AT_HIGH,
                reasoning="Buying absorbed at highs, likely reversal",
            ))
        
        # Check for initiative breakout
        if FootprintPattern.INITIATIVE_BUYING in state.patterns:
            if state.auction_state == AuctionState.INITIATING_LONG:
                entries.append(OrderFlowEntry(
                    direction='long',
                    confidence=0.65,
                    entry_price=current,
                    stop_price=profile.value_area_high if profile else current * 0.98,
                    target_price=current * 1.03,
                    signal_type=OrderFlowSignal.INITIATION,
                    supporting_signals=state.signals,
                    footprint_pattern=FootprintPattern.INITIATIVE_BUYING,
                    reasoning="Initiative buying breaking out of value area",
                ))
        
        elif FootprintPattern.INITIATIVE_SELLING in state.patterns:
            if state.auction_state == AuctionState.INITIATING_SHORT:
                entries.append(OrderFlowEntry(
                    direction='short',
                    confidence=0.65,
                    entry_price=current,
                    stop_price=profile.value_area_low if profile else current * 1.02,
                    target_price=current * 0.97,
                    signal_type=OrderFlowSignal.INITIATION,
                    supporting_signals=state.signals,
                    footprint_pattern=FootprintPattern.INITIATIVE_SELLING,
                    reasoning="Initiative selling breaking out of value area",
                ))
        
        # Check for divergence entry
        if OrderFlowSignal.DIVERGENCE in state.signals:
            # Get CVD bias for direction
            cvd_bias, cvd_conf = self.cvd.get_current_bias()
            
            if cvd_bias == 'bullish':
                entries.append(OrderFlowEntry(
                    direction='long',
                    confidence=0.60 + cvd_conf * 0.15,
                    entry_price=current,
                    stop_price=current * 0.98,
                    target_price=current * 1.025,
                    signal_type=OrderFlowSignal.DIVERGENCE,
                    supporting_signals=state.signals,
                    reasoning="Bullish CVD divergence - price down but delta improving",
                ))
            elif cvd_bias == 'bearish':
                entries.append(OrderFlowEntry(
                    direction='short',
                    confidence=0.60 + cvd_conf * 0.15,
                    entry_price=current,
                    stop_price=current * 1.02,
                    target_price=current * 0.975,
                    signal_type=OrderFlowSignal.DIVERGENCE,
                    supporting_signals=state.signals,
                    reasoning="Bearish CVD divergence - price up but delta weakening",
                ))
        
        # Check for POC test entry
        if OrderFlowSignal.POC_TEST in state.signals and profile:
            # POC tests often lead to continuation or reversal
            cvd_bias, _ = self.cvd.get_current_bias()
            
            if cvd_bias == 'bullish':
                entries.append(OrderFlowEntry(
                    direction='long',
                    confidence=0.55,
                    entry_price=current,
                    stop_price=profile.value_area_low,
                    target_price=profile.value_area_high,
                    signal_type=OrderFlowSignal.POC_TEST,
                    supporting_signals=state.signals,
                    reasoning="Testing POC with bullish CVD - expect value area rotation",
                ))
            elif cvd_bias == 'bearish':
                entries.append(OrderFlowEntry(
                    direction='short',
                    confidence=0.55,
                    entry_price=current,
                    stop_price=profile.value_area_high,
                    target_price=profile.value_area_low,
                    signal_type=OrderFlowSignal.POC_TEST,
                    supporting_signals=state.signals,
                    reasoning="Testing POC with bearish CVD - expect value area rotation",
                ))
        
        return entries
    
    def get_bias(self, symbol: str) -> Tuple[str, float, str]:
        """
        Get current order flow bias.
        
        Returns:
            Tuple of (direction, confidence, reasoning)
        """
        state = self._states.get(symbol)
        if not state:
            return ('neutral', 0.0, 'No order flow data')
        
        # Combine all signals
        bullish_score = 0.0
        bearish_score = 0.0
        reasons = []
        
        # Auction state
        if state.auction_state == AuctionState.INITIATING_LONG:
            bullish_score += 0.3
            reasons.append("Breaking out of value area upward")
        elif state.auction_state == AuctionState.INITIATING_SHORT:
            bearish_score += 0.3
            reasons.append("Breaking out of value area downward")
        
        # CVD bias
        cvd_bias, cvd_conf = self.cvd.get_current_bias()
        if cvd_bias == 'bullish':
            bullish_score += cvd_conf * 0.3
            reasons.append(f"CVD bullish (conf: {cvd_conf:.0%})")
        elif cvd_bias == 'bearish':
            bearish_score += cvd_conf * 0.3
            reasons.append(f"CVD bearish (conf: {cvd_conf:.0%})")
        
        # Footprint patterns
        for pattern in state.patterns:
            if pattern in [FootprintPattern.ABSORPTION_AT_LOW, FootprintPattern.INITIATIVE_BUYING]:
                bullish_score += 0.2
                reasons.append(f"Bullish {pattern.name}")
            elif pattern in [FootprintPattern.ABSORPTION_AT_HIGH, FootprintPattern.INITIATIVE_SELLING]:
                bearish_score += 0.2
                reasons.append(f"Bearish {pattern.name}")
        
        # Signal confluence
        bullish_score += state.confluence_score * 0.2 if bullish_score > bearish_score else 0
        bearish_score += state.confluence_score * 0.2 if bearish_score > bullish_score else 0
        
        # Determine bias
        if bullish_score > bearish_score and bullish_score > 0.2:
            return ('bullish', min(bullish_score, 1.0), "; ".join(reasons))
        elif bearish_score > bullish_score and bearish_score > 0.2:
            return ('bearish', min(bearish_score, 1.0), "; ".join(reasons))
        
        return ('neutral', max(bullish_score, bearish_score), "Mixed signals")
    
    def get_key_levels(self, symbol: str) -> Dict[str, Any]:
        """Get key order flow levels for a symbol."""
        profile = self._profiles.get(symbol)
        if not profile:
            return {}
        
        levels = self.profile.find_key_levels(profile)
        
        # Add CVD data
        cvd_data = self._cvd_data.get(symbol, [])
        if cvd_data:
            levels['current_cvd'] = cvd_data[-1].cumulative_delta
            levels['cvd_trend'] = 'positive' if cvd_data[-1].delta_type == DeltaType.POSITIVE else 'negative'
        
        return levels
    
    def get_state(self, symbol: str) -> Optional[OrderFlowState]:
        """Get current order flow state for a symbol."""
        return self._states.get(symbol)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_order_flow_engine(
    imbalance_threshold: float = 2.0,
    stacked_min_count: int = 3,
    cvd_smoothing: int = 14,
    value_area_percent: float = 0.70,
) -> OrderFlowEngine:
    """
    Factory function to create configured OrderFlowEngine.
    
    Args:
        imbalance_threshold: Ratio for bid/ask imbalance (default 2.0 = 200%)
        stacked_min_count: Minimum consecutive imbalances for stacked pattern
        cvd_smoothing: Period for CVD smoothing
        value_area_percent: Percentage of volume for value area (default 70%)
    
    Returns:
        Configured OrderFlowEngine instance
    """
    return OrderFlowEngine(
        footprint_config={
            'imbalance_threshold': imbalance_threshold,
            'stacked_min_count': stacked_min_count,
        },
        cvd_config={
            'smoothing_period': cvd_smoothing,
        },
        profile_config={
            'value_area_percent': value_area_percent,
        },
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Engine
    'OrderFlowEngine',
    'create_order_flow_engine',
    
    # Analyzers
    'FootprintAnalyzer',
    'CVDAnalyzer',
    'VolumeProfileAnalyzer',
    
    # Enums
    'OrderFlowSignal',
    'DeltaType',
    'VolumeNodeType',
    'AuctionState',
    'FootprintPattern',
    'ImbalanceType',
    'MarketParticipant',
    'OrderType',
    
    # Data Structures
    'FootprintCell',
    'FootprintBar',
    'CVDPoint',
    'VolumeNode',
    'VolumeProfile',
    'OrderFlowState',
    'OrderFlowEntry',
]
