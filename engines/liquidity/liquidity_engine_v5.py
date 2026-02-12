"""
Liquidity Engine V5 - Unified Market Quality + PENTA Methodology Engine

This is the main Liquidity Engine that combines:
1. Market Quality Analysis (bid-ask, depth, tradability)
2. PENTA Methodology (5 sub-engines):
   - Wyckoff (VSA, Phases, Events)
   - ICT (FVGs, Order Blocks, OTE)
   - Order Flow (Footprint, CVD, Volume Profile)
   - Supply & Demand (Zones, Strength)
   - Liquidity Concepts (Pools, Voids, Inducements)

Architecture:
    Data Adapters → LiquidityEngineV5 → LiquidityAgentV5 → Composer

The Liquidity Engine feeds the Liquidity Agent, which then feeds the Composer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# Import base market data components
try:
    from engines.inputs.market_data_adapter import MarketDataAdapter
    from schemas.core_schemas import LiquiditySnapshot
    MARKET_ADAPTER_AVAILABLE = True
except ImportError:
    MARKET_ADAPTER_AVAILABLE = False
    MarketDataAdapter = None
    LiquiditySnapshot = None

# Import PENTA methodology sub-engines
try:
    from engines.liquidity.liquidity_engine_v4 import LiquidityEngineV4  # Wyckoff
    WYCKOFF_AVAILABLE = True
except ImportError:
    WYCKOFF_AVAILABLE = False
    LiquidityEngineV4 = None

try:
    from engines.liquidity.ict_engine import ICTEngine
    ICT_AVAILABLE = True
except ImportError:
    ICT_AVAILABLE = False
    ICTEngine = None

try:
    from engines.liquidity.order_flow_engine import OrderFlowEngine
    ORDER_FLOW_AVAILABLE = True
except ImportError:
    ORDER_FLOW_AVAILABLE = False
    OrderFlowEngine = None

try:
    from engines.liquidity.supply_demand_engine import SupplyDemandEngine
    SUPPLY_DEMAND_AVAILABLE = True
except ImportError:
    SUPPLY_DEMAND_AVAILABLE = False
    SupplyDemandEngine = None

try:
    from engines.liquidity.liquidity_concepts_engine import (
        LiquidityConceptsEngine,
        LiquidityConceptsState,
    )
    LIQUIDITY_CONCEPTS_AVAILABLE = True
except ImportError:
    LIQUIDITY_CONCEPTS_AVAILABLE = False
    LiquidityConceptsEngine = None
    LiquidityConceptsState = None


class MarketQualityGrade(str, Enum):
    """Market quality classification."""
    EXCELLENT = "A+"
    GOOD = "B"
    FAIR = "C"
    POOR = "D"
    ILLIQUID = "F"


@dataclass
class PENTAState:
    """Combined state from all 5 PENTA methodology engines."""
    
    # Methodology availability
    wyckoff_enabled: bool = False
    ict_enabled: bool = False
    order_flow_enabled: bool = False
    supply_demand_enabled: bool = False
    liquidity_concepts_enabled: bool = False
    
    # Wyckoff Analysis
    wyckoff_phase: Optional[str] = None
    wyckoff_event: Optional[str] = None
    wyckoff_bias: Optional[str] = None
    wyckoff_confidence: float = 0.0
    
    # ICT Analysis
    ict_bias: Optional[str] = None
    ict_in_discount: bool = False
    ict_in_premium: bool = False
    ict_fvg_count: int = 0
    ict_order_block_count: int = 0
    ict_confidence: float = 0.0
    
    # Order Flow Analysis
    order_flow_bias: Optional[str] = None
    cvd_trend: Optional[str] = None
    absorption_detected: bool = False
    exhaustion_detected: bool = False
    order_flow_confidence: float = 0.0
    
    # Supply & Demand Analysis
    sd_bias: Optional[str] = None
    demand_zones: int = 0
    supply_zones: int = 0
    fresh_demand_zones: int = 0
    fresh_supply_zones: int = 0
    nearest_demand: Optional[float] = None
    nearest_supply: Optional[float] = None
    sd_confidence: float = 0.0
    
    # Liquidity Concepts Analysis
    lc_trend: Optional[str] = None
    buy_pools: int = 0
    sell_pools: int = 0
    liquidity_voids: int = 0
    active_inducement: Optional[str] = None
    inducement_direction: Optional[str] = None
    strong_highs: int = 0
    strong_lows: int = 0
    weak_highs: int = 0
    weak_lows: int = 0
    market_structure: Optional[str] = None
    lc_confidence: float = 0.0
    
    # Confluence
    confluence_level: Optional[str] = None  # PENTA, QUAD, TRIPLE, DOUBLE, SINGLE
    combined_bias: Optional[str] = None  # bullish, bearish, neutral
    combined_confidence: float = 0.0
    agreeing_methodologies: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            # Availability
            "wyckoff_enabled": self.wyckoff_enabled,
            "ict_enabled": self.ict_enabled,
            "order_flow_enabled": self.order_flow_enabled,
            "supply_demand_enabled": self.supply_demand_enabled,
            "liquidity_concepts_enabled": self.liquidity_concepts_enabled,
            # Wyckoff
            "wyckoff_phase": self.wyckoff_phase,
            "wyckoff_event": self.wyckoff_event,
            "wyckoff_bias": self.wyckoff_bias,
            "wyckoff_confidence": self.wyckoff_confidence,
            # ICT
            "ict_bias": self.ict_bias,
            "ict_in_discount": self.ict_in_discount,
            "ict_in_premium": self.ict_in_premium,
            "ict_fvg_count": self.ict_fvg_count,
            "ict_order_block_count": self.ict_order_block_count,
            "ict_confidence": self.ict_confidence,
            # Order Flow
            "order_flow_bias": self.order_flow_bias,
            "cvd_trend": self.cvd_trend,
            "absorption_detected": self.absorption_detected,
            "exhaustion_detected": self.exhaustion_detected,
            "order_flow_confidence": self.order_flow_confidence,
            # Supply & Demand
            "sd_bias": self.sd_bias,
            "demand_zones": self.demand_zones,
            "supply_zones": self.supply_zones,
            "fresh_demand_zones": self.fresh_demand_zones,
            "fresh_supply_zones": self.fresh_supply_zones,
            "nearest_demand": self.nearest_demand,
            "nearest_supply": self.nearest_supply,
            "sd_confidence": self.sd_confidence,
            # Liquidity Concepts
            "lc_trend": self.lc_trend,
            "buy_pools": self.buy_pools,
            "sell_pools": self.sell_pools,
            "liquidity_voids": self.liquidity_voids,
            "active_inducement": self.active_inducement,
            "inducement_direction": self.inducement_direction,
            "strong_highs": self.strong_highs,
            "strong_lows": self.strong_lows,
            "weak_highs": self.weak_highs,
            "weak_lows": self.weak_lows,
            "market_structure": self.market_structure,
            "lc_confidence": self.lc_confidence,
            # Confluence
            "confluence_level": self.confluence_level,
            "combined_bias": self.combined_bias,
            "combined_confidence": self.combined_confidence,
            "agreeing_methodologies": self.agreeing_methodologies,
        }


@dataclass
class LiquidityEngineV5Snapshot:
    """
    Complete snapshot from LiquidityEngineV5.
    
    Contains both market quality metrics and PENTA methodology analysis.
    """
    timestamp: datetime
    symbol: str
    
    # Market Quality Metrics
    liquidity_score: float = 0.5
    bid_ask_spread: float = 0.0
    volume: float = 0.0
    depth: float = 0.0
    impact_cost: float = 0.0
    quality_grade: MarketQualityGrade = MarketQualityGrade.FAIR
    
    # PENTA Methodology State
    penta_state: PENTAState = field(default_factory=PENTAState)
    
    # Combined Analysis
    trading_bias: Optional[str] = None  # bullish, bearish, neutral
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "liquidity_score": self.liquidity_score,
            "bid_ask_spread": self.bid_ask_spread,
            "volume": self.volume,
            "depth": self.depth,
            "impact_cost": self.impact_cost,
            "quality_grade": self.quality_grade.value,
            "penta_state": self.penta_state.to_dict(),
            "trading_bias": self.trading_bias,
            "confidence": self.confidence,
        }


class LiquidityEngineV5:
    """
    Liquidity Engine V5 - Unified Market Quality + PENTA Methodology.
    
    This engine combines:
    1. Market Quality Analysis (spreads, depth, tradability)
    2. PENTA Methodology (5 sub-engines for smart money analysis)
    
    The output feeds LiquidityAgentV5 which then feeds the Composer.
    """
    
    VERSION = "5.0.0"
    
    def __init__(
        self,
        market_adapter: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize LiquidityEngineV5.
        
        Args:
            market_adapter: Market data adapter for quotes/bars
            config: Engine configuration
        """
        self.market_adapter = market_adapter
        self.config = config or {}
        
        # Configuration
        self.min_volume_threshold = self.config.get("min_volume_threshold", 1_000_000)
        self.spread_threshold = self.config.get("spread_threshold", 0.5)
        
        # Initialize PENTA sub-engines
        self._init_penta_engines()
        
        logger.info(f"LiquidityEngineV5 v{self.VERSION} initialized")
        logger.info(f"  - Wyckoff: {'enabled' if self.wyckoff_engine else 'disabled'}")
        logger.info(f"  - ICT: {'enabled' if self.ict_engine else 'disabled'}")
        logger.info(f"  - Order Flow: {'enabled' if self.order_flow_engine else 'disabled'}")
        logger.info(f"  - Supply & Demand: {'enabled' if self.supply_demand_engine else 'disabled'}")
        logger.info(f"  - Liquidity Concepts: {'enabled' if self.liquidity_concepts_engine else 'disabled'}")
    
    def _init_penta_engines(self) -> None:
        """Initialize PENTA methodology sub-engines."""
        # Wyckoff Engine (requires adapters - skip if not available)
        self.wyckoff_engine = None
        if WYCKOFF_AVAILABLE and self.market_adapter is not None:
            try:
                # Wyckoff engine needs adapters
                self.wyckoff_engine = LiquidityEngineV4(
                    market_adapter=self.market_adapter,
                    options_adapter=None,  # Will use stub if needed
                    config=self.config or {},
                )
            except Exception as e:
                logger.debug(f"Wyckoff engine (V4) skipped: {e}")
        
        # ICT Engine
        self.ict_engine = None
        if ICT_AVAILABLE:
            try:
                ict_config = self.config.get("ict", {}) if self.config else {}
                self.ict_engine = ICTEngine(config=ict_config)
            except Exception as e:
                logger.debug(f"ICT engine not initialized: {e}")
        
        # Order Flow Engine
        self.order_flow_engine = None
        if ORDER_FLOW_AVAILABLE:
            try:
                self.order_flow_engine = OrderFlowEngine()
            except Exception as e:
                logger.warning(f"Failed to initialize Order Flow engine: {e}")
        
        # Supply & Demand Engine
        self.supply_demand_engine = None
        if SUPPLY_DEMAND_AVAILABLE:
            try:
                self.supply_demand_engine = SupplyDemandEngine()
            except Exception as e:
                logger.warning(f"Failed to initialize Supply & Demand engine: {e}")
        
        # Liquidity Concepts Engine
        self.liquidity_concepts_engine = None
        if LIQUIDITY_CONCEPTS_AVAILABLE:
            try:
                self.liquidity_concepts_engine = LiquidityConceptsEngine()
            except Exception as e:
                logger.warning(f"Failed to initialize Liquidity Concepts engine: {e}")
    
    def run(
        self,
        symbol: str,
        timestamp: datetime,
        bars: Optional[List[Any]] = None,
        current_price: Optional[float] = None,
    ) -> LiquidityEngineV5Snapshot:
        """
        Run complete liquidity analysis including PENTA methodology.
        
        Args:
            symbol: Trading symbol
            timestamp: Analysis timestamp
            bars: Optional OHLCV bars (if not provided, will fetch)
            current_price: Optional current price
            
        Returns:
            LiquidityEngineV5Snapshot with market quality and PENTA analysis
        """
        logger.debug(f"Running LiquidityEngineV5 for {symbol} at {timestamp}")
        
        # Initialize snapshot
        snapshot = LiquidityEngineV5Snapshot(
            timestamp=timestamp,
            symbol=symbol,
        )
        
        # Step 1: Market Quality Analysis
        self._analyze_market_quality(snapshot, symbol, timestamp)
        
        # Step 2: PENTA Methodology Analysis
        self._analyze_penta(snapshot, symbol, timestamp, bars, current_price)
        
        # Step 3: Combine for final trading bias
        self._calculate_combined_bias(snapshot)
        
        return snapshot
    
    def _analyze_market_quality(
        self,
        snapshot: LiquidityEngineV5Snapshot,
        symbol: str,
        timestamp: datetime,
    ) -> None:
        """Analyze market quality metrics."""
        if not self.market_adapter:
            return
        
        try:
            # Get current quote
            quote = self.market_adapter.get_quote(symbol)
            
            # Calculate bid-ask spread
            mid_price = (quote.bid + quote.ask) / 2
            spread_pct = ((quote.ask - quote.bid) / mid_price) * 100 if mid_price > 0 else 0.0
            
            # Get recent volume data
            bars = self.market_adapter.get_bars(
                symbol,
                timestamp - timedelta(days=5),
                timestamp,
                timeframe="1Day"
            )
            avg_volume = sum(bar.volume for bar in bars) / len(bars) if bars else 0.0
            
            # Calculate depth
            depth = quote.bid_size + quote.ask_size
            
            # Calculate impact cost
            impact_cost = spread_pct * 0.5
            
            # Calculate liquidity score
            volume_score = min(1.0, avg_volume / 10_000_000)
            spread_score = max(0.0, 1.0 - (spread_pct / 1.0))
            liquidity_score = (volume_score * 0.7 + spread_score * 0.3)
            
            # Determine grade
            if liquidity_score >= 0.8:
                grade = MarketQualityGrade.EXCELLENT
            elif liquidity_score >= 0.6:
                grade = MarketQualityGrade.GOOD
            elif liquidity_score >= 0.4:
                grade = MarketQualityGrade.FAIR
            elif liquidity_score >= 0.2:
                grade = MarketQualityGrade.POOR
            else:
                grade = MarketQualityGrade.ILLIQUID
            
            snapshot.liquidity_score = liquidity_score
            snapshot.bid_ask_spread = spread_pct
            snapshot.volume = avg_volume
            snapshot.depth = depth
            snapshot.impact_cost = impact_cost
            snapshot.quality_grade = grade
            
        except Exception as e:
            logger.debug(f"Market quality analysis failed: {e}")
    
    def _analyze_penta(
        self,
        snapshot: LiquidityEngineV5Snapshot,
        symbol: str,
        timestamp: datetime,
        bars: Optional[List[Any]] = None,
        current_price: Optional[float] = None,
    ) -> None:
        """Run PENTA methodology analysis using all 5 sub-engines."""
        penta = snapshot.penta_state
        
        # Get price data if not provided
        if bars is None and self.market_adapter:
            try:
                bars = self.market_adapter.get_bars(
                    symbol,
                    timestamp - timedelta(days=90),
                    timestamp,
                    timeframe="1Day"
                )
            except Exception:
                bars = []
        
        if not bars:
            return
        
        if current_price is None:
            current_price = bars[-1].close if bars else None
        
        if not current_price:
            return
        
        # Track biases for confluence
        biases = {"bullish": 0, "bearish": 0}
        
        # 1. Wyckoff Analysis
        if self.wyckoff_engine:
            try:
                wyckoff_result = self.wyckoff_engine.analyze(bars, current_price)
                if wyckoff_result:
                    penta.wyckoff_enabled = True
                    penta.wyckoff_phase = getattr(wyckoff_result, 'phase', None)
                    penta.wyckoff_event = getattr(wyckoff_result, 'event', None)
                    penta.wyckoff_bias = getattr(wyckoff_result, 'bias', None)
                    penta.wyckoff_confidence = getattr(wyckoff_result, 'confidence', 0.5)
                    
                    if penta.wyckoff_bias == "bullish":
                        biases["bullish"] += 1
                    elif penta.wyckoff_bias == "bearish":
                        biases["bearish"] += 1
            except Exception as e:
                logger.debug(f"Wyckoff analysis failed: {e}")
        
        # 2. ICT Analysis
        if self.ict_engine:
            try:
                ict_result = self.ict_engine.analyze(bars, current_price)
                if ict_result:
                    penta.ict_enabled = True
                    penta.ict_bias = getattr(ict_result, 'daily_bias', None)
                    if hasattr(ict_result, 'daily_bias') and ict_result.daily_bias:
                        penta.ict_bias = ict_result.daily_bias.value if hasattr(ict_result.daily_bias, 'value') else str(ict_result.daily_bias)
                    penta.ict_in_discount = getattr(ict_result, 'in_discount', False)
                    penta.ict_in_premium = getattr(ict_result, 'in_premium', False)
                    penta.ict_fvg_count = len(getattr(ict_result, 'fvgs', []))
                    penta.ict_order_block_count = len(getattr(ict_result, 'order_blocks', []))
                    penta.ict_confidence = getattr(ict_result, 'confidence', 0.5)
                    
                    if penta.ict_bias and "bullish" in str(penta.ict_bias).lower():
                        biases["bullish"] += 1
                    elif penta.ict_bias and "bearish" in str(penta.ict_bias).lower():
                        biases["bearish"] += 1
            except Exception as e:
                logger.debug(f"ICT analysis failed: {e}")
        
        # 3. Order Flow Analysis
        if self.order_flow_engine:
            try:
                of_result = self.order_flow_engine.analyze(bars, current_price)
                if of_result:
                    penta.order_flow_enabled = True
                    penta.order_flow_bias = getattr(of_result, 'bias', None)
                    penta.cvd_trend = getattr(of_result, 'cvd_trend', None)
                    penta.absorption_detected = getattr(of_result, 'absorption_detected', False)
                    penta.exhaustion_detected = getattr(of_result, 'exhaustion_detected', False)
                    penta.order_flow_confidence = getattr(of_result, 'confidence', 0.5)
                    
                    if penta.order_flow_bias == "bullish":
                        biases["bullish"] += 1
                    elif penta.order_flow_bias == "bearish":
                        biases["bearish"] += 1
            except Exception as e:
                logger.debug(f"Order Flow analysis failed: {e}")
        
        # 4. Supply & Demand Analysis
        if self.supply_demand_engine:
            try:
                sd_result = self.supply_demand_engine.analyze(bars, current_price)
                if sd_result:
                    penta.supply_demand_enabled = True
                    penta.sd_bias = getattr(sd_result, 'bias', None)
                    penta.demand_zones = len(getattr(sd_result, 'demand_zones', []))
                    penta.supply_zones = len(getattr(sd_result, 'supply_zones', []))
                    penta.fresh_demand_zones = getattr(sd_result, 'fresh_demand_count', 0)
                    penta.fresh_supply_zones = getattr(sd_result, 'fresh_supply_count', 0)
                    penta.nearest_demand = getattr(sd_result, 'nearest_demand', None)
                    penta.nearest_supply = getattr(sd_result, 'nearest_supply', None)
                    penta.sd_confidence = getattr(sd_result, 'confidence', 0.5)
                    
                    if penta.sd_bias == "bullish":
                        biases["bullish"] += 1
                    elif penta.sd_bias == "bearish":
                        biases["bearish"] += 1
            except Exception as e:
                logger.debug(f"Supply & Demand analysis failed: {e}")
        
        # 5. Liquidity Concepts Analysis
        if self.liquidity_concepts_engine:
            try:
                lc_result = self.liquidity_concepts_engine.analyze(bars, current_price)
                if lc_result:
                    penta.liquidity_concepts_enabled = True
                    penta.lc_trend = lc_result.trend.value if lc_result.trend else None
                    penta.buy_pools = len([p for p in lc_result.pools if p.side.value == "buy_side"])
                    penta.sell_pools = len([p for p in lc_result.pools if p.side.value == "sell_side"])
                    penta.liquidity_voids = len(lc_result.voids)
                    
                    if lc_result.inducements:
                        ind = lc_result.inducements[0]
                        penta.active_inducement = ind.inducement_type.value
                        penta.inducement_direction = ind.expected_direction
                    
                    penta.strong_highs = len(lc_result.strong_highs)
                    penta.strong_lows = len(lc_result.strong_lows)
                    penta.weak_highs = len(lc_result.weak_highs)
                    penta.weak_lows = len(lc_result.weak_lows)
                    penta.market_structure = lc_result.structure_type.value if lc_result.structure_type else None
                    penta.lc_confidence = lc_result.bias_confidence
                    
                    if penta.lc_trend == "bullish":
                        biases["bullish"] += 1
                    elif penta.lc_trend == "bearish":
                        biases["bearish"] += 1
            except Exception as e:
                logger.debug(f"Liquidity Concepts analysis failed: {e}")
        
        # Calculate confluence
        total_enabled = sum([
            penta.wyckoff_enabled,
            penta.ict_enabled,
            penta.order_flow_enabled,
            penta.supply_demand_enabled,
            penta.liquidity_concepts_enabled,
        ])
        
        agreeing = max(biases.values())
        penta.agreeing_methodologies = agreeing
        
        # Determine confluence level
        if agreeing >= 5:
            penta.confluence_level = "PENTA"
        elif agreeing >= 4:
            penta.confluence_level = "QUAD"
        elif agreeing >= 3:
            penta.confluence_level = "TRIPLE"
        elif agreeing >= 2:
            penta.confluence_level = "DOUBLE"
        elif agreeing >= 1:
            penta.confluence_level = "SINGLE"
        
        # Determine combined bias
        if biases["bullish"] > biases["bearish"]:
            penta.combined_bias = "bullish"
        elif biases["bearish"] > biases["bullish"]:
            penta.combined_bias = "bearish"
        else:
            penta.combined_bias = "neutral"
        
        # Calculate combined confidence with confluence bonus
        base_conf = sum([
            penta.wyckoff_confidence if penta.wyckoff_enabled else 0,
            penta.ict_confidence if penta.ict_enabled else 0,
            penta.order_flow_confidence if penta.order_flow_enabled else 0,
            penta.sd_confidence if penta.supply_demand_enabled else 0,
            penta.lc_confidence if penta.liquidity_concepts_enabled else 0,
        ]) / max(1, total_enabled)
        
        # Apply confluence bonus
        confluence_bonus = {
            "PENTA": 0.30,
            "QUAD": 0.25,
            "TRIPLE": 0.15,
            "DOUBLE": 0.08,
            "SINGLE": 0.0,
        }.get(penta.confluence_level, 0.0)
        
        penta.combined_confidence = min(1.0, base_conf * (1 + confluence_bonus))
    
    def _calculate_combined_bias(self, snapshot: LiquidityEngineV5Snapshot) -> None:
        """Calculate final trading bias combining market quality and PENTA."""
        penta = snapshot.penta_state
        
        # If market quality is poor, reduce confidence
        quality_multiplier = {
            MarketQualityGrade.EXCELLENT: 1.0,
            MarketQualityGrade.GOOD: 0.95,
            MarketQualityGrade.FAIR: 0.85,
            MarketQualityGrade.POOR: 0.7,
            MarketQualityGrade.ILLIQUID: 0.5,
        }.get(snapshot.quality_grade, 0.8)
        
        snapshot.trading_bias = penta.combined_bias
        snapshot.confidence = penta.combined_confidence * quality_multiplier
    
    def get_penta_engines(self) -> Dict[str, Any]:
        """Get references to all PENTA sub-engines."""
        return {
            "wyckoff": self.wyckoff_engine,
            "ict": self.ict_engine,
            "order_flow": self.order_flow_engine,
            "supply_demand": self.supply_demand_engine,
            "liquidity_concepts": self.liquidity_concepts_engine,
        }
