"""
Gnosis Alpha - Signal Generator

Generates directional trading signals using the FULL Gnosis engine architecture.
Optimized for short-term (0-7 day) stock trading on Robinhood/Webull.

Integrates:
- HedgeEngineV3: Dealer flow, gamma pressure, energy asymmetry analysis
- LiquidityEngineV1: Market liquidity, spreads, and depth analysis  
- SentimentEngineV1: Multi-source sentiment analysis
- ComposerAgentV1: Consensus building from all engine outputs
- TechnicalAnalyzer: Standalone technical analysis (always runs)

PENTA Methodology (LiquidityAgentV5):
- Wyckoff: VSA, Phases, Events, Structures
- ICT: FVGs, Order Blocks, OTE, Liquidity Sweeps
- Order Flow: Footprint, CVD, Volume Profile
- Supply & Demand: Zones, Strength, Status
- Liquidity Concepts: Pools, Voids, Strong/Weak Swings, Inducements

Output: BUY / SELL / HOLD signals with confidence scores and smart money analysis
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Gnosis components
try:
    from engines.hedge.hedge_engine_v3 import HedgeEngineV3
    from engines.liquidity.liquidity_engine_v1 import LiquidityEngineV1
    from engines.sentiment.sentiment_engine_v1 import SentimentEngineV1
    from engines.elasticity.elasticity_engine_v1 import ElasticityEngineV1
    from engines.inputs.adapter_factory import (
        create_options_adapter,
        create_market_data_adapter,
    )
    from agents.hedge_agent_v3 import HedgeAgentV3
    from agents.liquidity_agent_v1 import LiquidityAgentV1
    from agents.sentiment_agent_v1 import SentimentAgentV1
    from agents.composer.composer_agent_v1 import ComposerAgentV1
    from schemas.core_schemas import (
        DirectionEnum, 
        AgentSignal, 
        AgentSuggestion, 
        PipelineResult,
        HedgeSnapshot,
        LiquiditySnapshot,
        SentimentSnapshot,
        MLEnhancementSnapshot,
    )
    GNOSIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Gnosis engines not available: {e}")
    GNOSIS_AVAILABLE = False
    DirectionEnum = None
    HedgeSnapshot = None
    LiquiditySnapshot = None
    SentimentSnapshot = None
    PipelineResult = None

# Try to import PENTA methodology engines (Liquidity Concepts + 4 other methodologies)
try:
    from engines.engine_factory import create_unified_analysis_engines
    from agents.liquidity_agent_v5 import LiquidityAgentV5
    from engines.liquidity import (
        LiquidityEngineV5,
        LiquidityEngineV5Snapshot,
        PENTAState,
        LiquidityConceptsEngine,
        LiquidityConceptsState,
        LiquidityPoolType,
        LiquidityPoolSide,
        SwingStrength,
        LiquidityInducementType,
    )
    PENTA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PENTA methodology engines not available: {e}")
    PENTA_AVAILABLE = False
    LiquidityAgentV5 = None
    LiquidityConceptsEngine = None
    LiquidityEngineV5 = None

# Import Trade and Monitoring layers
try:
    from trade.gnosis_trade_agent import AlphaTradeAgent, AlphaSignal as TradeAlphaSignal
    from trade.gnosis_trade_agent_v2 import AlphaTradeAgentV2, AlphaSignalV2
    from agents.monitoring import AlphaMonitor
    TRADE_LAYER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Trade/Monitor layers not available: {e}")
    TRADE_LAYER_AVAILABLE = False
    AlphaTradeAgent = None
    AlphaTradeAgentV2 = None
    AlphaMonitor = None

# Import ComposerAgentV4 for proper architecture
try:
    from agents.composer.composer_agent_v4 import (
        ComposerAgentV4,
        ComposerOutput,
        ComposerMode,
    )
    COMPOSER_V4_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ComposerAgentV4 not available: {e}")
    COMPOSER_V4_AVAILABLE = False
    ComposerAgentV4 = None

from alpha.alpha_config import AlphaConfig
from alpha.pdt_tracker import PDTTracker
from alpha.technical_analyzer import TechnicalAnalyzer, TechnicalSignals


class SignalDirection(str, Enum):
    """Simple directional signal."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStrength(str, Enum):
    """Signal strength indicator."""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


@dataclass
class AlphaSignal:
    """A single trading signal from Gnosis Alpha."""
    
    # Core signal
    symbol: str
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    strength: SignalStrength
    
    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: Optional[datetime] = None  # Signal expiration
    holding_period_days: int = 3  # Suggested holding period
    
    # Price targets
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Support/Resistance from OPTIONS LIQUIDITY POOLS
    support_levels: List[float] = field(default_factory=list)  # PUT walls (dealer buy zones)
    resistance_levels: List[float] = field(default_factory=list)  # CALL walls (dealer sell zones)
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    # Liquidity pool details
    put_walls: List[Dict] = field(default_factory=list)  # PUT OI concentrations
    call_walls: List[Dict] = field(default_factory=list)  # CALL OI concentrations
    max_pain: Optional[float] = None  # Max pain strike price
    gamma_flip: Optional[float] = None  # Where dealer gamma flips
    
    # Engine contributions
    hedge_signal: Optional[str] = None
    hedge_confidence: float = 0.0
    liquidity_signal: Optional[str] = None
    liquidity_confidence: float = 0.0
    sentiment_signal: Optional[str] = None
    sentiment_confidence: float = 0.0
    
    # Detailed Engine Analysis (agentic reasoning)
    dealer_positioning: Optional[str] = None  # "Long gamma" / "Short gamma" / "Neutral"
    gamma_pressure: Optional[str] = None  # "High buying pressure" / "High selling pressure"
    energy_bias: Optional[str] = None  # "Upward momentum" / "Downward momentum"
    regime: Optional[str] = None  # Market regime from hedge engine
    
    # Liquidity Analysis
    liquidity_grade: Optional[str] = None  # "A+" to "F"
    spread_quality: Optional[str] = None  # "Tight" / "Wide" / "Very Wide"
    market_depth: Optional[str] = None  # "Deep" / "Shallow"
    
    # Technical Analysis Details
    trend_status: Optional[str] = None  # "Strong uptrend" / "Downtrend" / "Ranging"
    momentum_status: Optional[str] = None  # "Oversold bounce" / "Overbought reversal"
    price_vs_sma20: Optional[str] = None  # "Above" / "Below" with %
    price_vs_sma50: Optional[str] = None
    price_vs_sma200: Optional[str] = None
    bollinger_position: Optional[str] = None  # "Near upper band" / "Near lower band"
    
    # Reasoning - now structured
    reasoning: str = ""
    agent_insights: List[str] = field(default_factory=list)  # Individual agent observations
    risk_factors: List[str] = field(default_factory=list)
    catalysts: List[str] = field(default_factory=list)  # Potential catalysts for move
    
    # PDT info
    is_day_trade_candidate: bool = False
    day_trades_remaining: int = 3
    
    # Volume indicators
    unusual_volume: bool = False
    volume_ratio: float = 1.0
    volume_description: str = "Normal"
    
    # PENTA Methodology - Liquidity Concepts (Smart Money Analysis)
    penta_enabled: bool = False
    penta_confluence: Optional[str] = None  # "PENTA", "QUAD", "TRIPLE", "Double"
    penta_confidence: float = 0.0
    
    # Liquidity Concepts Analysis
    liquidity_pools_above: int = 0  # Buy-side pools (resistance)
    liquidity_pools_below: int = 0  # Sell-side pools (support)
    nearest_buy_pool: Optional[float] = None  # Nearest buy-side liquidity
    nearest_sell_pool: Optional[float] = None  # Nearest sell-side liquidity
    pool_type: Optional[str] = None  # "MAJOR", "MINOR", "CLUSTERED", "EQUAL_HIGHS/LOWS"
    
    # Smart Money Inducement
    active_inducement: Optional[str] = None  # "STOP_HUNT", "LIQUIDITY_SWEEP", etc.
    inducement_direction: Optional[str] = None  # Expected direction after sweep
    inducement_confidence: float = 0.0
    
    # Strong/Weak Swing Levels
    strong_highs: int = 0
    strong_lows: int = 0
    weak_highs: int = 0
    weak_lows: int = 0
    trend_from_bos: Optional[str] = None  # Trend direction from Break of Structure
    
    # Market Structure
    market_structure: Optional[str] = None  # "SMOOTH", "ROUGH", "MIXED"
    zone_reliability: Optional[str] = None  # Based on fractal analysis
    
    # Liquidity Voids
    voids_above: int = 0
    voids_below: int = 0
    nearest_void_target: Optional[float] = None  # Price target from void fill
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "confidence": round(self.confidence, 3),
            "strength": self.strength.value,
            "timestamp": self.timestamp.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "holding_period_days": self.holding_period_days,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            # Support/Resistance from Liquidity Pools
            "support_levels": self.support_levels,
            "resistance_levels": self.resistance_levels,
            "nearest_support": self.nearest_support,
            "nearest_resistance": self.nearest_resistance,
            # Liquidity pool details
            "put_walls": self.put_walls,
            "call_walls": self.call_walls,
            "max_pain": self.max_pain,
            "gamma_flip": self.gamma_flip,
            # Engine signals
            "hedge_signal": self.hedge_signal,
            "hedge_confidence": round(self.hedge_confidence, 3),
            "liquidity_signal": self.liquidity_signal,
            "liquidity_confidence": round(self.liquidity_confidence, 3),
            "sentiment_signal": self.sentiment_signal,
            "sentiment_confidence": round(self.sentiment_confidence, 3),
            # Detailed Analysis
            "dealer_positioning": self.dealer_positioning,
            "gamma_pressure": self.gamma_pressure,
            "energy_bias": self.energy_bias,
            "regime": self.regime,
            "liquidity_grade": self.liquidity_grade,
            "spread_quality": self.spread_quality,
            "market_depth": self.market_depth,
            "trend_status": self.trend_status,
            "momentum_status": self.momentum_status,
            "price_vs_sma20": self.price_vs_sma20,
            "price_vs_sma50": self.price_vs_sma50,
            "price_vs_sma200": self.price_vs_sma200,
            "bollinger_position": self.bollinger_position,
            # Reasoning
            "reasoning": self.reasoning,
            "agent_insights": self.agent_insights,
            "risk_factors": self.risk_factors,
            "catalysts": self.catalysts,
            # PDT
            "is_day_trade_candidate": self.is_day_trade_candidate,
            "day_trades_remaining": self.day_trades_remaining,
            # Volume
            "unusual_volume": self.unusual_volume,
            "volume_ratio": round(self.volume_ratio, 2),
            "volume_description": self.volume_description,
            # PENTA Methodology
            "penta_enabled": self.penta_enabled,
            "penta_confluence": self.penta_confluence,
            "penta_confidence": round(self.penta_confidence, 3),
            # Liquidity Concepts
            "liquidity_pools_above": self.liquidity_pools_above,
            "liquidity_pools_below": self.liquidity_pools_below,
            "nearest_buy_pool": self.nearest_buy_pool,
            "nearest_sell_pool": self.nearest_sell_pool,
            "pool_type": self.pool_type,
            # Inducement
            "active_inducement": self.active_inducement,
            "inducement_direction": self.inducement_direction,
            "inducement_confidence": round(self.inducement_confidence, 3),
            # Swing Analysis
            "strong_highs": self.strong_highs,
            "strong_lows": self.strong_lows,
            "weak_highs": self.weak_highs,
            "weak_lows": self.weak_lows,
            "trend_from_bos": self.trend_from_bos,
            # Market Structure
            "market_structure": self.market_structure,
            "zone_reliability": self.zone_reliability,
            # Voids
            "voids_above": self.voids_above,
            "voids_below": self.voids_below,
            "nearest_void_target": self.nearest_void_target,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AlphaSignal":
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            direction=SignalDirection(data["direction"]),
            confidence=data["confidence"],
            strength=SignalStrength(data["strength"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            valid_until=datetime.fromisoformat(data["valid_until"]) if data.get("valid_until") else None,
            holding_period_days=data.get("holding_period_days", 3),
            entry_price=data.get("entry_price"),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            hedge_signal=data.get("hedge_signal"),
            hedge_confidence=data.get("hedge_confidence", 0.0),
            liquidity_signal=data.get("liquidity_signal"),
            liquidity_confidence=data.get("liquidity_confidence", 0.0),
            sentiment_signal=data.get("sentiment_signal"),
            sentiment_confidence=data.get("sentiment_confidence", 0.0),
            reasoning=data.get("reasoning", ""),
            risk_factors=data.get("risk_factors", []),
            is_day_trade_candidate=data.get("is_day_trade_candidate", False),
            day_trades_remaining=data.get("day_trades_remaining", 3),
        )
    
    def to_robinhood_format(self) -> str:
        """
        Format signal for Robinhood-style display.
        Simple, clean output for mobile viewing.
        """
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}[self.direction.value]
        strength_emoji = {"STRONG": "💪", "MODERATE": "👍", "WEAK": "👋"}[self.strength.value]
        
        lines = [
            f"{emoji} {self.symbol}: {self.direction.value} {strength_emoji}",
            f"Confidence: {self.confidence * 100:.0f}%",
        ]
        
        if self.entry_price:
            lines.append(f"Entry: ${self.entry_price:.2f}")
        if self.stop_loss:
            lines.append(f"Stop: ${self.stop_loss:.2f}")
        if self.take_profit:
            lines.append(f"Target: ${self.take_profit:.2f}")
        
        lines.append(f"Hold: {self.holding_period_days} days")
        
        if self.risk_factors:
            lines.append(f"⚠️ {', '.join(self.risk_factors[:2])}")
        
        return "\n".join(lines)


class AlphaSignalGenerator:
    """
    Generates BUY/SELL/HOLD signals using FULL Gnosis engine architecture.
    
    Uses the complete Gnosis analysis pipeline including:
    - HedgeEngineV3 for dealer flow/gamma analysis
    - LiquidityEngineV1 for market liquidity analysis
    - SentimentEngineV1 for multi-source sentiment
    - ComposerAgentV1 for consensus building
    - TechnicalAnalyzer for standalone technical analysis
    
    Outputs simplified directional signals suitable for Robinhood/Webull users.
    """
    
    def __init__(
        self,
        config: Optional[AlphaConfig] = None,
        pdt_tracker: Optional[PDTTracker] = None,
    ):
        """
        Initialize Alpha Signal Generator with full Gnosis architecture.
        
        Args:
            config: Alpha configuration (uses defaults if None)
            pdt_tracker: PDT compliance tracker (creates new if None)
        """
        self.config = config or AlphaConfig.from_env()
        self.pdt_tracker = pdt_tracker or PDTTracker(
            max_day_trades=self.config.max_day_trades,
            lookback_days=self.config.pdt_lookback_days,
        )
        
        # Validate config
        issues = self.config.validate()
        if issues:
            logger.warning(f"Config issues: {issues}")
        
        # Initialize engines, agents, and adapters
        self.engines: Dict[str, Any] = {}
        self.agents: Dict[str, Any] = {}
        self.adapters: Dict[str, Any] = {}
        self.composer = None
        self.gnosis_enabled = False
        self.penta_enabled = False  # PENTA methodology (5 engines)
        
        # Initialize technical analyzer (always available as fallback/supplement)
        self.technical_analyzer = TechnicalAnalyzer(lookback_days=self.config.lookback_days)
        
        # Initialize Gnosis engines if available
        if GNOSIS_AVAILABLE:
            self._initialize_engines()
        else:
            logger.info("Gnosis engines not available - using standalone technical analysis mode")
        
        # Price cache for entry/stop/target calculations
        self._price_cache: Dict[str, float] = {}
        
        mode = "FULL GNOSIS" if self.gnosis_enabled else "TECHNICAL ANALYSIS"
        logger.info(f"AlphaSignalGenerator initialized with {len(self.config.universe)} symbols ({mode} mode)")
    
    def _initialize_engines(self) -> None:
        """Initialize FULL Gnosis engine architecture."""
        try:
            logger.info("Initializing FULL Gnosis engine architecture for Alpha...")
            
            # Create adapters for data access
            self.adapters["options"] = create_options_adapter(prefer_real=True)
            self.adapters["market_data"] = create_market_data_adapter(prefer_real=True)
            
            # Engine configurations
            hedge_config = {
                "regime_components": 3,
                "regime_history": 256,
                "regime_min_samples": 32,
                "gamma_weight": 0.6,
                "vanna_weight": 0.4,
                "lookback_days": self.config.lookback_days,
            }
            
            liquidity_config = {
                "min_volume_threshold": 1_000_000,
                "spread_threshold": 0.5,  # 0.5% spread threshold
            }
            
            sentiment_config = {
                "news_weight": 0.4,
                "flow_weight": 0.3,
                "technical_weight": 0.3,
            }
            
            # Initialize Gnosis Engines
            self.engines["hedge"] = HedgeEngineV3(
                options_adapter=self.adapters["options"],
                config=hedge_config,
            )
            
            self.engines["liquidity"] = LiquidityEngineV1(
                market_adapter=self.adapters["market_data"],
                config=liquidity_config,
            )
            
            # Sentiment engine uses processors (use empty list for now - tech signals will supplement)
            self.engines["sentiment"] = SentimentEngineV1(
                processors=[],  # Will use technical sentiment as fallback
                config=sentiment_config,
            )
            
            # Initialize Gnosis Agents
            agent_config = {
                "min_confidence": self.config.min_confidence,
                "use_lstm": False,  # Disable LSTM for Alpha (simplicity)
                "min_liquidity_score": 0.3,
                "min_sentiment_threshold": 0.2,
            }
            
            self.agents["hedge"] = HedgeAgentV3(config=agent_config)
            self.agents["liquidity"] = LiquidityAgentV1(config=agent_config)
            self.agents["sentiment"] = SentimentAgentV1(config=agent_config)
            
            # Composer agent for consensus building
            # Use ComposerAgentV4 if available for full architecture
            if COMPOSER_V4_AVAILABLE:
                self.composer = ComposerAgentV4(
                    weights={
                        'hedge': 0.4,      # Dealer flow/structure: 40%
                        'liquidity': 0.2,  # Liquidity quality: 20%
                        'sentiment': 0.4,  # Market sentiment: 40%
                    },
                    config={"min_confidence": self.config.min_confidence}
                )
                logger.info("   - ComposerAgentV4: Full architecture integration")
            else:
                # Fallback to V1
                weights = type('Weights', (), {
                    'hedge': 0.4,      # Dealer flow/structure: 40%
                    'liquidity': 0.2,  # Liquidity quality: 20%
                    'sentiment': 0.4,  # Market sentiment: 40%
                })()
                
                self.composer = ComposerAgentV1(
                    weights=weights,
                    config={"min_consensus_score": self.config.min_confidence}
                )
            
            self.gnosis_enabled = True
            logger.info("✅ FULL Gnosis engine architecture initialized successfully")
            logger.info(f"   - HedgeEngineV3: Dealer flow/gamma analysis")
            logger.info(f"   - LiquidityEngineV1: Market liquidity analysis")
            logger.info(f"   - SentimentEngineV1: Multi-source sentiment")
            logger.info(f"   - ComposerAgentV1: Consensus builder")
            
            # Initialize PENTA methodology engines (5 methodologies)
            self._initialize_penta_engines()
            
        except Exception as e:
            logger.error(f"Failed to initialize Gnosis engines: {e}")
            logger.info("Falling back to standalone technical analysis mode")
            self.gnosis_enabled = False
    
    def _initialize_penta_engines(self) -> None:
        """Initialize PENTA methodology engines for smart money analysis.
        
        Architecture:
            LiquidityEngineV5 (unified) → LiquidityAgentV5 → Composer
        
        The LiquidityEngineV5 contains all 5 PENTA sub-engines internally.
        """
        if not PENTA_AVAILABLE:
            logger.info("PENTA methodology not available - using standard Gnosis engines")
            self.penta_enabled = False
            return
        
        try:
            logger.info("Initializing PENTA methodology with unified LiquidityEngineV5...")
            
            # Create unified analysis engines with LiquidityEngineV5
            penta_engines = create_unified_analysis_engines(use_unified_v5=True)
            
            # Store the unified engine (preferred approach)
            self.engines["liquidity_v5"] = penta_engines.get("liquidity_engine_v5")
            
            # Also store individual sub-engines for backward compatibility
            self.engines["wyckoff"] = penta_engines.get("wyckoff_engine")
            self.engines["ict"] = penta_engines.get("ict_engine")
            self.engines["order_flow"] = penta_engines.get("order_flow_engine")
            self.engines["supply_demand"] = penta_engines.get("supply_demand_engine")
            self.engines["liquidity_concepts"] = penta_engines.get("liquidity_concepts_engine")
            
            # Create LiquidityAgentV5 with unified engine (correct architecture)
            penta_config = {
                "min_confidence": self.config.min_confidence,
                "wyckoff_weight": 0.18,
                "ict_weight": 0.18,
                "order_flow_weight": 0.18,
                "supply_demand_weight": 0.18,
                "liquidity_concepts_weight": 0.18,
                "base_weight": 0.10,
            }
            
            # Use unified engine (preferred)
            self.agents["penta"] = LiquidityAgentV5(
                config=penta_config,
                liquidity_engine_v5=self.engines["liquidity_v5"],  # Unified engine
            )
            
            self.penta_enabled = True
            logger.info("✅ PENTA methodology initialized with unified LiquidityEngineV5")
            logger.info("   Architecture: LiquidityEngineV5 → LiquidityAgentV5 → Composer")
            logger.info("   Sub-engines: Wyckoff, ICT, Order Flow, Supply/Demand, Liquidity Concepts")
            
        except Exception as e:
            logger.warning(f"Failed to initialize PENTA engines: {e}")
            logger.info("Continuing with standard Gnosis engines only")
            self.penta_enabled = False
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol using Alpaca or yfinance."""
        # Check cache first
        if symbol in self._price_cache:
            return self._price_cache[symbol]
        
        try:
            # Try yfinance first (free, no API key needed)
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                self._price_cache[symbol] = price
                return price
        except Exception as e:
            logger.debug(f"yfinance failed for {symbol}: {e}")
        
        try:
            # Fall back to Alpaca
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            
            client = StockHistoricalDataClient(
                api_key=self.config.alpaca_api_key,
                secret_key=self.config.alpaca_secret_key,
            )
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = client.get_stock_latest_quote(request)
            if symbol in quote:
                price = float(quote[symbol].ask_price)
                self._price_cache[symbol] = price
                return price
        except Exception as e:
            logger.debug(f"Alpaca failed for {symbol}: {e}")
        
        return None
    
    def _get_liquidity_pools(self, symbol: str, current_price: float, timestamp: datetime) -> Dict[str, Any]:
        """
        Calculate support/resistance levels from OPTIONS LIQUIDITY POOLS.
        
        Liquidity pools are strike prices with high open interest where:
        - High PUT OI below price = SUPPORT (dealers must buy to hedge)
        - High CALL OI above price = RESISTANCE (dealers must sell to hedge)
        
        Returns dict with support_levels, resistance_levels, and pool details.
        """
        result = {
            "support_levels": [],
            "resistance_levels": [],
            "put_walls": [],  # Major put OI concentrations
            "call_walls": [],  # Major call OI concentrations
            "max_pain": None,  # Strike with max total OI
            "gamma_flip": None,  # Where dealer gamma flips sign
        }
        
        if not self.gnosis_enabled or "options" not in self.adapters:
            return result
        
        try:
            # Get options chain
            chain = self.adapters["options"].get_chain(symbol, timestamp)
            if not chain:
                return result
            
            from collections import defaultdict
            
            # Aggregate OI by strike
            strike_total_oi = defaultdict(float)
            strike_call_oi = defaultdict(float)
            strike_put_oi = defaultdict(float)
            strike_call_gamma = defaultdict(float)
            strike_put_gamma = defaultdict(float)
            
            for contract in chain:
                strike = contract.strike
                oi = contract.open_interest or 0
                gamma = abs(contract.gamma or 0) * oi
                
                strike_total_oi[strike] += oi
                if contract.option_type == "call":
                    strike_call_oi[strike] += oi
                    strike_call_gamma[strike] += gamma
                else:
                    strike_put_oi[strike] += oi
                    strike_put_gamma[strike] += gamma
            
            if not strike_total_oi:
                return result
            
            # Find max pain (strike with highest total OI)
            max_pain_strike = max(strike_total_oi.keys(), key=lambda x: strike_total_oi[x])
            result["max_pain"] = max_pain_strike
            
            # Sort strikes by OI
            sorted_strikes = sorted(strike_total_oi.items(), key=lambda x: x[1], reverse=True)
            
            # Find major PUT walls below price (SUPPORT)
            put_walls_below = []
            for strike, _ in sorted_strikes:
                if strike < current_price and strike_put_oi[strike] > 0:
                    put_walls_below.append({
                        "strike": strike,
                        "put_oi": strike_put_oi[strike],
                        "call_oi": strike_call_oi[strike],
                        "total_oi": strike_total_oi[strike],
                        "pct_from_price": ((current_price - strike) / current_price) * 100,
                    })
            
            # Sort by put OI and take top 5
            put_walls_below.sort(key=lambda x: x["put_oi"], reverse=True)
            result["put_walls"] = put_walls_below[:5]
            result["support_levels"] = [w["strike"] for w in put_walls_below[:5]]
            
            # Find major CALL walls above price (RESISTANCE)
            call_walls_above = []
            for strike, _ in sorted_strikes:
                if strike > current_price and strike_call_oi[strike] > 0:
                    call_walls_above.append({
                        "strike": strike,
                        "call_oi": strike_call_oi[strike],
                        "put_oi": strike_put_oi[strike],
                        "total_oi": strike_total_oi[strike],
                        "pct_from_price": ((strike - current_price) / current_price) * 100,
                    })
            
            # Sort by call OI and take top 5
            call_walls_above.sort(key=lambda x: x["call_oi"], reverse=True)
            result["call_walls"] = call_walls_above[:5]
            result["resistance_levels"] = [w["strike"] for w in call_walls_above[:5]]
            
            # Calculate gamma flip point (where net gamma changes sign)
            # This is where dealer hedging behavior changes
            strikes_near_price = [s for s in strike_total_oi.keys() 
                                  if abs(s - current_price) / current_price < 0.15]
            
            if strikes_near_price:
                for strike in sorted(strikes_near_price):
                    net_gamma = strike_call_gamma[strike] - strike_put_gamma[strike]
                    # Simplified: gamma flip is near max pain typically
                    if strike >= current_price:
                        result["gamma_flip"] = strike
                        break
            
        except Exception as e:
            logger.debug(f"Error calculating liquidity pools for {symbol}: {e}")
        
        return result
    
    def _analyze_symbol(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        """
        Run FULL Gnosis engine analysis on a symbol with rich, agentic insights.
        
        Uses the complete Gnosis architecture:
        1. HedgeEngineV3 - Dealer flow, gamma pressure, energy asymmetry
        2. LiquidityEngineV1 - Market depth, spreads, tradability
        3. SentimentEngineV1 - Multi-source sentiment analysis
        4. TechnicalAnalyzer - Technical indicators, support/resistance
        5. ComposerAgentV1 - Consensus building
        
        Returns analysis dict with direction, confidence, detailed reasoning, and agent insights.
        """
        analysis = {
            "direction": SignalDirection.HOLD,
            "confidence": 0.0,
            "hedge_signal": None,
            "hedge_confidence": 0.0,
            "liquidity_signal": None, 
            "liquidity_confidence": 0.0,
            "sentiment_signal": None,
            "sentiment_confidence": 0.0,
            "reasoning": "",
            "risk_factors": [],
            "agent_insights": [],  # Detailed insights from each agent
            "catalysts": [],  # Potential catalysts
            # Volume indicators
            "unusual_volume": False,
            "volume_ratio": 1.0,
            "volume_description": "Normal",
            # Gnosis engine outputs
            "hedge_snapshot": None,
            "liquidity_snapshot": None,
            "sentiment_snapshot": None,
            # Detailed analysis fields
            "dealer_positioning": None,
            "gamma_pressure": None,
            "energy_bias": None,
            "regime": None,
            "liquidity_grade": None,
            "spread_quality": None,
            "market_depth": None,
            "trend_status": None,
            "momentum_status": None,
            "price_vs_sma20": None,
            "price_vs_sma50": None,
            "price_vs_sma200": None,
            "bollinger_position": None,
            # Support/Resistance from Liquidity Pools
            "support_levels": [],
            "resistance_levels": [],
            "nearest_support": None,
            "nearest_resistance": None,
            # Liquidity pool details
            "put_walls": [],  # PUT OI concentrations (support)
            "call_walls": [],  # CALL OI concentrations (resistance)
            "max_pain": None,  # Max pain strike
            "gamma_flip": None,  # Gamma flip point
        }
        
        # =====================================================
        # STEP 1: Technical Analysis (always runs)
        # =====================================================
        tech_signals = None
        try:
            tech_signals = self.technical_analyzer.analyze(symbol)
            
            if tech_signals.current_price:
                self._price_cache[symbol] = tech_signals.current_price
            
            # Volume metrics
            analysis["volume_ratio"] = tech_signals.volume_ratio
            analysis["unusual_volume"] = tech_signals.unusual_volume
            analysis["volume_description"] = tech_signals.volume_description
            
            # === TREND ANALYSIS ===
            if tech_signals.current_price and tech_signals.sma_20:
                pct_from_sma20 = ((tech_signals.current_price - tech_signals.sma_20) / tech_signals.sma_20) * 100
                if pct_from_sma20 > 0:
                    analysis["price_vs_sma20"] = f"+{pct_from_sma20:.1f}% above"
                else:
                    analysis["price_vs_sma20"] = f"{pct_from_sma20:.1f}% below"
            
            if tech_signals.current_price and tech_signals.sma_50:
                pct_from_sma50 = ((tech_signals.current_price - tech_signals.sma_50) / tech_signals.sma_50) * 100
                if pct_from_sma50 > 0:
                    analysis["price_vs_sma50"] = f"+{pct_from_sma50:.1f}% above"
                else:
                    analysis["price_vs_sma50"] = f"{pct_from_sma50:.1f}% below"
                    
            if tech_signals.current_price and tech_signals.sma_200:
                pct_from_sma200 = ((tech_signals.current_price - tech_signals.sma_200) / tech_signals.sma_200) * 100
                if pct_from_sma200 > 0:
                    analysis["price_vs_sma200"] = f"+{pct_from_sma200:.1f}% above"
                else:
                    analysis["price_vs_sma200"] = f"{pct_from_sma200:.1f}% below"
            
            # Determine trend status
            if tech_signals.sma_20 and tech_signals.sma_50 and tech_signals.sma_200:
                if tech_signals.sma_20 > tech_signals.sma_50 > tech_signals.sma_200:
                    analysis["trend_status"] = "Strong uptrend (SMAs aligned bullish)"
                elif tech_signals.sma_20 < tech_signals.sma_50 < tech_signals.sma_200:
                    analysis["trend_status"] = "Strong downtrend (SMAs aligned bearish)"
                elif tech_signals.current_price > tech_signals.sma_20:
                    analysis["trend_status"] = "Short-term uptrend"
                elif tech_signals.current_price < tech_signals.sma_20:
                    analysis["trend_status"] = "Short-term downtrend"
                else:
                    analysis["trend_status"] = "Ranging/Consolidating"
            
            # === MOMENTUM ANALYSIS ===
            if tech_signals.rsi_14:
                if tech_signals.rsi_14 < 20:
                    analysis["momentum_status"] = "Extremely oversold - potential bounce"
                    analysis["catalysts"].append("RSI extreme oversold - reversal candidate")
                elif tech_signals.rsi_14 < 30:
                    analysis["momentum_status"] = "Oversold - watching for reversal"
                elif tech_signals.rsi_14 > 80:
                    analysis["momentum_status"] = "Extremely overbought - potential pullback"
                    analysis["catalysts"].append("RSI extreme overbought - pullback likely")
                elif tech_signals.rsi_14 > 70:
                    analysis["momentum_status"] = "Overbought - caution advised"
                elif 45 <= tech_signals.rsi_14 <= 55:
                    analysis["momentum_status"] = "Neutral momentum"
                elif tech_signals.rsi_14 > 55:
                    analysis["momentum_status"] = "Bullish momentum building"
                else:
                    analysis["momentum_status"] = "Bearish momentum"
            
            # === BOLLINGER BANDS ===
            if tech_signals.bollinger_upper and tech_signals.bollinger_lower and tech_signals.current_price:
                bb_range = tech_signals.bollinger_upper - tech_signals.bollinger_lower
                bb_position = (tech_signals.current_price - tech_signals.bollinger_lower) / bb_range if bb_range > 0 else 0.5
                
                if bb_position > 0.95:
                    analysis["bollinger_position"] = "At upper band (overbought)"
                elif bb_position > 0.8:
                    analysis["bollinger_position"] = "Near upper band"
                elif bb_position < 0.05:
                    analysis["bollinger_position"] = "At lower band (oversold)"
                elif bb_position < 0.2:
                    analysis["bollinger_position"] = "Near lower band"
                else:
                    analysis["bollinger_position"] = "Middle of bands"
            
            # Get Support/Resistance from OPTIONS LIQUIDITY POOLS
            # These are strike prices with high open interest where dealer hedging creates price magnets
            if tech_signals.current_price:
                try:
                    liquidity_pools = self._get_liquidity_pools(symbol, tech_signals.current_price, timestamp)
                    
                    # PUT walls below price = SUPPORT (dealers buy to hedge)
                    analysis["support_levels"] = liquidity_pools.get("support_levels", [])[:5]
                    # CALL walls above price = RESISTANCE (dealers sell to hedge)
                    analysis["resistance_levels"] = liquidity_pools.get("resistance_levels", [])[:5]
                    
                    # Store detailed pool info
                    analysis["put_walls"] = liquidity_pools.get("put_walls", [])
                    analysis["call_walls"] = liquidity_pools.get("call_walls", [])
                    analysis["max_pain"] = liquidity_pools.get("max_pain")
                    analysis["gamma_flip"] = liquidity_pools.get("gamma_flip")
                    
                    # Find nearest levels
                    if analysis["support_levels"]:
                        # Sort supports descending (highest first = nearest to price)
                        sorted_supports = sorted(analysis["support_levels"], reverse=True)
                        analysis["nearest_support"] = sorted_supports[0] if sorted_supports else None
                    
                    if analysis["resistance_levels"]:
                        # Sort resistances ascending (lowest first = nearest to price)
                        sorted_resistances = sorted(analysis["resistance_levels"])
                        analysis["nearest_resistance"] = sorted_resistances[0] if sorted_resistances else None
                    
                    # Add liquidity pool insights
                    if analysis["put_walls"]:
                        top_put = analysis["put_walls"][0]
                        analysis["agent_insights"].append(
                            f"🛡️ PUT WALL (SUPPORT): ${top_put['strike']:.0f} has {top_put['put_oi']:,.0f} put OI. "
                            f"Dealers must BUY stock to hedge if price drops here ({top_put['pct_from_price']:.1f}% below)."
                        )
                    
                    if analysis["call_walls"]:
                        top_call = analysis["call_walls"][0]
                        analysis["agent_insights"].append(
                            f"🚧 CALL WALL (RESISTANCE): ${top_call['strike']:.0f} has {top_call['call_oi']:,.0f} call OI. "
                            f"Dealers must SELL stock to hedge if price rises here ({top_call['pct_from_price']:.1f}% above)."
                        )
                    
                    if analysis["max_pain"]:
                        mp_pct = ((analysis["max_pain"] - tech_signals.current_price) / tech_signals.current_price) * 100
                        direction = "above" if mp_pct > 0 else "below"
                        analysis["agent_insights"].append(
                            f"💰 MAX PAIN: ${analysis['max_pain']:.0f} ({abs(mp_pct):.1f}% {direction}). "
                            "Price tends to gravitate here by expiration as dealers hedge."
                        )
                        
                except Exception as e:
                    logger.debug(f"Liquidity pools error for {symbol}: {e}")
                    # Fallback to technical S/R if options data fails
                    try:
                        sr_levels = self.technical_analyzer.get_support_resistance(symbol)
                        analysis["support_levels"] = sr_levels.get("support", [])[:3]
                        analysis["resistance_levels"] = sr_levels.get("resistance", [])[:3]
                    except:
                        pass
                
        except Exception as e:
            logger.error(f"Technical analysis error for {symbol}: {e}")
        
        # =====================================================
        # STEP 2: Gnosis Engine Analysis
        # =====================================================
        hedge_snapshot = None
        liquidity_snapshot = None
        sentiment_snapshot = None
        agent_suggestions = []
        
        if self.gnosis_enabled:
            try:
                pipeline_result = PipelineResult(timestamp=timestamp, symbol=symbol)
                
                # === HEDGE ENGINE - Dealer Flow Analysis ===
                try:
                    hedge_snapshot = self.engines["hedge"].run(symbol, timestamp)
                    pipeline_result.hedge_snapshot = hedge_snapshot
                    analysis["hedge_snapshot"] = hedge_snapshot
                    
                    if hedge_snapshot:
                        # Dealer positioning from gamma sign
                        if hedge_snapshot.dealer_gamma_sign > 0.2:
                            analysis["dealer_positioning"] = "Dealers LONG gamma (will sell rallies, buy dips)"
                            analysis["agent_insights"].append(
                                f"🏦 DEALER FLOW: Market makers are long gamma ({hedge_snapshot.dealer_gamma_sign:.2f}). "
                                "They will hedge by selling into strength and buying weakness, creating mean reversion."
                            )
                        elif hedge_snapshot.dealer_gamma_sign < -0.2:
                            analysis["dealer_positioning"] = "Dealers SHORT gamma (will chase moves)"
                            analysis["agent_insights"].append(
                                f"🏦 DEALER FLOW: Market makers are short gamma ({hedge_snapshot.dealer_gamma_sign:.2f}). "
                                "They must hedge by chasing price moves, amplifying volatility."
                            )
                            analysis["risk_factors"].append("Short gamma environment - expect volatility")
                        else:
                            analysis["dealer_positioning"] = "Dealers neutral gamma"
                        
                        # Gamma pressure
                        if hedge_snapshot.gamma_pressure > 0:
                            pressure_desc = "high" if hedge_snapshot.gamma_pressure > 0.5 else "moderate"
                            analysis["gamma_pressure"] = f"{pressure_desc.title()} gamma pressure"
                            if hedge_snapshot.gamma_pressure > 0.5:
                                analysis["agent_insights"].append(
                                    f"⚡ GAMMA PRESSURE: High gamma concentration ({hedge_snapshot.gamma_pressure:.2f}) "
                                    "near current price. Expect price magnetism to key strikes."
                                )
                        
                        # Energy bias
                        if hedge_snapshot.energy_asymmetry > 0.3:
                            analysis["hedge_signal"] = "bullish"
                            analysis["energy_bias"] = "Upward energy building"
                            analysis["agent_insights"].append(
                                f"📈 ENERGY FLOW: Positive energy asymmetry ({hedge_snapshot.energy_asymmetry:.2f}). "
                                "Options flow suggests upward pressure building."
                            )
                        elif hedge_snapshot.energy_asymmetry < -0.3:
                            analysis["hedge_signal"] = "bearish"
                            analysis["energy_bias"] = "Downward energy building"
                            analysis["agent_insights"].append(
                                f"📉 ENERGY FLOW: Negative energy asymmetry ({hedge_snapshot.energy_asymmetry:.2f}). "
                                "Options flow suggests downward pressure building."
                            )
                        else:
                            analysis["hedge_signal"] = "neutral"
                            analysis["energy_bias"] = "Balanced energy"
                        
                        analysis["hedge_confidence"] = hedge_snapshot.confidence
                        analysis["regime"] = hedge_snapshot.regime
                        
                        # Regime insight
                        if hedge_snapshot.regime and hedge_snapshot.regime != "neutral":
                            analysis["agent_insights"].append(
                                f"🎯 REGIME: Market in '{hedge_snapshot.regime}' regime based on options structure."
                            )
                            
                except Exception as e:
                    logger.debug(f"HedgeEngine error for {symbol}: {e}")
                
                # === LIQUIDITY ENGINE - Market Quality Analysis ===
                try:
                    liquidity_snapshot = self.engines["liquidity"].run(symbol, timestamp)
                    pipeline_result.liquidity_snapshot = liquidity_snapshot
                    analysis["liquidity_snapshot"] = liquidity_snapshot
                    
                    if liquidity_snapshot:
                        # Liquidity grade
                        score = liquidity_snapshot.liquidity_score
                        if score >= 0.8:
                            analysis["liquidity_grade"] = "A+ (Excellent)"
                            analysis["agent_insights"].append(
                                f"💧 LIQUIDITY: Excellent ({score:.0%}). Easy entry/exit with minimal slippage."
                            )
                        elif score >= 0.6:
                            analysis["liquidity_grade"] = "B (Good)"
                            analysis["agent_insights"].append(
                                f"💧 LIQUIDITY: Good ({score:.0%}). Normal trading conditions."
                            )
                        elif score >= 0.4:
                            analysis["liquidity_grade"] = "C (Fair)"
                            analysis["agent_insights"].append(
                                f"💧 LIQUIDITY: Fair ({score:.0%}). Use limit orders."
                            )
                        else:
                            analysis["liquidity_grade"] = "D (Poor)"
                            analysis["risk_factors"].append("Poor liquidity - wide spreads")
                            analysis["agent_insights"].append(
                                f"⚠️ LIQUIDITY: Poor ({score:.0%}). Be careful with order sizes."
                            )
                        
                        # Spread quality
                        spread = liquidity_snapshot.bid_ask_spread
                        if spread < 0.05:
                            analysis["spread_quality"] = "Tight (< 0.05%)"
                        elif spread < 0.1:
                            analysis["spread_quality"] = "Normal (0.05-0.1%)"
                        elif spread < 0.5:
                            analysis["spread_quality"] = "Wide (0.1-0.5%)"
                        else:
                            analysis["spread_quality"] = "Very Wide (> 0.5%)"
                            analysis["risk_factors"].append("Wide bid-ask spread")
                        
                        # Market depth
                        if liquidity_snapshot.depth > 10000:
                            analysis["market_depth"] = "Deep order book"
                        elif liquidity_snapshot.depth > 1000:
                            analysis["market_depth"] = "Normal depth"
                        else:
                            analysis["market_depth"] = "Shallow order book"
                            
                        analysis["liquidity_signal"] = "neutral"
                        analysis["liquidity_confidence"] = score
                        
                except Exception as e:
                    logger.debug(f"LiquidityEngine error for {symbol}: {e}")
                
                # === SENTIMENT ENGINE ===
                try:
                    sentiment_snapshot = self.engines["sentiment"].run(symbol, timestamp)
                    pipeline_result.sentiment_snapshot = sentiment_snapshot
                    analysis["sentiment_snapshot"] = sentiment_snapshot
                    
                    if sentiment_snapshot:
                        sent_score = sentiment_snapshot.sentiment_score
                        if sent_score > 0.5:
                            analysis["sentiment_signal"] = "bullish"
                            analysis["agent_insights"].append(
                                f"📊 SENTIMENT: Strongly bullish ({sent_score:.2f}). Positive market sentiment."
                            )
                        elif sent_score > 0.2:
                            analysis["sentiment_signal"] = "bullish"
                            analysis["agent_insights"].append(
                                f"📊 SENTIMENT: Mildly bullish ({sent_score:.2f})."
                            )
                        elif sent_score < -0.5:
                            analysis["sentiment_signal"] = "bearish"
                            analysis["agent_insights"].append(
                                f"📊 SENTIMENT: Strongly bearish ({sent_score:.2f}). Negative market sentiment."
                            )
                        elif sent_score < -0.2:
                            analysis["sentiment_signal"] = "bearish"
                            analysis["agent_insights"].append(
                                f"📊 SENTIMENT: Mildly bearish ({sent_score:.2f})."
                            )
                        else:
                            analysis["sentiment_signal"] = "neutral"
                            
                        analysis["sentiment_confidence"] = sentiment_snapshot.confidence
                        
                except Exception as e:
                    logger.debug(f"SentimentEngine error for {symbol}: {e}")
                
                # Generate agent suggestions for composer
                if self.agents.get("hedge") and pipeline_result.hedge_snapshot:
                    hedge_suggestion = self.agents["hedge"].suggest(pipeline_result, timestamp)
                    if hedge_suggestion:
                        agent_suggestions.append(hedge_suggestion)
                
                if self.agents.get("liquidity") and pipeline_result.liquidity_snapshot:
                    liq_suggestion = self.agents["liquidity"].suggest(pipeline_result, timestamp)
                    if liq_suggestion:
                        agent_suggestions.append(liq_suggestion)
                
                if self.agents.get("sentiment") and pipeline_result.sentiment_snapshot:
                    sent_suggestion = self.agents["sentiment"].suggest(pipeline_result, timestamp)
                    if sent_suggestion:
                        agent_suggestions.append(sent_suggestion)
                
            except Exception as e:
                logger.error(f"Gnosis engine analysis error for {symbol}: {e}")
        
        # =====================================================
        # STEP 3: Technical Analysis Agent Insights
        # =====================================================
        if tech_signals:
            # Fallback signals if Gnosis didn't provide them
            if not analysis["hedge_signal"]:
                analysis["hedge_signal"] = tech_signals.trend_signal
                analysis["hedge_confidence"] = tech_signals.confidence if tech_signals.trend_signal != "neutral" else 0.3
            
            if not analysis["sentiment_signal"]:
                analysis["sentiment_signal"] = tech_signals.momentum_signal
                analysis["sentiment_confidence"] = tech_signals.confidence if tech_signals.momentum_signal != "neutral" else 0.3
            
            if not analysis["liquidity_signal"]:
                analysis["liquidity_signal"] = tech_signals.volume_signal
                analysis["liquidity_confidence"] = min(1.0, tech_signals.volume_ratio / 2) if tech_signals.volume_signal != "neutral" else 0.3
            
            # Technical insights
            tech_insight = "📈 TECHNICALS: "
            tech_parts = []
            
            if tech_signals.rsi_14:
                if tech_signals.rsi_14 < 30:
                    tech_parts.append(f"RSI oversold at {tech_signals.rsi_14:.0f}")
                elif tech_signals.rsi_14 > 70:
                    tech_parts.append(f"RSI overbought at {tech_signals.rsi_14:.0f}")
                else:
                    tech_parts.append(f"RSI neutral at {tech_signals.rsi_14:.0f}")
            
            if tech_signals.macd and tech_signals.macd_signal:
                if tech_signals.macd > tech_signals.macd_signal:
                    tech_parts.append("MACD bullish crossover")
                else:
                    tech_parts.append("MACD bearish crossover")
            
            if analysis["trend_status"]:
                tech_parts.append(analysis["trend_status"])
            
            if tech_parts:
                analysis["agent_insights"].append(tech_insight + ". ".join(tech_parts) + ".")
            
            # Support/Resistance insight
            if analysis["nearest_support"] or analysis["nearest_resistance"]:
                sr_insight = "🎯 KEY LEVELS: "
                sr_parts = []
                if analysis["nearest_support"] and tech_signals.current_price:
                    pct_to_support = ((tech_signals.current_price - analysis["nearest_support"]) / tech_signals.current_price) * 100
                    sr_parts.append(f"Support at ${analysis['nearest_support']:.2f} ({pct_to_support:.1f}% below)")
                if analysis["nearest_resistance"] and tech_signals.current_price:
                    pct_to_resistance = ((analysis["nearest_resistance"] - tech_signals.current_price) / tech_signals.current_price) * 100
                    sr_parts.append(f"Resistance at ${analysis['nearest_resistance']:.2f} ({pct_to_resistance:.1f}% above)")
                if sr_parts:
                    analysis["agent_insights"].append(sr_insight + ". ".join(sr_parts) + ".")
            
            # Volume insight
            if tech_signals.unusual_volume:
                vol_insight = f"🔊 VOLUME: Unusual activity ({tech_signals.volume_ratio:.1f}x average). "
                if tech_signals.day_change_pct and tech_signals.day_change_pct > 0:
                    vol_insight += "High volume on up day - bullish confirmation."
                    analysis["catalysts"].append("High volume buying")
                elif tech_signals.day_change_pct and tech_signals.day_change_pct < 0:
                    vol_insight += "High volume on down day - distribution."
                    analysis["catalysts"].append("High volume selling")
                analysis["agent_insights"].append(vol_insight)
            
            # Risk factors
            if tech_signals.rsi_14:
                if tech_signals.rsi_14 > 75:
                    analysis["risk_factors"].append("RSI extremely overbought")
                elif tech_signals.rsi_14 < 25:
                    analysis["risk_factors"].append("RSI extremely oversold")
            
            if tech_signals.day_change_pct and abs(tech_signals.day_change_pct) > 5:
                analysis["risk_factors"].append(f"High volatility today ({tech_signals.day_change_pct:+.1f}%)")
        
        # =====================================================
        # STEP 3.5: PENTA METHODOLOGY ANALYSIS (Smart Money)
        # =====================================================
        if self.penta_enabled and "penta" in self.agents:
            try:
                penta_analysis = self._get_penta_analysis(symbol, timestamp, tech_signals)
                analysis.update(penta_analysis)
                
                # Add PENTA insights if significant
                if penta_analysis.get("penta_confluence"):
                    analysis["agent_insights"].append(
                        f"🧠 SMART MONEY: {penta_analysis['penta_confluence']} confluence detected "
                        f"({penta_analysis.get('penta_confidence', 0):.0%})"
                    )
                
                if penta_analysis.get("active_inducement"):
                    analysis["agent_insights"].append(
                        f"⚡ INDUCEMENT: {penta_analysis['active_inducement']} detected - "
                        f"expect {penta_analysis.get('inducement_direction', 'reversal')}"
                    )
                    analysis["catalysts"].append(f"Smart money {penta_analysis['active_inducement'].lower()}")
                
                if penta_analysis.get("trend_from_bos"):
                    analysis["agent_insights"].append(
                        f"📊 BOS TREND: {penta_analysis['trend_from_bos']} (Break of Structure confirmed)"
                    )
                
                # Add liquidity pool levels as insights
                if penta_analysis.get("nearest_sell_pool"):
                    analysis["agent_insights"].append(
                        f"🎯 SELL-SIDE LIQUIDITY: ${penta_analysis['nearest_sell_pool']:.2f} "
                        f"({penta_analysis.get('pool_type', 'pool')})"
                    )
                    
            except Exception as e:
                logger.debug(f"PENTA analysis failed for {symbol}: {e}")
        
        # =====================================================
        # STEP 4: Calculate Final Direction
        # =====================================================
        bullish_weight = 0.0
        bearish_weight = 0.0
        neutral_weight = 0.0
        
        signals = [
            (analysis["hedge_signal"], analysis["hedge_confidence"], 0.4),
            (analysis["sentiment_signal"], analysis["sentiment_confidence"], 0.4),
            (analysis["liquidity_signal"], analysis["liquidity_confidence"], 0.2),
        ]
        
        for signal, conf, weight in signals:
            if signal == "bullish":
                bullish_weight += conf * weight
            elif signal == "bearish":
                bearish_weight += conf * weight
            else:
                neutral_weight += conf * weight
        
        total = bullish_weight + bearish_weight + neutral_weight
        if total > 0:
            bullish_pct = bullish_weight / total
            bearish_pct = bearish_weight / total
            
            if bullish_pct > 0.55:
                analysis["direction"] = SignalDirection.BUY
                analysis["confidence"] = bullish_pct
            elif bearish_pct > 0.55:
                analysis["direction"] = SignalDirection.SELL
                analysis["confidence"] = bearish_pct
            else:
                analysis["direction"] = SignalDirection.HOLD
                analysis["confidence"] = max(bullish_pct, bearish_pct, 1 - bullish_pct - bearish_pct)
        
        # =====================================================
        # STEP 5: Build Human-Readable Summary
        # =====================================================
        summary_parts = []
        
        # Direction summary
        if analysis["direction"] == SignalDirection.BUY:
            summary_parts.append(f"BULLISH ({analysis['confidence']:.0%} confidence)")
        elif analysis["direction"] == SignalDirection.SELL:
            summary_parts.append(f"BEARISH ({analysis['confidence']:.0%} confidence)")
        else:
            summary_parts.append(f"NEUTRAL ({analysis['confidence']:.0%} confidence)")
        
        # Key reasons
        if analysis["trend_status"]:
            summary_parts.append(analysis["trend_status"])
        if analysis["momentum_status"]:
            summary_parts.append(analysis["momentum_status"])
        if analysis["energy_bias"] and analysis["energy_bias"] != "Balanced energy":
            summary_parts.append(analysis["energy_bias"])
        
        analysis["reasoning"] = " | ".join(summary_parts) if summary_parts else "Insufficient data"
        
        return analysis
    
    def _get_penta_analysis(
        self, symbol: str, timestamp: datetime, tech_signals: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run PENTA methodology analysis using LiquidityConceptsEngine + 4 other methodologies.
        
        Returns:
            Dict with smart money analysis fields:
            - penta_confluence: Confluence level (PENTA/QUAD/TRIPLE/DOUBLE/SINGLE)
            - penta_confidence: Combined confidence from all methodologies
            - active_inducement: Detected inducement pattern
            - inducement_direction: Expected move direction from inducement
            - inducement_confidence: Confidence in inducement
            - strong_highs/lows: Strong swing levels (require BOS to break)
            - weak_highs/lows: Weak swing levels (targets for sweeps)
            - trend_from_bos: Trend direction from Break of Structure
            - market_structure: Smooth/Rough/Mixed structure
            - zone_reliability: How reliable current zone detection is
            - voids_above/below: Liquidity voids (gaps) in price
            - nearest_void_target: Nearest void for fill target
            - buy_pools/sell_pools: Major liquidity pool locations
            - nearest_buy_pool: Nearest buy-side liquidity
            - nearest_sell_pool: Nearest sell-side liquidity
            - pool_type: Type of nearest pool (CLUSTERED/MAJOR/MINOR)
        """
        result = {
            # Confluence
            "penta_confluence": None,
            "penta_confidence": 0.0,
            "penta_direction": None,
            # Inducement detection
            "active_inducement": None,
            "inducement_direction": None,
            "inducement_confidence": 0.0,
            # Strong/Weak swings
            "strong_highs": [],
            "strong_lows": [],
            "weak_highs": [],
            "weak_lows": [],
            "trend_from_bos": None,
            # Market structure
            "market_structure": None,
            "zone_reliability": None,
            # Liquidity voids
            "voids_above": [],
            "voids_below": [],
            "nearest_void_target": None,
            # Liquidity pools
            "buy_pools": [],
            "sell_pools": [],
            "nearest_buy_pool": None,
            "nearest_sell_pool": None,
            "pool_type": None,
        }
        
        if not self.penta_enabled or "liquidity_concepts" not in self.engines:
            return result
        
        try:
            # Get price data for Liquidity Concepts Engine
            current_price = tech_signals.current_price if tech_signals else self.get_current_price(symbol)
            if not current_price:
                return result
            
            # Prepare OHLCV data from technical analyzer if available
            bars = []
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="3mo")
                
                if not hist.empty:
                    from engines.liquidity.liquidity_concepts_engine import OHLCV
                    
                    for idx, row in hist.iterrows():
                        bars.append(OHLCV(
                            timestamp=idx.to_pydatetime(),
                            open=float(row['Open']),
                            high=float(row['High']),
                            low=float(row['Low']),
                            close=float(row['Close']),
                            volume=float(row['Volume']),
                        ))
            except Exception as e:
                logger.debug(f"Failed to get OHLCV data for PENTA: {e}")
                return result
            
            if len(bars) < 30:
                return result
            
            # Run Liquidity Concepts Engine analysis
            lc_engine = self.engines["liquidity_concepts"]
            lc_state = lc_engine.analyze(bars, current_price)
            
            if not lc_state:
                return result
            
            # Extract Liquidity Concepts results
            # Trend from BOS
            result["trend_from_bos"] = lc_state.trend.value if lc_state.trend else None
            
            # Market structure
            result["market_structure"] = (
                lc_state.structure_type.value if lc_state.structure_type else None
            )
            result["zone_reliability"] = lc_state.zone_reliability
            
            # Strong/Weak Swings
            if lc_state.strong_highs:
                result["strong_highs"] = [
                    {"price": s.price, "strength": s.strength.value}
                    for s in lc_state.strong_highs[:5]
                ]
            if lc_state.strong_lows:
                result["strong_lows"] = [
                    {"price": s.price, "strength": s.strength.value}
                    for s in lc_state.strong_lows[:5]
                ]
            if lc_state.weak_highs:
                result["weak_highs"] = [
                    {"price": s.price, "target": True}
                    for s in lc_state.weak_highs[:5]
                ]
            if lc_state.weak_lows:
                result["weak_lows"] = [
                    {"price": s.price, "target": True}
                    for s in lc_state.weak_lows[:5]
                ]
            
            # Liquidity Pools
            buy_pools = [p for p in lc_state.pools if p.side.value == "buy_side"]
            sell_pools = [p for p in lc_state.pools if p.side.value == "sell_side"]
            
            if buy_pools:
                result["buy_pools"] = [
                    {
                        "price": p.price,
                        "type": p.pool_type.value,
                        "strength": p.strength,
                    }
                    for p in buy_pools[:5]
                ]
                # Nearest buy pool (below current price)
                below_pools = [p for p in buy_pools if p.price < current_price]
                if below_pools:
                    nearest = max(below_pools, key=lambda x: x.price)
                    result["nearest_buy_pool"] = nearest.price
                    result["pool_type"] = nearest.pool_type.value
            
            if sell_pools:
                result["sell_pools"] = [
                    {
                        "price": p.price,
                        "type": p.pool_type.value,
                        "strength": p.strength,
                    }
                    for p in sell_pools[:5]
                ]
                # Nearest sell pool (above current price)
                above_pools = [p for p in sell_pools if p.price > current_price]
                if above_pools:
                    nearest = min(above_pools, key=lambda x: x.price)
                    result["nearest_sell_pool"] = nearest.price
                    result["pool_type"] = nearest.pool_type.value
            
            # Liquidity Voids
            voids_above = [v for v in lc_state.voids if v.low > current_price]
            voids_below = [v for v in lc_state.voids if v.high < current_price]
            
            if voids_above:
                result["voids_above"] = [
                    {"low": v.low, "high": v.high, "size": v.size}
                    for v in sorted(voids_above, key=lambda x: x.low)[:3]
                ]
                result["nearest_void_target"] = voids_above[0].low
            
            if voids_below:
                result["voids_below"] = [
                    {"low": v.low, "high": v.high, "size": v.size}
                    for v in sorted(voids_below, key=lambda x: x.high, reverse=True)[:3]
                ]
                if not result["nearest_void_target"]:
                    result["nearest_void_target"] = voids_below[0].high
            
            # Inducement detection
            if lc_state.inducements:
                latest_inducement = lc_state.inducements[0]
                result["active_inducement"] = latest_inducement.inducement_type.value
                result["inducement_direction"] = latest_inducement.expected_direction
                result["inducement_confidence"] = latest_inducement.confidence
            
            # Calculate PENTA confluence
            # Count methodology agreements
            agreeing_methods = 0
            direction_votes = {"bullish": 0, "bearish": 0}
            
            # Check each methodology
            # 1. Liquidity Concepts (from BOS/trend)
            if lc_state.trend:
                agreeing_methods += 1
                if lc_state.trend.value == "bullish":
                    direction_votes["bullish"] += 1
                elif lc_state.trend.value == "bearish":
                    direction_votes["bearish"] += 1
            
            # 2. Check inducement direction
            if lc_state.inducements:
                agreeing_methods += 1
                if latest_inducement.expected_direction in ["bullish", "long"]:
                    direction_votes["bullish"] += 1
                else:
                    direction_votes["bearish"] += 1
            
            # 3-5. Check other PENTA methodologies if available via agent
            if "penta" in self.agents:
                try:
                    # Use LiquidityAgentV5 for full PENTA confluence
                    pipeline_result = PipelineResult(timestamp=timestamp, symbol=symbol)
                    suggestion = self.agents["penta"].suggest(pipeline_result, timestamp)
                    
                    if suggestion and suggestion.confidence > 0.5:
                        # Add methodology agreements from agent
                        if suggestion.direction in [DirectionEnum.LONG, DirectionEnum.UP]:
                            direction_votes["bullish"] += 3  # Count as 3 more methodologies
                            agreeing_methods += 3
                        elif suggestion.direction in [DirectionEnum.SHORT, DirectionEnum.DOWN]:
                            direction_votes["bearish"] += 3
                            agreeing_methods += 3
                except Exception as e:
                    logger.debug(f"PENTA agent error: {e}")
            
            # Determine confluence level
            if agreeing_methods >= 5:
                result["penta_confluence"] = "PENTA"
            elif agreeing_methods >= 4:
                result["penta_confluence"] = "QUAD"
            elif agreeing_methods >= 3:
                result["penta_confluence"] = "TRIPLE"
            elif agreeing_methods >= 2:
                result["penta_confluence"] = "DOUBLE"
            elif agreeing_methods >= 1:
                result["penta_confluence"] = "SINGLE"
            
            # Calculate combined confidence
            total_votes = direction_votes["bullish"] + direction_votes["bearish"]
            if total_votes > 0:
                dominant = max(direction_votes.values())
                result["penta_confidence"] = dominant / 5  # Normalize to 5 methodologies
                
                if direction_votes["bullish"] > direction_votes["bearish"]:
                    result["penta_direction"] = "bullish"
                elif direction_votes["bearish"] > direction_votes["bullish"]:
                    result["penta_direction"] = "bearish"
                else:
                    result["penta_direction"] = "neutral"
            
        except Exception as e:
            logger.debug(f"PENTA analysis error for {symbol}: {e}")
        
        return result
    
    def generate_signal(self, symbol: str) -> AlphaSignal:
        """
        Generate a single trading signal for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            
        Returns:
            AlphaSignal with BUY/SELL/HOLD recommendation
        """
        timestamp = datetime.now(timezone.utc)
        
        # Run analysis
        analysis = self._analyze_symbol(symbol, timestamp)
        
        # Determine signal strength
        if analysis["confidence"] >= 0.8:
            strength = SignalStrength.STRONG
        elif analysis["confidence"] >= 0.65:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        # Get current price for targets
        current_price = self.get_current_price(symbol)
        
        # Calculate entry, stop, and target
        entry_price = current_price
        stop_loss = None
        take_profit = None
        
        if current_price:
            if analysis["direction"] == SignalDirection.BUY:
                stop_loss = current_price * (1 - self.config.stop_loss_pct)
                take_profit = current_price * (1 + self.config.take_profit_pct)
            elif analysis["direction"] == SignalDirection.SELL:
                # For short/exit signals
                stop_loss = current_price * (1 + self.config.stop_loss_pct)
                take_profit = current_price * (1 - self.config.take_profit_pct)
        
        # Calculate holding period based on confidence
        if strength == SignalStrength.STRONG:
            holding_days = min(7, self.config.max_holding_days)
        elif strength == SignalStrength.MODERATE:
            holding_days = min(5, self.config.max_holding_days)
        else:
            holding_days = min(3, self.config.max_holding_days)
        
        # Signal validity
        valid_until = timestamp + timedelta(hours=24)  # Signal valid for 24 hours
        
        # PDT tracking
        day_trades_remaining = self.pdt_tracker.day_trades_remaining
        is_day_trade_candidate = (
            strength == SignalStrength.STRONG and 
            day_trades_remaining > 0 and
            holding_days <= 1
        )
        
        # Check for risk factors
        risk_factors = analysis.get("risk_factors", [])
        if analysis["confidence"] < 0.6:
            risk_factors.append("Low confidence")
        if not current_price:
            risk_factors.append("Price data unavailable")
        
        return AlphaSignal(
            symbol=symbol,
            direction=analysis["direction"],
            confidence=analysis["confidence"],
            strength=strength,
            timestamp=timestamp,
            valid_until=valid_until,
            holding_period_days=holding_days,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            # Support/Resistance from Liquidity Pools
            support_levels=analysis.get("support_levels", []),
            resistance_levels=analysis.get("resistance_levels", []),
            nearest_support=analysis.get("nearest_support"),
            nearest_resistance=analysis.get("nearest_resistance"),
            # Detailed liquidity pool data
            put_walls=analysis.get("put_walls", []),  # PUT OI concentrations
            call_walls=analysis.get("call_walls", []),  # CALL OI concentrations
            max_pain=analysis.get("max_pain"),  # Price magnet at expiration
            gamma_flip=analysis.get("gamma_flip"),  # Where dealer hedging flips
            # Engine signals
            hedge_signal=analysis.get("hedge_signal"),
            hedge_confidence=analysis.get("hedge_confidence", 0.0),
            liquidity_signal=analysis.get("liquidity_signal"),
            liquidity_confidence=analysis.get("liquidity_confidence", 0.0),
            sentiment_signal=analysis.get("sentiment_signal"),
            sentiment_confidence=analysis.get("sentiment_confidence", 0.0),
            # Detailed analysis
            dealer_positioning=analysis.get("dealer_positioning"),
            gamma_pressure=analysis.get("gamma_pressure"),
            energy_bias=analysis.get("energy_bias"),
            regime=analysis.get("regime"),
            liquidity_grade=analysis.get("liquidity_grade"),
            spread_quality=analysis.get("spread_quality"),
            market_depth=analysis.get("market_depth"),
            trend_status=analysis.get("trend_status"),
            momentum_status=analysis.get("momentum_status"),
            price_vs_sma20=analysis.get("price_vs_sma20"),
            price_vs_sma50=analysis.get("price_vs_sma50"),
            price_vs_sma200=analysis.get("price_vs_sma200"),
            bollinger_position=analysis.get("bollinger_position"),
            # Reasoning
            reasoning=analysis.get("reasoning", ""),
            agent_insights=analysis.get("agent_insights", []),
            risk_factors=risk_factors,
            catalysts=analysis.get("catalysts", []),
            # PDT
            is_day_trade_candidate=is_day_trade_candidate,
            day_trades_remaining=day_trades_remaining,
            # Volume
            unusual_volume=analysis.get("unusual_volume", False),
            volume_ratio=analysis.get("volume_ratio", 1.0),
            volume_description=analysis.get("volume_description", "Normal"),
            # PENTA Methodology
            penta_enabled=self.penta_enabled,
            penta_confluence=analysis.get("penta_confluence"),
            penta_confidence=analysis.get("penta_confidence", 0.0),
            # Liquidity Concepts
            liquidity_pools_above=len(analysis.get("sell_pools", [])),
            liquidity_pools_below=len(analysis.get("buy_pools", [])),
            nearest_buy_pool=analysis.get("nearest_buy_pool"),
            nearest_sell_pool=analysis.get("nearest_sell_pool"),
            pool_type=analysis.get("pool_type"),
            # Inducement
            active_inducement=analysis.get("active_inducement"),
            inducement_direction=analysis.get("inducement_direction"),
            inducement_confidence=analysis.get("inducement_confidence", 0.0),
            # Strong/Weak Swings
            strong_highs=len(analysis.get("strong_highs", [])),
            strong_lows=len(analysis.get("strong_lows", [])),
            weak_highs=len(analysis.get("weak_highs", [])),
            weak_lows=len(analysis.get("weak_lows", [])),
            trend_from_bos=analysis.get("trend_from_bos"),
            # Market Structure
            market_structure=analysis.get("market_structure"),
            zone_reliability=analysis.get("zone_reliability"),
            # Liquidity Voids
            voids_above=len(analysis.get("voids_above", [])),
            voids_below=len(analysis.get("voids_below", [])),
            nearest_void_target=analysis.get("nearest_void_target"),
        )
    
    def scan_universe(self, min_confidence: Optional[float] = None) -> List[AlphaSignal]:
        """
        Scan all symbols in universe and generate signals.
        
        Args:
            min_confidence: Minimum confidence threshold (default: config value)
            
        Returns:
            List of signals sorted by confidence (highest first)
        """
        min_conf = min_confidence or self.config.min_confidence
        signals = []
        
        logger.info(f"Scanning {len(self.config.universe)} symbols...")
        
        for symbol in self.config.universe:
            try:
                signal = self.generate_signal(symbol)
                
                # Filter by confidence
                if signal.confidence >= min_conf:
                    signals.append(signal)
                    logger.info(
                        f"{signal.symbol}: {signal.direction.value} "
                        f"({signal.confidence:.0%} confidence)"
                    )
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        # Sort by confidence (highest first)
        signals.sort(key=lambda s: s.confidence, reverse=True)
        
        logger.info(f"Found {len(signals)} actionable signals")
        
        return signals
    
    def get_top_signals(
        self,
        n: int = 5,
        direction: Optional[SignalDirection] = None,
    ) -> List[AlphaSignal]:
        """
        Get top N signals, optionally filtered by direction.
        
        Args:
            n: Number of signals to return
            direction: Filter by BUY, SELL, or None for all
            
        Returns:
            Top N signals by confidence
        """
        all_signals = self.scan_universe()
        
        if direction:
            all_signals = [s for s in all_signals if s.direction == direction]
        
        return all_signals[:n]
    
    def save_signals(self, signals: List[AlphaSignal], filename: Optional[str] = None) -> Path:
        """
        Save signals to JSON file.
        
        Args:
            signals: List of signals to save
            filename: Optional filename (default: signals_YYYY-MM-DD.json)
            
        Returns:
            Path to saved file
        """
        output_dir = self.config.ensure_output_dir()
        
        if not filename:
            filename = f"signals_{date.today().isoformat()}.json"
        
        filepath = output_dir / filename
        
        data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.to_dict(),
            "pdt_status": self.pdt_tracker.get_status(),
            "signals": [s.to_dict() for s in signals],
        }
        
        filepath.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved {len(signals)} signals to {filepath}")
        
        return filepath
    
    def print_signals(self, signals: List[AlphaSignal]) -> None:
        """Print signals in Robinhood-friendly format."""
        print("\n" + "="*50)
        print("  GNOSIS ALPHA - Trading Signals")
        print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"  PDT Status: {self.pdt_tracker.day_trades_remaining}/3 day trades remaining")
        print("="*50 + "\n")
        
        buy_signals = [s for s in signals if s.direction == SignalDirection.BUY]
        sell_signals = [s for s in signals if s.direction == SignalDirection.SELL]
        
        if buy_signals:
            print("🟢 BUY SIGNALS:")
            print("-"*30)
            for signal in buy_signals:
                print(signal.to_robinhood_format())
                print()
        
        if sell_signals:
            print("🔴 SELL SIGNALS:")
            print("-"*30)
            for signal in sell_signals:
                print(signal.to_robinhood_format())
                print()
        
        if not buy_signals and not sell_signals:
            print("⚪ No actionable signals at this time.")
            print("   All positions: HOLD")
        
        print("="*50)
