"""Core Pydantic schemas for Super Gnosis / DHPE v3."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EngineSnapshot(BaseModel):
    """Base snapshot from an engine."""

    timestamp: datetime
    symbol: str
    data: Dict[str, Any] = Field(default_factory=dict)


class HedgeSnapshot(BaseModel):
    """Snapshot from the Hedge Engine."""

    timestamp: datetime
    symbol: str
    elasticity: float = 0.0
    movement_energy: float = 0.0
    energy_asymmetry: float = 0.0
    pressure_up: float = 0.0
    pressure_down: float = 0.0
    pressure_net: float = 0.0
    gamma_pressure: float = 0.0
    vanna_pressure: float = 0.0
    charm_pressure: float = 0.0
    dealer_gamma_sign: float = 0.0
    regime: str = "neutral"
    regime_features: Dict[str, float] = Field(default_factory=dict)
    regime_probabilities: Dict[str, float] = Field(default_factory=dict)
    jump_intensity: float = 0.0
    liquidity_friction: float = 0.0
    adaptive_weights: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.5
    directional_elasticity: Dict[str, float] = Field(default_factory=dict)


class LiquiditySnapshot(BaseModel):
    """Snapshot from the Liquidity Engine."""

    timestamp: datetime
    symbol: str
    liquidity_score: float = 0.5
    bid_ask_spread: float = 0.0
    volume: float = 0.0
    depth: float = 0.0
    impact_cost: float = 0.0
    # NEW: optional v2+ liquidity metrics
    impact_lambda: Optional[float] = None
    friction: Optional[float] = None
    forecast_depth: List[float] = Field(default_factory=list)
    percentile_score: Optional[float] = None
    liquidity_friction: Optional[float] = None


class SentimentSnapshot(BaseModel):
    """Snapshot from the Sentiment Engine."""

    timestamp: datetime
    symbol: str
    sentiment_score: float = 0.0
    news_sentiment: float = 0.0
    flow_sentiment: float = 0.0
    technical_sentiment: float = 0.0
    confidence: float = 0.5
    intensity: Optional[float] = None
    relevance: Optional[float] = None
    mtf_score: Optional[float] = None


class ElasticitySnapshot(BaseModel):
    """Snapshot from the Elasticity Engine."""

    timestamp: datetime
    symbol: str
    volatility: float = 0.0
    volatility_regime: str = "moderate"
    trend_strength: float = 0.0


class ForecastSnapshot(BaseModel):
    """Time-series forecast payload (Kats-inspired)."""

    model: str = "unavailable"
    horizon: int = 0
    forecast: List[float] = Field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LSTMLookaheadSnapshot(BaseModel):
    """LSTM Lookahead prediction snapshot with multi-horizon forecasts."""

    timestamp: datetime
    symbol: str
    horizons: List[int] = Field(default_factory=list)  # [1, 5, 15, 60] minutes
    predictions: Dict[int, float] = Field(default_factory=dict)  # horizon -> predicted return %
    uncertainties: Dict[int, float] = Field(default_factory=dict)  # horizon -> uncertainty
    direction: str = "neutral"  # up, down, neutral
    direction_probs: Dict[str, float] = Field(default_factory=dict)  # {up, down, neutral} -> probability
    confidence: float = 0.0
    model_version: str = "lstm_lookahead_v1"
    # NEW: optional forward-looking returns curve to preserve backward compatibility
    forecast_returns: Optional[List[float]] = None


class RegimeSimilaritySnapshot(BaseModel):
    """Nearest-neighbor regime retrieval diagnostics."""

    similarity_score: float = 0.0
    neighbors: List[Dict[str, Any]] = Field(default_factory=list)
    feature_vector: List[float] = Field(default_factory=list)


class AnomalySnapshot(BaseModel):
    """Anomaly detector output based on isolation forests."""

    score: float = 0.0
    flagged: bool = False
    feature_vector: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PolicyRecommendation(BaseModel):
    """Curriculum-style policy suggestion for execution/hedging."""

    action: str = "hold"
    risk_multiplier: float = 1.0
    rationale: str = ""
    curriculum_stage: str = "warmup"


class MLEnhancementSnapshot(BaseModel):
    """Aggregate ML augmentations across forecasting, similarity, and RL."""

    timestamp: datetime
    symbol: str
    forecast: Optional[ForecastSnapshot] = None
    regime_similarity: Optional[RegimeSimilaritySnapshot] = None
    anomaly: Optional[AnomalySnapshot] = None
    policy_recommendation: Optional[PolicyRecommendation] = None


class DirectionEnum(str, Enum):
    """Trade direction enum."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class AgentSignal(BaseModel):
    """
    Structured trading signal produced by an agent.
    
    Used by composer for consensus building and by ML forecasting agents.
    Compatible with both Pydantic validation and dataclass-style usage.
    """

    timestamp: datetime
    symbol: str
    signal: str = "neutral"  # "bullish", "bearish", "neutral"
    confidence: float = 0.5
    reasoning: str = ""
    # Optional fields for extended use cases
    agent_id: Optional[str] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Composer-specific fields
    strength: float = 0.0
    consensus_score: float = 0.0


class AgentSuggestion(BaseModel):
    """Suggestion from a primary agent."""

    agent_name: str
    timestamp: datetime
    symbol: str
    direction: DirectionEnum = DirectionEnum.NEUTRAL
    confidence: float = 0.5
    reasoning: str = ""
    target_allocation: float = 0.0


class StrategyType(str, Enum):
    """Strategy type enum."""

    DIRECTIONAL = "directional"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    OPTIONS_SPREAD = "options_spread"
    HEDGED = "hedged"


class OptionsLeg(BaseModel):
    """Single leg of an options strategy."""

    symbol: str
    ratio: int
    side: str  # "buy" or "sell"
    type: str  # "call" or "put"
    strike: float
    expiration: str  # YYYY-MM-DD
    action: str  # "buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"


class OptionsOrderRequest(BaseModel):
    """Request for a multi-leg options order."""

    symbol: str  # Underlying symbol
    strategy_name: str
    legs: List[OptionsLeg]
    max_loss: float
    max_profit: float
    bpr: float  # Buying Power Reduction
    rationale: str
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.now)


class TradeIdea(BaseModel):
    """Trade idea from the trade agent."""

    timestamp: datetime
    symbol: str
    strategy_type: StrategyType
    direction: DirectionEnum
    confidence: float
    size: float = 0.0
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    options_request: Optional[OptionsOrderRequest] = None


class WatchlistEntry(BaseModel):
    """Adaptive watchlist ranking details for a symbol."""

    symbol: str
    score: float
    timestamp: datetime
    passes_filters: bool = True
    reasons: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)


class OrderStatus(str, Enum):
    """Order status enum."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderResult(BaseModel):
    """Result of an order execution."""

    timestamp: datetime
    symbol: str
    status: OrderStatus
    order_id: Optional[str] = None
    filled_qty: float = 0.0
    filled_price: Optional[float] = None
    message: str = ""


class PositionState(BaseModel):
    """Snapshot of a single open position used for tracking."""

    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: str


class TrackingSnapshot(BaseModel):
    """Aggregate tracking data for the current pipeline iteration."""

    timestamp: datetime
    positions: List[PositionState] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class AdaptationUpdate(BaseModel):
    """Record of adaptive parameter changes applied after feedback."""

    timestamp: datetime
    changes: Dict[str, float] = Field(default_factory=dict)
    rationale: str = ""


class LedgerEntry(BaseModel):
    """Entry in the ledger store."""

    timestamp: datetime
    symbol: str
    event_type: str
    data: Dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """Complete result from a pipeline run."""

    timestamp: datetime
    symbol: str
    hedge_snapshot: Optional[HedgeSnapshot] = None
    liquidity_snapshot: Optional[LiquiditySnapshot] = None
    sentiment_snapshot: Optional[SentimentSnapshot] = None
    elasticity_snapshot: Optional[ElasticitySnapshot] = None
    suggestions: List[AgentSuggestion] = Field(default_factory=list)
    trade_ideas: List[TradeIdea] = Field(default_factory=list)
    order_results: List[OrderResult] = Field(default_factory=list)
    consensus: Optional[Dict[str, Any]] = None
    watchlist_entry: Optional[WatchlistEntry] = None
    watchlist_snapshot: List[WatchlistEntry] = Field(default_factory=list)
    tracking_snapshot: Optional[TrackingSnapshot] = None
    adaptation_update: Optional[AdaptationUpdate] = None
    ml_snapshot: Optional[MLEnhancementSnapshot] = None


class OptionsPosition(BaseModel):
    """Options position tracking."""

    symbol: str  # Option symbol
    underlying: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: str
    expiration: datetime
    strike: float
    option_type: str  # "call" or "put"
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None


class PortfolioGreeks(BaseModel):
    """Aggregate portfolio Greeks."""

    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    timestamp: datetime


class Position(BaseModel):
    """Position tracking for UnifiedTradingBot."""

    symbol: str
    side: str  # "long" or "short"
    size: float  # Position size in $ (or BPR for options)
    entry_price: float
    entry_time: datetime
    quantity: float = 0.0

    # Risk Management
    highest_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    trailing_stop_active: bool = False

    # Metadata
    asset_class: str = "equity"  # "equity", "option", "option_strategy"
    option_symbol: Optional[str] = None

    def update_highest_price(self, current_price: float) -> None:
        """Update highest price seen for trailing stop."""
        if self.highest_price is None:
            self.highest_price = current_price
        else:
            if self.side == "long":
                self.highest_price = max(self.highest_price, current_price)
            else:
                self.highest_price = min(self.highest_price, current_price)
