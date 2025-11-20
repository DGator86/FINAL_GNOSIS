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
    confidence: float = 0.5


class LiquiditySnapshot(BaseModel):
    """Snapshot from the Liquidity Engine."""
    
    timestamp: datetime
    symbol: str
    liquidity_score: float = 0.5
    bid_ask_spread: float = 0.0
    volume: float = 0.0
    depth: float = 0.0
    impact_cost: float = 0.0


class SentimentSnapshot(BaseModel):
    """Snapshot from the Sentiment Engine."""
    
    timestamp: datetime
    symbol: str
    sentiment_score: float = 0.0
    news_sentiment: float = 0.0
    flow_sentiment: float = 0.0
    technical_sentiment: float = 0.0
    confidence: float = 0.5


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
