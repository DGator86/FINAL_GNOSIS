"""Pydantic schemas for trade decision tracking.

These schemas define the API contract for:
- Creating trade decisions (from GNOSIS pipeline)
- Updating execution details (from broker responses)
- Reading trade decisions (for analytics/ML)
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field

Mode = Literal["live", "paper", "backtest"]
Direction = Literal["long", "short", "neutral"]


class TradeDecisionCreate(BaseModel):
    """
    Payload from GNOSIS pipeline when it decides to take a trade.

    This captures the complete state of the world at decision time:
    - Universe filter results
    - Engine snapshots (dealer, liquidity, sentiment)
    - Agent votes
    - Composer decision
    - Portfolio context

    Execution fields are optional and can remain null initially.
    """

    # Meta
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    mode: Mode
    symbol: str
    direction: Direction
    structure: str
    config_version: str

    # Universe / Filters
    universe_eligible: bool
    universe_reasons: List[str]
    price: float
    adv: float
    iv_rank: float
    realized_vol_30d: float
    options_liq_score: float

    # Engine snapshots
    dealer_features: Dict[str, Any] = Field(
        description="Dealer Hedge Pressure Field Engine snapshot: GEX, vanna, charm, gamma pivot, etc."
    )
    liquidity_features: Dict[str, Any] = Field(
        description="Liquidity Engine snapshot: liquidity zones, dark pool, HVN/LVN, VWAP, etc."
    )
    sentiment_features: Dict[str, Any] = Field(
        description="Sentiment & Regime Engine snapshot: Wyckoff, micro/macro regime, sentiment scores"
    )

    # Agent logic
    hedge_agent_vote: Dict[str, Any] = Field(
        description="Hedge Agent vote: bias, direction_bias, confidence"
    )
    liquidity_agent_vote: Dict[str, Any] = Field(
        description="Liquidity Agent vote: zone, confidence"
    )
    sentiment_agent_vote: Dict[str, Any] = Field(
        description="Sentiment Agent vote: risk_posture, trend_alignment, confidence"
    )
    composer_decision: Dict[str, Any] = Field(
        description="Composer Agent decision: final_direction, structure, sizing, invalidation, reason_codes"
    )

    # Portfolio context
    portfolio_context: Dict[str, Any] = Field(
        description="Portfolio state: exposure_before/after, risk_per_trade, max_dd_limit, etc."
    )

    # Optional initial execution values
    order_id: Optional[str] = None
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    slippage_bps: Optional[float] = None
    status: Optional[str] = None


class TradeDecisionUpdateExecution(BaseModel):
    """
    Patch execution details after broker response or at exit.

    This is called after:
    - Order is submitted (order_id)
    - Order is filled (entry_price, slippage)
    - Targets/stops are set
    - Trade is closed (status)
    """

    order_id: Optional[str] = None
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    slippage_bps: Optional[float] = None
    status: Optional[str] = None


class TradeDecisionRead(BaseModel):
    """
    Full trade decision record returned from the API.

    This includes all fields from create + execution updates + audit timestamps.
    """

    # Identity
    id: UUID

    # Meta
    timestamp: datetime
    mode: Mode
    symbol: str
    direction: Direction
    structure: str
    config_version: str

    # Universe / Filters
    universe_eligible: bool
    universe_reasons: List[str]
    price: float
    adv: float
    iv_rank: float
    realized_vol_30d: float
    options_liq_score: float

    # Engine snapshots
    dealer_features: Dict[str, Any]
    liquidity_features: Dict[str, Any]
    sentiment_features: Dict[str, Any]

    # Agent logic
    hedge_agent_vote: Dict[str, Any]
    liquidity_agent_vote: Dict[str, Any]
    sentiment_agent_vote: Dict[str, Any]
    composer_decision: Dict[str, Any]

    # Portfolio context
    portfolio_context: Dict[str, Any]

    # Execution outcome
    order_id: Optional[str]
    entry_price: Optional[float]
    target_price: Optional[float]
    stop_price: Optional[float]
    slippage_bps: Optional[float]
    status: Optional[str]

    # Audit
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # SQLAlchemy -> Pydantic compatibility
