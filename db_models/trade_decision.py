"""SQLAlchemy model for trade_decisions table.

This model captures the full decision context for every trade:
- Universe and filter state
- Engine snapshots (Dealer Hedge, Liquidity, Sentiment)
- Agent votes and logic
- Composer decision
- Portfolio context
- Execution outcome
"""

import uuid

from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    DateTime,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from db import Base


class TradeDecision(Base):
    """
    ORM model mapping the trade_decisions table.

    This table is the cornerstone of the GNOSIS ML pipeline.
    Every trade decision is logged with full context, enabling:
    - Performance attribution
    - Regime diagnostics
    - Simulation vs live sanity checks
    - ML training dataset generation
    """

    __tablename__ = "trade_decisions"

    # ========== Identity ==========
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    # ========== Multi-Tenancy ==========
    user_id = Column(
        String,
        nullable=False,
        index=True,
        default="system"  # Backward compatibility for existing system trades
    )

    # ========== Meta ==========
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    mode = Column(
        String,
        nullable=False,
        index=True,
    )  # 'live' | 'paper' | 'backtest'
    symbol = Column(
        String,
        nullable=False,
        index=True,
    )
    direction = Column(
        String,
        nullable=False,
    )  # 'long' | 'short' | 'neutral'
    structure = Column(
        String,
        nullable=False,
    )  # 'long_call', 'call_spread', 'iron_condor', etc.
    config_version = Column(
        String,
        nullable=False,
    )

    # ========== Universe / Filter State ==========
    universe_eligible = Column(
        Boolean,
        nullable=False,
    )
    universe_reasons = Column(
        ARRAY(Text),
        nullable=False,
    )
    price = Column(
        Numeric(18, 8),
        nullable=False,
    )
    adv = Column(
        Numeric(18, 2),
        nullable=False,
    )  # Average Daily Volume
    iv_rank = Column(
        Numeric(6, 3),
        nullable=False,
    )  # Implied Volatility Rank
    realized_vol_30d = Column(
        Numeric(8, 4),
        nullable=False,
    )
    options_liq_score = Column(
        Numeric(8, 4),
        nullable=False,
    )

    # ========== Engine Snapshots ==========
    # These are JSONB to allow flexible evolution of features
    dealer_features = Column(
        JSONB,
        nullable=False,
    )  # GEX, vanna, charm, gamma pivot, etc.
    liquidity_features = Column(
        JSONB,
        nullable=False,
    )  # Liquidity zones, dark pool, HVN/LVN, etc.
    sentiment_features = Column(
        JSONB,
        nullable=False,
    )  # Wyckoff, micro/macro regime, sentiment scores

    # ========== Agent Logic ==========
    hedge_agent_vote = Column(
        JSONB,
        nullable=False,
    )  # bias, direction_bias, confidence
    liquidity_agent_vote = Column(
        JSONB,
        nullable=False,
    )  # zone, confidence
    sentiment_agent_vote = Column(
        JSONB,
        nullable=False,
    )  # risk_posture, trend_alignment, confidence
    composer_decision = Column(
        JSONB,
        nullable=False,
    )  # final_direction, structure, sizing, invalidation, reason_codes

    # ========== Portfolio Context ==========
    portfolio_context = Column(
        JSONB,
        nullable=False,
    )  # exposure_before/after, risk_per_trade, max_dd_limit, etc.

    # ========== Execution Outcome ==========
    # May be NULL at decision time, filled later for live/paper
    order_id = Column(
        String,
        nullable=True,
    )
    entry_price = Column(
        Numeric(18, 8),
        nullable=True,
    )
    target_price = Column(
        Numeric(18, 8),
        nullable=True,
    )
    stop_price = Column(
        Numeric(18, 8),
        nullable=True,
    )
    slippage_bps = Column(
        Numeric(10, 4),
        nullable=True,
    )
    status = Column(
        String,
        nullable=True,
        index=True,
    )  # 'filled', 'partial', 'rejected', etc.

    # ========== Audit Timestamps ==========
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __repr__(self) -> str:
        return (
            f"<TradeDecision("
            f"id={self.id}, "
            f"user_id={self.user_id}, "
            f"symbol={self.symbol}, "
            f"direction={self.direction}, "
            f"mode={self.mode}, "
            f"timestamp={self.timestamp}"
            f")>"
        )
