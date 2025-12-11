"""CRUD operations for trade_decisions table.

These functions handle all database interactions for trade decisions:
- Creating new trade decisions
- Retrieving trade decisions by ID or filters
- Updating execution details
"""

from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from db_models.trade_decision import TradeDecision
from schemas.trade_decision import (
    TradeDecisionCreate,
    TradeDecisionUpdateExecution,
)


def create_trade_decision(
    db: Session,
    payload: TradeDecisionCreate,
) -> TradeDecision:
    """
    Insert a new trade decision row into the database.

    This is called right before sending an order to the broker,
    capturing the full GNOSIS state at decision time.

    Args:
        db: Database session
        payload: Trade decision creation payload from GNOSIS pipeline

    Returns:
        Created TradeDecision ORM object with auto-generated ID
    """
    obj = TradeDecision(
        timestamp=payload.timestamp,
        mode=payload.mode,
        symbol=payload.symbol,
        direction=payload.direction,
        structure=payload.structure,
        config_version=payload.config_version,
        universe_eligible=payload.universe_eligible,
        universe_reasons=payload.universe_reasons,
        price=payload.price,
        adv=payload.adv,
        iv_rank=payload.iv_rank,
        realized_vol_30d=payload.realized_vol_30d,
        options_liq_score=payload.options_liq_score,
        dealer_features=payload.dealer_features,
        liquidity_features=payload.liquidity_features,
        sentiment_features=payload.sentiment_features,
        hedge_agent_vote=payload.hedge_agent_vote,
        liquidity_agent_vote=payload.liquidity_agent_vote,
        sentiment_agent_vote=payload.sentiment_agent_vote,
        composer_decision=payload.composer_decision,
        portfolio_context=payload.portfolio_context,
        order_id=payload.order_id,
        entry_price=payload.entry_price,
        target_price=payload.target_price,
        stop_price=payload.stop_price,
        slippage_bps=payload.slippage_bps,
        status=payload.status,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def get_trade_decision_by_id(
    db: Session,
    trade_id: UUID,
) -> Optional[TradeDecision]:
    """
    Fetch a single trade decision by ID.

    Args:
        db: Database session
        trade_id: UUID of the trade decision

    Returns:
        TradeDecision ORM object or None if not found
    """
    stmt = select(TradeDecision).where(TradeDecision.id == trade_id)
    return db.scalar(stmt)


def list_trade_decisions(
    db: Session,
    symbol: Optional[str] = None,
    mode: Optional[str] = None,
    direction: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[TradeDecision]:
    """
    Retrieve recent trade decisions with optional filters.

    This is the primary query interface for:
    - Analytics dashboards
    - Performance analysis
    - ML dataset construction

    Args:
        db: Database session
        symbol: Filter by symbol (optional)
        mode: Filter by mode ('live', 'paper', 'backtest') (optional)
        direction: Filter by direction ('long', 'short', 'neutral') (optional)
        status: Filter by execution status (optional)
        limit: Maximum number of results (default 100, max 1000)
        offset: Number of results to skip (for pagination)

    Returns:
        List of TradeDecision ORM objects, ordered by timestamp descending
    """
    stmt = select(TradeDecision).order_by(desc(TradeDecision.timestamp))

    if symbol:
        stmt = stmt.where(TradeDecision.symbol == symbol.upper())
    if mode:
        stmt = stmt.where(TradeDecision.mode == mode)
    if direction:
        stmt = stmt.where(TradeDecision.direction == direction)
    if status:
        stmt = stmt.where(TradeDecision.status == status)

    stmt = stmt.limit(min(limit, 1000)).offset(offset)
    return list(db.scalars(stmt))


def update_trade_execution(
    db: Session,
    trade_id: UUID,
    update: TradeDecisionUpdateExecution,
) -> Optional[TradeDecision]:
    """
    Patch execution-specific fields for an existing trade decision.

    This is called after:
    - Broker responds with order_id
    - Order is filled (entry_price, slippage)
    - Trade is closed (status)

    Args:
        db: Database session
        trade_id: UUID of the trade decision to update
        update: Execution update payload

    Returns:
        Updated TradeDecision ORM object or None if not found
    """
    obj = get_trade_decision_by_id(db, trade_id)
    if obj is None:
        return None

    # Only update fields that are explicitly set
    for field, value in update.model_dump(exclude_unset=True).items():
        setattr(obj, field, value)

    db.commit()
    db.refresh(obj)
    return obj
