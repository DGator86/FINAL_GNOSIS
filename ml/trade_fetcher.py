"""Trade decision fetcher for ML dataset construction.

This module provides functions to query trade_decisions
from the database with filters optimized for ML training.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import select

from db_models.trade_decision import TradeDecision


def fetch_trade_decisions_for_ml(
    db: Session,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    mode: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[TradeDecision]:
    """
    Fetch trade decisions for ML dataset construction.

    This is optimized for ML use cases:
    - Time range filtering for train/test splits
    - Mode filtering (backtest vs live vs paper)
    - Symbol filtering for per-symbol models
    - Row limits for sampling

    Args:
        db: Database session
        start_time: Filter for trades after this time (inclusive)
        end_time: Filter for trades before this time (inclusive)
        mode: Filter by mode ('live', 'paper', 'backtest')
        symbol: Filter by symbol
        limit: Maximum number of rows to return

    Returns:
        List of TradeDecision ORM objects, ordered by timestamp ascending
    """
    stmt = select(TradeDecision)

    if start_time:
        stmt = stmt.where(TradeDecision.timestamp >= start_time)
    if end_time:
        stmt = stmt.where(TradeDecision.timestamp <= end_time)
    if mode:
        stmt = stmt.where(TradeDecision.mode == mode)
    if symbol:
        stmt = stmt.where(TradeDecision.symbol == symbol.upper())

    stmt = stmt.order_by(TradeDecision.timestamp.asc())

    if limit:
        stmt = stmt.limit(limit)

    return list(db.scalars(stmt))
