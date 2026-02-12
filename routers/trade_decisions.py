"""FastAPI router for trade decision endpoints.

This router exposes the trade decision tracking API:
- POST /trades/decisions - Log a trade decision from GNOSIS pipeline
- GET /trades/decisions/{trade_id} - Retrieve a single trade decision
- GET /trades/decisions - List trade decisions with filters
- PATCH /trades/decisions/{trade_id}/execution - Update execution details
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from crud.trade_decision import (
    create_trade_decision,
    get_trade_decision_by_id,
    list_trade_decisions,
    update_trade_execution,
)
from db import get_db
from schemas.trade_decision import (
    TradeDecisionCreate,
    TradeDecisionRead,
    TradeDecisionUpdateExecution,
)

router = APIRouter(prefix="/trades", tags=["trades"])


@router.post(
    "/decisions",
    response_model=TradeDecisionRead,
    status_code=status.HTTP_201_CREATED,
    summary="Log a trade decision",
    description="""
    GNOSIS pipeline calls this endpoint when it decides to take a trade.

    This endpoint persists the complete decision context:
    - Universe filter state
    - Engine snapshots (Dealer Hedge, Liquidity, Sentiment)
    - Agent votes (Hedge, Liquidity, Sentiment)
    - Composer decision
    - Portfolio context
    - Optional initial execution info

    Returns the created trade decision with auto-generated ID.
    """,
)
def log_trade_decision(
    payload: TradeDecisionCreate,
    db: Session = Depends(get_db),
) -> TradeDecisionRead:
    """Log a new trade decision from GNOSIS pipeline."""
    obj = create_trade_decision(db, payload)
    return obj


@router.get(
    "/decisions/{trade_id}",
    response_model=TradeDecisionRead,
    summary="Get a single trade decision",
    description="Retrieve a trade decision by its unique ID.",
)
def get_trade_decision(
    trade_id: UUID,
    db: Session = Depends(get_db),
) -> TradeDecisionRead:
    """Retrieve a single trade decision by ID."""
    obj = get_trade_decision_by_id(db, trade_id)
    if obj is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trade decision {trade_id} not found",
        )
    return obj


@router.get(
    "/decisions",
    response_model=List[TradeDecisionRead],
    summary="List trade decisions",
    description="""
    List recent trade decisions with optional filters.

    This endpoint supports:
    - Filtering by symbol, mode, direction, status
    - Pagination via limit and offset
    - Ordering by timestamp descending (most recent first)

    Use this for:
    - Analytics dashboards
    - Performance analysis
    - Building ML datasets
    """,
)
def get_trade_decisions(
    symbol: str | None = Query(
        default=None,
        description="Filter by symbol (e.g., 'SPY')"
    ),
    mode: str | None = Query(
        default=None,
        description="Filter by mode: 'live', 'paper', or 'backtest'"
    ),
    direction: str | None = Query(
        default=None,
        description="Filter by direction: 'long', 'short', or 'neutral'"
    ),
    status: str | None = Query(
        default=None,
        description="Filter by execution status"
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of results (1-1000)"
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of results to skip (for pagination)"
    ),
    db: Session = Depends(get_db),
) -> List[TradeDecisionRead]:
    """List trade decisions with optional filters."""
    objs = list_trade_decisions(
        db,
        symbol=symbol,
        mode=mode,
        direction=direction,
        status=status,
        limit=limit,
        offset=offset,
    )
    return objs


@router.patch(
    "/decisions/{trade_id}/execution",
    response_model=TradeDecisionRead,
    summary="Update execution details",
    description="""
    Update execution-specific fields after broker responses or trade close.

    This endpoint is called:
    - After order is submitted (to set order_id)
    - After order is filled (to set entry_price, slippage)
    - After targets/stops are set
    - After trade is closed (to set final status)

    Only fields provided in the payload will be updated.
    """,
)
def patch_trade_execution(
    trade_id: UUID,
    payload: TradeDecisionUpdateExecution,
    db: Session = Depends(get_db),
) -> TradeDecisionRead:
    """Update execution details for a trade decision."""
    obj = update_trade_execution(db, trade_id, payload)
    if obj is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trade decision {trade_id} not found",
        )
    return obj
