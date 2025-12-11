"""FastAPI router for ML trade dataset endpoints.

This router exposes the ML dataset generation API:
- GET /ml/trades/dataset - Build and return ML-ready trade examples
"""

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session

from db import get_db
from ml.schemas import TradeMLExample
from ml.dataset_builder import build_ml_examples_from_trades


router = APIRouter(prefix="/ml/trades", tags=["ml-trades"])


def get_price_series_adapter(
    symbol: str,
    start: datetime,
    end: datetime
) -> List[float]:
    """
    Price series provider adapter.

    IMPORTANT: This is a stub implementation.
    Replace with real implementation that queries your price store.

    Options:
    - Query bars_1m/bars_5m/bars_1d tables
    - Call Alpaca historical API
    - Query your feature store
    - Use yfinance for backtesting

    Args:
        symbol: Stock symbol
        start: Start time
        end: End time

    Returns:
        List of prices (open, high, low, close, or just close prices)
    """
    # TODO: Wire this to your historical price store
    # For now, raise NotImplementedError with helpful message
    raise NotImplementedError(
        "Price series provider not implemented. "
        "Wire get_price_series_adapter to your historical price store:\n"
        "  - Query bars_1m/bars_5m/bars_1d tables\n"
        "  - Call Alpaca historical API\n"
        "  - Query your feature store\n"
        "  - Use yfinance for backtesting\n"
        f"Requested: {symbol} from {start} to {end}"
    )


@router.get(
    "/dataset",
    response_model=List[TradeMLExample],
    summary="Build ML trade dataset",
    description="""
    Build and return ML-ready trade examples directly from the API.

    This endpoint:
    1. Fetches trade decisions from DB with optional filters
    2. For each trade:
       - Computes labels from price evolution
       - Flattens engine/agent features
    3. Returns list of ML examples

    Use this for:
    - Small/interactive dataset pulls
    - Dashboard/analytics integration
    - Quick experimentation

    For production ML training:
    - Consider writing to Parquet/S3 instead
    - Use background jobs for large datasets
    - Implement pagination for >1000 examples

    IMPORTANT: The price series provider is currently a stub.
    Wire it to your historical price store before using this endpoint.
    """,
)
def get_ml_trade_dataset(
    start_time: Optional[datetime] = Query(
        default=None,
        description="Filter trades after this time (for train/test splits)"
    ),
    end_time: Optional[datetime] = Query(
        default=None,
        description="Filter trades before this time (for train/test splits)"
    ),
    mode: Optional[str] = Query(
        default=None,
        description="Filter by mode: 'live', 'paper', or 'backtest'"
    ),
    symbol: Optional[str] = Query(
        default=None,
        description="Filter by symbol (for per-symbol models)"
    ),
    horizon_days: int = Query(
        default=5,
        ge=1,
        le=60,
        description="Horizon for computing labels (1-60 days)"
    ),
    limit: Optional[int] = Query(
        default=5000,
        ge=1,
        le=50_000,
        description="Maximum number of trades to fetch (1-50000)"
    ),
    db: Session = Depends(get_db),
) -> List[TradeMLExample]:
    """Build and return ML-ready trade examples."""
    horizon = timedelta(days=horizon_days)

    try:
        examples = build_ml_examples_from_trades(
            db=db,
            get_price_series=get_price_series_adapter,
            start_time=start_time,
            end_time=end_time,
            mode=mode,
            symbol=symbol,
            horizon=horizon,
            limit=limit,
        )
    except NotImplementedError as e:
        # Re-raise with HTTP 501 Not Implemented
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(e)
        )

    return examples


@router.get(
    "/dataset/export",
    summary="Export ML dataset to Parquet",
    description="""
    Export ML trade dataset to Parquet file.

    This endpoint:
    1. Builds ML examples (same as /dataset)
    2. Converts to pandas DataFrame
    3. Saves to Parquet file
    4. Returns file path or download link

    Use this for:
    - Production ML training pipelines
    - Archiving datasets
    - Offline analysis

    TODO: Implement this endpoint for production use.
    """,
)
def export_ml_trade_dataset():
    """Export ML dataset to Parquet file."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "Parquet export not implemented yet. "
            "To implement:\n"
            "  1. Call build_ml_examples_from_trades()\n"
            "  2. Call ml_examples_to_dataframe()\n"
            "  3. Call df.to_parquet(path)\n"
            "  4. Return file path or serve file download"
        )
    )
