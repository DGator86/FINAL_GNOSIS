"""FastAPI router for ML trade dataset endpoints.

This router exposes the ML dataset generation API:
- GET /ml/trades/dataset - Build and return ML-ready trade examples
- GET /ml/trades/dataset/export - Export dataset to Parquet

Version: 2.0.0 - Full price provider integration
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import tempfile
import os

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from db import get_db
from ml.dataset_builder import build_ml_examples_from_trades
from ml.schemas import TradeMLExample

# Import price provider
try:
    from utils.price_provider import get_price_series_for_ml, get_price_provider
    PRICE_PROVIDER_AVAILABLE = True
except ImportError:
    PRICE_PROVIDER_AVAILABLE = False

router = APIRouter(prefix="/ml/trades", tags=["ml-trades"])


def get_price_series_adapter(
    symbol: str,
    start: datetime,
    end: datetime
) -> List[float]:
    """
    Price series provider adapter.

    Uses the unified price provider with automatic fallback:
    1. Alpaca Historical API (if credentials available)
    2. yfinance (free fallback)
    
    Args:
        symbol: Stock symbol
        start: Start time
        end: End time

    Returns:
        List of close prices for the period
    """
    if not PRICE_PROVIDER_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Price provider module not available. Install required dependencies."
        )
    
    provider = get_price_provider()
    
    if not provider.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No price data source available. "
                "Configure Alpaca API credentials or install yfinance."
            )
        )
    
    prices = get_price_series_for_ml(symbol, start, end)
    
    if not prices:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No price data found for {symbol} from {start} to {end}"
        )
    
    return prices


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
    Export ML trade dataset to Parquet file for download.

    This endpoint:
    1. Builds ML examples (same as /dataset)
    2. Converts to pandas DataFrame
    3. Saves to Parquet file
    4. Returns file for download

    Use this for:
    - Production ML training pipelines
    - Archiving datasets
    - Offline analysis
    """,
)
def export_ml_trade_dataset(
    start_time: Optional[datetime] = Query(
        default=None,
        description="Filter trades after this time"
    ),
    end_time: Optional[datetime] = Query(
        default=None,
        description="Filter trades before this time"
    ),
    mode: Optional[str] = Query(
        default=None,
        description="Filter by mode: 'live', 'paper', or 'backtest'"
    ),
    symbol: Optional[str] = Query(
        default=None,
        description="Filter by symbol"
    ),
    horizon_days: int = Query(
        default=5,
        ge=1,
        le=60,
        description="Horizon for computing labels (1-60 days)"
    ),
    limit: Optional[int] = Query(
        default=10000,
        ge=1,
        le=100_000,
        description="Maximum number of trades to export"
    ),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Export ML dataset to Parquet file."""
    try:
        import pandas as pd
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="pandas is required for Parquet export"
        )
    
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
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(e)
        )
    
    if not examples:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No trade examples found matching criteria"
        )
    
    # Convert to DataFrame
    records = []
    for ex in examples:
        record = {
            "trade_id": ex.trade_id,
            "symbol": ex.symbol,
            "timestamp": ex.timestamp,
            "direction": ex.direction,
            "entry_price": ex.entry_price,
        }
        # Flatten features
        for k, v in ex.features.items():
            record[f"feature_{k}"] = v
        # Add labels
        for k, v in ex.labels.items():
            record[f"label_{k}"] = v
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Create temp file
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ml_trades_dataset_{timestamp_str}.parquet"
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    
    df.to_parquet(temp_path, index=False, engine="pyarrow")
    
    return FileResponse(
        path=temp_path,
        filename=filename,
        media_type="application/octet-stream",
    )


@router.get(
    "/providers/status",
    summary="Check price provider status",
    description="Check which price data providers are available.",
)
def get_price_provider_status():
    """Get status of price data providers."""
    if not PRICE_PROVIDER_AVAILABLE:
        return {
            "available": False,
            "providers": [],
            "message": "Price provider module not installed",
        }
    
    provider = get_price_provider()
    
    return {
        "available": provider.is_available(),
        "providers": [
            {
                "name": "alpaca",
                "available": hasattr(provider, 'provider') and 
                           hasattr(provider.provider, 'providers') and
                           any(p.__class__.__name__ == 'AlpacaPriceProvider' 
                               for p in getattr(provider.provider, 'providers', [])),
            },
            {
                "name": "yfinance", 
                "available": hasattr(provider, 'provider') and
                           hasattr(provider.provider, 'providers') and
                           any(p.__class__.__name__ == 'YFinancePriceProvider'
                               for p in getattr(provider.provider, 'providers', [])),
            },
        ],
        "message": "Price provider is operational" if provider.is_available() else "No providers available",
    }
