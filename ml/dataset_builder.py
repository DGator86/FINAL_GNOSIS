"""ML dataset builder for GNOSIS trade decisions.

This module ties together fetching, labeling, and feature extraction
to produce ML-ready training examples and DataFrames.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from db_models.trade_decision import TradeDecision
from ml.feature_extractor import extract_features_from_trade
from ml.labeling import PriceSeriesProvider, compute_trade_labels
from ml.schemas import TradeMLExample
from ml.trade_fetcher import fetch_trade_decisions_for_ml


def build_ml_examples_from_trades(
    db: Session,
    get_price_series: PriceSeriesProvider,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    mode: Optional[str] = None,
    symbol: Optional[str] = None,
    horizon: timedelta = timedelta(days=5),
    limit: Optional[int] = None,
) -> List[TradeMLExample]:
    """
    High-level pipeline: TradeDecision rows -> TradeMLExample objects.

    This is the core function that plugs GNOSIS into your ML layer.

    It:
    1. Fetches trade decisions from DB
    2. For each trade:
       - Computes labels from price evolution
       - Flattens engine/agent features
       - Builds a TradeMLExample
    3. Returns list of ML examples

    Args:
        db: Database session
        get_price_series: Function that returns price series for (symbol, start, end)
        start_time: Filter trades after this time
        end_time: Filter trades before this time
        mode: Filter by mode ('live', 'paper', 'backtest')
        symbol: Filter by symbol
        horizon: Time window for computing horizon_return
        limit: Maximum number of trades to fetch

    Returns:
        List of TradeMLExample objects ready for ML training
    """
    # Fetch trade decisions
    trades: List[TradeDecision] = fetch_trade_decisions_for_ml(
        db=db,
        start_time=start_time,
        end_time=end_time,
        mode=mode,
        symbol=symbol,
        limit=limit,
    )

    examples: List[TradeMLExample] = []

    for trade in trades:
        # Compute labels
        labels = compute_trade_labels(trade, get_price_series, horizon=horizon)

        # Extract and flatten features
        features = extract_features_from_trade(trade)

        # Build ML example
        example = TradeMLExample(
            trade_id=str(trade.id),
            timestamp=trade.timestamp,
            mode=trade.mode,
            symbol=trade.symbol,
            direction=trade.direction,
            structure=trade.structure,
            config_version=trade.config_version,
            price=float(trade.price),
            adv=float(trade.adv),
            iv_rank=float(trade.iv_rank),
            realized_vol_30d=float(trade.realized_vol_30d),
            options_liq_score=float(trade.options_liq_score),
            features=features,
            **labels,
        )
        examples.append(example)

    return examples


def ml_examples_to_dataframe(
    examples: List[TradeMLExample],
) -> pd.DataFrame:
    """
    Convert TradeMLExample objects into a flat pandas DataFrame.

    Features dict is exploded into columns; labels and meta stay as columns.

    Args:
        examples: List of TradeMLExample objects

    Returns:
        Pandas DataFrame with:
        - Meta columns (trade_id, timestamp, symbol, etc.)
        - Universe metric columns (price, adv, iv_rank, etc.)
        - Flattened feature columns (dealer.*, liq.*, sentiment.*, agent_*, etc.)
        - Label columns (realized_return, r_multiple, hit_target, etc.)
    """
    rows: List[Dict[str, Any]] = []

    for ex in examples:
        # Base fields
        base = {
            "trade_id": ex.trade_id,
            "timestamp": ex.timestamp,
            "mode": ex.mode,
            "symbol": ex.symbol,
            "direction": ex.direction,
            "structure": ex.structure,
            "config_version": ex.config_version,
            "price": ex.price,
            "adv": ex.adv,
            "iv_rank": ex.iv_rank,
            "realized_vol_30d": ex.realized_vol_30d,
            "options_liq_score": ex.options_liq_score,
            "realized_return": ex.realized_return,
            "r_multiple": ex.r_multiple,
            "max_drawdown_pct": ex.max_drawdown_pct,
            "hit_target": ex.hit_target,
            "stopped_out": ex.stopped_out,
            "horizon_return": ex.horizon_return,
        }

        # Merge in flattened features
        base.update(ex.features)
        rows.append(base)

    df = pd.DataFrame(rows)
    return df
