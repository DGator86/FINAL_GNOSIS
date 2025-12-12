"""Label computation for trade decisions.

This module computes outcome labels (PnL, R-multiple, hit_target, etc.)
from trade decisions by analyzing price evolution after the trade.
"""

from datetime import datetime, timedelta
from typing import Callable, Dict, List

from db_models.trade_decision import TradeDecision

# Type alias for price series provider
# Signature: (symbol, start_time, end_time) -> List[float]
PriceSeriesProvider = Callable[[str, datetime, datetime], List[float]]


def compute_trade_labels(
    trade: TradeDecision,
    get_price_series: PriceSeriesProvider,
    horizon: timedelta = timedelta(days=5),
) -> Dict[str, float]:
    """
    Compute labels for a single trade decision.

    This function:
    - Uses trade.price (or entry_price) as reference
    - Gets price path from timestamp to timestamp + horizon
    - Treats direction (long/short) to compute directional returns
    - Approximates risk as |entry - stop| if available
    - Computes hit_target and stopped_out flags

    Args:
        trade: TradeDecision ORM object
        get_price_series: Function that returns price series for a symbol/time range
        horizon: Time window for computing horizon_return

    Returns:
        Dictionary with label fields:
        - realized_return: Final return at horizon
        - r_multiple: Return / risk
        - max_drawdown_pct: Worst intratrade drawdown
        - hit_target: 1 if target hit, 0 otherwise
        - stopped_out: 1 if stop hit, 0 otherwise
        - horizon_return: Return at horizon endpoint
    """
    start = trade.timestamp
    end = trade.timestamp + horizon

    # Get price series from provider
    prices = get_price_series(trade.symbol, start, end)

    # Handle missing data
    if not prices or len(prices) < 2:
        # No data available, return zeros/NaNs
        # Caller can filter these out later
        return {
            "realized_return": 0.0,
            "r_multiple": 0.0,
            "max_drawdown_pct": 0.0,
            "hit_target": 0,
            "stopped_out": 0,
            "horizon_return": 0.0,
        }

    # Use entry_price if available, otherwise use reference price
    entry_price = float(trade.entry_price or trade.price)
    stop_price = float(trade.stop_price or entry_price * 0.9)
    target_price = float(trade.target_price or entry_price * 1.1)

    # Determine direction sign for return calculation
    direction_sign = (
        1.0 if trade.direction == "long"
        else -1.0 if trade.direction == "short"
        else 0.0
    )

    # Compute path-based metrics
    max_price = max(prices)
    min_price = min(prices)
    final_price = prices[-1]

    def dir_return(px: float) -> float:
        """Compute directional return from entry to price."""
        if direction_sign == 0.0:
            return 0.0
        return direction_sign * (px - entry_price) / entry_price

    horizon_return = dir_return(final_price)
    dir_return(max_price if direction_sign > 0 else min_price)
    worst_return = dir_return(min_price if direction_sign > 0 else max_price)

    # Compute R-multiple
    # Risk = distance from entry to stop in percentage terms
    risk = abs(entry_price - stop_price) / entry_price if stop_price != entry_price else 1e-6
    realized_return = horizon_return
    r_multiple = realized_return / risk

    # Check if target or stop was hit during the path
    hit_target = 0
    stopped_out = 0
    target_return = (target_price - entry_price) * direction_sign / entry_price
    stop_return = (stop_price - entry_price) * direction_sign / entry_price

    for px in prices:
        r = dir_return(px)
        if r >= target_return:
            hit_target = 1
            break
        if r <= stop_return:
            stopped_out = 1
            break

    max_drawdown_pct = worst_return

    return {
        "realized_return": float(realized_return),
        "r_multiple": float(r_multiple),
        "max_drawdown_pct": float(max_drawdown_pct),
        "hit_target": int(hit_target),
        "stopped_out": int(stopped_out),
        "horizon_return": float(horizon_return),
    }
