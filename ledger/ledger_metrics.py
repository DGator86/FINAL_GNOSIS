"""Performance metrics for ledger entries."""

from __future__ import annotations

import json
from typing import Dict

import pandas as pd
from loguru import logger

try:  # pragma: no cover - optional dependency
    import pyfolio
except ImportError:  # pragma: no cover
    pyfolio = None


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute Sharpe/Calmar etc. from ledger dataframe."""

    if df.empty:
        return {}

    returns = []
    for _, row in df.iterrows():
        try:
            payload = json.loads(row.get("payload") or "{}")
            pnl = payload.get("consensus", {}).get("pnl", 0.0)
        except json.JSONDecodeError:
            pnl = 0.0
        returns.append(pnl)

    series = pd.Series(returns)
    stats: Dict[str, float] = {}
    if pyfolio:
        perf = pyfolio.timeseries.perf_stats(series)
        stats["sharpe"] = float(perf.get("Sharpe ratio", perf.get("sharpe", 0.0)))
        stats["calmar"] = float(perf.get("Calmar ratio", 0.0))
    else:
        stats["sharpe"] = float(series.mean() / (series.std() + 1e-8))
    stats["mean_return"] = float(series.mean())
    stats["max_drawdown"] = float(series.min())
    return stats


# Test: with mock df ensure compute_metrics returns sharpe key
