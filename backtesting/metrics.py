"""Utility metrics for backtesting routines."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def _to_numpy(values: Iterable[float]) -> np.ndarray:
    """Convert iterable to numpy array of floats."""
    return np.asarray(list(values), dtype=float)


def compute_sharpe_ratio(returns: Iterable[float], risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio for a series of returns."""
    arr = _to_numpy(returns)
    if arr.size == 0 or np.isclose(arr.std(), 0):
        return 0.0
    excess = arr.mean() - risk_free_rate / 252
    return float(excess / arr.std() * np.sqrt(252))


def max_drawdown(returns: Iterable[float]) -> float:
    """Calculate maximum drawdown from a returns or equity curve series."""
    arr = _to_numpy(returns)
    if arr.size == 0:
        return 0.0
    equity_curve = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return float(abs(drawdown.min())) if drawdown.size else 0.0


def compute_directional_accuracy(signals: Iterable[float], returns: Iterable[float]) -> float:
    """Compute directional accuracy given signals and subsequent returns."""
    sig = _to_numpy(signals)
    ret = _to_numpy(returns)
    if sig.size == 0 or ret.size == 0:
        return 0.0
    n = min(sig.size, ret.size)
    sig = sig[:n]
    ret = ret[:n]
    correct = np.sign(sig) == np.sign(ret)
    return float(correct.mean()) if correct.size else 0.0
