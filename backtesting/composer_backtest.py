"""Simplified Composer backtesting utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from backtesting.metrics import compute_directional_accuracy, compute_sharpe_ratio, max_drawdown


@dataclass
class BacktestConfig:
    """Configuration for running a composer backtest."""

    symbols: List[str]
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float = 100_000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    seed: Optional[int] = 42
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Result container for composer backtests."""

    total_return: float
    sharpe: float
    max_drawdown: float
    directional_accuracy: float
    equity_curve: List[float]
    returns: List[float]


_PERIODS_PER_YEAR = {
    "1D": 252,
    "1H": 24 * 252,
    "1h": 24 * 252,
    "15m": 24 * 4 * 252,
    "15M": 24 * 4 * 252,
}


def _infer_periods(timeframe: str) -> int:
    """Infer approximate periods per year based on timeframe string."""
    return _PERIODS_PER_YEAR.get(timeframe, 252)


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _generate_returns(num_periods: int, seed: Optional[int]) -> np.SimpleArray:
    """Generate synthetic returns for the backtest."""
    rng = np.random.default_rng(seed)
    return np.SimpleArray([rng.normal(loc=0.0005, scale=0.01) for _ in range(num_periods)])


def _generate_signals(returns: Iterable[float]) -> np.SimpleArray:
    """Generate basic trading signals from returns."""
    arr = np.asarray(list(returns), dtype=float)
    signals = np.sign(arr)
    noise = np.random.default_rng().choice([-1.0, 0.0, 1.0], size=len(arr), p=[0.05, 0.1, 0.85])
    return np.sign(signals + noise)


def run_composer_backtest(config: BacktestConfig) -> BacktestResult:
    """Run a simple synthetic backtest based on configuration."""

    start = _parse_date(config.start_date)
    end = _parse_date(config.end_date)
    periods = max((end - start).days, 1)
    freq = _infer_periods(config.timeframe)
    num_periods = min(int(periods * freq / 252), freq * 5)
    num_periods = max(num_periods, 10)

    returns = _generate_returns(num_periods, config.seed)
    signals = _generate_signals(returns)

    if len(returns) > 1:
        positions = np.sign(signals)
        transaction_costs = np.abs(np.diff(positions, prepend=positions[0])) * (
            config.transaction_cost + config.slippage
        )
        net_returns = returns - transaction_costs
    else:
        net_returns = returns

    equity_curve = list(np.cumprod([1 + r for r in net_returns]))
    total_return = float(equity_curve[-1] - 1) if equity_curve else 0.0
    sharpe = compute_sharpe_ratio(net_returns)
    mdd = max_drawdown(net_returns)
    directional_accuracy = compute_directional_accuracy(signals, net_returns)

    return BacktestResult(
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=mdd,
        directional_accuracy=directional_accuracy,
        equity_curve=equity_curve,
        returns=list(net_returns),
    )


__all__ = ["BacktestConfig", "BacktestResult", "run_composer_backtest"]
