"""Backtesting utilities for Gnosis."""
from backtesting.composer_backtest import (
    BacktestConfig,
    BacktestResult,
    run_composer_backtest,
)

# ML-enabled backtest engine imports (lazy to avoid import errors)
try:
    from backtesting.ml_backtest_engine import (
        MLBacktestConfig,
        MLBacktestEngine,
        MLBacktestResults,
        run_ml_backtest,
        print_results_summary,
    )
    __all__ = [
        "BacktestConfig",
        "BacktestResult",
        "run_composer_backtest",
        "MLBacktestConfig",
        "MLBacktestEngine",
        "MLBacktestResults",
        "run_ml_backtest",
        "print_results_summary",
    ]
except ImportError:
    # ML dependencies not installed
    __all__ = ["BacktestConfig", "BacktestResult", "run_composer_backtest"]
