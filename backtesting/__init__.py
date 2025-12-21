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
    
    # ML Hyperparameter backtest integration
    from backtesting.ml_hyperparameter_backtest import (
        MLBacktestMode,
        MLBacktestHyperparameters,
        MLBacktestPeriod,
        MLHyperparameterBacktestConfig,
        MLHyperparameterBacktestResults,
        MLHyperparameterBacktester,
        create_ml_hyperparameter_backtester,
        run_ml_parameter_backtest,
    )
    
    __all__ = [
        # Composer backtest
        "BacktestConfig",
        "BacktestResult",
        "run_composer_backtest",
        # ML backtest engine
        "MLBacktestConfig",
        "MLBacktestEngine",
        "MLBacktestResults",
        "run_ml_backtest",
        "print_results_summary",
        # ML Hyperparameter backtest
        "MLBacktestMode",
        "MLBacktestHyperparameters",
        "MLBacktestPeriod",
        "MLHyperparameterBacktestConfig",
        "MLHyperparameterBacktestResults",
        "MLHyperparameterBacktester",
        "create_ml_hyperparameter_backtester",
        "run_ml_parameter_backtest",
    ]
except ImportError:
    # ML dependencies not installed
    __all__ = ["BacktestConfig", "BacktestResult", "run_composer_backtest"]
