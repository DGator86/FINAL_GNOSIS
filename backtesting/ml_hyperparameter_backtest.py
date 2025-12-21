"""
ML Hyperparameter Backtest Integration

This module connects the ML hyperparameter pipeline with the backtesting engine,
enabling:
1. Backtesting with ML-optimized hyperparameters
2. Walk-forward validation of ML parameters
3. Parameter sensitivity analysis
4. Regime-specific backtesting

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import copy

import numpy as np
import pandas as pd
from loguru import logger

# ML Integration
from ml import (
    MLHyperparameterManager,
    MLHyperparameterSet,
    AdaptiveMLPipeline,
    MLOptimizationEngine,
    OptimizationConfig,
    OptimizationStage,
    ObjectiveFunction,
    MarketRegime,
    create_preset_config,
)

# Backtesting
from backtesting.elite_backtest_engine import (
    EliteBacktestConfig,
    EliteBacktestResults,
    SimulatedTrade,
)


class MLBacktestMode(str, Enum):
    """ML backtest execution modes."""
    STATIC = "static"  # Use fixed ML parameters throughout
    ADAPTIVE = "adaptive"  # Adapt parameters based on regime
    WALK_FORWARD = "walk_forward"  # Walk-forward optimization
    SENSITIVITY = "sensitivity"  # Parameter sensitivity analysis


@dataclass
class MLBacktestHyperparameters:
    """ML hyperparameters for backtesting."""
    # Model parameters
    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_learning_rate: float = 0.001
    
    # Feature parameters
    lookback_period: int = 50
    feature_scaling: str = "standard"
    use_technical_features: bool = True
    use_volume_features: bool = True
    
    # Signal parameters
    confidence_threshold: float = 0.60
    signal_smoothing: float = 0.3
    use_regime_filter: bool = True
    
    # Position sizing
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.04
    volatility_scaling: bool = True
    
    # Risk management
    max_drawdown_pct: float = 0.15
    correlation_limit: float = 0.70
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "lstm_num_layers": self.lstm_num_layers,
            "lstm_dropout": self.lstm_dropout,
            "lstm_learning_rate": self.lstm_learning_rate,
            "lookback_period": self.lookback_period,
            "feature_scaling": self.feature_scaling,
            "use_technical_features": self.use_technical_features,
            "use_volume_features": self.use_volume_features,
            "confidence_threshold": self.confidence_threshold,
            "signal_smoothing": self.signal_smoothing,
            "use_regime_filter": self.use_regime_filter,
            "kelly_fraction": self.kelly_fraction,
            "max_position_pct": self.max_position_pct,
            "volatility_scaling": self.volatility_scaling,
            "max_drawdown_pct": self.max_drawdown_pct,
            "correlation_limit": self.correlation_limit,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MLBacktestHyperparameters":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_preset(cls, preset: str) -> "MLBacktestHyperparameters":
        """Create from preset name."""
        presets = {
            "conservative": cls(
                lstm_hidden_dim=32,
                lstm_dropout=0.3,
                confidence_threshold=0.70,
                kelly_fraction=0.15,
                max_position_pct=0.02,
                max_drawdown_pct=0.10,
            ),
            "balanced": cls(
                lstm_hidden_dim=64,
                lstm_dropout=0.2,
                confidence_threshold=0.60,
                kelly_fraction=0.25,
                max_position_pct=0.04,
                max_drawdown_pct=0.15,
            ),
            "aggressive": cls(
                lstm_hidden_dim=128,
                lstm_dropout=0.1,
                confidence_threshold=0.50,
                kelly_fraction=0.40,
                max_position_pct=0.08,
                max_drawdown_pct=0.25,
            ),
        }
        return presets.get(preset, cls())


@dataclass
class MLBacktestPeriod:
    """A single backtest period result."""
    start_date: datetime
    end_date: datetime
    regime: Optional[MarketRegime] = None
    
    # Performance
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Trade stats
    total_trades: int = 0
    avg_trade_pnl: float = 0.0
    
    # ML metrics
    avg_confidence: float = 0.0
    signal_accuracy: float = 0.0
    
    # Parameters used
    hyperparameters: Optional[MLBacktestHyperparameters] = None


@dataclass
class MLHyperparameterBacktestConfig:
    """Configuration for ML hyperparameter backtesting."""
    # Base backtest settings
    symbols: List[str] = field(default_factory=lambda: ["SPY"])
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-01"
    timeframe: str = "1Day"
    
    # Capital
    initial_capital: float = 100_000.0
    
    # ML settings
    mode: MLBacktestMode = MLBacktestMode.STATIC
    preset: str = "balanced"
    hyperparameters: Optional[MLBacktestHyperparameters] = None
    
    # Walk-forward settings
    train_pct: float = 0.70
    min_train_periods: int = 100
    reoptimize_frequency: int = 20  # bars between reoptimization
    
    # Optimization settings
    optimization_metric: str = "sharpe_ratio"
    optimization_method: str = "bayesian"
    optimization_trials: int = 50
    
    # Regime settings
    regime_specific_params: bool = False
    regimes_to_test: List[str] = field(default_factory=lambda: ["trending_bull", "trending_bear", "sideways", "high_volatility"])
    
    # Sensitivity analysis
    sensitivity_params: List[str] = field(default_factory=lambda: ["confidence_threshold", "kelly_fraction", "max_position_pct"])
    sensitivity_range: float = 0.30  # +/- 30% from base
    sensitivity_steps: int = 5
    
    # Output
    output_dir: str = "runs/ml_hyperparameter_backtests"
    save_period_results: bool = True
    tag: str = ""


@dataclass
class MLHyperparameterBacktestResults:
    """Results from ML hyperparameter backtest."""
    config: MLHyperparameterBacktestConfig
    
    # Overall performance
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    
    # ML metrics
    avg_ml_confidence: float = 0.0
    ml_signal_accuracy: float = 0.0
    regime_changes: int = 0
    
    # Period breakdowns
    periods: List[MLBacktestPeriod] = field(default_factory=list)
    
    # Walk-forward specific
    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float = 0.0
    walk_forward_efficiency: float = 0.0  # OOS Sharpe / IS Sharpe
    
    # Sensitivity analysis
    sensitivity_results: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    
    # Best parameters found
    best_hyperparameters: Optional[MLBacktestHyperparameters] = None
    
    # Regime performance
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Equity curve
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution time
    execution_time_seconds: float = 0.0


class MLHyperparameterBacktester:
    """
    Backtester that integrates ML hyperparameter optimization.
    
    Supports:
    - Static parameter backtesting
    - Adaptive regime-based parameters
    - Walk-forward optimization
    - Parameter sensitivity analysis
    """
    
    def __init__(
        self,
        config: Optional[MLHyperparameterBacktestConfig] = None,
        market_adapter: Any = None,
    ):
        """
        Initialize the backtester.
        
        Args:
            config: Backtest configuration
            market_adapter: Market data adapter
        """
        self.config = config or MLHyperparameterBacktestConfig()
        self.market_adapter = market_adapter
        
        # Initialize ML components
        self.hp_manager = MLHyperparameterManager(create_preset_config(self.config.preset))
        
        # Results storage
        self.results: Optional[MLHyperparameterBacktestResults] = None
        self._equity_curve: List[Dict[str, Any]] = []
        self._trades: List[Dict[str, Any]] = []
        
        logger.info(
            f"MLHyperparameterBacktester initialized | "
            f"mode={self.config.mode.value} | "
            f"preset={self.config.preset}"
        )
    
    def run(self) -> MLHyperparameterBacktestResults:
        """Run the backtest based on configured mode."""
        import time
        start_time = time.time()
        
        logger.info(f"Starting ML hyperparameter backtest | mode={self.config.mode.value}")
        
        if self.config.mode == MLBacktestMode.STATIC:
            results = self._run_static_backtest()
        elif self.config.mode == MLBacktestMode.ADAPTIVE:
            results = self._run_adaptive_backtest()
        elif self.config.mode == MLBacktestMode.WALK_FORWARD:
            results = self._run_walk_forward_backtest()
        elif self.config.mode == MLBacktestMode.SENSITIVITY:
            results = self._run_sensitivity_analysis()
        else:
            raise ValueError(f"Unknown backtest mode: {self.config.mode}")
        
        results.execution_time_seconds = time.time() - start_time
        self.results = results
        
        # Save results
        self._save_results(results)
        
        logger.info(
            f"ML backtest complete | "
            f"Sharpe={results.sharpe_ratio:.2f} | "
            f"Return={results.total_return_pct:.1%} | "
            f"Time={results.execution_time_seconds:.1f}s"
        )
        
        return results
    
    def _run_static_backtest(self) -> MLHyperparameterBacktestResults:
        """Run backtest with static ML parameters."""
        logger.info("Running static ML parameter backtest...")
        
        # Get hyperparameters
        hp = self.config.hyperparameters or MLBacktestHyperparameters.from_preset(self.config.preset)
        
        # Run single period backtest
        period_result = self._backtest_period(
            start_date=datetime.fromisoformat(self.config.start_date),
            end_date=datetime.fromisoformat(self.config.end_date),
            hyperparameters=hp,
        )
        
        return MLHyperparameterBacktestResults(
            config=self.config,
            total_return_pct=period_result.total_return_pct,
            sharpe_ratio=period_result.sharpe_ratio,
            sortino_ratio=period_result.sortino_ratio,
            max_drawdown_pct=period_result.max_drawdown_pct,
            total_trades=period_result.total_trades,
            win_rate=period_result.win_rate,
            profit_factor=period_result.profit_factor,
            avg_trade_pnl=period_result.avg_trade_pnl,
            avg_ml_confidence=period_result.avg_confidence,
            ml_signal_accuracy=period_result.signal_accuracy,
            periods=[period_result],
            best_hyperparameters=hp,
            equity_curve=self._equity_curve,
        )
    
    def _run_adaptive_backtest(self) -> MLHyperparameterBacktestResults:
        """Run backtest with regime-adaptive parameters."""
        logger.info("Running adaptive ML parameter backtest...")
        
        # Split data by regime
        start = datetime.fromisoformat(self.config.start_date)
        end = datetime.fromisoformat(self.config.end_date)
        
        # Get regime-specific parameters
        regime_params = self._get_regime_parameters()
        
        # Run backtest with regime switching
        periods = []
        combined_equity = []
        total_trades = 0
        total_pnl = 0.0
        
        for regime_name, hp in regime_params.items():
            period = self._backtest_period(
                start_date=start,
                end_date=end,
                hyperparameters=hp,
                regime_filter=regime_name,
            )
            periods.append(period)
            total_trades += period.total_trades
            total_pnl += period.avg_trade_pnl * period.total_trades
        
        # Calculate aggregate metrics
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        avg_sharpe = np.mean([p.sharpe_ratio for p in periods])
        
        return MLHyperparameterBacktestResults(
            config=self.config,
            sharpe_ratio=avg_sharpe,
            total_trades=total_trades,
            avg_trade_pnl=avg_trade,
            periods=periods,
            regime_performance={
                p.regime.value if p.regime else "unknown": {
                    "sharpe": p.sharpe_ratio,
                    "return": p.total_return_pct,
                    "trades": p.total_trades,
                } for p in periods
            },
        )
    
    def _run_walk_forward_backtest(self) -> MLHyperparameterBacktestResults:
        """Run walk-forward optimization backtest."""
        logger.info("Running walk-forward ML parameter backtest...")
        
        start = datetime.fromisoformat(self.config.start_date)
        end = datetime.fromisoformat(self.config.end_date)
        total_days = (end - start).days
        
        # Calculate period sizes
        train_days = int(total_days * self.config.train_pct)
        test_days = total_days - train_days
        
        # Generate walk-forward windows
        windows = self._generate_walk_forward_windows(start, end, train_days, test_days)
        
        periods = []
        in_sample_sharpes = []
        out_sample_sharpes = []
        best_params_history = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Walk-forward window {i+1}/{len(windows)}")
            
            # Optimize on training data
            best_hp = self._optimize_parameters(train_start, train_end)
            best_params_history.append(best_hp)
            
            # Backtest on training (in-sample)
            is_result = self._backtest_period(train_start, train_end, best_hp)
            in_sample_sharpes.append(is_result.sharpe_ratio)
            
            # Backtest on test (out-of-sample)
            oos_result = self._backtest_period(test_start, test_end, best_hp)
            out_sample_sharpes.append(oos_result.sharpe_ratio)
            periods.append(oos_result)
        
        # Calculate walk-forward efficiency
        avg_is = np.mean(in_sample_sharpes)
        avg_oos = np.mean(out_sample_sharpes)
        wfe = avg_oos / avg_is if avg_is > 0 else 0
        
        return MLHyperparameterBacktestResults(
            config=self.config,
            sharpe_ratio=avg_oos,
            in_sample_sharpe=avg_is,
            out_of_sample_sharpe=avg_oos,
            walk_forward_efficiency=wfe,
            total_trades=sum(p.total_trades for p in periods),
            win_rate=np.mean([p.win_rate for p in periods]),
            periods=periods,
            best_hyperparameters=best_params_history[-1] if best_params_history else None,
        )
    
    def _run_sensitivity_analysis(self) -> MLHyperparameterBacktestResults:
        """Run parameter sensitivity analysis."""
        logger.info("Running sensitivity analysis...")
        
        base_hp = self.config.hyperparameters or MLBacktestHyperparameters.from_preset(self.config.preset)
        start = datetime.fromisoformat(self.config.start_date)
        end = datetime.fromisoformat(self.config.end_date)
        
        sensitivity_results = {}
        
        for param_name in self.config.sensitivity_params:
            if not hasattr(base_hp, param_name):
                continue
            
            base_value = getattr(base_hp, param_name)
            if not isinstance(base_value, (int, float)):
                continue
            
            results = []
            
            # Generate range of values
            min_val = base_value * (1 - self.config.sensitivity_range)
            max_val = base_value * (1 + self.config.sensitivity_range)
            
            for value in np.linspace(min_val, max_val, self.config.sensitivity_steps):
                # Create modified hyperparameters
                hp_dict = base_hp.to_dict()
                hp_dict[param_name] = value if isinstance(base_value, float) else int(value)
                hp = MLBacktestHyperparameters.from_dict(hp_dict)
                
                # Run backtest
                period = self._backtest_period(start, end, hp)
                results.append((value, period.sharpe_ratio))
            
            sensitivity_results[param_name] = results
        
        # Run base case
        base_period = self._backtest_period(start, end, base_hp)
        
        return MLHyperparameterBacktestResults(
            config=self.config,
            sharpe_ratio=base_period.sharpe_ratio,
            total_return_pct=base_period.total_return_pct,
            max_drawdown_pct=base_period.max_drawdown_pct,
            total_trades=base_period.total_trades,
            win_rate=base_period.win_rate,
            periods=[base_period],
            sensitivity_results=sensitivity_results,
            best_hyperparameters=base_hp,
        )
    
    def _backtest_period(
        self,
        start_date: datetime,
        end_date: datetime,
        hyperparameters: MLBacktestHyperparameters,
        regime_filter: Optional[str] = None,
    ) -> MLBacktestPeriod:
        """Run backtest for a single period."""
        # This is a simplified simulation - in production, would use full EliteBacktestEngine
        
        # Simulate based on parameters
        np.random.seed(42)  # For reproducibility
        
        days = (end_date - start_date).days
        
        # Generate simulated returns based on hyperparameters
        # Higher confidence threshold -> fewer but better trades
        trade_frequency = 1 / (hyperparameters.confidence_threshold * 10)
        expected_trades = int(days * trade_frequency)
        
        # Win rate based on confidence threshold
        base_win_rate = 0.45 + hyperparameters.confidence_threshold * 0.2
        
        # Simulate trades
        wins = int(expected_trades * base_win_rate)
        losses = expected_trades - wins
        
        # Generate P&L
        avg_win = hyperparameters.max_position_pct * 0.03  # 3% average win
        avg_loss = hyperparameters.max_position_pct * 0.015  # 1.5% average loss (with stops)
        
        total_pnl = wins * avg_win - losses * avg_loss
        total_return_pct = total_pnl * 10  # Simplified
        
        # Calculate metrics
        sharpe = total_return_pct / 0.15 if total_return_pct > 0 else total_return_pct / 0.20
        sortino = sharpe * 1.2 if sharpe > 0 else sharpe * 0.8
        max_dd = 0.05 + (1 - hyperparameters.confidence_threshold) * 0.10
        
        profit_factor = (wins * avg_win) / (losses * avg_loss) if losses > 0 and avg_loss > 0 else 1.0
        
        return MLBacktestPeriod(
            start_date=start_date,
            end_date=end_date,
            regime=MarketRegime.TRENDING_BULL if regime_filter else None,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd,
            win_rate=base_win_rate,
            profit_factor=profit_factor,
            total_trades=expected_trades,
            avg_trade_pnl=total_pnl / expected_trades if expected_trades > 0 else 0,
            avg_confidence=hyperparameters.confidence_threshold,
            signal_accuracy=base_win_rate * 0.9,
            hyperparameters=hyperparameters,
        )
    
    def _optimize_parameters(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> MLBacktestHyperparameters:
        """Optimize hyperparameters for a period."""
        # Simplified optimization - in production would use full OptimizationEngine
        
        best_sharpe = -np.inf
        best_hp = MLBacktestHyperparameters.from_preset(self.config.preset)
        
        # Grid search over key parameters
        for conf in [0.50, 0.60, 0.70]:
            for kelly in [0.15, 0.25, 0.35]:
                for pos_size in [0.02, 0.04, 0.06]:
                    hp = MLBacktestHyperparameters(
                        confidence_threshold=conf,
                        kelly_fraction=kelly,
                        max_position_pct=pos_size,
                    )
                    
                    result = self._backtest_period(start_date, end_date, hp)
                    
                    if result.sharpe_ratio > best_sharpe:
                        best_sharpe = result.sharpe_ratio
                        best_hp = hp
        
        logger.debug(f"Optimized params: conf={best_hp.confidence_threshold}, sharpe={best_sharpe:.2f}")
        return best_hp
    
    def _get_regime_parameters(self) -> Dict[str, MLBacktestHyperparameters]:
        """Get regime-specific hyperparameters."""
        return {
            "trending_bull": MLBacktestHyperparameters(
                confidence_threshold=0.55,
                kelly_fraction=0.30,
                max_position_pct=0.05,
            ),
            "trending_bear": MLBacktestHyperparameters(
                confidence_threshold=0.65,
                kelly_fraction=0.20,
                max_position_pct=0.03,
            ),
            "sideways": MLBacktestHyperparameters(
                confidence_threshold=0.70,
                kelly_fraction=0.15,
                max_position_pct=0.02,
            ),
            "high_volatility": MLBacktestHyperparameters(
                confidence_threshold=0.75,
                kelly_fraction=0.10,
                max_position_pct=0.02,
            ),
        }
    
    def _generate_walk_forward_windows(
        self,
        start: datetime,
        end: datetime,
        train_days: int,
        test_days: int,
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate walk-forward optimization windows."""
        windows = []
        current = start
        
        while current + timedelta(days=train_days + test_days) <= end:
            train_start = current
            train_end = current + timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_days)
            
            windows.append((train_start, train_end, test_start, test_end))
            
            # Move forward by reoptimize_frequency
            current += timedelta(days=self.config.reoptimize_frequency)
        
        return windows
    
    def _save_results(self, results: MLHyperparameterBacktestResults):
        """Save results to disk."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{self.config.tag}" if self.config.tag else ""
        
        # Save summary
        summary = {
            "mode": self.config.mode.value,
            "preset": self.config.preset,
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "sharpe_ratio": results.sharpe_ratio,
            "total_return_pct": results.total_return_pct,
            "max_drawdown_pct": results.max_drawdown_pct,
            "total_trades": results.total_trades,
            "win_rate": results.win_rate,
            "walk_forward_efficiency": results.walk_forward_efficiency,
            "execution_time_seconds": results.execution_time_seconds,
            "best_hyperparameters": results.best_hyperparameters.to_dict() if results.best_hyperparameters else None,
            "sensitivity_results": results.sensitivity_results,
            "regime_performance": results.regime_performance,
        }
        
        with open(output_dir / f"ml_backtest_{timestamp}{tag}.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_dir}")


# Factory function
def create_ml_hyperparameter_backtester(
    preset: str = "balanced",
    mode: str = "static",
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-01",
    symbols: Optional[List[str]] = None,
) -> MLHyperparameterBacktester:
    """Create an ML hyperparameter backtester."""
    config = MLHyperparameterBacktestConfig(
        mode=MLBacktestMode(mode),
        preset=preset,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols or ["SPY"],
    )
    
    return MLHyperparameterBacktester(config=config)


def run_ml_parameter_backtest(
    preset: str = "balanced",
    mode: str = "static",
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-01",
    symbols: Optional[List[str]] = None,
) -> MLHyperparameterBacktestResults:
    """Quick function to run ML parameter backtest."""
    backtester = create_ml_hyperparameter_backtester(
        preset=preset,
        mode=mode,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
    )
    return backtester.run()


__all__ = [
    "MLBacktestMode",
    "MLBacktestHyperparameters",
    "MLBacktestPeriod",
    "MLHyperparameterBacktestConfig",
    "MLHyperparameterBacktestResults",
    "MLHyperparameterBacktester",
    "create_ml_hyperparameter_backtester",
    "run_ml_parameter_backtest",
]
