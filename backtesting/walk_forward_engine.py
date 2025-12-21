"""
Walk-Forward Validation Engine

Walk-forward analysis is a rigorous backtesting methodology that:
1. Divides data into multiple train/test windows
2. Optimizes parameters on training data (in-sample)
3. Validates on unseen test data (out-of-sample)
4. Rolls forward and repeats
5. Aggregates results to detect overfitting

This prevents curve-fitting and provides realistic performance expectations.

Example:
    |------ Train 1 ------|-- Test 1 --|
                |------ Train 2 ------|-- Test 2 --|
                          |------ Train 3 ------|-- Test 3 --|

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import json
import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import concurrent.futures

import numpy as np
import pandas as pd
from loguru import logger

from backtesting.elite_backtest_engine import (
    EliteBacktestConfig,
    EliteBacktestResults,
    EliteBacktestEngine,
    SimulatedTrade,
)


class OptimizationMethod(str, Enum):
    """Parameter optimization method."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"  # Future enhancement


class OptimizationObjective(str, Enum):
    """Objective function for optimization."""
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    TOTAL_RETURN = "total_return"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"  # Return / Max DD


@dataclass
class ParameterRange:
    """Define a parameter range for optimization."""
    name: str
    min_value: float
    max_value: float
    step: float = None  # For grid search
    n_samples: int = 10  # For random search
    
    def grid_values(self) -> List[float]:
        """Generate grid values for this parameter."""
        if self.step is None:
            # Default: 5 steps
            step = (self.max_value - self.min_value) / 4
        else:
            step = self.step
        
        values = []
        current = self.min_value
        while current <= self.max_value + 1e-9:
            values.append(round(current, 6))
            current += step
        return values
    
    def random_values(self, n: int = None) -> List[float]:
        """Generate random values for this parameter."""
        n = n or self.n_samples
        return list(np.random.uniform(self.min_value, self.max_value, n))


@dataclass
class WalkForwardWindow:
    """A single walk-forward window with train and test periods."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Results
    best_params: Dict[str, float] = field(default_factory=dict)
    train_results: Optional[EliteBacktestResults] = None
    test_results: Optional[EliteBacktestResults] = None
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days
    
    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    
    # Base backtest config (will be modified per window)
    base_config: EliteBacktestConfig = None
    
    # Walk-forward settings
    n_windows: int = 5  # Number of walk-forward windows
    train_pct: float = 0.70  # 70% train, 30% test per window
    anchored: bool = False  # If True, training always starts from beginning
    overlap_pct: float = 0.50  # 50% overlap between windows
    
    # Optimization settings
    optimization_method: OptimizationMethod = OptimizationMethod.GRID_SEARCH
    optimization_objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO
    parameter_ranges: List[ParameterRange] = field(default_factory=list)
    max_optimization_iterations: int = 100  # For random/bayesian search
    
    # Parallel processing
    n_jobs: int = 1  # Number of parallel jobs (1 = sequential)
    
    # Output settings
    save_results: bool = True
    output_dir: str = "runs/walk_forward"
    tag: str = ""
    
    def __post_init__(self):
        if self.base_config is None:
            self.base_config = EliteBacktestConfig()
        
        # Default parameter ranges if none provided
        if not self.parameter_ranges:
            self.parameter_ranges = [
                ParameterRange("min_confidence", 0.25, 0.60, step=0.05),
                ParameterRange("atr_stop_mult", 1.5, 3.0, step=0.5),
                ParameterRange("atr_target_mult", 2.0, 4.0, step=0.5),
            ]


@dataclass
class WalkForwardResults:
    """Aggregated results from walk-forward validation."""
    
    config: Optional[WalkForwardConfig] = None
    windows: List[WalkForwardWindow] = field(default_factory=list)
    
    # Aggregated metrics (out-of-sample only)
    total_return: float = 0.0
    total_return_pct: float = 0.0
    avg_return_per_window: float = 0.0
    
    # Risk metrics
    avg_sharpe: float = 0.0
    avg_sortino: float = 0.0
    avg_max_drawdown: float = 0.0
    worst_drawdown: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    overall_win_rate: float = 0.0
    avg_profit_factor: float = 0.0
    
    # Robustness metrics
    pct_profitable_windows: float = 0.0
    return_consistency: float = 0.0  # Std dev of window returns
    parameter_stability: float = 0.0  # How stable are optimal params
    
    # In-sample vs Out-of-sample comparison
    is_avg_sharpe: float = 0.0  # In-sample average Sharpe
    oos_avg_sharpe: float = 0.0  # Out-of-sample average Sharpe
    overfitting_ratio: float = 0.0  # IS Sharpe / OOS Sharpe (>1 = overfitting)
    
    # Combined equity curve
    combined_equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    all_trades: List[SimulatedTrade] = field(default_factory=list)


class WalkForwardEngine:
    """
    Walk-Forward Validation Engine.
    
    Performs rigorous out-of-sample testing by:
    1. Creating rolling train/test windows
    2. Optimizing parameters on training data
    3. Testing optimal params on unseen data
    4. Aggregating results for robustness analysis
    """
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.windows: List[WalkForwardWindow] = []
        self.historical_data: Dict[str, pd.DataFrame] = {}
        
        logger.info(
            f"WalkForwardEngine initialized | "
            f"n_windows={config.n_windows} | "
            f"train_pct={config.train_pct:.0%} | "
            f"method={config.optimization_method.value}"
        )
    
    def _fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols once."""
        if self.historical_data:
            return self.historical_data
        
        from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
        
        adapter = AlpacaMarketDataAdapter()
        base = self.config.base_config
        
        start = datetime.strptime(base.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(base.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
        for symbol in base.symbols:
            try:
                bars = adapter.get_bars(
                    symbol=symbol,
                    start=start,
                    end=end,
                    timeframe=base.timeframe,
                )
                
                if bars:
                    df = pd.DataFrame([
                        {
                            'timestamp': bar.timestamp,
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume,
                        }
                        for bar in bars
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    df['symbol'] = symbol
                    self.historical_data[symbol] = df
                    logger.info(f"Fetched {len(df)} bars for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        
        return self.historical_data
    
    def _create_windows(self) -> List[WalkForwardWindow]:
        """Create walk-forward windows based on configuration."""
        base = self.config.base_config
        
        start = datetime.strptime(base.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(base.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
        total_days = (end - start).days
        n_windows = self.config.n_windows
        
        # Calculate window sizes
        if self.config.anchored:
            # Anchored: training always starts from beginning
            # Each window adds more training data
            test_days = int(total_days * (1 - self.config.train_pct) / n_windows)
            windows = []
            
            for i in range(n_windows):
                train_start = start
                test_end = end - timedelta(days=(n_windows - i - 1) * test_days)
                test_start = test_end - timedelta(days=test_days)
                train_end = test_start - timedelta(days=1)
                
                if train_end <= train_start:
                    continue
                
                windows.append(WalkForwardWindow(
                    window_id=i + 1,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                ))
        else:
            # Rolling: overlapping windows
            window_size = int(total_days / (1 + (n_windows - 1) * (1 - self.config.overlap_pct)))
            step_size = int(window_size * (1 - self.config.overlap_pct))
            
            train_days = int(window_size * self.config.train_pct)
            test_days = window_size - train_days
            
            windows = []
            for i in range(n_windows):
                window_start = start + timedelta(days=i * step_size)
                train_start = window_start
                train_end = train_start + timedelta(days=train_days)
                test_start = train_end + timedelta(days=1)
                test_end = test_start + timedelta(days=test_days)
                
                if test_end > end:
                    test_end = end
                if test_start >= test_end:
                    continue
                
                windows.append(WalkForwardWindow(
                    window_id=i + 1,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                ))
        
        self.windows = windows
        
        logger.info(f"Created {len(windows)} walk-forward windows:")
        for w in windows:
            logger.info(
                f"  Window {w.window_id}: "
                f"Train {w.train_start.date()} to {w.train_end.date()} ({w.train_days}d) | "
                f"Test {w.test_start.date()} to {w.test_end.date()} ({w.test_days}d)"
            )
        
        return windows
    
    def _get_objective_value(self, results: EliteBacktestResults) -> float:
        """Extract objective value from results."""
        obj = self.config.optimization_objective
        
        if results.total_trades == 0:
            return -999.0  # Penalize no trades
        
        if obj == OptimizationObjective.SHARPE_RATIO:
            return results.sharpe_ratio if not np.isnan(results.sharpe_ratio) else -999.0
        elif obj == OptimizationObjective.SORTINO_RATIO:
            return results.sortino_ratio if not np.isnan(results.sortino_ratio) else -999.0
        elif obj == OptimizationObjective.CALMAR_RATIO:
            return results.calmar_ratio if not np.isnan(results.calmar_ratio) else -999.0
        elif obj == OptimizationObjective.TOTAL_RETURN:
            return results.total_return_pct
        elif obj == OptimizationObjective.PROFIT_FACTOR:
            return results.profit_factor if results.profit_factor != float('inf') else 10.0
        elif obj == OptimizationObjective.WIN_RATE:
            return results.win_rate
        elif obj == OptimizationObjective.RISK_ADJUSTED_RETURN:
            if results.max_drawdown_pct > 0:
                return results.total_return_pct / results.max_drawdown_pct
            return results.total_return_pct
        else:
            return results.sharpe_ratio
    
    def _run_single_backtest(
        self,
        start_date: str,
        end_date: str,
        params: Dict[str, float],
    ) -> EliteBacktestResults:
        """Run a single backtest with given parameters."""
        # Create config with modified params
        config_dict = {
            "symbols": self.config.base_config.symbols,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": self.config.base_config.initial_capital,
            "max_positions": self.config.base_config.max_positions,
            "max_position_pct": self.config.base_config.max_position_pct,
            "use_agent_signals": self.config.base_config.use_agent_signals,
            "disable_event_risk": self.config.base_config.disable_event_risk,
            "save_trades": False,
            "save_equity_curve": False,
            "monte_carlo_runs": 0,  # Skip MC for optimization speed
        }
        
        # Apply optimizable params
        for name, value in params.items():
            config_dict[name] = value
        
        config = EliteBacktestConfig(**config_dict)
        engine = EliteBacktestEngine(config)
        
        # Use cached data if available
        if self.historical_data:
            # Filter data to window dates
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            
            def mock_fetch(symbol):
                if symbol in self.historical_data:
                    df = self.historical_data[symbol]
                    mask = (df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)
                    return df[mask].copy().reset_index(drop=True)
                return pd.DataFrame()
            
            # Patch the fetch method
            original_fetch = engine.fetch_historical_data
            engine.fetch_historical_data = mock_fetch
        
        try:
            results = engine.run_backtest()
            return results
        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            return EliteBacktestResults()
    
    def _optimize_window(self, window: WalkForwardWindow) -> WalkForwardWindow:
        """Optimize parameters for a single window."""
        logger.info(f"Optimizing Window {window.window_id}...")
        
        train_start = window.train_start.strftime("%Y-%m-%d")
        train_end = window.train_end.strftime("%Y-%m-%d")
        
        best_params = {}
        best_score = -999.0
        optimization_history = []
        
        if self.config.optimization_method == OptimizationMethod.GRID_SEARCH:
            # Generate all parameter combinations
            param_names = [p.name for p in self.config.parameter_ranges]
            param_values = [p.grid_values() for p in self.config.parameter_ranges]
            
            all_combinations = list(itertools.product(*param_values))
            logger.info(f"  Grid search: {len(all_combinations)} combinations")
            
            for combo in all_combinations:
                params = dict(zip(param_names, combo))
                results = self._run_single_backtest(train_start, train_end, params)
                score = self._get_objective_value(results)
                
                optimization_history.append({
                    "params": params.copy(),
                    "score": score,
                    "return": results.total_return_pct,
                    "trades": results.total_trades,
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
        
        elif self.config.optimization_method == OptimizationMethod.RANDOM_SEARCH:
            n_iter = self.config.max_optimization_iterations
            logger.info(f"  Random search: {n_iter} iterations")
            
            for i in range(n_iter):
                params = {}
                for p in self.config.parameter_ranges:
                    params[p.name] = np.random.uniform(p.min_value, p.max_value)
                
                results = self._run_single_backtest(train_start, train_end, params)
                score = self._get_objective_value(results)
                
                optimization_history.append({
                    "params": params.copy(),
                    "score": score,
                    "return": results.total_return_pct,
                    "trades": results.total_trades,
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
        
        # Run best params on training data for final train results
        window.best_params = best_params
        window.optimization_history = optimization_history
        
        if best_params:
            window.train_results = self._run_single_backtest(train_start, train_end, best_params)
            logger.info(
                f"  Best params: {best_params} | "
                f"Train {self.config.optimization_objective.value}: {best_score:.3f}"
            )
        
        return window
    
    def _validate_window(self, window: WalkForwardWindow) -> WalkForwardWindow:
        """Validate optimized parameters on test data."""
        if not window.best_params:
            logger.warning(f"Window {window.window_id} has no optimized params")
            return window
        
        test_start = window.test_start.strftime("%Y-%m-%d")
        test_end = window.test_end.strftime("%Y-%m-%d")
        
        logger.info(f"Validating Window {window.window_id} on test data...")
        
        window.test_results = self._run_single_backtest(
            test_start, test_end, window.best_params
        )
        
        if window.test_results:
            test_score = self._get_objective_value(window.test_results)
            logger.info(
                f"  Test {self.config.optimization_objective.value}: {test_score:.3f} | "
                f"Return: {window.test_results.total_return_pct:.2%} | "
                f"Trades: {window.test_results.total_trades}"
            )
        
        return window
    
    def _aggregate_results(self) -> WalkForwardResults:
        """Aggregate results from all windows."""
        results = WalkForwardResults(config=self.config, windows=self.windows)
        
        # Collect out-of-sample metrics
        oos_returns = []
        oos_sharpes = []
        oos_sortinos = []
        oos_drawdowns = []
        oos_profit_factors = []
        is_sharpes = []
        
        total_trades = 0
        total_wins = 0
        all_trades = []
        combined_equity = []
        
        for window in self.windows:
            if window.test_results:
                tr = window.test_results
                oos_returns.append(tr.total_return_pct)
                oos_sharpes.append(tr.sharpe_ratio if not np.isnan(tr.sharpe_ratio) else 0)
                oos_sortinos.append(tr.sortino_ratio if not np.isnan(tr.sortino_ratio) else 0)
                oos_drawdowns.append(tr.max_drawdown_pct)
                
                if tr.profit_factor != float('inf'):
                    oos_profit_factors.append(tr.profit_factor)
                
                total_trades += tr.total_trades
                total_wins += tr.winning_trades
                
                # Collect trades
                all_trades.extend(tr.trades)
                
                # Collect equity curve
                for eq in tr.equity_curve:
                    eq['window_id'] = window.window_id
                    combined_equity.append(eq)
            
            if window.train_results:
                is_sharpes.append(
                    window.train_results.sharpe_ratio 
                    if not np.isnan(window.train_results.sharpe_ratio) else 0
                )
        
        # Calculate aggregated metrics
        if oos_returns:
            results.total_return_pct = np.prod([1 + r for r in oos_returns]) - 1
            results.total_return = results.total_return_pct * self.config.base_config.initial_capital
            results.avg_return_per_window = np.mean(oos_returns)
            results.return_consistency = np.std(oos_returns) if len(oos_returns) > 1 else 0
            
            profitable_windows = sum(1 for r in oos_returns if r > 0)
            results.pct_profitable_windows = profitable_windows / len(oos_returns)
        
        if oos_sharpes:
            results.avg_sharpe = np.mean(oos_sharpes)
            results.oos_avg_sharpe = results.avg_sharpe
        
        if oos_sortinos:
            results.avg_sortino = np.mean(oos_sortinos)
        
        if oos_drawdowns:
            results.avg_max_drawdown = np.mean(oos_drawdowns)
            results.worst_drawdown = max(oos_drawdowns)
        
        if oos_profit_factors:
            results.avg_profit_factor = np.mean(oos_profit_factors)
        
        if is_sharpes:
            results.is_avg_sharpe = np.mean(is_sharpes)
            if results.oos_avg_sharpe != 0:
                results.overfitting_ratio = results.is_avg_sharpe / results.oos_avg_sharpe
        
        results.total_trades = total_trades
        if total_trades > 0:
            results.overall_win_rate = total_wins / total_trades
        
        # Parameter stability (how much do optimal params vary?)
        if len(self.windows) > 1:
            param_stds = []
            for p in self.config.parameter_ranges:
                values = [w.best_params.get(p.name, 0) for w in self.windows if w.best_params]
                if values:
                    # Normalize by parameter range
                    norm_std = np.std(values) / (p.max_value - p.min_value)
                    param_stds.append(norm_std)
            if param_stds:
                results.parameter_stability = 1 - np.mean(param_stds)  # Higher = more stable
        
        results.all_trades = all_trades
        results.combined_equity_curve = combined_equity
        
        return results
    
    def run(self) -> WalkForwardResults:
        """Run full walk-forward validation."""
        logger.info("=" * 60)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info("=" * 60)
        
        # Fetch all data upfront
        logger.info("Fetching historical data...")
        self._fetch_all_data()
        
        if not self.historical_data:
            raise ValueError("No historical data available")
        
        # Create windows
        self._create_windows()
        
        if not self.windows:
            raise ValueError("No valid windows created")
        
        # Optimize and validate each window
        for i, window in enumerate(self.windows):
            logger.info("-" * 40)
            logger.info(f"Processing Window {window.window_id}/{len(self.windows)}")
            logger.info("-" * 40)
            
            # Optimize on training data
            self._optimize_window(window)
            
            # Validate on test data
            self._validate_window(window)
        
        # Aggregate results
        results = self._aggregate_results()
        
        # Save results
        if self.config.save_results:
            self._save_results(results)
        
        return results
    
    def _save_results(self, results: WalkForwardResults):
        """Save walk-forward results to disk."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tag = self.config.tag or f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Summary
        summary = {
            "tag": tag,
            "n_windows": len(self.windows),
            "optimization_method": self.config.optimization_method.value,
            "optimization_objective": self.config.optimization_objective.value,
            "symbols": self.config.base_config.symbols,
            "total_return_pct": results.total_return_pct,
            "avg_return_per_window": results.avg_return_per_window,
            "avg_sharpe": results.avg_sharpe,
            "avg_sortino": results.avg_sortino,
            "avg_max_drawdown": results.avg_max_drawdown,
            "worst_drawdown": results.worst_drawdown,
            "total_trades": results.total_trades,
            "overall_win_rate": results.overall_win_rate,
            "avg_profit_factor": results.avg_profit_factor,
            "pct_profitable_windows": results.pct_profitable_windows,
            "return_consistency": results.return_consistency,
            "parameter_stability": results.parameter_stability,
            "is_avg_sharpe": results.is_avg_sharpe,
            "oos_avg_sharpe": results.oos_avg_sharpe,
            "overfitting_ratio": results.overfitting_ratio,
            "windows": [],
        }
        
        for w in self.windows:
            window_data = {
                "window_id": w.window_id,
                "train_start": str(w.train_start.date()),
                "train_end": str(w.train_end.date()),
                "test_start": str(w.test_start.date()),
                "test_end": str(w.test_end.date()),
                "best_params": w.best_params,
                "train_return": w.train_results.total_return_pct if w.train_results else None,
                "train_sharpe": w.train_results.sharpe_ratio if w.train_results else None,
                "test_return": w.test_results.total_return_pct if w.test_results else None,
                "test_sharpe": w.test_results.sharpe_ratio if w.test_results else None,
                "test_trades": w.test_results.total_trades if w.test_results else 0,
            }
            summary["windows"].append(window_data)
        
        summary_path = output_dir / f"{tag}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {summary_path}")


def print_walk_forward_results(results: WalkForwardResults):
    """Print formatted walk-forward results."""
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 70)
    
    if results.config:
        print(f"Symbols: {', '.join(results.config.base_config.symbols)}")
        print(f"Windows: {len(results.windows)}")
        print(f"Method: {results.config.optimization_method.value}")
        print(f"Objective: {results.config.optimization_objective.value}")
    
    print("-" * 70)
    print("OUT-OF-SAMPLE PERFORMANCE (Aggregate)")
    print(f"  Total Return:         {results.total_return_pct*100:+.2f}%")
    print(f"  Avg Return/Window:    {results.avg_return_per_window*100:+.2f}%")
    print(f"  Profitable Windows:   {results.pct_profitable_windows*100:.0f}%")
    print(f"  Return Consistency:   {results.return_consistency*100:.2f}% (std dev)")
    
    print("-" * 70)
    print("RISK METRICS (Out-of-Sample)")
    print(f"  Avg Sharpe Ratio:     {results.avg_sharpe:.2f}")
    print(f"  Avg Sortino Ratio:    {results.avg_sortino:.2f}")
    print(f"  Avg Max Drawdown:     {results.avg_max_drawdown*100:.2f}%")
    print(f"  Worst Drawdown:       {results.worst_drawdown*100:.2f}%")
    
    print("-" * 70)
    print("TRADE STATISTICS")
    print(f"  Total Trades:         {results.total_trades}")
    print(f"  Overall Win Rate:     {results.overall_win_rate*100:.1f}%")
    print(f"  Avg Profit Factor:    {results.avg_profit_factor:.2f}")
    
    print("-" * 70)
    print("ROBUSTNESS ANALYSIS")
    print(f"  In-Sample Sharpe:     {results.is_avg_sharpe:.2f}")
    print(f"  Out-of-Sample Sharpe: {results.oos_avg_sharpe:.2f}")
    print(f"  Overfitting Ratio:    {results.overfitting_ratio:.2f} {'⚠️ OVERFIT' if results.overfitting_ratio > 2 else '✓ OK'}")
    print(f"  Parameter Stability:  {results.parameter_stability*100:.0f}%")
    
    print("-" * 70)
    print("WINDOW BREAKDOWN")
    print(f"  {'Window':<8} {'Train Period':<25} {'Test Period':<25} {'Test Return':<12} {'Test Sharpe':<12}")
    print(f"  {'-'*8} {'-'*25} {'-'*25} {'-'*12} {'-'*12}")
    
    for w in results.windows:
        train_period = f"{w.train_start.date()} - {w.train_end.date()}"
        test_period = f"{w.test_start.date()} - {w.test_end.date()}"
        test_ret = f"{w.test_results.total_return_pct*100:+.2f}%" if w.test_results else "N/A"
        test_sharpe = f"{w.test_results.sharpe_ratio:.2f}" if w.test_results else "N/A"
        print(f"  {w.window_id:<8} {train_period:<25} {test_period:<25} {test_ret:<12} {test_sharpe:<12}")
    
    print("=" * 70)
    
    # Verdict
    print("\n🎯 VERDICT:")
    if results.overfitting_ratio > 2:
        print("  ❌ HIGH OVERFITTING - Strategy is curve-fitted to historical data")
    elif results.overfitting_ratio > 1.5:
        print("  ⚠️  MODERATE OVERFITTING - Some curve-fitting detected")
    elif results.pct_profitable_windows < 0.5:
        print("  ⚠️  LOW WIN RATE - Less than 50% of windows profitable")
    elif results.avg_sharpe < 0.5:
        print("  ⚠️  LOW RISK-ADJUSTED RETURNS - Sharpe below 0.5")
    elif results.avg_sharpe >= 1.0 and results.pct_profitable_windows >= 0.6:
        print("  ✅ ROBUST - Strategy shows consistent out-of-sample performance")
    else:
        print("  ⚠️  MARGINAL - Strategy may need improvement")


def run_walk_forward(
    symbols: List[str] = None,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-01",
    n_windows: int = 5,
    train_pct: float = 0.70,
    optimization_objective: str = "sharpe_ratio",
    tag: str = "",
    **kwargs
) -> WalkForwardResults:
    """Convenience function to run walk-forward validation."""
    
    if symbols is None:
        symbols = ["SPY"]
    
    base_config = EliteBacktestConfig(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        use_agent_signals=False,
        disable_event_risk=True,
        **{k: v for k, v in kwargs.items() if hasattr(EliteBacktestConfig, k)}
    )
    
    wf_config = WalkForwardConfig(
        base_config=base_config,
        n_windows=n_windows,
        train_pct=train_pct,
        optimization_objective=OptimizationObjective(optimization_objective),
        tag=tag or f"wf_{'-'.join(symbols)}_{n_windows}windows",
    )
    
    engine = WalkForwardEngine(wf_config)
    return engine.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Walk-Forward Validation")
    parser.add_argument("--symbols", type=str, default="SPY", help="Comma-separated symbols")
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-12-01", help="End date")
    parser.add_argument("--windows", type=int, default=5, help="Number of windows")
    parser.add_argument("--train-pct", type=float, default=0.70, help="Training percentage")
    parser.add_argument("--objective", type=str, default="sharpe_ratio", 
                       choices=["sharpe_ratio", "sortino_ratio", "total_return", "profit_factor"])
    parser.add_argument("--tag", type=str, default="", help="Run identifier")
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    results = run_walk_forward(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        n_windows=args.windows,
        train_pct=args.train_pct,
        optimization_objective=args.objective,
        tag=args.tag,
    )
    
    print_walk_forward_results(results)
