"""
Strategy Parameter Optimizer - Institutional-Grade Parameter Tuning

This module provides comprehensive parameter optimization for EliteTradeAgent:
- Kelly fraction and position sizing optimization
- Stop loss/take profit multiplier tuning
- IV threshold boundary optimization
- DTE preferences optimization
- Walk-forward validated optimization
- Multi-objective optimization support

Key Features:
- Uses walk-forward validation to prevent overfitting
- Supports grid search, random search, and Bayesian optimization
- Provides parameter stability analysis
- Generates actionable recommendations

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

import numpy as np
import pandas as pd
from loguru import logger

from backtesting.elite_backtest_engine import (
    EliteBacktestConfig,
    EliteBacktestResults,
    EliteBacktestEngine,
)
from backtesting.walk_forward_engine import (
    WalkForwardEngine,
    WalkForwardConfig,
    WalkForwardResults,
    ParameterRange,
    OptimizationMethod,
    OptimizationObjective,
    print_walk_forward_results,
)


class ParameterCategory(str, Enum):
    """Categories of parameters for optimization."""
    POSITION_SIZING = "position_sizing"
    RISK_MANAGEMENT = "risk_management"
    SIGNAL_THRESHOLDS = "signal_thresholds"
    IV_SETTINGS = "iv_settings"
    DTE_SETTINGS = "dte_settings"
    STRATEGY_SELECTION = "strategy_selection"


@dataclass
class OptimizableParameter:
    """Definition of an optimizable parameter with metadata."""
    name: str
    category: ParameterCategory
    min_value: float
    max_value: float
    default_value: float
    step: Optional[float] = None
    description: str = ""
    unit: str = ""
    
    # Constraints
    integer_only: bool = False
    dependent_on: Optional[str] = None  # Parameter this depends on
    
    def to_range(self) -> ParameterRange:
        """Convert to ParameterRange for walk-forward engine."""
        return ParameterRange(
            name=self.name,
            min_value=self.min_value,
            max_value=self.max_value,
            step=self.step,
        )
    
    def validate(self, value: float) -> float:
        """Validate and constrain value."""
        value = max(self.min_value, min(self.max_value, value))
        if self.integer_only:
            value = int(round(value))
        return value


@dataclass
class OptimizationResult:
    """Result from a single optimization run."""
    parameters: Dict[str, float]
    objective_value: float
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    
    # Walk-forward specific
    oos_return: float = 0.0  # Out-of-sample return
    overfitting_ratio: float = 0.0
    parameter_stability: float = 0.0


@dataclass
class StrategyOptimizationResults:
    """Complete results from strategy optimization."""
    
    # Configuration
    optimization_objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO
    parameters_optimized: List[str] = field(default_factory=list)
    total_combinations: int = 0
    
    # Best parameters
    best_parameters: Dict[str, float] = field(default_factory=dict)
    best_objective_value: float = 0.0
    
    # Walk-forward results (if applicable)
    walk_forward_results: Optional[WalkForwardResults] = None
    
    # Optimization history
    all_results: List[OptimizationResult] = field(default_factory=list)
    
    # Analysis
    parameter_sensitivity: Dict[str, float] = field(default_factory=dict)  # How sensitive is objective to each param
    parameter_correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    optimal_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # Recommended ranges
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    start_time: datetime = None
    end_time: datetime = None
    duration_seconds: float = 0.0


class StrategyOptimizer:
    """
    Comprehensive strategy parameter optimizer.
    
    Optimizes EliteTradeAgent parameters using walk-forward validation
    to prevent overfitting and ensure robust parameter selection.
    """
    
    # ==========================================================================
    # PREDEFINED PARAMETER SETS
    # ==========================================================================
    
    POSITION_SIZING_PARAMS: Dict[str, OptimizableParameter] = {
        "kelly_fraction": OptimizableParameter(
            name="kelly_fraction",
            category=ParameterCategory.POSITION_SIZING,
            min_value=0.10,
            max_value=0.50,
            default_value=0.25,
            step=0.05,
            description="Fraction of Kelly Criterion to use",
            unit="%",
        ),
        "max_position_pct": OptimizableParameter(
            name="max_position_pct",
            category=ParameterCategory.POSITION_SIZING,
            min_value=0.01,
            max_value=0.10,
            default_value=0.04,
            step=0.01,
            description="Maximum position size as % of portfolio",
            unit="%",
        ),
        "max_portfolio_heat": OptimizableParameter(
            name="max_portfolio_heat",
            category=ParameterCategory.POSITION_SIZING,
            min_value=0.10,
            max_value=0.40,
            default_value=0.20,
            step=0.05,
            description="Maximum total portfolio risk exposure",
            unit="%",
        ),
    }
    
    RISK_MANAGEMENT_PARAMS: Dict[str, OptimizableParameter] = {
        "atr_stop_mult": OptimizableParameter(
            name="atr_stop_mult",
            category=ParameterCategory.RISK_MANAGEMENT,
            min_value=1.0,
            max_value=4.0,
            default_value=2.0,
            step=0.5,
            description="ATR multiplier for stop loss",
            unit="x",
        ),
        "atr_target_mult": OptimizableParameter(
            name="atr_target_mult",
            category=ParameterCategory.RISK_MANAGEMENT,
            min_value=1.5,
            max_value=5.0,
            default_value=3.0,
            step=0.5,
            description="ATR multiplier for profit target",
            unit="x",
        ),
        "min_reward_risk": OptimizableParameter(
            name="min_reward_risk",
            category=ParameterCategory.RISK_MANAGEMENT,
            min_value=1.0,
            max_value=3.0,
            default_value=1.5,
            step=0.25,
            description="Minimum reward:risk ratio required",
            unit=":1",
        ),
        "max_loss_pct": OptimizableParameter(
            name="max_loss_pct",
            category=ParameterCategory.RISK_MANAGEMENT,
            min_value=0.005,
            max_value=0.05,
            default_value=0.02,
            step=0.005,
            description="Maximum loss per trade as % of portfolio",
            unit="%",
        ),
    }
    
    SIGNAL_THRESHOLD_PARAMS: Dict[str, OptimizableParameter] = {
        "min_confidence": OptimizableParameter(
            name="min_confidence",
            category=ParameterCategory.SIGNAL_THRESHOLDS,
            min_value=0.20,
            max_value=0.70,
            default_value=0.30,
            step=0.05,
            description="Minimum signal confidence to trade",
            unit="%",
        ),
        "min_mtf_alignment": OptimizableParameter(
            name="min_mtf_alignment",
            category=ParameterCategory.SIGNAL_THRESHOLDS,
            min_value=0.25,
            max_value=0.75,
            default_value=0.40,
            step=0.05,
            description="Minimum multi-timeframe alignment",
            unit="%",
        ),
    }
    
    IV_SETTING_PARAMS: Dict[str, OptimizableParameter] = {
        "high_iv_threshold": OptimizableParameter(
            name="high_iv_threshold",
            category=ParameterCategory.IV_SETTINGS,
            min_value=40.0,
            max_value=70.0,
            default_value=50.0,
            step=5.0,
            description="IV rank threshold for 'high IV' environment",
            unit="rank",
        ),
        "low_iv_threshold": OptimizableParameter(
            name="low_iv_threshold",
            category=ParameterCategory.IV_SETTINGS,
            min_value=15.0,
            max_value=40.0,
            default_value=30.0,
            step=5.0,
            description="IV rank threshold for 'low IV' environment",
            unit="rank",
        ),
    }
    
    DTE_SETTING_PARAMS: Dict[str, OptimizableParameter] = {
        "preferred_dte_min": OptimizableParameter(
            name="preferred_dte_min",
            category=ParameterCategory.DTE_SETTINGS,
            min_value=3,
            max_value=21,
            default_value=7,
            step=1,
            description="Minimum preferred DTE for options",
            unit="days",
            integer_only=True,
        ),
        "preferred_dte_max": OptimizableParameter(
            name="preferred_dte_max",
            category=ParameterCategory.DTE_SETTINGS,
            min_value=21,
            max_value=90,
            default_value=45,
            step=7,
            description="Maximum preferred DTE for options",
            unit="days",
            integer_only=True,
        ),
    }
    
    def __init__(
        self,
        symbols: List[str] = None,
        start_date: str = "2022-01-01",
        end_date: str = "2024-12-01",
        initial_capital: float = 100_000.0,
        optimization_objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO,
        use_walk_forward: bool = True,
        n_windows: int = 5,
        train_pct: float = 0.70,
    ):
        """Initialize the Strategy Optimizer.
        
        Args:
            symbols: Symbols to backtest
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            optimization_objective: What metric to optimize
            use_walk_forward: Use walk-forward validation
            n_windows: Number of walk-forward windows
            train_pct: Training percentage per window
        """
        self.symbols = symbols or ["SPY"]
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.optimization_objective = optimization_objective
        self.use_walk_forward = use_walk_forward
        self.n_windows = n_windows
        self.train_pct = train_pct
        
        # Collect all available parameters
        self.all_parameters: Dict[str, OptimizableParameter] = {}
        self.all_parameters.update(self.POSITION_SIZING_PARAMS)
        self.all_parameters.update(self.RISK_MANAGEMENT_PARAMS)
        self.all_parameters.update(self.SIGNAL_THRESHOLD_PARAMS)
        self.all_parameters.update(self.IV_SETTING_PARAMS)
        self.all_parameters.update(self.DTE_SETTING_PARAMS)
        
        logger.info(
            f"StrategyOptimizer initialized | "
            f"symbols={self.symbols} | "
            f"objective={optimization_objective.value} | "
            f"walk_forward={use_walk_forward}"
        )
    
    def get_parameter_info(self, category: Optional[ParameterCategory] = None) -> Dict[str, Dict]:
        """Get information about optimizable parameters.
        
        Args:
            category: Filter by category, or None for all
            
        Returns:
            Dictionary with parameter info
        """
        result = {}
        for name, param in self.all_parameters.items():
            if category is None or param.category == category:
                result[name] = {
                    "category": param.category.value,
                    "min": param.min_value,
                    "max": param.max_value,
                    "default": param.default_value,
                    "step": param.step,
                    "description": param.description,
                    "unit": param.unit,
                }
        return result
    
    def optimize_parameters(
        self,
        parameters: List[str] = None,
        categories: List[ParameterCategory] = None,
        method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
        max_iterations: int = 100,
        save_results: bool = True,
        output_dir: str = "runs/optimizations",
        tag: str = "",
    ) -> StrategyOptimizationResults:
        """Run parameter optimization.
        
        Args:
            parameters: List of parameter names to optimize (or use categories)
            categories: Parameter categories to optimize
            method: Optimization method
            max_iterations: Max iterations for random/Bayesian
            save_results: Save results to disk
            output_dir: Output directory
            tag: Run identifier
            
        Returns:
            StrategyOptimizationResults with best parameters and analysis
        """
        start_time = datetime.now()
        
        # Determine which parameters to optimize
        params_to_optimize = self._get_parameters_to_optimize(parameters, categories)
        
        if not params_to_optimize:
            raise ValueError("No parameters specified for optimization")
        
        logger.info(f"Optimizing {len(params_to_optimize)} parameters: {list(params_to_optimize.keys())}")
        
        # Create parameter ranges
        param_ranges = [p.to_range() for p in params_to_optimize.values()]
        
        # Run optimization
        if self.use_walk_forward:
            results = self._optimize_with_walk_forward(
                param_ranges=param_ranges,
                method=method,
                max_iterations=max_iterations,
            )
        else:
            results = self._optimize_direct(
                param_ranges=param_ranges,
                method=method,
                max_iterations=max_iterations,
            )
        
        # Add metadata
        results.parameters_optimized = list(params_to_optimize.keys())
        results.optimization_objective = self.optimization_objective
        results.start_time = start_time
        results.end_time = datetime.now()
        results.duration_seconds = (results.end_time - start_time).total_seconds()
        
        # Analyze results
        self._analyze_results(results, params_to_optimize)
        
        # Generate recommendations
        self._generate_recommendations(results, params_to_optimize)
        
        # Save results
        if save_results:
            self._save_results(results, output_dir, tag)
        
        return results
    
    def optimize_position_sizing(self, **kwargs) -> StrategyOptimizationResults:
        """Convenience method to optimize position sizing parameters."""
        return self.optimize_parameters(
            categories=[ParameterCategory.POSITION_SIZING],
            **kwargs
        )
    
    def optimize_risk_management(self, **kwargs) -> StrategyOptimizationResults:
        """Convenience method to optimize risk management parameters."""
        return self.optimize_parameters(
            categories=[ParameterCategory.RISK_MANAGEMENT],
            **kwargs
        )
    
    def optimize_signal_thresholds(self, **kwargs) -> StrategyOptimizationResults:
        """Convenience method to optimize signal threshold parameters."""
        return self.optimize_parameters(
            categories=[ParameterCategory.SIGNAL_THRESHOLDS],
            **kwargs
        )
    
    def optimize_all(self, **kwargs) -> StrategyOptimizationResults:
        """Optimize all parameter categories."""
        return self.optimize_parameters(
            categories=[
                ParameterCategory.POSITION_SIZING,
                ParameterCategory.RISK_MANAGEMENT,
                ParameterCategory.SIGNAL_THRESHOLDS,
            ],
            **kwargs
        )
    
    def _get_parameters_to_optimize(
        self,
        parameters: Optional[List[str]],
        categories: Optional[List[ParameterCategory]],
    ) -> Dict[str, OptimizableParameter]:
        """Get parameters to optimize based on input."""
        result = {}
        
        if parameters:
            for name in parameters:
                if name in self.all_parameters:
                    result[name] = self.all_parameters[name]
                else:
                    logger.warning(f"Unknown parameter: {name}")
        
        if categories:
            for name, param in self.all_parameters.items():
                if param.category in categories and name not in result:
                    result[name] = param
        
        # Default: optimize key parameters
        if not result:
            result = {
                "min_confidence": self.all_parameters["min_confidence"],
                "atr_stop_mult": self.all_parameters["atr_stop_mult"],
                "atr_target_mult": self.all_parameters["atr_target_mult"],
            }
        
        return result
    
    def _optimize_with_walk_forward(
        self,
        param_ranges: List[ParameterRange],
        method: OptimizationMethod,
        max_iterations: int,
    ) -> StrategyOptimizationResults:
        """Run optimization with walk-forward validation."""
        
        logger.info("Running walk-forward optimization...")
        
        # Create base backtest config
        base_config = EliteBacktestConfig(
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            use_agent_signals=False,  # Use simple signals for speed
            disable_event_risk=True,
            monte_carlo_runs=0,  # Skip MC for speed
        )
        
        # Create walk-forward config
        wf_config = WalkForwardConfig(
            base_config=base_config,
            n_windows=self.n_windows,
            train_pct=self.train_pct,
            optimization_method=method,
            optimization_objective=self.optimization_objective,
            parameter_ranges=param_ranges,
            max_optimization_iterations=max_iterations,
            save_results=False,  # We'll save our own results
        )
        
        # Run walk-forward optimization
        wf_engine = WalkForwardEngine(wf_config)
        wf_results = wf_engine.run()
        
        # Build optimization results
        results = StrategyOptimizationResults()
        results.walk_forward_results = wf_results
        
        # Collect best parameters across windows
        param_values = {p.name: [] for p in param_ranges}
        for window in wf_results.windows:
            if window.best_params:
                for name, value in window.best_params.items():
                    if name in param_values:
                        param_values[name].append(value)
        
        # Average best parameters (with option to use last window)
        results.best_parameters = {}
        for name, values in param_values.items():
            if values:
                results.best_parameters[name] = np.mean(values)
        
        results.best_objective_value = wf_results.oos_avg_sharpe
        results.total_combinations = sum(
            len(w.optimization_history) for w in wf_results.windows
        )
        
        # Collect all results for analysis
        for window in wf_results.windows:
            for hist_entry in window.optimization_history:
                opt_result = OptimizationResult(
                    parameters=hist_entry["params"],
                    objective_value=hist_entry["score"],
                    total_return=hist_entry.get("return", 0),
                    sharpe_ratio=hist_entry.get("score", 0) if self.optimization_objective == OptimizationObjective.SHARPE_RATIO else 0,
                    sortino_ratio=0,
                    max_drawdown=0,
                    win_rate=0,
                    profit_factor=0,
                    total_trades=hist_entry.get("trades", 0),
                )
                results.all_results.append(opt_result)
        
        return results
    
    def _optimize_direct(
        self,
        param_ranges: List[ParameterRange],
        method: OptimizationMethod,
        max_iterations: int,
    ) -> StrategyOptimizationResults:
        """Run direct optimization without walk-forward (faster, but may overfit)."""
        
        logger.info("Running direct optimization (no walk-forward)...")
        results = StrategyOptimizationResults()
        
        best_params = {}
        best_score = -999.0
        all_results = []
        
        # Generate parameter combinations
        if method == OptimizationMethod.GRID_SEARCH:
            param_names = [p.name for p in param_ranges]
            param_values = [p.grid_values() for p in param_ranges]
            combinations = list(itertools.product(*param_values))
            
            logger.info(f"Grid search: {len(combinations)} combinations")
            
            for combo in combinations:
                params = dict(zip(param_names, combo))
                score, metrics = self._evaluate_parameters(params)
                
                opt_result = OptimizationResult(
                    parameters=params.copy(),
                    objective_value=score,
                    **metrics
                )
                all_results.append(opt_result)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
        
        elif method == OptimizationMethod.RANDOM_SEARCH:
            logger.info(f"Random search: {max_iterations} iterations")
            
            for i in range(max_iterations):
                params = {}
                for p in param_ranges:
                    params[p.name] = np.random.uniform(p.min_value, p.max_value)
                
                score, metrics = self._evaluate_parameters(params)
                
                opt_result = OptimizationResult(
                    parameters=params.copy(),
                    objective_value=score,
                    **metrics
                )
                all_results.append(opt_result)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  Iteration {i+1}/{max_iterations} | Best: {best_score:.3f}")
        
        results.best_parameters = best_params
        results.best_objective_value = best_score
        results.all_results = all_results
        results.total_combinations = len(all_results)
        
        return results
    
    def _evaluate_parameters(self, params: Dict[str, float]) -> Tuple[float, Dict]:
        """Evaluate a parameter set with backtesting."""
        
        # Create config with params
        config_dict = {
            "symbols": self.symbols,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "use_agent_signals": False,
            "disable_event_risk": True,
            "monte_carlo_runs": 0,
            "save_trades": False,
            "save_equity_curve": False,
        }
        
        # Apply optimization params
        for name, value in params.items():
            if name in ["high_iv_threshold", "low_iv_threshold"]:
                # These are custom params not in EliteBacktestConfig
                continue
            config_dict[name] = value
        
        config = EliteBacktestConfig(**config_dict)
        engine = EliteBacktestEngine(config)
        
        try:
            results = engine.run_backtest()
            
            # Get objective value
            if self.optimization_objective == OptimizationObjective.SHARPE_RATIO:
                score = results.sharpe_ratio if not np.isnan(results.sharpe_ratio) else -999.0
            elif self.optimization_objective == OptimizationObjective.SORTINO_RATIO:
                score = results.sortino_ratio if not np.isnan(results.sortino_ratio) else -999.0
            elif self.optimization_objective == OptimizationObjective.TOTAL_RETURN:
                score = results.total_return_pct
            elif self.optimization_objective == OptimizationObjective.PROFIT_FACTOR:
                score = results.profit_factor if results.profit_factor != float('inf') else 10.0
            elif self.optimization_objective == OptimizationObjective.WIN_RATE:
                score = results.win_rate
            else:
                score = results.sharpe_ratio
            
            metrics = {
                "total_return": results.total_return_pct,
                "sharpe_ratio": results.sharpe_ratio if not np.isnan(results.sharpe_ratio) else 0,
                "sortino_ratio": results.sortino_ratio if not np.isnan(results.sortino_ratio) else 0,
                "max_drawdown": results.max_drawdown_pct,
                "win_rate": results.win_rate,
                "profit_factor": results.profit_factor if results.profit_factor != float('inf') else 0,
                "total_trades": results.total_trades,
            }
            
            return score, metrics
            
        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            return -999.0, {
                "total_return": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_trades": 0,
            }
    
    def _analyze_results(
        self,
        results: StrategyOptimizationResults,
        params_optimized: Dict[str, OptimizableParameter],
    ):
        """Analyze optimization results for sensitivity and patterns."""
        
        if not results.all_results:
            return
        
        # Parameter sensitivity analysis
        for param_name in params_optimized.keys():
            param_values = [r.parameters.get(param_name, 0) for r in results.all_results]
            objective_values = [r.objective_value for r in results.all_results]
            
            if len(set(param_values)) > 1:
                # Calculate correlation between parameter and objective
                correlation = np.corrcoef(param_values, objective_values)[0, 1]
                results.parameter_sensitivity[param_name] = abs(correlation) if not np.isnan(correlation) else 0
        
        # Find optimal ranges (top 10% of results)
        sorted_results = sorted(results.all_results, key=lambda x: x.objective_value, reverse=True)
        top_results = sorted_results[:max(1, len(sorted_results) // 10)]
        
        for param_name in params_optimized.keys():
            values = [r.parameters.get(param_name, 0) for r in top_results]
            if values:
                results.optimal_ranges[param_name] = (min(values), max(values))
    
    def _generate_recommendations(
        self,
        results: StrategyOptimizationResults,
        params_optimized: Dict[str, OptimizableParameter],
    ):
        """Generate actionable recommendations from optimization results."""
        
        recommendations = []
        warnings = []
        
        # Check walk-forward overfitting
        if results.walk_forward_results:
            wf = results.walk_forward_results
            
            if wf.overfitting_ratio > 2.0:
                warnings.append(
                    f"HIGH OVERFITTING DETECTED: In-sample/out-of-sample ratio = {wf.overfitting_ratio:.2f}. "
                    "Consider using more conservative parameters or longer training periods."
                )
            elif wf.overfitting_ratio > 1.5:
                warnings.append(
                    f"Moderate overfitting: ratio = {wf.overfitting_ratio:.2f}. "
                    "Monitor live performance closely."
                )
            
            if wf.pct_profitable_windows < 0.5:
                warnings.append(
                    f"Less than 50% of walk-forward windows were profitable ({wf.pct_profitable_windows:.0%}). "
                    "Strategy may not be robust."
                )
            
            if wf.parameter_stability < 0.5:
                warnings.append(
                    f"Low parameter stability ({wf.parameter_stability:.0%}). "
                    "Optimal parameters vary significantly across windows."
                )
        
        # Parameter-specific recommendations
        for param_name, param in params_optimized.items():
            if param_name in results.best_parameters:
                best_value = results.best_parameters[param_name]
                
                # Check if at boundary
                if best_value <= param.min_value * 1.1:
                    recommendations.append(
                        f"{param_name}: Optimal value ({best_value:.3f}) is near minimum. "
                        "Consider extending the search range lower."
                    )
                elif best_value >= param.max_value * 0.9:
                    recommendations.append(
                        f"{param_name}: Optimal value ({best_value:.3f}) is near maximum. "
                        "Consider extending the search range higher."
                    )
                
                # Check sensitivity
                if param_name in results.parameter_sensitivity:
                    sensitivity = results.parameter_sensitivity[param_name]
                    if sensitivity > 0.7:
                        recommendations.append(
                            f"{param_name}: High sensitivity ({sensitivity:.2f}). "
                            "Small changes significantly impact performance."
                        )
        
        # General recommendations
        if results.total_combinations < 50:
            recommendations.append(
                "Consider increasing the number of parameter combinations tested for more robust results."
            )
        
        if results.best_objective_value < 0.5 and self.optimization_objective == OptimizationObjective.SHARPE_RATIO:
            warnings.append(
                f"Best Sharpe ratio ({results.best_objective_value:.2f}) is below 0.5. "
                "Strategy may not be viable for live trading."
            )
        elif results.best_objective_value >= 1.0:
            recommendations.append(
                f"Strong risk-adjusted returns (Sharpe {results.best_objective_value:.2f}). "
                "Verify with additional out-of-sample testing before live deployment."
            )
        
        results.recommendations = recommendations
        results.warnings = warnings
    
    def _save_results(
        self,
        results: StrategyOptimizationResults,
        output_dir: str,
        tag: str,
    ):
        """Save optimization results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        tag = tag or f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Summary JSON
        summary = {
            "tag": tag,
            "symbols": self.symbols,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "objective": self.optimization_objective.value,
            "use_walk_forward": self.use_walk_forward,
            "parameters_optimized": results.parameters_optimized,
            "total_combinations": results.total_combinations,
            "best_parameters": results.best_parameters,
            "best_objective_value": results.best_objective_value,
            "parameter_sensitivity": results.parameter_sensitivity,
            "optimal_ranges": {k: list(v) for k, v in results.optimal_ranges.items()},
            "recommendations": results.recommendations,
            "warnings": results.warnings,
            "duration_seconds": results.duration_seconds,
        }
        
        if results.walk_forward_results:
            summary["walk_forward"] = {
                "n_windows": len(results.walk_forward_results.windows),
                "oos_sharpe": results.walk_forward_results.oos_avg_sharpe,
                "is_sharpe": results.walk_forward_results.is_avg_sharpe,
                "overfitting_ratio": results.walk_forward_results.overfitting_ratio,
                "pct_profitable_windows": results.walk_forward_results.pct_profitable_windows,
                "parameter_stability": results.walk_forward_results.parameter_stability,
            }
        
        summary_path = output_path / f"{tag}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {summary_path}")


def print_optimization_results(results: StrategyOptimizationResults):
    """Print formatted optimization results."""
    print("\n" + "=" * 70)
    print("STRATEGY PARAMETER OPTIMIZATION RESULTS")
    print("=" * 70)
    
    print(f"Objective: {results.optimization_objective.value}")
    print(f"Parameters Optimized: {', '.join(results.parameters_optimized)}")
    print(f"Total Combinations Tested: {results.total_combinations}")
    print(f"Duration: {results.duration_seconds:.1f} seconds")
    
    print("-" * 70)
    print("BEST PARAMETERS")
    for name, value in results.best_parameters.items():
        print(f"  {name}: {value:.4f}")
    print(f"\nBest {results.optimization_objective.value}: {results.best_objective_value:.4f}")
    
    if results.parameter_sensitivity:
        print("-" * 70)
        print("PARAMETER SENSITIVITY (correlation with objective)")
        for name, sensitivity in sorted(results.parameter_sensitivity.items(), key=lambda x: -x[1]):
            bar = "█" * int(sensitivity * 20)
            print(f"  {name}: {sensitivity:.2f} {bar}")
    
    if results.optimal_ranges:
        print("-" * 70)
        print("OPTIMAL RANGES (top 10% performers)")
        for name, (low, high) in results.optimal_ranges.items():
            print(f"  {name}: [{low:.4f}, {high:.4f}]")
    
    if results.walk_forward_results:
        print("-" * 70)
        print("WALK-FORWARD VALIDATION")
        wf = results.walk_forward_results
        print(f"  Out-of-Sample Sharpe: {wf.oos_avg_sharpe:.2f}")
        print(f"  In-Sample Sharpe: {wf.is_avg_sharpe:.2f}")
        print(f"  Overfitting Ratio: {wf.overfitting_ratio:.2f} {'⚠️' if wf.overfitting_ratio > 1.5 else '✓'}")
        print(f"  Profitable Windows: {wf.pct_profitable_windows:.0%}")
        print(f"  Parameter Stability: {wf.parameter_stability:.0%}")
    
    if results.recommendations:
        print("-" * 70)
        print("📋 RECOMMENDATIONS")
        for rec in results.recommendations:
            print(f"  • {rec}")
    
    if results.warnings:
        print("-" * 70)
        print("⚠️  WARNINGS")
        for warn in results.warnings:
            print(f"  • {warn}")
    
    print("=" * 70)


def optimize_strategy(
    symbols: List[str] = None,
    parameters: List[str] = None,
    objective: str = "sharpe_ratio",
    method: str = "grid_search",
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-01",
    use_walk_forward: bool = True,
    n_windows: int = 5,
    tag: str = "",
) -> StrategyOptimizationResults:
    """Convenience function to run strategy optimization.
    
    Args:
        symbols: Symbols to backtest
        parameters: Parameter names to optimize (or None for defaults)
        objective: Optimization objective
        method: Optimization method (grid_search, random_search)
        start_date: Backtest start
        end_date: Backtest end
        use_walk_forward: Use walk-forward validation
        n_windows: Walk-forward windows
        tag: Run identifier
        
    Returns:
        StrategyOptimizationResults
    """
    optimizer = StrategyOptimizer(
        symbols=symbols or ["SPY"],
        start_date=start_date,
        end_date=end_date,
        optimization_objective=OptimizationObjective(objective),
        use_walk_forward=use_walk_forward,
        n_windows=n_windows,
    )
    
    return optimizer.optimize_parameters(
        parameters=parameters,
        method=OptimizationMethod(method),
        tag=tag,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize Strategy Parameters")
    parser.add_argument("--symbols", type=str, default="SPY", help="Comma-separated symbols")
    parser.add_argument("--params", type=str, default=None, 
                       help="Comma-separated parameter names (or 'all')")
    parser.add_argument("--objective", type=str, default="sharpe_ratio",
                       choices=["sharpe_ratio", "sortino_ratio", "total_return", "profit_factor", "win_rate"])
    parser.add_argument("--method", type=str, default="grid_search",
                       choices=["grid_search", "random_search"])
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-12-01", help="End date")
    parser.add_argument("--no-walk-forward", action="store_true", help="Disable walk-forward")
    parser.add_argument("--windows", type=int, default=5, help="Walk-forward windows")
    parser.add_argument("--tag", type=str, default="", help="Run identifier")
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(",")]
    params = args.params.split(",") if args.params and args.params != "all" else None
    
    results = optimize_strategy(
        symbols=symbols,
        parameters=params,
        objective=args.objective,
        method=args.method,
        start_date=args.start,
        end_date=args.end,
        use_walk_forward=not args.no_walk_forward,
        n_windows=args.windows,
        tag=args.tag,
    )
    
    print_optimization_results(results)
