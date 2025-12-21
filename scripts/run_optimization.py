#!/usr/bin/env python3
"""
Strategy Optimization CLI - Tune EliteTradeAgent Parameters

This script provides a command-line interface for running strategy
parameter optimization with walk-forward validation.

Usage:
    # Optimize default parameters (confidence, stop/target multipliers)
    python scripts/run_optimization.py
    
    # Optimize specific parameters
    python scripts/run_optimization.py --params kelly_fraction,max_position_pct
    
    # Optimize by category
    python scripts/run_optimization.py --category position_sizing
    
    # Quick optimization without walk-forward
    python scripts/run_optimization.py --no-walk-forward
    
    # Custom date range and symbols
    python scripts/run_optimization.py --symbols SPY,QQQ --start 2023-01-01

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    )


def list_parameters():
    """List all available parameters for optimization."""
    from backtesting.strategy_optimizer import StrategyOptimizer, ParameterCategory
    
    optimizer = StrategyOptimizer()
    
    print("\n" + "=" * 70)
    print("AVAILABLE PARAMETERS FOR OPTIMIZATION")
    print("=" * 70)
    
    for category in ParameterCategory:
        params = optimizer.get_parameter_info(category)
        if params:
            print(f"\n{category.value.upper()}")
            print("-" * 50)
            for name, info in params.items():
                print(f"  {name}")
                print(f"    Range: [{info['min']}, {info['max']}] (step: {info['step']})")
                print(f"    Default: {info['default']}")
                print(f"    {info['description']}")
    
    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Strategy Parameter Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default optimization
    python scripts/run_optimization.py
    
    # Optimize position sizing
    python scripts/run_optimization.py --category position_sizing
    
    # Optimize specific parameters
    python scripts/run_optimization.py --params kelly_fraction,atr_stop_mult
    
    # Quick test (no walk-forward)
    python scripts/run_optimization.py --no-walk-forward
        """
    )
    
    parser.add_argument(
        "--symbols", "-s",
        type=str,
        default="SPY",
        help="Comma-separated symbols to backtest (default: SPY)"
    )
    
    parser.add_argument(
        "--params", "-p",
        type=str,
        default=None,
        help="Comma-separated parameter names to optimize"
    )
    
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        choices=["position_sizing", "risk_management", "signal_thresholds", "iv_settings", "dte_settings"],
        help="Parameter category to optimize"
    )
    
    parser.add_argument(
        "--objective", "-o",
        type=str,
        default="sharpe_ratio",
        choices=["sharpe_ratio", "sortino_ratio", "total_return", "profit_factor", "win_rate"],
        help="Optimization objective (default: sharpe_ratio)"
    )
    
    parser.add_argument(
        "--method", "-m",
        type=str,
        default="grid_search",
        choices=["grid_search", "random_search"],
        help="Optimization method (default: grid_search)"
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default="2022-01-01",
        help="Backtest start date (default: 2022-01-01)"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-01",
        help="Backtest end date (default: 2024-12-01)"
    )
    
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital (default: 100000)"
    )
    
    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Disable walk-forward validation (faster but may overfit)"
    )
    
    parser.add_argument(
        "--windows", "-w",
        type=int,
        default=5,
        help="Number of walk-forward windows (default: 5)"
    )
    
    parser.add_argument(
        "--train-pct",
        type=float,
        default=0.70,
        help="Training percentage per window (default: 0.70)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Max iterations for random search (default: 100)"
    )
    
    parser.add_argument(
        "--tag", "-t",
        type=str,
        default="",
        help="Run identifier tag"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/optimizations",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="List available parameters and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Handle list params
    if args.list_params:
        list_parameters()
        return
    
    # Parse symbols and parameters
    symbols = [s.strip() for s in args.symbols.split(",")]
    params = [p.strip() for p in args.params.split(",")] if args.params else None
    
    # Print banner
    print("\n" + "=" * 70)
    print("🔧 STRATEGY PARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Date Range: {args.start} to {args.end}")
    print(f"Objective: {args.objective}")
    print(f"Method: {args.method}")
    print(f"Walk-Forward: {'Disabled' if args.no_walk_forward else f'{args.windows} windows'}")
    
    if params:
        print(f"Parameters: {', '.join(params)}")
    elif args.category:
        print(f"Category: {args.category}")
    else:
        print("Parameters: Default (min_confidence, atr_stop_mult, atr_target_mult)")
    
    print("=" * 70 + "\n")
    
    # Import optimizer
    from backtesting.strategy_optimizer import (
        StrategyOptimizer,
        ParameterCategory,
        OptimizationMethod,
        OptimizationObjective,
        print_optimization_results,
    )
    
    # Create optimizer
    optimizer = StrategyOptimizer(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        optimization_objective=OptimizationObjective(args.objective),
        use_walk_forward=not args.no_walk_forward,
        n_windows=args.windows,
        train_pct=args.train_pct,
    )
    
    # Determine categories
    categories = None
    if args.category:
        categories = [ParameterCategory(args.category)]
    
    # Run optimization
    try:
        results = optimizer.optimize_parameters(
            parameters=params,
            categories=categories,
            method=OptimizationMethod(args.method),
            max_iterations=args.max_iterations,
            save_results=True,
            output_dir=args.output_dir,
            tag=args.tag,
        )
        
        # Print results
        print_optimization_results(results)
        
        # Print recommended config
        print("\n📋 RECOMMENDED CONFIGURATION")
        print("-" * 40)
        print("Add these to your EliteTradeAgent config:")
        print()
        print("config = {")
        for name, value in results.best_parameters.items():
            print(f'    "{name}": {value:.4f},')
        print("}")
        print()
        
        logger.info("Optimization complete!")
        
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise


if __name__ == "__main__":
    main()
