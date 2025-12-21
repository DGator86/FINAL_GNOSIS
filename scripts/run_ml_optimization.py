#!/usr/bin/env python3
"""
ML Optimization CLI Script

Run hyperparameter optimization for the ML trading pipeline.

Usage:
    python scripts/run_ml_optimization.py --preset balanced --n-trials 100
    python scripts/run_ml_optimization.py --method bayesian --stage signals
    python scripts/run_ml_optimization.py --optimize-all --output results.json

Author: Super Gnosis Elite Trading System
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from ml import (
    MLHyperparameterManager,
    MLOptimizationEngine,
    OptimizationConfig,
    OptimizationStage,
    OptimizationMetric,
    MarketRegime,
    create_preset_config,
    create_optimization_engine,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ML Hyperparameter Optimization for Trading Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick optimization with defaults
    python scripts/run_ml_optimization.py
    
    # Full Bayesian optimization
    python scripts/run_ml_optimization.py --method bayesian --n-trials 200
    
    # Optimize specific stages
    python scripts/run_ml_optimization.py --stage signals position
    
    # Use aggressive preset and save results
    python scripts/run_ml_optimization.py --preset aggressive --output results.json
    
    # Optimize for specific metric
    python scripts/run_ml_optimization.py --metric sortino_ratio
    
    # Regime-specific optimization
    python scripts/run_ml_optimization.py --per-regime --regimes high_volatility crisis
        """,
    )
    
    # Preset configuration
    parser.add_argument(
        "--preset",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="Base parameter preset (default: balanced)",
    )
    
    # Optimization method
    parser.add_argument(
        "--method",
        choices=["bayesian", "grid", "random"],
        default="bayesian",
        help="Optimization method (default: bayesian)",
    )
    
    # Number of trials
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of optimization trials (default: 100)",
    )
    
    # Timeout
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Optimization timeout in seconds (default: 3600)",
    )
    
    # Stages to optimize
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=["model", "features", "signals", "position", "risk", "strategy", "full"],
        default=["full"],
        help="Stages to optimize (default: full)",
    )
    
    # Optimization metric
    parser.add_argument(
        "--metric",
        choices=[
            "sharpe_ratio",
            "sortino_ratio",
            "total_return",
            "risk_adjusted_return",
            "profit_factor",
            "win_rate",
        ],
        default="sharpe_ratio",
        help="Primary optimization metric (default: sharpe_ratio)",
    )
    
    # Walk-forward validation
    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Disable walk-forward validation",
    )
    
    parser.add_argument(
        "--wf-windows",
        type=int,
        default=5,
        help="Number of walk-forward windows (default: 5)",
    )
    
    # Regime-specific optimization
    parser.add_argument(
        "--per-regime",
        action="store_true",
        help="Optimize parameters for each market regime",
    )
    
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=[
            "trending_bull",
            "trending_bear",
            "range_bound",
            "high_volatility",
            "low_volatility",
            "crisis",
        ],
        help="Specific regimes to optimize for (default: all)",
    )
    
    # Constraints
    parser.add_argument(
        "--min-trades",
        type=int,
        default=30,
        help="Minimum trades for valid strategy (default: 30)",
    )
    
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.25,
        help="Maximum allowed drawdown (default: 0.25)",
    )
    
    # Output
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file for results (JSON)",
    )
    
    parser.add_argument(
        "--save-params",
        type=str,
        help="Save optimized parameters to config directory with this name",
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-essential output",
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool, quiet: bool):
    """Configure logging."""
    if quiet:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    elif verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")


def print_banner():
    """Print application banner."""
    print("\n" + "=" * 60)
    print("       ML HYPERPARAMETER OPTIMIZATION")
    print("       Super Gnosis Elite Trading System")
    print("=" * 60 + "\n")


def print_config(config: OptimizationConfig, preset: str):
    """Print optimization configuration."""
    print("Configuration:")
    print(f"  Preset:       {preset}")
    print(f"  Method:       {config.method}")
    print(f"  Trials:       {config.n_trials}")
    print(f"  Timeout:      {config.timeout_seconds}s")
    print(f"  Stages:       {[s.value for s in config.stages]}")
    print(f"  Metric:       {config.primary_metric.value}")
    print(f"  Walk-Forward: {config.use_walk_forward}")
    if config.use_walk_forward:
        print(f"  WF Windows:   {config.n_walk_forward_windows}")
    print(f"  Per-Regime:   {config.optimize_per_regime}")
    print()


def print_results(result):
    """Print optimization results."""
    print("\n" + "=" * 60)
    print("                   RESULTS")
    print("=" * 60 + "\n")
    
    print(f"Best Score:          {result.best_score:.4f}")
    print(f"Duration:            {result.total_duration_seconds:.1f}s")
    
    if result.walk_forward_sharpe:
        print(f"\nWalk-Forward Results:")
        print(f"  OOS Sharpe:        {result.walk_forward_sharpe:.4f}")
        print(f"  OOS Return:        {result.walk_forward_return:.2%}")
        print(f"  Overfitting Ratio: {result.overfitting_ratio:.2%}")
    
    if result.stage_results:
        print(f"\nStage Results:")
        for stage, sr in result.stage_results.items():
            print(f"  {stage.value}:")
            print(f"    Best Score: {sr.best_score:.4f}")
            print(f"    Trials:     {sr.n_trials}")
            print(f"    Duration:   {sr.duration_seconds:.1f}s")
    
    if result.global_importance:
        print(f"\nTop Important Parameters:")
        sorted_imp = sorted(
            result.global_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        for param, imp in sorted_imp:
            print(f"  {param}: {imp:.3f}")
    
    if result.recommendations:
        print(f"\nRecommendations:")
        for rec in result.recommendations:
            print(f"  • {rec}")
    
    if result.warnings:
        print(f"\nWarnings:")
        for warn in result.warnings:
            print(f"  ⚠ {warn}")
    
    if result.best_params:
        print(f"\nBest Parameters (selected):")
        hp = result.best_params
        print(f"  LSTM hidden_dim:    {hp.lstm.hidden_dim}")
        print(f"  Signal confidence:  {hp.signals.min_confidence:.2f}")
        print(f"  Kelly fraction:     {hp.position_sizing.kelly_fraction:.2f}")
        print(f"  Stop loss ATR:      {hp.risk_management.stop_loss_atr_multiple:.1f}")
    
    print()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose, args.quiet)
    
    if not args.quiet:
        print_banner()
    
    # Map stage names to enums
    stage_map = {
        "model": OptimizationStage.MODEL,
        "features": OptimizationStage.FEATURES,
        "signals": OptimizationStage.SIGNALS,
        "position": OptimizationStage.POSITION,
        "risk": OptimizationStage.RISK,
        "strategy": OptimizationStage.STRATEGY,
        "full": OptimizationStage.FULL,
    }
    
    metric_map = {
        "sharpe_ratio": OptimizationMetric.SHARPE_RATIO,
        "sortino_ratio": OptimizationMetric.SORTINO_RATIO,
        "total_return": OptimizationMetric.TOTAL_RETURN,
        "risk_adjusted_return": OptimizationMetric.RISK_ADJUSTED_RETURN,
        "profit_factor": OptimizationMetric.PROFIT_FACTOR,
        "win_rate": OptimizationMetric.WIN_RATE,
    }
    
    regime_map = {
        "trending_bull": MarketRegime.TRENDING_BULL,
        "trending_bear": MarketRegime.TRENDING_BEAR,
        "range_bound": MarketRegime.RANGE_BOUND,
        "high_volatility": MarketRegime.HIGH_VOLATILITY,
        "low_volatility": MarketRegime.LOW_VOLATILITY,
        "crisis": MarketRegime.CRISIS,
    }
    
    # Build configuration
    stages = [stage_map[s] for s in args.stage]
    target_regimes = [regime_map[r] for r in args.regimes] if args.regimes else []
    
    config = OptimizationConfig(
        method=args.method,
        n_trials=args.n_trials,
        timeout_seconds=args.timeout,
        stages=stages,
        primary_metric=metric_map[args.metric],
        use_walk_forward=not args.no_walk_forward,
        n_walk_forward_windows=args.wf_windows,
        optimize_per_regime=args.per_regime,
        target_regimes=target_regimes,
        min_trades=args.min_trades,
        max_drawdown_limit=args.max_drawdown,
    )
    
    if not args.quiet:
        print_config(config, args.preset)
    
    # Create optimization engine
    print("Initializing optimization engine...")
    engine = create_optimization_engine(preset=args.preset)
    
    # Run optimization
    print(f"Starting optimization with {args.n_trials} trials...")
    start_time = datetime.now()
    
    try:
        result = engine.optimize(config)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)
    
    # Print results
    if not args.quiet:
        print_results(result)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        result.save(output_path)
        print(f"Results saved to: {output_path}")
    
    # Save optimized parameters
    if args.save_params and result.best_params:
        config_dir = Path("config/hyperparameters")
        config_dir.mkdir(parents=True, exist_ok=True)
        param_path = config_dir / f"{args.save_params}.json"
        result.best_params.save(param_path)
        print(f"Parameters saved to: {param_path}")
    
    # Summary
    print(f"\n✅ Optimization completed in {result.total_duration_seconds:.1f}s")
    print(f"   Best {args.metric}: {result.best_score:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
