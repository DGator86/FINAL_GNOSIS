#!/usr/bin/env python3
"""
Elite Backtest Runner - Command-line interface for backtesting

Usage:
    python scripts/run_elite_backtest.py --symbols SPY,NVDA,TSLA --start 2023-01-01 --end 2024-12-01
    python scripts/run_elite_backtest.py --quick  # Quick test with SPY
    python scripts/run_elite_backtest.py --preset momentum  # Use preset configuration

Presets:
    - quick: Fast test with SPY, 6 months
    - momentum: High-confidence momentum trades
    - conservative: Low-risk settings
    - aggressive: Higher risk tolerance
    - multi_asset: Diverse portfolio of 10 stocks

Author: Super Gnosis Elite Trading System
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


# ============================================================================
# PRESETS
# ============================================================================

PRESETS = {
    "quick": {
        "description": "Quick test with SPY",
        "symbols": ["SPY"],
        "start_date": (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "initial_capital": 100_000,
        "max_positions": 3,
        "min_confidence": 0.25,
    },
    "momentum": {
        "description": "High-confidence momentum trades",
        "symbols": ["SPY", "QQQ", "NVDA", "TSLA", "AMD"],
        "start_date": "2023-01-01",
        "end_date": "2024-12-01",
        "initial_capital": 100_000,
        "max_positions": 5,
        "min_confidence": 0.40,
        "min_mtf_alignment": 0.50,
        "atr_stop_mult": 1.5,
        "atr_target_mult": 3.5,
    },
    "conservative": {
        "description": "Low-risk, steady returns",
        "symbols": ["SPY", "QQQ", "IWM"],
        "start_date": "2022-01-01",
        "end_date": "2024-12-01",
        "initial_capital": 100_000,
        "max_positions": 2,
        "max_position_pct": 0.02,  # 2% per trade
        "min_confidence": 0.50,
        "min_reward_risk": 2.0,
        "atr_stop_mult": 1.5,
        "atr_target_mult": 2.5,
    },
    "aggressive": {
        "description": "Higher risk tolerance for larger gains",
        "symbols": ["NVDA", "TSLA", "AMD", "META", "MSFT", "GOOGL", "AMZN"],
        "start_date": "2023-01-01",
        "end_date": "2024-12-01",
        "initial_capital": 100_000,
        "max_positions": 7,
        "max_position_pct": 0.06,  # 6% per trade
        "max_portfolio_heat": 0.30,  # 30% total risk
        "min_confidence": 0.25,
        "kelly_fraction": 0.35,  # More aggressive Kelly
        "atr_stop_mult": 2.5,
        "atr_target_mult": 4.0,
    },
    "multi_asset": {
        "description": "Diversified portfolio across sectors",
        "symbols": [
            "SPY", "QQQ",           # Index
            "AAPL", "MSFT", "GOOGL",  # Tech
            "JPM", "GS",             # Finance
            "XOM", "CVX",            # Energy
            "UNH",                   # Healthcare
        ],
        "start_date": "2022-01-01",
        "end_date": "2024-12-01",
        "initial_capital": 250_000,
        "max_positions": 10,
        "max_position_pct": 0.03,
        "min_confidence": 0.35,
    },
}


def print_banner():
    """Print a nice banner."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║                    🎯 ELITE BACKTEST ENGINE 🎯                       ║
║                                                                      ║
║           Institutional-Grade Backtesting with EliteTradeAgent       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def print_presets():
    """Print available presets."""
    print("\nAvailable Presets:")
    print("-" * 50)
    for name, config in PRESETS.items():
        desc = config.get("description", "No description")
        symbols = ", ".join(config.get("symbols", [])[:5])
        if len(config.get("symbols", [])) > 5:
            symbols += "..."
        print(f"  {name:15s} - {desc}")
        print(f"                    Symbols: {symbols}")
    print("-" * 50)


def run_backtest(args):
    """Run the backtest with given arguments."""
    
    from backtesting.elite_backtest_engine import (
        EliteBacktestConfig,
        EliteBacktestEngine,
        print_elite_results,
    )
    
    # Build config
    config_kwargs = {
        "symbols": args.symbols,
        "start_date": args.start,
        "end_date": args.end,
        "initial_capital": args.capital,
        "timeframe": args.timeframe,
        "max_positions": args.max_positions,
        "max_position_pct": args.max_position_pct,
        "min_confidence": args.min_confidence,
        "tag": args.tag,
        "monte_carlo_runs": args.monte_carlo,
    }
    
    # Add optional parameters if specified
    if args.min_reward_risk:
        config_kwargs["min_reward_risk"] = args.min_reward_risk
    if args.kelly_fraction:
        config_kwargs["kelly_fraction"] = args.kelly_fraction
    if args.atr_stop_mult:
        config_kwargs["atr_stop_mult"] = args.atr_stop_mult
    if args.atr_target_mult:
        config_kwargs["atr_target_mult"] = args.atr_target_mult
    if args.max_portfolio_heat:
        config_kwargs["max_portfolio_heat"] = args.max_portfolio_heat
    
    config = EliteBacktestConfig(**config_kwargs)
    
    # Print configuration
    print("\n📊 Configuration:")
    print(f"  Symbols:        {', '.join(config.symbols)}")
    print(f"  Period:         {config.start_date} to {config.end_date}")
    print(f"  Capital:        ${config.initial_capital:,.2f}")
    print(f"  Max Positions:  {config.max_positions}")
    print(f"  Position Size:  {config.max_position_pct:.1%}")
    print(f"  Min Confidence: {config.min_confidence:.1%}")
    print(f"  ATR Stop:       {config.atr_stop_mult}x")
    print(f"  ATR Target:     {config.atr_target_mult}x")
    print()
    
    # Run backtest
    print("🚀 Starting backtest...")
    engine = EliteBacktestEngine(config)
    
    try:
        results = engine.run_backtest()
        
        # Print results
        print_elite_results(results)
        
        # Print summary stats
        print("\n📈 Quick Summary:")
        print(f"  Return: {results.total_return_pct*100:+.2f}%")
        print(f"  Sharpe: {results.sharpe_ratio:.2f}")
        print(f"  Max DD: {results.max_drawdown_pct*100:.2f}%")
        print(f"  Trades: {results.total_trades} ({results.win_rate*100:.0f}% win rate)")
        
        if results.total_trades > 0:
            # Verdict
            print("\n🏆 Verdict:")
            if results.sharpe_ratio >= 2.0 and results.max_drawdown_pct < 0.15:
                print("  ✅ EXCELLENT - Strong risk-adjusted returns with controlled drawdown")
            elif results.sharpe_ratio >= 1.0 and results.max_drawdown_pct < 0.25:
                print("  ✅ GOOD - Solid performance with acceptable risk")
            elif results.sharpe_ratio >= 0.5:
                print("  ⚠️  MODERATE - Room for improvement in risk management")
            elif results.total_return_pct > 0:
                print("  ⚠️  MARGINAL - Profitable but high risk")
            else:
                print("  ❌ POOR - Strategy needs significant adjustment")
        
        return results
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Elite Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_elite_backtest.py --quick
  python scripts/run_elite_backtest.py --preset momentum
  python scripts/run_elite_backtest.py --symbols SPY,NVDA --start 2023-01-01 --end 2024-12-01
  python scripts/run_elite_backtest.py --symbols TSLA --capital 50000 --max-positions 3
        """
    )
    
    # Preset options
    parser.add_argument("--quick", action="store_true", help="Run quick test with SPY")
    parser.add_argument("--preset", type=str, choices=list(PRESETS.keys()), help="Use preset configuration")
    parser.add_argument("--list-presets", action="store_true", help="List available presets")
    
    # Main options
    parser.add_argument("--symbols", type=str, default="SPY", help="Comma-separated symbols")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-12-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--timeframe", type=str, default="1Day", help="Timeframe (1Day, 1Hour, etc)")
    
    # Position sizing
    parser.add_argument("--max-positions", type=int, default=5, help="Max concurrent positions")
    parser.add_argument("--max-position-pct", type=float, default=0.04, help="Max position size (0.04 = 4%%)")
    parser.add_argument("--max-portfolio-heat", type=float, help="Max portfolio heat (0.20 = 20%%)")
    
    # Risk parameters
    parser.add_argument("--min-confidence", type=float, default=0.30, help="Min confidence threshold")
    parser.add_argument("--min-reward-risk", type=float, help="Min reward-to-risk ratio")
    parser.add_argument("--kelly-fraction", type=float, help="Kelly fraction (0.25 = 25%%)")
    parser.add_argument("--atr-stop-mult", type=float, help="ATR stop multiplier")
    parser.add_argument("--atr-target-mult", type=float, help="ATR target multiplier")
    
    # Analysis
    parser.add_argument("--monte-carlo", type=int, default=1000, help="Monte Carlo simulation runs")
    parser.add_argument("--tag", type=str, default="", help="Run identifier tag")
    
    # Output
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Print banner
    if not args.quiet:
        print_banner()
    
    # List presets
    if args.list_presets:
        print_presets()
        return
    
    # Handle quick mode
    if args.quick:
        args.preset = "quick"
    
    # Apply preset
    if args.preset:
        preset = PRESETS.get(args.preset, {})
        if not args.quiet:
            print(f"\n📋 Using preset: {args.preset}")
            print(f"   {preset.get('description', '')}")
        
        # Apply preset values (args override preset)
        for key, value in preset.items():
            if key == "description":
                continue
            arg_name = key.replace("_", "-")
            if hasattr(args, key.replace("-", "_")):
                current = getattr(args, key.replace("-", "_"))
                # Only apply preset if arg wasn't explicitly set
                if current == parser.get_default(key.replace("-", "_")) or current is None:
                    setattr(args, key.replace("-", "_"), value)
    
    # Parse symbols
    if isinstance(args.symbols, str):
        args.symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    # Configure logging
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    elif args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
    
    # Run backtest
    results = run_backtest(args)
    
    if results is None:
        sys.exit(1)
    
    print("\n✅ Backtest complete!")
    if results.config and results.config.save_trades:
        print(f"   Results saved to: {results.config.output_dir}")


if __name__ == "__main__":
    main()
