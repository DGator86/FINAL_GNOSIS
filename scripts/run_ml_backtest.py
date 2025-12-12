#!/usr/bin/env python
"""
Run ML-Enabled Gnosis Backtest

This script runs the full Gnosis trading system backtest with:
- Real historical data from Alpaca (5+ years available)
- All 4 engines (Hedge, Liquidity, Sentiment, Elasticity)
- Composer consensus voting
- Optional LSTM predictions
- Comprehensive metrics and trade analysis

Prerequisites:
  - Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables
  - Free Alpaca account: https://app.alpaca.markets/signup

Usage:
  # Basic backtest (5 years of SPY)
  python scripts/run_ml_backtest.py --symbol SPY --start 2019-01-01 --end 2024-12-01

  # Shorter timeframe with intraday data
  python scripts/run_ml_backtest.py --symbol AAPL --timeframe 1Hour --start 2024-01-01

  # High-frequency test
  python scripts/run_ml_backtest.py --symbol TSLA --timeframe 15Min --start 2024-06-01

  # With LSTM (requires trained model)
  python scripts/run_ml_backtest.py --symbol SPY --use-lstm --lstm-path models/trained/lstm_spy.pt

Examples:
  # Quick test (1 year daily)
  python scripts/run_ml_backtest.py --symbol SPY --start 2024-01-01 --end 2024-12-01 --tag quick_test

  # Full historical test (5 years)
  python scripts/run_ml_backtest.py --symbol SPY --start 2019-01-01 --end 2024-12-01 --tag full_history

  # Multi-symbol test (run for each)
  for sym in SPY QQQ IWM; do
    python scripts/run_ml_backtest.py --symbol $sym --start 2020-01-01 --tag ${sym}_test
  done
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def check_alpaca_credentials():
    """Check if Alpaca credentials are configured."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        print("\n" + "="*60)
        print("ERROR: Alpaca API credentials not found!")
        print("="*60)
        print("\nTo run backtests with real historical data, you need Alpaca credentials.")
        print("\nSteps to set up:")
        print("1. Create a free account at https://app.alpaca.markets/signup")
        print("2. Go to Paper Trading -> API Keys")
        print("3. Generate new API keys")
        print("4. Set environment variables:")
        print("   export ALPACA_API_KEY='your-api-key'")
        print("   export ALPACA_SECRET_KEY='your-secret-key'")
        print("\nOr add to your .bashrc/.zshrc for persistence.")
        print("="*60 + "\n")
        return False
    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ML-enabled Gnosis backtest with real historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (1 year, daily bars)
  python scripts/run_ml_backtest.py --symbol SPY --start 2024-01-01

  # Full history test (5 years)
  python scripts/run_ml_backtest.py --symbol SPY --start 2019-01-01 --tag full_test

  # Intraday test (hourly bars, 6 months)
  python scripts/run_ml_backtest.py --symbol AAPL --timeframe 1Hour --start 2024-06-01

  # Custom parameters
  python scripts/run_ml_backtest.py --symbol TSLA --start 2023-01-01 \\
    --position-size 0.15 --stop-loss 0.03 --take-profit 0.06
        """
    )

    # Required
    parser.add_argument(
        "--symbol",
        type=str,
        default="SPY",
        help="Trading symbol (default: SPY)"
    )

    # Date range
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date YYYY-MM-DD (default: 2020-01-01)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD (default: today)"
    )

    # Timeframe
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1Day",
        choices=["1Min", "5Min", "15Min", "30Min", "1Hour", "4Hour", "1Day"],
        help="Bar timeframe (default: 1Day)"
    )

    # Capital
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)"
    )

    # Position sizing
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.10,
        help="Position size as fraction of capital (default: 0.10 = 10%%)"
    )

    # Risk management
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.02,
        help="Stop loss percentage (default: 0.02 = 2%%)"
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.04,
        help="Take profit percentage (default: 0.04 = 4%%)"
    )

    # Signal thresholds
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.3,
        help="Minimum consensus for entry (default: 0.3)"
    )

    # Engine weights
    parser.add_argument(
        "--hedge-weight",
        type=float,
        default=0.4,
        help="Hedge engine weight (default: 0.4)"
    )
    parser.add_argument(
        "--sentiment-weight",
        type=float,
        default=0.4,
        help="Sentiment engine weight (default: 0.4)"
    )
    parser.add_argument(
        "--liquidity-weight",
        type=float,
        default=0.2,
        help="Liquidity engine weight (default: 0.2)"
    )

    # LSTM options
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help="Enable LSTM predictions (requires trained model)"
    )
    parser.add_argument(
        "--lstm-path",
        type=str,
        default=None,
        help="Path to trained LSTM model"
    )
    parser.add_argument(
        "--lstm-weight",
        type=float,
        default=0.3,
        help="LSTM weight in final signal (default: 0.3)"
    )

    # Output
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Tag for this backtest run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/backtests",
        help="Output directory (default: runs/backtests)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk"
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output (just final metrics)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Check credentials
    if not check_alpaca_credentials():
        sys.exit(1)

    # Import after checking credentials
    from backtesting.ml_backtest_engine import (
        MLBacktestConfig,
        MLBacktestEngine,
        print_results_summary,
    )
    from loguru import logger

    # Configure logging
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    elif not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    # Build config
    config = MLBacktestConfig(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        timeframe=args.timeframe,
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        entry_threshold=args.entry_threshold,
        hedge_weight=args.hedge_weight,
        sentiment_weight=args.sentiment_weight,
        liquidity_weight=args.liquidity_weight,
        use_lstm=args.use_lstm,
        lstm_model_path=args.lstm_path,
        lstm_weight=args.lstm_weight,
        tag=args.tag or f"{args.symbol}_{args.start}_{args.end}",
        output_dir=args.output_dir,
        save_trades=not args.no_save,
        save_equity_curve=not args.no_save,
    )

    if not args.quiet:
        print("\n" + "="*60)
        print("GNOSIS ML BACKTEST")
        print("="*60)
        print(f"Symbol:     {config.symbol}")
        print(f"Period:     {config.start_date} to {config.end_date}")
        print(f"Timeframe:  {config.timeframe}")
        print(f"Capital:    ${config.initial_capital:,.0f}")
        print(f"LSTM:       {'Enabled' if config.use_lstm else 'Disabled'}")
        print("="*60)
        print("Running backtest...")
        print()

    try:
        # Run backtest
        engine = MLBacktestEngine(config)
        results = engine.run_backtest()

        # Print results
        print_results_summary(results)

        # Print output location
        if not args.no_save:
            print(f"\nResults saved to: {config.output_dir}/{config.tag}_*.json")

        # Exit with appropriate code
        if results.total_return >= 0:
            sys.exit(0)
        else:
            sys.exit(0)  # Still success, just negative returns

    except ValueError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Backtest failed - {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
