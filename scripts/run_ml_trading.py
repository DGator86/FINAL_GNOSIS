#!/usr/bin/env python3
"""
Run ML-Integrated Trading Engine.

This script demonstrates how to use the ML trading engine with
full safety controls for paper/live trading.

Usage:
    # Dry run mode (no actual orders)
    python scripts/run_ml_trading.py --dry-run
    
    # Paper trading
    python scripts/run_ml_trading.py --paper
    
    # With custom settings
    python scripts/run_ml_trading.py --preset aggressive --symbols SPY,QQQ,NVDA
    
    # With custom risk limits
    python scripts/run_ml_trading.py --max-loss 2000 --max-positions 5

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

import argparse
import sys
import signal
from typing import Optional

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)


def parse_args():
    parser = argparse.ArgumentParser(description="ML-Integrated Trading Engine")
    
    # Mode
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no actual orders)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Paper trading mode (default)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Live trading mode (CAUTION: real money)",
    )
    
    # ML Settings
    parser.add_argument(
        "--preset",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="ML hyperparameter preset",
    )
    parser.add_argument(
        "--ml-confidence",
        type=float,
        default=0.60,
        help="Minimum ML confidence threshold",
    )
    
    # Symbols
    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,QQQ,AAPL,NVDA,MSFT",
        help="Comma-separated list of symbols to trade",
    )
    
    # Risk Limits
    parser.add_argument(
        "--max-loss",
        type=float,
        default=5000.0,
        help="Maximum daily loss (USD)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=10,
        help="Maximum number of positions",
    )
    parser.add_argument(
        "--max-position-size",
        type=float,
        default=0.04,
        help="Maximum position size (fraction of portfolio)",
    )
    
    # Scanning
    parser.add_argument(
        "--scan-interval",
        type=int,
        default=60,
        help="Signal scanning interval (seconds)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Import here to avoid circular imports
    from trade.ml_trading_engine import MLTradingConfig, MLTradingEngine
    
    # Determine mode
    if args.live:
        paper_mode = False
        logger.warning("⚠️ LIVE TRADING MODE - REAL MONEY AT RISK")
        confirm = input("Type 'YES' to confirm live trading: ")
        if confirm != "YES":
            logger.info("Cancelled.")
            return
    else:
        paper_mode = True
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    # Create configuration
    config = MLTradingConfig(
        paper_mode=paper_mode,
        dry_run=args.dry_run,
        ml_preset=args.preset,
        ml_confidence_threshold=args.ml_confidence,
        max_daily_loss_usd=args.max_loss,
        max_positions=args.max_positions,
        max_position_size_pct=args.max_position_size,
        scan_interval_seconds=args.scan_interval,
        symbols=symbols,
    )
    
    # Create engine
    engine = MLTradingEngine(config=config)
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal...")
        engine.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Print configuration
    print("\n" + "=" * 70)
    print("🤖 ML TRADING ENGINE CONFIGURATION")
    print("=" * 70)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'PAPER' if paper_mode else 'LIVE'}")
    print(f"ML Preset: {args.preset}")
    print(f"ML Confidence Threshold: {args.ml_confidence:.0%}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Max Daily Loss: ${args.max_loss:,.0f}")
    print(f"Max Positions: {args.max_positions}")
    print(f"Max Position Size: {args.max_position_size:.1%}")
    print(f"Scan Interval: {args.scan_interval}s")
    print("=" * 70 + "\n")
    
    # Start engine
    try:
        engine.start(blocking=True)
    except Exception as e:
        logger.error(f"Engine error: {e}")
        engine.stop()


if __name__ == "__main__":
    main()
