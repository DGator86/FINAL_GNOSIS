#!/usr/bin/env python3
"""
Paper Trading CLI - Run the EliteTradeAgent with Alpaca Paper Trading

This script provides a CLI interface to start the paper trading engine
with various configuration options.

Usage:
    # Default: Trade dynamic universe
    python scripts/run_paper_trading.py
    
    # Trade specific symbols
    python scripts/run_paper_trading.py --symbols SPY,QQQ,AAPL
    
    # Dry run mode (no actual orders)
    python scripts/run_paper_trading.py --dry-run
    
    # Custom scan interval
    python scripts/run_paper_trading.py --interval 120
    
    # With verbose logging
    python scripts/run_paper_trading.py --verbose

Safety:
    - Always runs in PAPER mode (enforced)
    - Pre-flight checks before starting
    - Graceful shutdown on Ctrl+C
    - Circuit breakers for daily loss

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger


def setup_logging(verbose: bool = False, log_file: str = None):
    """Setup logging configuration."""
    # Remove default handler
    logger.remove()
    
    # Console handler
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    )
    
    # File handler
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger.add(
        log_file,
        rotation="100 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )
    
    return log_file


def run_preflight_checks() -> bool:
    """Run pre-flight checks before starting."""
    from scripts.preflight_check import PreflightChecker
    
    logger.info("Running pre-flight checks...")
    checker = PreflightChecker()
    result = checker.run_all_checks()
    
    if result == 1:
        logger.error("Pre-flight check FAILED - cannot start trading")
        return False
    elif result == 2:
        logger.warning("Pre-flight check passed with warnings")
    else:
        logger.info("Pre-flight check PASSED")
    
    return True


def get_dynamic_universe(top_n: int = 10) -> list:
    """Get dynamic trading universe."""
    try:
        from engines.scanner import get_dynamic_universe
        from config import load_config
        
        config = load_config()
        symbols = get_dynamic_universe(config.scanner.model_dump(), top_n)
        
        if symbols:
            return symbols
    except Exception as e:
        logger.warning(f"Could not get dynamic universe: {e}")
    
    # Fallback to defaults
    return ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "AMD"][:top_n]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Super Gnosis Paper Trading Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_paper_trading.py
    python scripts/run_paper_trading.py --symbols SPY,QQQ,AAPL
    python scripts/run_paper_trading.py --dry-run --interval 30
    python scripts/run_paper_trading.py --top 5 --verbose
        """
    )
    
    parser.add_argument(
        "--symbols", "-s",
        type=str,
        default=None,
        help="Comma-separated list of symbols to trade (overrides dynamic universe)"
    )
    
    parser.add_argument(
        "--top", "-t",
        type=int,
        default=10,
        help="Number of symbols from dynamic universe (default: 10)"
    )
    
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=60,
        help="Seconds between market scans (default: 60)"
    )
    
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Dry run mode - simulate without actual orders"
    )
    
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip pre-flight checks (not recommended)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging"
    )
    
    parser.add_argument(
        "--max-positions",
        type=int,
        default=10,
        help="Maximum number of concurrent positions (default: 10)"
    )
    
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=5000.0,
        help="Maximum daily loss circuit breaker (default: $5000)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(verbose=args.verbose)
    
    # Print banner
    print("\n" + "="*70)
    print("🚀 SUPER GNOSIS PAPER TRADING ENGINE")
    print("="*70)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'PAPER TRADING'}")
    print(f"Log file: {log_file}")
    print("="*70 + "\n")
    
    # Run pre-flight checks
    if not args.skip_preflight:
        if not run_preflight_checks():
            sys.exit(1)
    else:
        logger.warning("Skipping pre-flight checks (--skip-preflight)")
    
    # Determine symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        logger.info(f"Using specified symbols: {', '.join(symbols)}")
    else:
        logger.info(f"Getting dynamic universe (top {args.top})...")
        symbols = get_dynamic_universe(args.top)
        logger.info(f"Dynamic universe: {', '.join(symbols)}")
    
    # Configuration
    config = {
        "scan_interval": args.interval,
        "position_check_interval": 30,
        "max_positions": args.max_positions,
        "max_daily_loss": args.max_daily_loss,
    }
    
    # Import and create engine
    from trade.paper_trading_engine import PaperTradingEngine
    
    engine = PaperTradingEngine(
        symbols=symbols,
        config=config,
        dry_run=args.dry_run,
    )
    
    # Setup signal handlers for graceful shutdown
    def handle_shutdown(signum, frame):
        logger.info(f"Received shutdown signal ({signum})")
        engine.stop()
    
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Print configuration
    print("\n" + "-"*50)
    print("Configuration:")
    print(f"  Symbols: {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
    print(f"  Scan Interval: {args.interval}s")
    print(f"  Max Positions: {args.max_positions}")
    print(f"  Max Daily Loss: ${args.max_daily_loss:,.0f}")
    print(f"  Dry Run: {args.dry_run}")
    print("-"*50)
    print("Press Ctrl+C to stop")
    print("-"*50 + "\n")
    
    # Start engine
    try:
        engine.start(blocking=True)
    except Exception as e:
        logger.error(f"Engine error: {e}")
        sys.exit(1)
    
    logger.info("Paper trading engine stopped")


if __name__ == "__main__":
    main()
