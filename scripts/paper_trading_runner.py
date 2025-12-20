#!/usr/bin/env python3
"""Production-ready paper trading runner with safety features.

This is the MAIN ENTRY POINT for paper trading on Alpaca.

DEFAULT MODE: Dynamic Universe Trading (Top 25 most active options underlyings)
The system automatically ranks and trades the hottest options names based on:
- Options volume, Open interest, Gamma exposure, Liquidity, Unusual flow

Features:
- Pre-flight checks before starting
- Dynamic universe scanning (top N most active symbols)
- Market hours awareness (only trades when market is open)
- Health monitoring with automatic recovery
- Safe shutdown handling (Ctrl+C)
- Session summary on exit
- Logging to file and console

Usage:
    # DEFAULT: Trade full dynamic universe (top 25 symbols)
    python scripts/paper_trading_runner.py
    
    # Trade top 10 from dynamic universe
    python scripts/paper_trading_runner.py --top 10
    
    # Single symbol only (override default)
    python scripts/paper_trading_runner.py --symbol SPY
    
    # Dry run (no actual trades)
    python scripts/paper_trading_runner.py --dry-run

Safety:
    - Always runs pre-flight check first
    - Respects market hours (waits when closed)
    - Circuit breaker on daily loss limit
    - Graceful shutdown on Ctrl+C
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.add(
    log_file,
    rotation="100 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
)


class TradingSession:
    """Manages a paper trading session with safety features."""
    
    def __init__(
        self,
        symbol: Optional[str] = None,
        multi_symbol: bool = False,
        top_n: int = 5,
        interval: int = 60,
        dry_run: bool = False,
        skip_preflight: bool = False,
    ):
        self.symbol = symbol or "SPY"
        self.multi_symbol = multi_symbol
        self.top_n = top_n
        self.interval = interval
        self.dry_run = dry_run
        self.skip_preflight = skip_preflight
        
        # Session state
        self.running = False
        self.start_time: Optional[datetime] = None
        self.iteration_count = 0
        self.trade_count = 0
        self.error_count = 0
        self.start_equity: Optional[float] = None
        self.current_equity: Optional[float] = None
        
        # Components (initialized in start())
        self.broker = None
        self.config = None
        self.health_monitor = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received shutdown signal ({signum})")
        self.running = False
    
    def run_preflight(self) -> bool:
        """Run pre-flight checks."""
        if self.skip_preflight:
            logger.warning("Skipping pre-flight checks (--skip-preflight)")
            return True
        
        logger.info("Running pre-flight checks...")
        
        from scripts.preflight_check import PreflightChecker
        checker = PreflightChecker()
        result = checker.run_all_checks()
        
        if result == 1:  # Critical failure
            logger.error("Pre-flight check FAILED - cannot start trading")
            return False
        elif result == 2:  # Warnings
            logger.warning("Pre-flight check passed with warnings")
            # Continue anyway for paper trading
        else:
            logger.info("Pre-flight check PASSED")
        
        return True
    
    def initialize(self) -> bool:
        """Initialize trading components."""
        try:
            from config import load_config
            from engines.inputs.adapter_factory import create_broker_adapter
            from execution.broker_adapters.settings import get_alpaca_paper_setting
            
            self.config = load_config()
            
            if not self.dry_run:
                paper_mode = get_alpaca_paper_setting()
                if not paper_mode:
                    logger.error("REFUSING TO RUN IN LIVE MODE - set ALPACA_PAPER=true")
                    return False
                
                self.broker = create_broker_adapter(paper=True, prefer_real=True)
                if not self.broker:
                    logger.error("Failed to create broker adapter")
                    return False
                
                # Get initial equity
                account = self.broker.get_account()
                self.start_equity = account.equity
                self.current_equity = account.equity
                logger.info(f"Initial equity: ${self.start_equity:,.2f}")
                
                # Initialize health monitor
                try:
                    from utils.health_monitor import create_trading_health_monitor
                    self.health_monitor = create_trading_health_monitor(
                        broker=self.broker,
                        check_interval=60.0,
                    )
                    self.health_monitor.start()
                    logger.info("Health monitor started")
                except Exception as e:
                    logger.warning(f"Health monitor initialization failed: {e}")
                    # Continue without health monitoring
            else:
                logger.info("DRY RUN MODE - no broker connection")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def wait_for_market_open(self) -> bool:
        """Wait for market to open, checking periodically."""
        if self.dry_run:
            return True
        
        try:
            clock = self.broker.get_market_clock()
            if not clock:
                logger.warning("Could not get market clock, proceeding anyway")
                return True
            
            if clock["is_open"]:
                logger.info("Market is OPEN")
                return True
            
            next_open = clock.get("next_open")
            if next_open:
                logger.info(f"Market is CLOSED. Next open: {next_open}")
                logger.info("Waiting for market to open (checking every 60s)...")
                
                while not self.broker.is_market_open() and self.running:
                    time.sleep(60)
                    logger.debug("Still waiting for market open...")
                
                if self.running:
                    logger.info("Market is now OPEN!")
                    return True
            
            return self.running
            
        except Exception as e:
            logger.warning(f"Market check failed: {e}, proceeding anyway")
            return True
    
    def run_single_iteration(self) -> Dict[str, Any]:
        """Run a single trading iteration."""
        from main import build_pipeline
        
        result_summary = {
            "timestamp": datetime.now(timezone.utc),
            "symbol": self.symbol,
            "trade_ideas": 0,
            "orders_placed": 0,
            "errors": [],
        }
        
        try:
            # Build adapters dict
            adapters = {}
            if self.broker and not self.dry_run:
                adapters["broker"] = self.broker
            
            # Build and run pipeline
            runner = build_pipeline(self.symbol, self.config, adapters)
            timestamp = datetime.now(timezone.utc)
            
            pipeline_result = runner.run_once(timestamp)
            
            # Collect stats
            if pipeline_result.trade_ideas:
                result_summary["trade_ideas"] = len(pipeline_result.trade_ideas)
            
            if pipeline_result.order_results:
                result_summary["orders_placed"] = len(pipeline_result.order_results)
                self.trade_count += len(pipeline_result.order_results)
            
            # Log consensus
            if pipeline_result.consensus:
                direction = pipeline_result.consensus.get("direction", "neutral")
                confidence = pipeline_result.consensus.get("confidence", 0)
                logger.info(f"Consensus: {direction} (confidence: {confidence:.2%})")
            
        except Exception as e:
            logger.error(f"Iteration error: {e}")
            result_summary["errors"].append(str(e))
            self.error_count += 1
        
        return result_summary
    
    def run_multi_symbol_iteration(self) -> List[Dict[str, Any]]:
        """Run iteration across multiple symbols."""
        from engines.scanner import OpportunityScanner, get_dynamic_universe
        from engines.inputs.adapter_factory import (
            create_market_data_adapter,
            create_options_adapter,
            create_news_adapter,
        )
        from engines.hedge.hedge_engine_v3 import HedgeEngineV3
        from engines.liquidity.liquidity_engine_v1 import LiquidityEngineV1
        from engines.sentiment.sentiment_engine_v1 import SentimentEngineV1
        from engines.sentiment.processors import (
            NewsSentimentProcessor,
            FlowSentimentProcessor,
            TechnicalSentimentProcessor,
        )
        from engines.elasticity.elasticity_engine_v1 import ElasticityEngineV1
        
        results = []
        
        try:
            # Get dynamic universe
            logger.info(f"Scanning for top {self.top_n} opportunities...")
            symbols = get_dynamic_universe(
                self.config.scanner.model_dump(), 
                self.top_n
            )
            
            if not symbols:
                logger.warning("No symbols from scanner, using default")
                symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"][:self.top_n]
            
            logger.info(f"Trading symbols: {', '.join(symbols)}")
            
            # Run pipeline for each symbol
            for symbol in symbols:
                self.symbol = symbol
                result = self.run_single_iteration()
                results.append(result)
                
                # Small delay between symbols
                if symbol != symbols[-1]:
                    time.sleep(2)
            
        except Exception as e:
            logger.error(f"Multi-symbol iteration error: {e}")
            self.error_count += 1
        
        return results
    
    def update_session_stats(self):
        """Update session statistics."""
        if self.broker and not self.dry_run:
            try:
                account = self.broker.get_account()
                self.current_equity = account.equity
            except Exception as e:
                logger.warning(f"Failed to update equity: {e}")
    
    def check_circuit_breaker(self) -> bool:
        """Check if daily loss limit has been hit."""
        if self.dry_run or not self.start_equity or not self.current_equity:
            return True  # OK to continue
        
        max_daily_loss = float(os.getenv("MAX_DAILY_LOSS_USD", "5000.0"))
        session_pnl = self.current_equity - self.start_equity
        
        if session_pnl < -max_daily_loss:
            logger.error(
                f"CIRCUIT BREAKER: Daily loss ${-session_pnl:,.2f} exceeds "
                f"limit ${max_daily_loss:,.2f}"
            )
            return False
        
        return True
    
    def print_session_summary(self):
        """Print end-of-session summary."""
        duration = datetime.now(timezone.utc) - self.start_time if self.start_time else timedelta(0)
        
        print("\n" + "="*70)
        print("📊 TRADING SESSION SUMMARY")
        print("="*70)
        print(f"Duration: {duration}")
        print(f"Iterations: {self.iteration_count}")
        print(f"Trades Placed: {self.trade_count}")
        print(f"Errors: {self.error_count}")
        
        if self.start_equity and self.current_equity:
            pnl = self.current_equity - self.start_equity
            pnl_pct = (pnl / self.start_equity) * 100
            print(f"\nStarting Equity: ${self.start_equity:,.2f}")
            print(f"Ending Equity: ${self.current_equity:,.2f}")
            print(f"Session P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        
        if self.broker and not self.dry_run:
            try:
                positions = self.broker.get_positions()
                print(f"\nOpen Positions: {len(positions)}")
                for pos in positions:
                    print(f"  {pos.symbol}: {pos.quantity} @ ${pos.avg_entry_price:.2f} "
                          f"| P&L: ${pos.unrealized_pnl:+,.2f}")
            except Exception:
                pass
        
        print(f"\nLog file: {log_file}")
        print("="*70 + "\n")
    
    def start(self):
        """Start the trading session."""
        logger.info("="*50)
        logger.info("STARTING PAPER TRADING SESSION")
        logger.info("="*50)
        
        # Run pre-flight
        if not self.run_preflight():
            return
        
        # Initialize
        if not self.initialize():
            return
        
        # Wait for market if needed
        self.running = True
        if not self.wait_for_market_open():
            return
        
        self.start_time = datetime.now(timezone.utc)
        
        mode = "MULTI-SYMBOL" if self.multi_symbol else f"SINGLE ({self.symbol})"
        logger.info(f"Mode: {mode}")
        logger.info(f"Interval: {self.interval}s")
        logger.info(f"Dry Run: {self.dry_run}")
        logger.info("Press Ctrl+C to stop")
        logger.info("-"*50)
        
        # Main trading loop
        try:
            while self.running:
                self.iteration_count += 1
                
                # Check market hours
                if self.broker and not self.dry_run:
                    if not self.broker.is_market_open():
                        logger.info("Market closed, waiting...")
                        self.wait_for_market_open()
                        if not self.running:
                            break
                
                # Check circuit breaker
                if not self.check_circuit_breaker():
                    logger.error("Circuit breaker triggered, stopping")
                    break
                
                # Check health status
                if self.health_monitor:
                    health_status = self.health_monitor.get_status()
                    if not health_status.get("overall_healthy", True):
                        unhealthy = [
                            name for name, comp in health_status.get("components", {}).items()
                            if comp.get("status") != "healthy"
                        ]
                        logger.warning(f"System health degraded: {', '.join(unhealthy)}")
                
                # Run iteration
                logger.info(f"\n--- Iteration {self.iteration_count} ---")
                
                if self.multi_symbol:
                    results = self.run_multi_symbol_iteration()
                    total_ideas = sum(r.get("trade_ideas", 0) for r in results)
                    total_orders = sum(r.get("orders_placed", 0) for r in results)
                    logger.info(f"Summary: {total_ideas} ideas, {total_orders} orders")
                else:
                    result = self.run_single_iteration()
                    logger.info(
                        f"Summary: {result['trade_ideas']} ideas, "
                        f"{result['orders_placed']} orders"
                    )
                
                # Update stats
                self.update_session_stats()
                
                # Log session PnL
                if self.start_equity and self.current_equity:
                    pnl = self.current_equity - self.start_equity
                    logger.info(f"Session P&L: ${pnl:+,.2f}")
                
                # Wait for next iteration
                if self.running:
                    logger.info(f"Next iteration in {self.interval}s...")
                    
                    # Sleep in small chunks for responsive shutdown
                    for _ in range(self.interval):
                        if not self.running:
                            break
                        time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.running = False
            logger.info("Shutting down...")
            
            # Stop health monitor
            if self.health_monitor:
                try:
                    self.health_monitor.stop()
                    logger.info("Health monitor stopped")
                except Exception:
                    pass
            
            self.update_session_stats()
            self.print_session_summary()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Super Gnosis Paper Trading Runner"
    )
    
    parser.add_argument(
        "--symbol", "-s",
        default=None,
        help="Single symbol to trade (overrides multi-symbol mode)"
    )
    
    parser.add_argument(
        "--single",
        action="store_true",
        help="Force single-symbol mode (default is multi-symbol)"
    )
    
    parser.add_argument(
        "--top", "-t",
        type=int,
        default=25,
        help="Number of top symbols from dynamic universe (default: 25)"
    )
    
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=60,
        help="Seconds between iterations (default: 60)"
    )
    
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Dry run mode (no actual trades)"
    )
    
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip pre-flight checks (not recommended)"
    )
    
    args = parser.parse_args()
    
    # Determine mode: multi-symbol is default unless --single or --symbol is specified
    if args.symbol:
        # Explicit symbol provided - single symbol mode
        multi_symbol = False
        symbol = args.symbol
    elif args.single:
        # Forced single mode without symbol - default to SPY
        multi_symbol = False
        symbol = "SPY"
    else:
        # Default: multi-symbol dynamic universe trading
        multi_symbol = True
        symbol = "SPY"  # Fallback for single iterations
    
    session = TradingSession(
        symbol=symbol,
        multi_symbol=multi_symbol,
        top_n=args.top,
        interval=args.interval,
        dry_run=args.dry_run,
        skip_preflight=args.skip_preflight,
    )
    
    session.start()


if __name__ == "__main__":
    main()
