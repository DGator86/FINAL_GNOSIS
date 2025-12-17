#!/usr/bin/env python3
"""
DYNAMIC UNIVERSE TRADING SYSTEM
Scanner-driven adaptive trading across top 25 opportunities
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment for full trading
os.environ["ENABLE_TRADING"] = "true"

from typing import Any, Optional

import yaml  # type: ignore
from loguru import logger

from engines.engine_factory import EngineFactory
from execution.broker_adapters.settings import get_alpaca_paper_setting
from gnosis.dynamic_universe_manager import DynamicUniverseManager
from gnosis.unified_trading_bot import UnifiedTradingBot

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
)
logger.add("logs/dynamic_trading_{time}.log", rotation="1 day", retention="7 days")


class DynamicTradingSystem:
    """Dynamic universe trading system driven by OpportunityScanner."""

    def __init__(self) -> None:
        self.universe_mgr: Optional[DynamicUniverseManager] = None
        self.trading_bot: Optional[UnifiedTradingBot] = None
        self.running: bool = False
        self.paper_mode: bool = get_alpaca_paper_setting()
        self.config: Any

        # Load config
        with open("config/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

    async def initialize(self) -> None:
        """Initialize scanner, universe manager, and trading bot."""

        mode_label = "PAPER" if self.paper_mode else "LIVE"

        print(f"""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║         🎯 DYNAMIC UNIVERSE TRADING SYSTEM 🎯                         ║
║                                                                        ║
║  Status: {mode_label} TRADING ENABLED                                 ║
║  Mode: Scanner-Driven Dynamic Universe                                ║
║  Account: Alpaca {mode_label}                                         ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

📊 SYSTEM CONFIGURATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

        # Initialize scanner using engine factory
        logger.info("Initializing opportunity scanner with all engines...")
        factory = EngineFactory(self.config)
        scanner = factory.create_scanner()

        # Create universe manager (top 25, refresh every 15 min)
        logger.info("Initializing universe manager...")
        self.universe_mgr = DynamicUniverseManager(
            scanner=scanner,
            top_n=25,
            refresh_interval_seconds=900,  # 15 minutes
            min_score_threshold=0.5,
        )

        # Create unified trading bot
        logger.info("Initializing unified trading bot...")
        self.trading_bot = UnifiedTradingBot(
            config=self.config, enable_trading=True, paper_mode=self.paper_mode
        )

        # Initial universe scan
        logger.info("Performing initial universe scan...")
        initial_symbols = self._get_candidate_symbols()
        initial_update = await self.universe_mgr.refresh_universe(initial_symbols)

        logger.info(f"Initial universe: {len(initial_update.current)} symbols")
        if initial_update.current:
            for i, symbol in enumerate(initial_update.current[:10], 1):
                logger.info(f"  {i}. {symbol}")
            if len(initial_update.current) > 10:
                logger.info(f"  ... and {len(initial_update.current) - 10} more")
        else:
            logger.warning(
                "Initial universe is empty. No trades will be placed "
                "until valid opportunities are detected."
            )
            logger.info(
                "Validate data sources: ALPACA_API_KEY/SECRET, "
                "ALPACA_DATA_FEED (IEX vs SIP), and Unusual Whales token if using flow data."
            )

        # Load universe into trading bot
        await self.trading_bot.update_universe(initial_update)

        if initial_update.current:
            trading_banner = "⚡ TRADING STATUS: ENABLED - WILL PLACE PAPER ORDERS"
        else:
            trading_banner = (
                "⚠️  TRADING STATUS: STANDBY - UNIVERSE IS EMPTY (no orders will be placed)"
            )

        print(f"""
✅ SYSTEM INITIALIZED:
   • Universe: {len(initial_update.current)} top opportunities
   • Refresh: Every 15 minutes
   • Max Positions: 5 concurrent
   • Risk Management: Stop-loss, Take-profit, Trailing stops
   • Close on exit: Losing positions only

{trading_banner}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

    def _get_candidate_symbols(self) -> list[str]:
        """Get candidate symbols for scanning.

        Returns high-liquidity, optionable stocks for scanning.
        """
        # Start with known liquid symbols
        # In production, this would pull from a dynamic universe manager
        # or access a database of all optionable stocks

        candidates = [
            # Indices
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            # Mega Tech
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            # Tech
            "AMD",
            "INTC",
            "CRM",
            "ORCL",
            "ADBE",
            "NFLX",
            "AVGO",
            # Finance
            "JPM",
            "BAC",
            "GS",
            "MS",
            "C",
            "WFC",
            # Healthcare
            "UNH",
            "JNJ",
            "PFE",
            "ABBV",
            "MRK",
            # Consumer
            "WMT",
            "HD",
            "MCD",
            "NKE",
            "SBUX",
            # Energy
            "XOM",
            "CVX",
            "COP",
            "SLB",
            # Industrial
            "BA",
            "CAT",
            "GE",
            "HON",
        ]

        return candidates

    async def universe_refresh_loop(self) -> None:
        """Periodically refresh universe and update trading bot."""

        while self.running:
            try:
                # Wait for refresh interval
                await asyncio.sleep(self.universe_mgr.refresh_interval)

                # Rescan universe
                logger.info("Refreshing universe...")
                assert self.universe_mgr is not None
                candidate_symbols = self._get_candidate_symbols()
                update = await self.universe_mgr.refresh_universe(candidate_symbols)

                # Update trading bot
                if update.added or update.removed:
                    assert self.trading_bot is not None
                    await self.trading_bot.update_universe(update)

                    # Log top 10
                    logger.info("Universe refreshed - Top 10:")
                    for i, symbol in enumerate(update.current[:10], 1):
                        opp = self.universe_mgr.get_opportunity_for_symbol(symbol)
                        if opp:
                            logger.info(f"  {i}. {symbol} (score: {opp.score:.2f})")

            except Exception as e:
                logger.error(f"Universe refresh error: {e}")
                await asyncio.sleep(60)  # Brief pause before retry

    async def monitor_performance(self) -> None:
        """Monitor and report performance."""

        report_interval = 300  # Every 5 minutes

        while self.running:
            try:
                await asyncio.sleep(report_interval)

                # Get account status
                from alpaca.trading.client import TradingClient

                api_key = os.getenv("ALPACA_API_KEY")
                secret_key = os.getenv("ALPACA_SECRET_KEY")
                client = TradingClient(api_key, secret_key, paper=self.paper_mode)
                account = client.get_account()

                # Get positions from client (actual broker positions)
                broker_positions = client.get_all_positions()

                print(f"""
📈 PERFORMANCE UPDATE ({datetime.now().strftime("%H:%M:%S")}):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Portfolio Value: ${float(getattr(account, "portfolio_value", 0)):,.2f}
   Equity: ${float(getattr(account, "equity", 0)):,.2f}
   Cash: ${float(getattr(account, "cash", 0)):,.2f}
   Buying Power: ${float(getattr(account, "buying_power", 0)):,.2f}
   P&L Today: ${float(getattr(account, "portfolio_value", 0)) - 30000:+,.2f}
   Open Positions: {len(broker_positions)}
   Active Universe: {len(self.trading_bot.active_symbols) if self.trading_bot else 0}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

                if broker_positions:
                    logger.info("Open positions:")
                    for pos in broker_positions[:5]:  # type: ignore[index]
                        unrealized_pnl = (
                            float(pos.unrealized_pl) if hasattr(pos, "unrealized_pl") else 0
                        )
                        logger.info(
                            f"   {pos.symbol}: {pos.qty} shares | P&L: ${unrealized_pnl:+,.2f}"
                        )

                # Show universe status
                if self.universe_mgr:
                    logger.info(
                        f"Universe: {len(self.universe_mgr.active_universe)} symbols active"
                    )

            except Exception as e:
                logger.error(f"Performance monitor error: {e}")

    async def run(self) -> None:
        """Main execution loop."""

        tasks: list[asyncio.Task[Any]] = []
        try:
            # Initialize system
            await self.initialize()

            # Set running flag
            self.running = True

            # Create tasks
            assert self.universe_mgr is not None
            assert self.trading_bot is not None
            tasks = [
                asyncio.create_task(self.universe_refresh_loop()),
                asyncio.create_task(self.trading_bot.run()),
                asyncio.create_task(self.monitor_performance()),
            ]

            print("""
🚀 DYNAMIC TRADING SYSTEM RUNNING!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Monitoring:
• Alpaca Dashboard: https://app.alpaca.markets/paper
• Logs: ./logs/dynamic_trading_*.log

Press Ctrl+C to stop gracefully.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

            # Run all tasks
            await asyncio.gather(*tasks)

        except KeyboardInterrupt:
            logger.warning("Shutdown signal received...")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.shutdown()
        except asyncio.CancelledError:
            logger.info("Tasks cancelled; shutting down gracefully")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.shutdown()
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            import traceback

            traceback.print_exc()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown."""

        logger.info("Initiating graceful shutdown...")
        self.running = False

        # Stop trading bot
        if self.trading_bot:
            await self.trading_bot.stop()  # type: ignore

        # Final report
        try:
            from alpaca.trading.client import TradingClient

            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            client = TradingClient(api_key, secret_key, paper=self.paper_mode)
            account = client.get_account()

            final_value = float(account.portfolio_value)  # type: ignore[union-attr, arg-type]
            final_pnl = final_value - 30000

            print(f"""
📊 FINAL SESSION REPORT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Final Portfolio Value: ${final_value:,.2f}
   Session P&L: ${final_pnl:+,.2f}
   Universe Size: {len(self.universe_mgr.active_universe) if self.universe_mgr else 0}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ System shutdown complete. All positions remain open in paper account.
""")
        except Exception as e:
            logger.warning(f"Error during shutdown summary: {e}")

        logger.info("Shutdown complete.")


async def main() -> None:
    """Main entry point."""

    # Load environment
    from dotenv import load_dotenv

    load_dotenv()

    # Verify market hours
    from alpaca.trading.client import TradingClient

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    client = TradingClient(api_key, secret_key, paper=True)
    clock = client.get_clock()

    print(f"""
⚠️  STARTING DYNAMIC UNIVERSE TRADING SYSTEM
   This will scan for top 25 opportunities and trade them!
   Press Ctrl+C at any time to stop.

⏰ MARKET STATUS:
   Current Time: {datetime.now().strftime("%H:%M:%S ET")}
   Market Open: {"✅ YES" if clock.is_open else "❌ NO (will trade when market opens)"}
   Next Close: {clock.next_close}  # type: ignore[union-attr]
""")

    # Create and run system
    system = DynamicTradingSystem()
    await system.run()


if __name__ == "__main__":
    asyncio.run(main())
