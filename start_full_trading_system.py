#!/usr/bin/env python3
"""
FULL TRADING SYSTEM LAUNCHER
With Multi-Symbol Trading, Options Flow, and All Capabilities
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import signal

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment for full trading
os.environ["ENABLE_TRADING"] = "true"

from gnosis.scanner import MultiTimeframeScanner
from gnosis.trading.live_bot import LiveTradingBot
from loguru import logger
import yaml

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
logger.add("logs/trading_{time}.log", rotation="1 day", retention="7 days")


class FullTradingSystem:
    """
    Full production trading system with all capabilities
    """
    
    def __init__(self):
        self.scanner = None
        self.trading_bots = {}
        self.running = False
        self.trades_executed = 0
        self.alerts_triggered = 0
        
        # Load watchlist
        with open("config/watchlist.yaml", 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Build full symbol list
        self.symbols = self._build_symbol_list()
        
    def _build_symbol_list(self):
        """Get all symbols from watchlist"""
        symbols = []
        watchlist = self.config['watchlist']
        for category in ['indices', 'mega_tech', 'financials', 'high_volume', 'strategic']:
            if category in watchlist:
                for item in watchlist[category]:
                    if item['symbol'] != 'VIX':  # VIX is not tradeable
                        symbols.append(item['symbol'])
        return symbols
    
    async def initialize(self):
        """Initialize all components"""
        
        print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║          🚀 FULL PAPER TRADING SYSTEM ACTIVATED 🚀                    ║
║                                                                        ║
║  Status: LIVE PAPER TRADING ENABLED                                   ║
║  Mode: FULL CAPABILITIES                                              ║
║  Account: Alpaca Paper ($30,000)                                      ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

📊 SYSTEM CONFIGURATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
        
        # Initialize scanner
        logger.info("Initializing multi-timeframe scanner...")
        self.scanner = MultiTimeframeScanner()
        
        # Initialize primary trading bots for top symbols
        # Start with SPY, QQQ, and top tech stocks
        primary_symbols = ['SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA']
        
        for symbol in primary_symbols:
            logger.info(f"Initializing trading bot for {symbol}...")
            self.trading_bots[symbol] = LiveTradingBot(
                symbol=symbol,
                bar_interval="1Min",
                enable_memory=True,
                enable_trading=True,  # TRADING IS ENABLED
                paper_mode=True
            )
        
        print(f"""
✅ SYSTEM INITIALIZED:
   • Scanner: {len(self.symbols)} symbols configured
   • Trading Bots: {len(self.trading_bots)} active ({', '.join(self.trading_bots.keys())})
   • Timeframes: 7 (1m, 5m, 15m, 30m, 1h, 4h, 1d)
   • Data Source: Unusual Whales (primary) + Alpaca (market data)
   • Risk Management: Active (2% stop-loss, 3 position max)
   
⚡ TRADING STATUS: ENABLED - WILL PLACE PAPER ORDERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
    
    async def run_scanner_loop(self):
        """Continuous scanning with trading decisions"""
        
        scan_interval = 60  # Every 60 seconds
        
        while self.running:
            try:
                timestamp = datetime.now().strftime("%H:%M:%S")
                logger.info(f"Starting market scan at {timestamp}...")
                
                # Run comprehensive scan
                results = await self.scanner.scan_all(priority_only=False)  # Full scan
                
                if results:
                    # Process alerts
                    alerts = [r for r in results if r.alert_triggered]
                    if alerts:
                        self.alerts_triggered += len(alerts)
                        logger.warning(f"🚨 {len(alerts)} ALERTS TRIGGERED!")
                        
                        for alert in alerts[:5]:
                            logger.info(f"   {alert.symbol}: {', '.join(alert.alert_reasons)}")
                            
                            # If we have a bot for this symbol, it will handle trading
                            # Otherwise, log for potential future action
                            if alert.symbol not in self.trading_bots:
                                logger.info(f"   → Consider adding {alert.symbol} to active trading")
                    
                    # Show top opportunities
                    top_opps = [r for r in results if r.regime_confidence > 0.7][:5]
                    if top_opps:
                        logger.info("🎯 Top opportunities detected:")
                        for opp in top_opps:
                            logger.info(f"   {opp.symbol}: {opp.regime} (conf: {opp.regime_confidence:.2f})")
                
                # Brief pause before next scan
                await asyncio.sleep(scan_interval)
                
            except Exception as e:
                logger.error(f"Scanner error: {e}")
                await asyncio.sleep(30)
    
    async def monitor_performance(self):
        """Monitor and report performance"""
        
        report_interval = 300  # Every 5 minutes
        
        while self.running:
            try:
                await asyncio.sleep(report_interval)
                
                # Get account status
                from alpaca.trading.client import TradingClient
                api_key = os.getenv("ALPACA_API_KEY")
                secret_key = os.getenv("ALPACA_SECRET_KEY")
                client = TradingClient(api_key, secret_key, paper=True)
                account = client.get_account()
                
                # Get positions
                positions = client.get_all_positions()
                
                print(f"""
📈 PERFORMANCE UPDATE ({datetime.now().strftime('%H:%M:%S')}):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Portfolio Value: ${float(account.portfolio_value):,.2f}
   Cash: ${float(account.cash):,.2f}
   P&L Today: ${float(account.portfolio_value) - 30000:+,.2f}
   Open Positions: {len(positions)}
   Alerts Triggered: {self.alerts_triggered}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
                
                if positions:
                    logger.info("Open positions:")
                    for pos in positions[:5]:  # Show first 5
                        unrealized_pnl = float(pos.unrealized_pl) if hasattr(pos, 'unrealized_pl') else 0
                        logger.info(f"   {pos.symbol}: {pos.qty} shares | P&L: ${unrealized_pnl:+,.2f}")
                
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
    
    async def run(self):
        """Main execution loop"""
        
        try:
            # Initialize system
            await self.initialize()
            
            # Set running flag
            self.running = True
            
            # Create all tasks
            tasks = []
            
            # Scanner task
            scanner_task = asyncio.create_task(self.run_scanner_loop())
            tasks.append(scanner_task)
            
            # Trading bot tasks
            for symbol, bot in self.trading_bots.items():
                logger.info(f"Starting {symbol} trading bot...")
                bot_task = asyncio.create_task(bot.run())
                tasks.append(bot_task)
            
            # Performance monitor
            monitor_task = asyncio.create_task(self.monitor_performance())
            tasks.append(monitor_task)
            
            print("""
🚀 SYSTEM RUNNING - PAPER TRADING ACTIVE!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Monitoring:
• Alpaca Dashboard: https://app.alpaca.markets/paper
• Logs: ./logs/trading_*.log

Press Ctrl+C to stop gracefully.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
            
            # Run all tasks
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            logger.warning("Shutdown signal received...")
            await self.shutdown()
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            import traceback
            traceback.print_exc()
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        
        logger.info("Initiating graceful shutdown...")
        self.running = False
        
        # Stop all trading bots
        for symbol, bot in self.trading_bots.items():
            logger.info(f"Stopping {symbol} bot...")
            bot.running = False
        
        # Final report
        try:
            from alpaca.trading.client import TradingClient
            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            client = TradingClient(api_key, secret_key, paper=True)
            account = client.get_account()
            
            final_value = float(account.portfolio_value)
            final_pnl = final_value - 30000
            
            print(f"""
📊 FINAL SESSION REPORT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Final Portfolio Value: ${final_value:,.2f}
   Session P&L: ${final_pnl:+,.2f}
   Total Alerts: {self.alerts_triggered}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ System shutdown complete. All positions remain open in paper account.
""")
        except:
            pass
        
        logger.info("Shutdown complete.")


async def main():
    """Main entry point"""
    
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
⏰ MARKET STATUS:
   Current Time: {datetime.now().strftime('%H:%M:%S ET')}
   Market Open: {'✅ YES' if clock.is_open else '❌ NO (will trade when market opens)'}
   Next Close: {clock.next_close}
""")
    
    # Create and run system
    system = FullTradingSystem()
    await system.run()


if __name__ == "__main__":
    # Set up signal handlers
    def signal_handler(signum, frame):
        logger.warning("Interrupt received, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Run the system
    try:
        print("""
⚠️  STARTING FULL PAPER TRADING SYSTEM
   This will begin placing paper trades immediately!
   Press Ctrl+C at any time to stop.
""")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Trading system stopped by user.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)