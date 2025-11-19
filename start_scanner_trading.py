#!/usr/bin/env python3
"""
Start Multi-Timeframe Scanner with Trading Bot

Launches the enhanced trading system with:
- Multi-timeframe scanning (1m to 1d)
- 25+ stock watchlist
- Unusual Whales integration
- Automated paper trading
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import signal

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from gnosis.scanner import MultiTimeframeScanner
from gnosis.trading.live_bot import LiveTradingBot
from gnosis.dashboard.dashboard_server import app
import uvicorn
from threading import Thread


class TradingSystem:
    """
    Combined trading system with scanner and execution
    """
    
    def __init__(self):
        self.scanner = None
        self.trading_bots = {}
        self.dashboard_thread = None
        self.running = False
        
    async def initialize(self):
        """Initialize all components"""
        
        print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║              🚀 SUPER GNOSIS TRADING SYSTEM v3.0                      ║
║                                                                        ║
║  Components:                                                           ║
║    • Multi-Timeframe Scanner (7 timeframes)                          ║
║    • 30 High-Liquidity Stocks                                        ║
║    • Unusual Whales Options Flow                                     ║
║    • Alpaca Paper Trading                                            ║
║    • Real-time Dashboard                                             ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
        """)
        
        print("📊 Initializing components...")
        
        # Initialize scanner
        print("   • Loading scanner...")
        self.scanner = MultiTimeframeScanner()
        
        # Initialize primary trading bot (SPY for now)
        print("   • Setting up trading bot...")
        self.trading_bots['SPY'] = LiveTradingBot(
            symbol="SPY",
            bar_interval="1Min",
            enable_memory=True,
            enable_trading=True,   # TRADING ENABLED - Full paper trading active
            paper_mode=True
        )
        
        print("\n✅ System initialized successfully!")
        
    async def run_scanner_loop(self):
        """Continuous scanning loop"""
        
        scan_interval = 60  # Scan every 60 seconds
        
        while self.running:
            try:
                print(f"\n📡 Starting scan at {datetime.now().strftime('%H:%M:%S')}...")
                
                # Run scan (priority items only for speed)
                results = await self.scanner.scan_all(priority_only=True)
                
                # Show summary
                if results:
                    self.scanner.print_summary(results, top_n=10)
                    
                    # Check for actionable alerts
                    alerts = [r for r in results if r.alert_triggered]
                    if alerts:
                        print(f"\n🚨 {len(alerts)} ALERTS REQUIRE ATTENTION!")
                        for alert in alerts[:3]:
                            print(f"   • {alert.symbol}: {', '.join(alert.alert_reasons)}")
                    
                    # Update trading decisions based on scan
                    await self.update_trading_decisions(results)
                
                # Wait before next scan
                print(f"\n⏰ Next scan in {scan_interval} seconds...")
                await asyncio.sleep(scan_interval)
                
            except Exception as e:
                print(f"❌ Scanner error: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def update_trading_decisions(self, scan_results):
        """Update trading decisions based on scan results"""
        
        # For now, just focus on SPY if it has strong signals
        spy_results = [r for r in scan_results if r.symbol == "SPY"]
        
        if spy_results:
            # Find the most relevant timeframe (5Min for day trading)
            five_min = next((r for r in spy_results if r.timeframe == "5Min"), None)
            
            if five_min and five_min.alert_triggered:
                print(f"\n💡 SPY Alert: {', '.join(five_min.alert_reasons)}")
                # Trading bot will handle this in its own loop
    
    def start_dashboard(self):
        """Start the web dashboard in a separate thread"""
        
        def run_dashboard():
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=8080,
                log_level="error"  # Reduce noise
            )
        
        self.dashboard_thread = Thread(target=run_dashboard, daemon=True)
        self.dashboard_thread.start()
        print("🌐 Dashboard available at http://localhost:8080")
    
    async def run(self):
        """Main execution loop"""
        
        try:
            # Initialize system
            await self.initialize()
            
            # Start dashboard
            self.start_dashboard()
            
            # Set running flag
            self.running = True
            
            # Create tasks
            tasks = []
            
            # Scanner task
            scanner_task = asyncio.create_task(self.run_scanner_loop())
            tasks.append(scanner_task)
            
            # Trading bot task (SPY)
            bot_task = asyncio.create_task(self.trading_bots['SPY'].run())
            tasks.append(bot_task)
            
            print("\n🚀 System running! Press Ctrl+C to stop.")
            print("-" * 60)
            
            # Run all tasks
            await asyncio.gather(*tasks)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Shutdown signal received...")
            await self.shutdown()
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        
        print("🛑 Shutting down system...")
        self.running = False
        
        # Stop trading bots
        for symbol, bot in self.trading_bots.items():
            print(f"   • Stopping {symbol} bot...")
            bot.running = False
        
        print("✅ Shutdown complete. Goodbye!")


async def main():
    """Main entry point"""
    
    # Print configuration summary
    print("""
📋 CONFIGURATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Data Sources:
  • Primary: Unusual Whales (options flow, sentiment)
  • Secondary: Alpaca (market data, execution)
  • Tertiary: yfinance (backup)

Watchlist (30 symbols):
  • Indices: SPY, QQQ, IWM, DIA, VIX, TLT, GLD
  • Mega Tech: AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, AMD
  • Financials: JPM, BAC, GS, XLF
  • High Volume: PLTR, NIO, SOFI, F, COIN, MARA
  • Strategic: NFLX, DIS, BA, V, UNH

Timeframes:
  • 1-minute (scalping, momentum)
  • 5-minute (day trading, breakouts)
  • 15-minute (intraday swings)
  • 30-minute (trend confirmation)
  • 1-hour (swing trading)
  • 4-hour (position trading)
  • 1-day (portfolio positioning)

Risk Management:
  • Max positions: 3
  • Position size: 2-15% per trade
  • Stop loss: 2% (dynamic)
  • Take profit: 4% (2:1 R:R)
  • Daily loss limit: -5%
  • Circuit breaker: -10%

Trading Mode: PAPER TRADING (Alpaca)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    
    # Confirm start
    try:
        response = input("\n🤔 Start trading system? (y/n): ")
        if response.lower() != 'y':
            print("❌ Cancelled.")
            return
    except KeyboardInterrupt:
        print("\n❌ Cancelled.")
        return
    
    # Create and run system
    system = TradingSystem()
    await system.run()


if __name__ == "__main__":
    # Set up signal handlers
    def signal_handler(signum, frame):
        print("\n⚠️  Interrupt received, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the system
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Stopped by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)