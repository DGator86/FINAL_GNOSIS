#!/usr/bin/env python3
"""
Start Paper Trading Bot

Quick launcher for live trading with sensible defaults.
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from gnosis.trading.live_bot import LiveTradingBot


async def main():
    """Run bot with configuration"""
    
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║                  📄 PAPER TRADING BOT LAUNCHER                        ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

Configuration:
  Symbol:   SPY
  Interval: 1-minute bars
  Memory:   ✅ Enabled (episodic learning)
  Trading:  ❌ DRY RUN (set enable_trading=True to trade)
  Mode:     📄 Paper (Alpaca paper account)
  
Risk Controls:
  Max Position:     15% per trade
  Max Positions:    3 simultaneous
  Daily Loss Limit: -5%
  Stop Loss:        -2% (dynamic based on confidence)
  Take Profit:      +4% (2:1 reward:risk)
  Circuit Breaker:  -10% drawdown

Press Ctrl+C to stop gracefully.
""")
    
    # Confirm start
    try:
        response = input("Start bot? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    except KeyboardInterrupt:
        print("\nCancelled.")
        return
    
    # Create and run bot
    bot = LiveTradingBot(
        symbol="SPY",
        bar_interval="1Min",
        enable_memory=True,      # Use episodic memory
        enable_trading=True,     # TRADING ENABLED - Will place paper orders
        paper_mode=True          # Paper trading account
    )
    
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Stopped by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
