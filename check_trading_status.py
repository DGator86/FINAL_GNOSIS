#!/usr/bin/env python3
"""Quick trading system health check"""

import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

try:
    from alpaca.trading.client import TradingClient

    # Initialize client
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        print("❌ ERROR: Alpaca credentials not found in environment")
        exit(1)

    client = TradingClient(api_key, secret_key, paper=True)

    # Check market status
    print("=" * 80)
    print("🔍 TRADING SYSTEM HEALTH CHECK")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}\n")

    # Get clock
    clock = client.get_clock()
    market_status = "🟢 OPEN" if clock.is_open else "🔴 CLOSED"
    print(f"Market Status: {market_status}")
    print(f"Next Open:  {clock.next_open}")
    print(f"Next Close: {clock.next_close}\n")

    # Get account
    account = client.get_account()
    print(f"Account Status: {account.status}")
    print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"Cash: ${float(account.cash):,.2f}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}\n")

    # Get positions
    positions = client.get_all_positions()
    print(f"Open Positions: {len(positions)}")

    if positions:
        print("\nCurrent Holdings:")
        print("-" * 80)
        for pos in positions:
            pnl = float(pos.unrealized_pl)
            pnl_pct = float(pos.unrealized_plpc) * 100
            symbol_display = f"{pos.symbol:8}"
            qty_display = f"{float(pos.qty):>8.2f}"
            price_display = f"${float(pos.current_price):>8.2f}"
            pnl_display = f"${pnl:>+10.2f}"
            pct_display = f"{pnl_pct:>+6.2f}%"

            print(
                f"{symbol_display} {qty_display} @ {price_display}  |  P&L: {pnl_display} ({pct_display})"
            )

    print("\n" + "=" * 80)
    print("✅ All systems operational - Ready to trade!")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback

    traceback.print_exc()
    exit(1)
