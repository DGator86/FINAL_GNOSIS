#!/usr/bin/env python3
"""
Portfolio & Positions Monitor - Shows your current positions, orders, and account status
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus

# Load environment
load_dotenv()

def format_currency(value):
    """Format currency values"""
    return f"${value:,.2f}"

def format_percent(value):
    """Format percentage values"""
    return f"{value:+.2f}%"

def main():
    # Initialize Alpaca client
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')

    if not api_key or api_key == 'your_alpaca_api_key_here':
        print("❌ Alpaca API credentials not configured in .env")
        return

    client = TradingClient(api_key, api_secret, paper=True)

    print("=" * 120)
    print("📊 PORTFOLIO & POSITIONS MONITOR".center(120))
    print("=" * 120)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)
    print()

    # Get account info
    try:
        account = client.get_account()

        print("┌" + "─" * 118 + "┐")
        print("│ 💰 ACCOUNT SUMMARY" + " " * 98 + "│")
        print("├" + "─" * 118 + "┤")

        equity = float(account.equity)
        cash = float(account.cash)
        buying_power = float(account.buying_power)
        portfolio_value = float(account.portfolio_value)

        # Calculate P&L
        last_equity = float(account.last_equity)
        pnl = equity - last_equity
        pnl_pct = (pnl / last_equity * 100) if last_equity > 0 else 0

        print(f"│   Portfolio Value:    {format_currency(portfolio_value):>15}                                                            │")
        print(f"│   Cash:               {format_currency(cash):>15}                                                            │")
        print(f"│   Buying Power:       {format_currency(buying_power):>15}                                                            │")
        print(f"│   Day P&L:            {format_currency(pnl):>15}  ({format_percent(pnl_pct):>8})                                          │")
        print(f"│   Status:             {account.status:>15}                                                            │")
        print("└" + "─" * 118 + "┘")
        print()

    except Exception as e:
        print(f"❌ Error getting account info: {str(e)}")
        print()

    # Get positions
    try:
        positions = client.get_all_positions()

        if positions:
            print("┌" + "─" * 118 + "┐")
            print("│ 📈 OPEN POSITIONS" + " " * 100 + "│")
            print("├" + "─" * 118 + "┤")
            print("│ Symbol    │ Qty    │ Entry Price │ Current Price │ Market Value │ P&L         │ P&L %      │ Side     │")
            print("├" + "─" * 118 + "┤")

            total_pnl = 0
            for pos in positions:
                symbol = pos.symbol
                qty = float(pos.qty)
                entry = float(pos.avg_entry_price)
                current = float(pos.current_price)
                market_value = float(pos.market_value)
                unrealized_pl = float(pos.unrealized_pl)
                unrealized_plpc = float(pos.unrealized_plpc) * 100
                side = pos.side

                total_pnl += unrealized_pl

                pnl_color = "🟢" if unrealized_pl >= 0 else "🔴"

                print(f"│ {symbol:9} │ {qty:6.0f} │ {format_currency(entry):>11} │ {format_currency(current):>13} │ "
                      f"{format_currency(market_value):>12} │ {pnl_color} {format_currency(unrealized_pl):>9} │ "
                      f"{format_percent(unrealized_plpc):>9} │ {side:8} │")

            print("├" + "─" * 118 + "┤")
            print(f"│ TOTAL UNREALIZED P&L: {format_currency(total_pnl):>15}" + " " * 85 + "│")
            print("└" + "─" * 118 + "┘")
            print()
        else:
            print("📭 No open positions")
            print()

    except Exception as e:
        print(f"❌ Error getting positions: {str(e)}")
        print()

    # Get recent orders
    try:
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=20
        )
        orders = client.get_orders(filter=request)

        if orders:
            print("┌" + "─" * 118 + "┐")
            print("│ 📝 RECENT ORDERS (Last 20)" + " " * 90 + "│")
            print("├" + "─" * 118 + "┤")
            print("│ Symbol    │ Side  │ Qty    │ Type      │ Status      │ Filled Price │ Submitted Time      │")
            print("├" + "─" * 118 + "┤")

            for order in orders[:20]:
                symbol = order.symbol
                side = "BUY " if order.side == OrderSide.BUY else "SELL"
                qty = float(order.qty)
                order_type = str(order.type.value) if order.type else "N/A"
                status = str(order.status.value) if order.status else "N/A"
                filled_price = format_currency(float(order.filled_avg_price)) if order.filled_avg_price else "N/A"
                submitted = order.submitted_at.strftime('%Y-%m-%d %H:%M:%S') if order.submitted_at else "N/A"

                status_emoji = "✅" if status == "filled" else "⏳" if status == "pending_new" else "❌"

                print(f"│ {symbol:9} │ {side:5} │ {qty:6.0f} │ {order_type:9} │ {status_emoji} {status:9} │ "
                      f"{filled_price:>12} │ {submitted:19} │")

            print("└" + "─" * 118 + "┘")
            print()
        else:
            print("📭 No recent orders")
            print()

    except Exception as e:
        print(f"❌ Error getting orders: {str(e)}")
        print()

    print("=" * 120)
    print("💡 Refresh: python3 monitor_positions.py")
    print("=" * 120)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✅ Monitor stopped")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
