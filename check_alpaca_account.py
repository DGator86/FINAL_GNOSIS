#!/usr/bin/env python3
"""
Comprehensive Alpaca Account Information Tool
Shows account status, balances, positions, orders, and portfolio allocation
"""

import os
import sys
from datetime import datetime, timedelta

try:
    from alpaca_trade_api import REST
except ImportError:
    print("Error: alpaca-trade-api not installed")
    print("Install with: pip install alpaca-trade-api")
    sys.exit(1)


def format_currency(value):
    """Format value as currency"""
    try:
        return f"${float(value):,.2f}"
    except (ValueError, TypeError):
        return "N/A"


def format_percent(value):
    """Format value as percentage"""
    try:
        return f"{float(value) * 100:.2f}%"
    except (ValueError, TypeError):
        return "N/A"


def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def check_alpaca_account():
    """Check and display Alpaca account information"""

    # Get credentials from environment
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not secret_key:
        print("❌ Error: Missing Alpaca credentials")
        print("\nRequired environment variables:")
        print("  - ALPACA_API_KEY")
        print("  - ALPACA_SECRET_KEY (or ALPACA_API_SECRET)")
        print("  - ALPACA_BASE_URL (optional, defaults to paper trading)")
        print("\nPlease set these in your .env file")
        return False

    try:
        # Initialize API client
        api = REST(api_key, secret_key, base_url)

        # Determine if paper or live
        is_paper = "paper" in base_url.lower()
        mode = "PAPER TRADING" if is_paper else "LIVE TRADING"

        print_section(f"Alpaca Account Overview ({mode})")
        print(f"API Endpoint: {base_url}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Get account information
        print_section("Account Status")
        account = api.get_account()

        print(f"Account ID: {account.id}")
        print(f"Account Status: {account.status}")
        print(f"Account Number: {account.account_number}")
        print(f"Created At: {account.created_at}")
        print(f"Trading Blocked: {account.trading_blocked}")
        print(f"Transfers Blocked: {account.transfers_blocked}")
        print(f"Pattern Day Trader: {account.pattern_day_trader}")

        # Account balances
        print_section("Account Balances")
        print(f"Equity: {format_currency(account.equity)}")
        print(f"Cash: {format_currency(account.cash)}")
        print(f"Buying Power: {format_currency(account.buying_power)}")
        print(f"Portfolio Value: {format_currency(account.portfolio_value)}")
        print(f"Long Market Value: {format_currency(account.long_market_value)}")
        print(f"Short Market Value: {format_currency(account.short_market_value)}")

        # P&L Information
        print_section("Profit & Loss")

        # Calculate daily P&L
        try:
            equity = float(account.equity)
            last_equity = float(account.last_equity)
            daily_pl = equity - last_equity
            daily_pl_pct = (daily_pl / last_equity * 100) if last_equity > 0 else 0

            pl_symbol = "📈" if daily_pl >= 0 else "📉"
            print(f"Today's P/L: {format_currency(daily_pl)} ({daily_pl_pct:.2f}%) {pl_symbol}")
            print(f"Last Equity: {format_currency(last_equity)}")
        except (ValueError, TypeError, AttributeError):
            print("Today's P/L: N/A")

        # Positions
        print_section("Open Positions")
        positions = api.list_positions()

        if not positions:
            print("No open positions")
        else:
            print(f"Total Positions: {len(positions)}\n")
            print(f"{'Symbol':<10} {'Qty':<8} {'Type':<8} {'Entry':<12} {'Current':<12} {'P/L $':<12} {'P/L %':<10}")
            print("-" * 80)

            total_unrealized_pl = 0
            for p in positions:
                try:
                    pl_dollar = float(p.unrealized_pl)
                    pl_pct = float(p.unrealized_plpc) * 100
                    total_unrealized_pl += pl_dollar

                    position_type = "Long" if float(p.qty) > 0 else "Short"

                    print(
                        f"{p.symbol:<10} {p.qty:<8} {position_type:<8} "
                        f"{format_currency(p.avg_entry_price):<12} "
                        f"{format_currency(p.current_price):<12} "
                        f"{format_currency(pl_dollar):<12} "
                        f"{pl_pct:>9.2f}%"
                    )
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"{p.symbol:<10} Error: {e}")

            print("-" * 80)
            print(f"{'Total Unrealized P/L:':<50} {format_currency(total_unrealized_pl)}")

        # Recent orders
        print_section("Recent Orders (Last 7 Days)")
        try:
            # Get orders from last 7 days
            after = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            orders = api.list_orders(
                status='all',
                limit=50,
                after=after,
                direction='desc'
            )

            if not orders:
                print("No recent orders")
            else:
                print(f"Total Orders: {len(orders)}\n")
                print(f"{'Time':<20} {'Symbol':<10} {'Side':<6} {'Qty':<8} {'Type':<10} {'Status':<12}")
                print("-" * 80)

                for order in orders[:10]:  # Show last 10
                    try:
                        order_time = datetime.fromisoformat(str(order.created_at).replace('Z', '+00:00'))
                        time_str = order_time.strftime('%Y-%m-%d %H:%M')

                        print(
                            f"{time_str:<20} {order.symbol:<10} {order.side:<6} "
                            f"{order.qty:<8} {order.type:<10} {order.status:<12}"
                        )
                    except (ValueError, TypeError, AttributeError) as e:
                        print(f"Order error: {e}")

                if len(orders) > 10:
                    print(f"\n... and {len(orders) - 10} more orders")

        except Exception as e:
            print(f"Could not fetch recent orders: {e}")

        # Portfolio allocation
        if positions:
            print_section("Portfolio Allocation")
            total_value = float(account.portfolio_value)

            for p in positions:
                try:
                    position_value = float(p.market_value)
                    allocation_pct = (abs(position_value) / total_value * 100) if total_value > 0 else 0

                    print(f"{p.symbol:<10} {format_currency(position_value):<15} {allocation_pct:>6.2f}%")
                except (ValueError, TypeError, AttributeError):
                    continue

        print("\n" + "=" * 80)
        print("✅ Account check completed successfully")
        print("=" * 80 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ Error connecting to Alpaca: {e}")
        print("\nPlease verify:")
        print("1. Your API credentials are correct")
        print("2. Your API key has not expired")
        print("3. You have internet connectivity")
        return False


if __name__ == "__main__":
    check_alpaca_account()
