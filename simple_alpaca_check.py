#!/usr/bin/env python3
"""Simple Alpaca Account Checker using requests library"""

import os
import sys
import requests
from datetime import datetime

def check_alpaca_account():
    """Check Alpaca account using direct API calls"""

    # Get credentials
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not secret_key:
        print("❌ Error: Missing Alpaca credentials")
        print("Required: ALPACA_API_KEY and ALPACA_SECRET_KEY")
        return False

    # Set up headers
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }

    try:
        is_paper = "paper" in base_url.lower()
        mode = "PAPER TRADING" if is_paper else "LIVE TRADING"

        print("=" * 80)
        print(f" Alpaca Account Overview ({mode})")
        print("=" * 80)
        print(f"API Endpoint: {base_url}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Get account information
        print("\n" + "=" * 80)
        print(" Account Status")
        print("=" * 80)

        response = requests.get(f"{base_url}/v2/account", headers=headers)

        if response.status_code == 401:
            print("❌ Authentication failed!")
            print("Please verify your API credentials are correct.")
            return False
        elif response.status_code != 200:
            print(f"❌ API error: {response.status_code}")
            print(f"Response: {response.text}")
            return False

        account = response.json()

        print(f"Account ID: {account.get('id', 'N/A')}")
        print(f"Account Status: {account.get('status', 'N/A')}")
        print(f"Account Number: {account.get('account_number', 'N/A')}")
        print(f"Created At: {account.get('created_at', 'N/A')}")
        print(f"Trading Blocked: {account.get('trading_blocked', 'N/A')}")
        print(f"Pattern Day Trader: {account.get('pattern_day_trader', 'N/A')}")

        # Account balances
        print("\n" + "=" * 80)
        print(" Account Balances")
        print("=" * 80)

        try:
            equity = float(account.get('equity', 0))
            cash = float(account.get('cash', 0))
            buying_power = float(account.get('buying_power', 0))
            portfolio_value = float(account.get('portfolio_value', 0))

            print(f"Equity: ${equity:,.2f}")
            print(f"Cash: ${cash:,.2f}")
            print(f"Buying Power: ${buying_power:,.2f}")
            print(f"Portfolio Value: ${portfolio_value:,.2f}")
            print(f"Long Market Value: ${float(account.get('long_market_value', 0)):,.2f}")
            print(f"Short Market Value: ${float(account.get('short_market_value', 0)):,.2f}")
        except (ValueError, TypeError) as e:
            print(f"Error parsing balances: {e}")

        # P&L Information
        print("\n" + "=" * 80)
        print(" Profit & Loss")
        print("=" * 80)

        try:
            last_equity = float(account.get('last_equity', equity))
            daily_pl = equity - last_equity
            daily_pl_pct = (daily_pl / last_equity * 100) if last_equity > 0 else 0

            pl_symbol = "📈" if daily_pl >= 0 else "📉"
            print(f"Today's P/L: ${daily_pl:,.2f} ({daily_pl_pct:.2f}%) {pl_symbol}")
            print(f"Last Equity: ${last_equity:,.2f}")
        except (ValueError, TypeError):
            print("Today's P/L: N/A")

        # Get positions
        print("\n" + "=" * 80)
        print(" Open Positions")
        print("=" * 80)

        positions_response = requests.get(f"{base_url}/v2/positions", headers=headers)

        if positions_response.status_code == 200:
            positions = positions_response.json()

            if not positions:
                print("No open positions")
            else:
                print(f"Total Positions: {len(positions)}\n")
                print(f"{'Symbol':<10} {'Qty':<8} {'Type':<8} {'Entry':<12} {'Current':<12} {'P/L $':<12} {'P/L %':<10}")
                print("-" * 80)

                total_unrealized_pl = 0
                for p in positions:
                    try:
                        symbol = p.get('symbol', 'N/A')
                        qty = p.get('qty', '0')
                        avg_entry_price = float(p.get('avg_entry_price', 0))
                        current_price = float(p.get('current_price', 0))
                        unrealized_pl = float(p.get('unrealized_pl', 0))
                        unrealized_plpc = float(p.get('unrealized_plpc', 0)) * 100

                        total_unrealized_pl += unrealized_pl
                        position_type = "Long" if float(qty) > 0 else "Short"

                        print(f"{symbol:<10} {qty:<8} {position_type:<8} "
                              f"${avg_entry_price:<11.2f} ${current_price:<11.2f} "
                              f"${unrealized_pl:<11.2f} {unrealized_plpc:>9.2f}%")
                    except (ValueError, TypeError) as e:
                        print(f"{symbol:<10} Error: {e}")

                print("-" * 80)
                print(f"{'Total Unrealized P/L:':<50} ${total_unrealized_pl:,.2f}")
        else:
            print(f"Could not fetch positions: {positions_response.status_code}")

        # Get recent orders
        print("\n" + "=" * 80)
        print(" Recent Orders (Last 10)")
        print("=" * 80)

        orders_response = requests.get(
            f"{base_url}/v2/orders",
            headers=headers,
            params={"status": "all", "limit": 10, "direction": "desc"}
        )

        if orders_response.status_code == 200:
            orders = orders_response.json()

            if not orders:
                print("No recent orders")
            else:
                print(f"{'Time':<20} {'Symbol':<10} {'Side':<6} {'Qty':<8} {'Type':<10} {'Status':<12}")
                print("-" * 80)

                for order in orders:
                    try:
                        created_at = order.get('created_at', '')[:19].replace('T', ' ')
                        symbol = order.get('symbol', 'N/A')
                        side = order.get('side', 'N/A')
                        qty = order.get('qty', '0')
                        order_type = order.get('type', 'N/A')
                        status = order.get('status', 'N/A')

                        print(f"{created_at:<20} {symbol:<10} {side:<6} "
                              f"{qty:<8} {order_type:<10} {status:<12}")
                    except (ValueError, TypeError) as e:
                        print(f"Order error: {e}")
        else:
            print(f"Could not fetch orders: {orders_response.status_code}")

        print("\n" + "=" * 80)
        print("✅ Account check completed successfully")
        print("=" * 80 + "\n")

        return True

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error connecting to Alpaca: {e}")
        print("\nPlease verify:")
        print("1. Your API credentials are correct")
        print("2. Your API key has not expired")
        print("3. You have internet connectivity")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    check_alpaca_account()
