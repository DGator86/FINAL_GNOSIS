#!/usr/bin/env python3
"""
Test script to fetch 50-bar historical data and current positions.
Tests the system's ability to pull live data from Alpaca.
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient

# Load environment variables
load_dotenv()

def fetch_historical_bars(symbols, timeframes):
    """Fetch 50 bars of historical data for each timeframe."""

    # Initialize data client
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        print("❌ ERROR: Alpaca API credentials not found in .env")
        return

    data_client = StockHistoricalDataClient(api_key, secret_key)

    print("=" * 80)
    print("📊 HISTORICAL BAR DATA TEST")
    print("=" * 80)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Bars per timeframe: 50")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # Map timeframe names to Alpaca TimeFrame objects
    tf_map = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "1Hour": TimeFrame.Hour,
        "4Hour": TimeFrame(4, TimeFrameUnit.Hour),
        "1Day": TimeFrame.Day,
    }

    for tf_name, tf_obj in tf_map.items():
        print(f"\n{'─' * 80}")
        print(f"📈 TIMEFRAME: {tf_name}")
        print(f"{'─' * 80}")

        try:
            # Calculate date range based on timeframe
            end = datetime.now()
            if tf_name == "1Min":
                start = end - timedelta(hours=2)  # 50 minutes of 1min bars
            elif tf_name == "5Min":
                start = end - timedelta(hours=5)  # ~50 5min bars
            elif tf_name == "1Hour":
                start = end - timedelta(days=3)  # 50 hours
            elif tf_name == "4Hour":
                start = end - timedelta(days=10)  # ~50 4hour bars
            else:  # 1Day
                start = end - timedelta(days=70)  # 50 trading days

            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=tf_obj,
                start=start,
                end=end,
                limit=50
            )

            # Fetch bars
            bars = data_client.get_stock_bars(request)

            # Display results for each symbol
            for symbol in symbols:
                if symbol in bars:
                    symbol_bars = bars[symbol]
                    bar_count = len(symbol_bars)

                    print(f"\n  {symbol}:")
                    print(f"    Bars Retrieved: {bar_count}")

                    if bar_count > 0:
                        # Show first 3 bars
                        print(f"    First 3 bars:")
                        for i, bar in enumerate(symbol_bars[:3]):
                            print(f"      [{i+1}] {bar.timestamp} | O: ${bar.open:.2f} | H: ${bar.high:.2f} | L: ${bar.low:.2f} | C: ${bar.close:.2f} | V: {bar.volume:,}")

                        # Show last bar
                        if bar_count > 3:
                            print(f"    ...")
                            last_bar = symbol_bars[-1]
                            print(f"    Last bar:")
                            print(f"      [{bar_count}] {last_bar.timestamp} | O: ${last_bar.open:.2f} | H: ${last_bar.high:.2f} | L: ${last_bar.low:.2f} | C: ${last_bar.close:.2f} | V: {last_bar.volume:,}")

                        # Calculate some stats
                        prices = [bar.close for bar in symbol_bars]
                        avg_price = sum(prices) / len(prices)
                        price_change = ((prices[-1] - prices[0]) / prices[0]) * 100

                        print(f"    Avg Close: ${avg_price:.2f}")
                        print(f"    Price Change: {price_change:+.2f}%")
                else:
                    print(f"\n  {symbol}: ❌ No data available")

        except Exception as e:
            print(f"  ❌ Error fetching {tf_name} data: {e}")


def fetch_current_positions():
    """Fetch and display current account and position information."""

    # Initialize trading client
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not secret_key:
        print("❌ ERROR: Alpaca API credentials not found in .env")
        return

    trading_client = TradingClient(api_key, secret_key, paper=True)

    print("\n\n" + "=" * 80)
    print("💼 CURRENT POSITIONS & ACCOUNT STATUS")
    print("=" * 80)

    try:
        # Get account info
        account = trading_client.get_account()

        print(f"\n📊 Account Overview:")
        print(f"  Account ID: {account.id}")
        print(f"  Status: {account.status}")
        print(f"  Portfolio Value: ${float(account.equity):,.2f}")
        print(f"  Cash: ${float(account.cash):,.2f}")
        print(f"  Buying Power: ${float(account.buying_power):,.2f}")

        # Calculate P&L
        pnl = float(account.equity) - float(account.last_equity)
        pnl_pct = (pnl / float(account.last_equity)) * 100 if float(account.last_equity) > 0 else 0

        print(f"  Today's P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")

        # Get positions
        positions = trading_client.get_all_positions()

        print(f"\n📈 Open Positions ({len(positions)}):")

        if len(positions) == 0:
            print("  No open positions")
        else:
            print()
            print(f"  {'Symbol':<8} {'Qty':>8} {'Entry':>10} {'Current':>10} {'P&L':>12} {'P&L %':>10}")
            print(f"  {'-' * 68}")

            total_position_value = 0.0
            total_unrealized_pl = 0.0

            for pos in positions:
                symbol = pos.symbol
                qty = float(pos.qty)
                avg_entry = float(pos.avg_entry_price)
                current = float(pos.current_price)
                unrealized_pl = float(pos.unrealized_pl)
                unrealized_plpc = float(pos.unrealized_plpc) * 100
                position_value = float(pos.market_value)

                total_position_value += position_value
                total_unrealized_pl += unrealized_pl

                side = pos.side
                side_symbol = "🟢" if side == "long" else "🔴"

                print(f"  {symbol:<8} {side_symbol}{qty:>7.0f} ${avg_entry:>9.2f} ${current:>9.2f} ${unrealized_pl:>+10.2f} {unrealized_plpc:>+9.2f}%")

            print(f"  {'-' * 68}")
            print(f"  {'TOTAL':<8} {'':<8} {'':<10} {'':<10} ${total_unrealized_pl:>+10.2f}")
            print(f"\n  Total Position Value: ${total_position_value:,.2f}")
            print(f"  Cash Available: ${float(account.cash):,.2f}")
            print(f"  Portfolio Utilization: {(total_position_value / float(account.equity)) * 100:.1f}%")

    except Exception as e:
        print(f"❌ Error fetching positions: {e}")
        import traceback
        traceback.print_exc()


def get_dynamic_universe_symbols():
    """Get symbols from dynamic universe configuration."""
    # Default to top actively traded symbols if we can't access the scanner
    return ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]


if __name__ == "__main__":
    print("\n🚀 SUPER GNOSIS DATA FETCH TEST")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Get test symbols
    symbols = get_dynamic_universe_symbols()

    # Fetch historical data
    timeframes = ["1Min", "5Min", "1Hour", "4Hour", "1Day"]
    fetch_historical_bars(symbols, timeframes)

    # Fetch current positions
    fetch_current_positions()

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)
