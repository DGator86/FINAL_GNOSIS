#!/usr/bin/env python3
"""
Live Unusual Whales Data Monitor - Shows REAL data from your API
"""

import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Add engines to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'engines'))

from engines.inputs.unusual_whales_adapter import UnusualWhalesOptionsAdapter as UnusualWhalesAdapter

# Load environment variables
load_dotenv()

# Your tickers
TICKERS = ['SPY', 'QQQ', 'IWM', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']


class LiveUWMonitor:
    """Monitor showing REAL data from Unusual Whales API."""

    def __init__(self):
        token = os.getenv("UNUSUAL_WHALES_API_TOKEN")
        if not token or token == "your_unusual_whales_api_token_here":
            print("❌ UNUSUAL_WHALES_API_TOKEN not configured in .env")
            sys.exit(1)

        self.uw_adapter = UnusualWhalesAdapter(token=token)
        print(f"✅ Connected to Unusual Whales API")

    def fetch_live_data(self, ticker: str) -> dict:
        """Fetch live data for a ticker."""
        data = {
            'ticker': ticker,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'options_count': 0,
            'greeks': None,
            'dark_pool': None,
            'flow_alerts': None,
        }

        try:
            # Get options chain
            print(f"   Fetching options chain for {ticker}...", end=" ", flush=True)
            options = self.uw_adapter.get_chain(ticker, datetime.now())
            data['options_count'] = len(options)
            print(f"✅ {len(options)} contracts")

            # Get Greek exposure
            print(f"   Fetching Greek exposure for {ticker}...", end=" ", flush=True)
            greeks = self.uw_adapter.get_greek_exposure(ticker)
            if greeks:
                data['greeks'] = {
                    'gex': greeks.get('total_gamma_exposure', 0),
                    'vex': greeks.get('total_vanna_exposure', 0),
                    'charm': greeks.get('total_charm_exposure', 0),
                }
                print(f"✅ GEX: {data['greeks']['gex']:.2f}")
            else:
                print("⚠️  No data")

            # Get dark pool
            print(f"   Fetching dark pool for {ticker}...", end=" ", flush=True)
            dark_pool = self.uw_adapter.get_dark_pool(ticker, limit=10)
            if dark_pool:
                data['dark_pool'] = {
                    'count': len(dark_pool),
                    'latest': dark_pool[0] if dark_pool else None
                }
                print(f"✅ {len(dark_pool)} trades")
            else:
                print("⚠️  No data")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

        return data

    def display_dashboard(self, ticker_data: dict):
        """Display comprehensive dashboard."""
        os.system('clear')

        print("=" * 120)
        print("🔴 LIVE UNUSUAL WHALES DATA MONITOR".center(120))
        print("=" * 120)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 120)
        print()

        for ticker, data in ticker_data.items():
            print(f"\n┌{'─' * 118}┐")
            print(f"│ 📊 {ticker:6} │ Last Update: {data['timestamp']:8} │ Options Contracts: {data['options_count']:5} │")
            print(f"├{'─' * 118}┤")

            # Greek Exposure
            if data['greeks']:
                gex = data['greeks']['gex']
                vex = data['greeks']['vex']
                charm = data['greeks']['charm']
                print(f"│   📊 Greek Exposure:")
                print(f"│      • GEX (Gamma):   {gex:>15,.2f}   │ Measures dealer hedging pressure")
                print(f"│      • VEX (Vanna):   {vex:>15,.2f}   │ Sensitivity to volatility & spot")
                print(f"│      • Charm:         {charm:>15,.2f}   │ Theta decay + delta impact")
            else:
                print(f"│   📊 Greek Exposure:   No data available")

            # Dark Pool
            if data['dark_pool'] and data['dark_pool']['count'] > 0:
                dp_count = data['dark_pool']['count']
                latest = data['dark_pool']['latest']
                if latest:
                    volume = latest.get('volume', 0)
                    price = latest.get('price', 0)
                    print(f"│   🌑 Dark Pool:")
                    print(f"│      • Recent Trades: {dp_count:>5}       │ Latest: {volume:,} shares @ ${price:.2f}")
            else:
                print(f"│   🌑 Dark Pool:        No recent trades")

            print(f"└{'─' * 118}┘")

        print()
        print("=" * 120)
        print("🟢 This is LIVE data from your Unusual Whales API")
        print("=" * 120)
        print()

    def run(self, refresh_seconds: int = 30):
        """Run the live monitor."""
        print("🎬 Starting Live Unusual Whales Data Monitor...")
        print(f"⏱️  Refreshing every {refresh_seconds} seconds")
        print("Press Ctrl+C to stop")
        print()

        while True:
            try:
                ticker_data = {}

                print(f"\n🔄 Fetching data for {len(TICKERS)} tickers...")
                for ticker in TICKERS:
                    print(f"\n📡 {ticker}:")
                    ticker_data[ticker] = self.fetch_live_data(ticker)

                # Display dashboard
                self.display_dashboard(ticker_data)

                # Wait before next refresh
                print(f"⏳ Next refresh in {refresh_seconds} seconds...\n")
                time.sleep(refresh_seconds)

            except KeyboardInterrupt:
                print("\n\n✅ Monitor stopped")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print(f"⏳ Retrying in {refresh_seconds} seconds...")
                time.sleep(refresh_seconds)


if __name__ == "__main__":
    monitor = LiveUWMonitor()
    # Refresh every 30 seconds (API rate limits)
    monitor.run(refresh_seconds=30)
