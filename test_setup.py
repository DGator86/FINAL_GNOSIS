#!/usr/bin/env python3
"""
Test the complete trading system setup
"""

import os
import sys
from pathlib import Path

import pytest

dotenv = pytest.importorskip("dotenv")
load_dotenv = dotenv.load_dotenv

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL")


def _require_alpaca_credentials() -> None:
    """Skip tests that need Alpaca credentials when none are configured."""

    if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
        pytest.skip("Alpaca credentials are not configured; skipping live Alpaca tests")


def test_environment():
    """Assert that critical environment variables are present when required."""

    print("\n" + "=" * 60)
    print("1. TESTING ENVIRONMENT VARIABLES")
    print("=" * 60)

    if ALPACA_API_KEY and ALPACA_SECRET_KEY:
        print("✅ Alpaca credentials found")
        print(f"   API Key: {ALPACA_API_KEY[:10]}...")
        print(f"   Base URL: {ALPACA_BASE_URL}")
    else:
        pytest.skip("Alpaca credentials missing; skipping environment-dependent checks")

    # Check Unusual Whales
    uw_key = os.getenv("UNUSUAL_WHALES_API_KEY")
    if uw_key and uw_key != "your_unusual_whales_key_here":
        print("✅ Unusual Whales API key found")
    else:
        print("⚠️  Unusual Whales API key not configured (will use test key)")

    assert True


def test_alpaca_connection():
    """Test Alpaca API connection"""
    _require_alpaca_credentials()

    print("\n" + "="*60)
    print("2. TESTING ALPACA CONNECTION")
    print("="*60)

    try:
        from alpaca.trading.client import TradingClient

        # Create client
        client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

        # Get account info
        account = client.get_account()
        
        print("✅ Successfully connected to Alpaca Paper Trading")
        print(f"\n📊 Account Information:")
        print(f"   Account ID: {account.id}")
        print(f"   Status: {account.status}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Pattern Day Trader: {account.pattern_day_trader}")
        print(f"   Trade Count: {account.daytrade_count}")
        
        # Check if market is open
        clock = client.get_clock()
        print(f"\n⏰ Market Status:")
        print(f"   Market Open: {'✅ YES' if clock.is_open else '❌ NO'}")
        print(f"   Next Open: {clock.next_open}")
        print(f"   Next Close: {clock.next_close}")
        
    except Exception as e:
        print(f"❌ Failed to connect to Alpaca: {e}")
        pytest.fail("Alpaca connection failed")


def test_data_fetch():
    """Test fetching market data"""
    _require_alpaca_credentials()

    print("\n" + "="*60)
    print("3. TESTING MARKET DATA FETCH")
    print("="*60)

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest

        # Create data client
        client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        
        # Test symbols
        symbols = ["SPY", "AAPL", "NVDA"]
        
        # Get latest quotes
        request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
        quotes = client.get_stock_latest_quote(request)
        
        print("✅ Successfully fetched market data")
        print(f"\n📈 Latest Quotes:")
        for symbol in symbols:
            if symbol in quotes:
                quote = quotes[symbol]
                print(f"   {symbol}: ${quote.ask_price:.2f} (ask) / ${quote.bid_price:.2f} (bid)")
                print(f"          Size: {quote.ask_size} x {quote.bid_size}")
        
    except Exception as e:
        print(f"❌ Failed to fetch market data: {e}")
        pytest.fail("Alpaca data fetch failed")


def test_watchlist():
    """Test watchlist configuration"""
    print("\n" + "="*60)
    print("4. TESTING WATCHLIST CONFIGURATION")
    print("="*60)

    try:
        import yaml
        
        config_path = Path(__file__).parent / "config" / "watchlist.yaml"
        
        if not config_path.exists():
            pytest.fail(f"Watchlist config not found at {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Count symbols
        total_symbols = 0
        categories = ['indices', 'mega_tech', 'financials', 'high_volume', 'strategic']
        
        print("✅ Watchlist loaded successfully")
        print(f"\n📋 Symbol Categories:")
        
        for category in categories:
            if category in config['watchlist']:
                count = len(config['watchlist'][category])
                total_symbols += count
                print(f"   {category:15}: {count:2} symbols")
        
        print(f"\n   Total symbols: {total_symbols}")
        
        # Show timeframes
        print(f"\n⏱️  Configured Timeframes:")
        for tf in config['timeframes']:
            print(f"   {tf['interval']:8} - {tf['bars']} bars - Priority: {tf['priority']}")
        
    except Exception as e:
        print(f"❌ Failed to load watchlist: {e}")
        pytest.fail("Watchlist configuration invalid")


@pytest.mark.asyncio
@pytest.mark.skip(reason="Async scanner test requires pytest-asyncio or similar plugin.")
async def test_scanner():
    """Test the multi-timeframe scanner"""
    print("\n" + "="*60)
    print("5. TESTING MULTI-TIMEFRAME SCANNER")
    print("="*60)
    
    try:
        from gnosis.scanner import MultiTimeframeScanner
        
        print("🔄 Initializing scanner...")
        scanner = MultiTimeframeScanner()
        
        print(f"✅ Scanner initialized with {len(scanner.symbols)} symbols")
        
        # Test scan on SPY
        print("\n🔍 Testing scan on SPY/5Min...")
        result = await scanner.scan_symbol_timeframe("SPY", "5Min", 100)
        
        if result:
            print("✅ Scan successful!")
            print(f"   Price: ${result.price:.2f}")
            print(f"   Volume: {result.volume:,.0f}")
            print(f"   Regime: {result.regime} (confidence: {result.regime_confidence:.2f})")
            print(f"   Signals: {result.signals}")
        else:
            print("⚠️  Scan returned no data (market may be closed)")
        
        return True
        
    except Exception as e:
        print(f"❌ Scanner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unusual_whales():
    """Test Unusual Whales connection"""
    print("\n" + "="*60)
    print("6. TESTING UNUSUAL WHALES API")
    print("="*60)

    if not os.getenv("RUN_UW_LIVE"):
        pytest.skip("Unusual Whales live test disabled; set RUN_UW_LIVE=1 to enable")

    try:
        from engines.inputs.unusual_whales_adapter import UnusualWhalesAdapter

        # Try with configured key or test key
        uw_key = os.getenv("UNUSUAL_WHALES_API_KEY")
        if uw_key and uw_key != "your_unusual_whales_key_here":
            adapter = UnusualWhalesAdapter(api_key=uw_key)
            print("🔄 Testing with configured API key...")
        else:
            adapter = UnusualWhalesAdapter(use_test_key=True)
            print("🔄 Testing with public test key...")
        
        # Test market tide
        tide = adapter.get_market_tide()
        if tide and 'data' in tide:
            print("✅ Successfully connected to Unusual Whales")
            print(f"   Market Tide: {tide.get('data', 'N/A')}")
        else:
            print("⚠️  Unusual Whales returned empty data")
        
    except Exception as e:
        print(f"⚠️  Unusual Whales test failed: {e}")
        pytest.fail("Unusual Whales live test failed")


async def main():
    """Run all tests"""
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║                    TRADING SYSTEM SETUP TEST                          ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    all_passed = True
    
    # Test environment
    if not test_environment():
        all_passed = False
    
    # Test Alpaca
    if not test_alpaca_connection():
        all_passed = False
    
    # Test data fetch
    if not test_data_fetch():
        all_passed = False
    
    # Test watchlist
    if not test_watchlist():
        all_passed = False
    
    # Test scanner
    if not await test_scanner():
        all_passed = False
    
    # Test Unusual Whales (non-critical)
    test_unusual_whales()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if all_passed:
        print("""
✅ ALL CRITICAL TESTS PASSED!

The trading system is properly configured and ready to run.

To start the system:
1. For basic paper trading: python start_paper_trading.py
2. For full scanner + trading: python start_scanner_trading.py
3. With dashboard: python start_with_dashboard.py

Note: Set enable_trading=True in the scripts to enable actual paper trading.
Currently in DRY RUN mode for safety.
        """)
    else:
        print("""
❌ SOME TESTS FAILED!

Please fix the issues above before running the trading system.
Common issues:
1. Missing .env file - Created, should be working now
2. Invalid API keys - Check your Alpaca credentials
3. Market closed - Some data may not be available outside market hours
        """)
    
    return all_passed


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())