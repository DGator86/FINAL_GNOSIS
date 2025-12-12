#!/usr/bin/env python3
"""Test API connections for Alpaca and Unusual Whales."""

import pytest

# Skip in environments without external API dependencies or credentials
pytest.skip(
    "External API connectivity tests require Alpaca/Unusual Whales dependencies and credentials.",
    allow_module_level=True,
)

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment
load_dotenv()

from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
from engines.inputs.unusual_whales_adapter import UnusualWhalesAdapter
from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter


def test_alpaca_trading():
    """Test Alpaca trading API."""
    print("\n" + "="*80)
    print("🧪 TESTING ALPACA TRADING API")
    print("="*80)
    
    try:
        # Initialize adapter
        adapter = AlpacaBrokerAdapter(paper=True)
        
        # Get account info
        account = adapter.get_account()
        print(f"\n✅ Account Connection Successful:")
        print(f"   Account ID: {account.account_id}")
        print(f"   Cash: ${account.cash:,.2f}")
        print(f"   Buying Power: ${account.buying_power:,.2f}")
        print(f"   Portfolio Value: ${account.portfolio_value:,.2f}")
        print(f"   Equity: ${account.equity:,.2f}")
        
        # Get positions
        positions = adapter.get_positions()
        print(f"\n📊 Current Positions: {len(positions)}")
        if positions:
            for pos in positions:
                print(f"   {pos.symbol}: {pos.quantity} @ ${pos.avg_entry_price:.2f} | P&L: ${pos.unrealized_pnl:+,.2f}")
        else:
            print("   No open positions")
        
        # Get a quote
        quote = adapter.get_latest_quote("SPY")
        if quote:
            print(f"\n💹 Latest Quote for SPY:")
            print(f"   Bid: ${quote['bid']:.2f} x {quote['bid_size']:.0f}")
            print(f"   Ask: ${quote['ask']:.2f} x {quote['ask_size']:.0f}")
        
        print("\n✅ Alpaca Trading API: WORKING")
        return True
        
    except Exception as e:
        print(f"\n❌ Alpaca Trading API Error: {e}")
        return False


def test_alpaca_market_data():
    """Test Alpaca market data API."""
    print("\n" + "="*80)
    print("🧪 TESTING ALPACA MARKET DATA API")
    print("="*80)
    
    try:
        # Initialize adapter
        adapter = AlpacaMarketDataAdapter()
        
        # Get historical bars
        end = datetime.now()
        start = end - timedelta(days=5)
        bars = adapter.get_bars("SPY", start, end, timeframe="1Day")
        
        print(f"\n✅ Retrieved {len(bars)} daily bars for SPY:")
        for bar in bars[-3:]:  # Show last 3
            print(f"   {bar.timestamp.strftime('%Y-%m-%d')}: "
                  f"O=${bar.open:.2f} H=${bar.high:.2f} "
                  f"L=${bar.low:.2f} C=${bar.close:.2f} "
                  f"V={bar.volume:,.0f}")
        
        # Get current quote
        quote = adapter.get_quote("SPY")
        print(f"\n💹 Current Quote for SPY:")
        print(f"   Bid: ${quote.bid:.2f} x {quote.bid_size:.0f}")
        print(f"   Ask: ${quote.ask:.2f} x {quote.ask_size:.0f}")
        print(f"   Last: ${quote.last:.2f}")
        
        print("\n✅ Alpaca Market Data API: WORKING")
        return True
        
    except Exception as e:
        print(f"\n❌ Alpaca Market Data API Error: {e}")
        return False


def test_unusual_whales():
    """Test Unusual Whales API."""
    print("\n" + "="*80)
    print("🧪 TESTING UNUSUAL WHALES API")
    print("="*80)
    
    try:
        # Initialize adapter
        with UnusualWhalesAdapter() as adapter:
            
            # Get options chain
            print("\n🔗 Fetching options chain for SPY...")
            chain = adapter.get_chain("SPY", datetime.now())
            
            if chain:
                print(f"\n✅ Retrieved {len(chain)} option contracts:")
                
                # Show some call examples
                calls = [c for c in chain if c.option_type == "call"][:3]
                puts = [c for c in chain if c.option_type == "put"][:3]
                
                print("\n   Sample Calls:")
                for c in calls:
                    print(f"   {c.symbol}: Strike ${c.strike:.2f}, "
                          f"Exp {c.expiration.strftime('%Y-%m-%d')}, "
                          f"IV {c.implied_volatility:.2%}, "
                          f"Delta {c.delta:.3f}, "
                          f"OI {c.open_interest:.0f}")
                
                print("\n   Sample Puts:")
                for p in puts:
                    print(f"   {p.symbol}: Strike ${p.strike:.2f}, "
                          f"Exp {p.expiration.strftime('%Y-%m-%d')}, "
                          f"IV {p.implied_volatility:.2%}, "
                          f"Delta {p.delta:.3f}, "
                          f"OI {p.open_interest:.0f}")
            else:
                print("   ⚠️  No options chain data returned")
            
            # Get unusual activity
            print("\n🚨 Fetching unusual options activity...")
            activity = adapter.get_unusual_activity()
            if activity:
                print(f"\n✅ Retrieved {len(activity)} unusual activity records")
                # Show first few
                for act in activity[:3]:
                    ticker = act.get("ticker", "N/A")
                    strike = act.get("strike", 0)
                    exp = act.get("expiration_date", "N/A")
                    print(f"   {ticker} ${strike} {exp}")
            else:
                print("   ⚠️  No unusual activity data returned")
            
            # Get IV
            print("\n📊 Fetching implied volatility for SPY...")
            iv = adapter.get_implied_volatility("SPY")
            if iv:
                print(f"\n✅ SPY IV: {iv:.2%}")
            else:
                print("   ⚠️  No IV data returned")
        
        print("\n✅ Unusual Whales API: WORKING")
        return True
        
    except Exception as e:
        print(f"\n❌ Unusual Whales API Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all API tests."""
    print("\n" + "="*80)
    print("🚀 SUPER GNOSIS API CONNECTION TESTS")
    print("="*80)
    
    results = {
        "Alpaca Trading": test_alpaca_trading(),
        "Alpaca Market Data": test_alpaca_market_data(),
        "Unusual Whales": test_unusual_whales(),
    }
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for api, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {api}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All API connections working!")
    else:
        print("\n⚠️  Some API connections failed. Check credentials and network.")
    
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
