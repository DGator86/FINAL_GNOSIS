#!/usr/bin/env python3
"""
Test script for Unusual Whales API fix.

Tests:
1. API token detection
2. Real endpoint connectivity
3. Options chain retrieval with greeks
4. Stub fallback mechanism
"""

import pytest

pytest.skip(
    "Unusual Whales integration test requires external connectivity and credentials.",
    allow_module_level=True,
)

import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from adapters.unusual_whales_adapter import UnusualWhalesAdapter


def main():
    print("="*80)
    print("🐋 UNUSUAL WHALES API FIX TEST")
    print("="*80)
    print()
    
    # Check token
    token = os.getenv("UNUSUAL_WHALES_API_TOKEN") or os.getenv("UNUSUAL_WHALES_TOKEN") or os.getenv("UNUSUAL_WHALES_API_KEY")
    if token:
        print(f"✅ Token found: {token[:20]}..." if len(token) > 20 else f"✅ Token found: {token}")
    else:
        print("⚠️  No token found - will use stub fallback")
    print()
    
    # Initialize adapter
    print("Initializing adapter...")
    adapter = UnusualWhalesAdapter()
    print()
    
    # Test symbols
    test_symbols = ["SPY", "NVDA", "TSLA"]
    
    for symbol in test_symbols:
        print(f"📊 Testing {symbol}...")
        print("-" * 80)
        
        try:
            contracts = adapter.get_chain(symbol, datetime.now())
            
            if contracts:
                print(f"✅ Retrieved {len(contracts)} contracts")
                
                # Show sample contract
                if len(contracts) > 0:
                    sample = contracts[0]
                    print(f"\nSample Contract:")
                    print(f"   Symbol: {sample.symbol}")
                    print(f"   Strike: ${sample.strike:.2f}")
                    print(f"   Type: {sample.option_type.upper()}")
                    print(f"   Expiration: {sample.expiration.strftime('%Y-%m-%d')}")
                    print(f"   Bid/Ask: ${sample.bid:.2f} / ${sample.ask:.2f}")
                    print(f"   Volume: {sample.volume:,.0f}")
                    print(f"   OI: {sample.open_interest:,.0f}")
                    print(f"   IV: {sample.implied_volatility:.2%}")
                    print(f"   Greeks:")
                    print(f"      Delta: {sample.delta:.4f}")
                    print(f"      Gamma: {sample.gamma:.4f}")
                    print(f"      Theta: {sample.theta:.4f}")
                    print(f"      Vega: {sample.vega:.4f}")
                
                # Check if it's real or stub data
                if any("STUB" in c.symbol for c in contracts[:5]):
                    print("\n⚠️  Using STUB data (API not accessible)")
                else:
                    print("\n✅ REAL API DATA RECEIVED!")
            else:
                print("❌ No contracts returned")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    print("="*80)
    print("🎯 TEST COMPLETE")
    print("="*80)
    print()
    
    if token:
        print("💡 If you see STUB data despite having a token:")
        print("   1. Verify token is valid at https://unusualwhales.com/settings/api")
        print("   2. Check your subscription includes API access")
        print("   3. Endpoint may have changed - check Unusual Whales API docs")
    else:
        print("💡 To use real data:")
        print("   1. Get API token from https://unusualwhales.com")
        print("   2. Add to .env: UNUSUAL_WHALES_API_TOKEN=your_token_here")
        print("   3. Re-run this test")
    print()

if __name__ == "__main__":
    main()
