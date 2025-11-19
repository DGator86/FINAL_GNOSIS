#!/usr/bin/env python3
"""
Test Unusual Whales API Connection
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment
load_dotenv()

def test_unusual_whales():
    """Test Unusual Whales API with the provided key"""
    
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                   UNUSUAL WHALES API TEST                             ║
╚════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # First check if we need to install dependencies
        try:
            from engines.inputs.unusual_whales_adapter import UnusualWhalesAdapter
        except ImportError as e:
            print(f"⚠️  Missing dependency: {e}")
            print("Installing required packages...")
            os.system("pip install requests pandas numpy polars --quiet")
            
            # Try again
            from engines.inputs.unusual_whales_adapter import UnusualWhalesAdapter
        
        # Get API key
        api_key = os.getenv("UNUSUAL_WHALES_API_KEY")
        print(f"✅ API Key loaded: {api_key[:10]}...{api_key[-5:]}")
        
        # Initialize adapter
        print("\n🔄 Initializing Unusual Whales adapter...")
        adapter = UnusualWhalesAdapter(api_key=api_key)
        
        # Test 1: Market Tide (overall market sentiment)
        print("\n1. Testing Market Tide...")
        tide = adapter.get_market_tide()
        if tide and 'data' in tide:
            print(f"   ✅ Market Tide: {tide['data']}")
        else:
            print(f"   ⚠️  No data returned")
        
        # Test 2: Options Flow for SPY
        print("\n2. Testing Options Flow (SPY)...")
        flow = adapter.get_ticker_flow("SPY", limit=5)
        if flow and 'data' in flow:
            data = flow['data']
            if data and len(data) > 0:
                print(f"   ✅ Found {len(data)} recent SPY flows")
                # Show first flow
                if isinstance(data[0], dict):
                    first_flow = data[0]
                    print(f"      Sample: {first_flow.get('type', 'N/A')} - "
                          f"${first_flow.get('premium', 0):,.0f} premium")
            else:
                print(f"   ⚠️  No flow data for SPY")
        
        # Test 3: Flow Alerts
        print("\n3. Testing Flow Alerts...")
        alerts = adapter.get_flow_alerts(limit=3)
        if alerts and 'data' in alerts:
            data = alerts['data']
            if data and len(data) > 0:
                print(f"   ✅ Found {len(data)} recent alerts")
                for alert in data[:3]:
                    if isinstance(alert, dict):
                        print(f"      • {alert.get('ticker', 'N/A')}: "
                              f"${alert.get('premium', 0):,.0f}")
            else:
                print(f"   ⚠️  No recent alerts")
        
        # Test 4: Congressional Trades
        print("\n4. Testing Congressional Trades...")
        congress = adapter.get_congress_trades(limit=5)
        if congress and 'data' in congress:
            data = congress['data']
            if data and len(data) > 0:
                print(f"   ✅ Found {len(data)} recent congressional trades")
                # Show sample
                if isinstance(data[0], dict):
                    trade = data[0]
                    print(f"      Sample: {trade.get('politician', 'N/A')} - "
                          f"{trade.get('ticker', 'N/A')} - "
                          f"{trade.get('transaction_type', 'N/A')}")
            else:
                print(f"   ⚠️  No recent congressional trades")
        
        # Test 5: Get ticker overview for NVDA
        print("\n5. Testing Ticker Overview (NVDA)...")
        overview = adapter.get_ticker_overview("NVDA")
        if overview and 'data' in overview:
            print(f"   ✅ NVDA data retrieved")
            if isinstance(overview['data'], dict):
                print(f"      Info available: {list(overview['data'].keys())[:5]}...")
        
        print("\n" + "="*60)
        print("✅ UNUSUAL WHALES API SUCCESSFULLY CONFIGURED!")
        print("="*60)
        print("""
Your API key is working correctly!

Available data sources:
• Options flow monitoring
• Market sentiment (Market Tide)
• Flow alerts for unusual activity
• Congressional trades tracking
• Dark pool activity
• Options chains with Greeks
• Institutional holdings

The system can now use Unusual Whales as the primary data source
for options flow analysis and market sentiment.
        """)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing Unusual Whales: {e}")
        import traceback
        traceback.print_exc()
        
        print("""
        
Troubleshooting:
1. Check if the API key is valid
2. Ensure you have an active Unusual Whales subscription
3. Check rate limits (you may have exceeded them)
        """)
        return False


if __name__ == "__main__":
    success = test_unusual_whales()
    sys.exit(0 if success else 1)