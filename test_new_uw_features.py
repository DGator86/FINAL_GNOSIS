#!/usr/bin/env python3
"""
Test script for new Unusual Whales features:
- Greek Exposure (Gamma, Delta, Vanna, Charm)
- Dark Pool trades
"""

import os
from dotenv import load_dotenv
from engines.inputs.unusual_whales_adapter import UnusualWhalesOptionsAdapter

load_dotenv()

# Test symbols
SYMBOLS = ["SPY", "NVDA", "TSLA"]

def main():
    print("="*80)
    print("🐋 UNUSUAL WHALES - GREEK EXPOSURE & DARK POOL TEST")
    print("="*80)
    print()

    # Initialize adapter
    adapter = UnusualWhalesOptionsAdapter()

    if not adapter.client:
        print("❌ Adapter disabled - check UNUSUAL_WHALES_API_TOKEN")
        return

    print(f"✅ Adapter initialized")
    print(f"   Base URL: {adapter.base_url}")
    print(f"   Token: {adapter.api_token[:20]}...")
    print()

    for symbol in SYMBOLS:
        print("="*80)
        print(f"📊 Testing: {symbol}")
        print("="*80)
        print()

        # Test Greek Exposure
        print(f"1️⃣  Greek Exposure (GEX/VEX/Charm):")
        print("-" * 60)
        greek_exposure = adapter.get_greek_exposure(symbol)

        if greek_exposure:
            print(f"   ✅ Successfully retrieved greek exposure data")
            print()
            print(f"   📈 CALL GREEKS:")
            print(f"      Gamma (GEX):  {float(greek_exposure.get('call_gamma', 0)):,.2f}")
            print(f"      Delta:        {float(greek_exposure.get('call_delta', 0)):,.2f}")
            print(f"      Vanna (VEX):  {float(greek_exposure.get('call_vanna', 0)):,.2f}")
            print(f"      Charm:        {float(greek_exposure.get('call_charm', 0)):,.2f}")
            print(f"      Vega:         {float(greek_exposure.get('call_vega', 0)):,.2f}")
            print(f"      Theta:        {float(greek_exposure.get('call_theta', 0)):,.2f}")
            print()
            print(f"   📉 PUT GREEKS:")
            print(f"      Gamma (GEX):  {float(greek_exposure.get('put_gamma', 0)):,.2f}")
            print(f"      Delta:        {float(greek_exposure.get('put_delta', 0)):,.2f}")
            print(f"      Vanna (VEX):  {float(greek_exposure.get('put_vanna', 0)):,.2f}")
            print(f"      Charm:        {float(greek_exposure.get('put_charm', 0)):,.2f}")
            print(f"      Vega:         {float(greek_exposure.get('put_vega', 0)):,.2f}")
            print(f"      Theta:        {float(greek_exposure.get('put_theta', 0)):,.2f}")
            print()

            # Calculate net exposure
            net_gamma = float(greek_exposure.get('call_gamma', 0)) + float(greek_exposure.get('put_gamma', 0))
            net_vanna = float(greek_exposure.get('call_vanna', 0)) + float(greek_exposure.get('put_vanna', 0))
            net_charm = float(greek_exposure.get('call_charm', 0)) + float(greek_exposure.get('put_charm', 0))

            print(f"   🎯 NET EXPOSURE:")
            print(f"      Net GEX:   {net_gamma:,.2f} {'(positive = volatility suppression)' if net_gamma > 0 else '(negative = volatility amplification)'}")
            print(f"      Net VEX:   {net_vanna:,.2f}")
            print(f"      Net Charm: {net_charm:,.2f}")
        else:
            print(f"   ⚠️  No greek exposure data available")

        print()

        # Test Dark Pool
        print(f"2️⃣  Dark Pool Trades (≥$1M):")
        print("-" * 60)
        dark_pool_trades = adapter.get_dark_pool(symbol, min_premium=1_000_000, limit=10)

        if dark_pool_trades:
            print(f"   ✅ Retrieved {len(dark_pool_trades)} large dark pool trades")
            print()
            total_premium = sum(float(t.get("premium", 0)) for t in dark_pool_trades)
            total_size = sum(float(t.get("size", 0)) for t in dark_pool_trades)

            print(f"   💰 Total Premium: ${total_premium:,.2f} (${total_premium/1_000_000:.2f}M)")
            print(f"   📦 Total Size:    {total_size:,.0f} shares")
            print()

            # Show top 3 trades
            print(f"   📊 Top 3 Trades:")
            for i, trade in enumerate(dark_pool_trades[:3], 1):
                price = float(trade.get("price", 0))
                size = float(trade.get("size", 0))
                premium = float(trade.get("premium", 0))
                timestamp = trade.get("date_time", trade.get("date", "N/A"))

                print(f"      {i}. ${price:.2f} × {size:,.0f} shares = ${premium:,.2f} | {timestamp}")
        else:
            print(f"   ⚠️  No dark pool data available")

        print()

    # Test Flow Alerts
    print("="*80)
    print("3️⃣  Flow Alerts (Market-Wide)")
    print("="*80)
    print()

    flow_alerts = adapter.get_unusual_activity()
    if flow_alerts:
        print(f"✅ Retrieved {len(flow_alerts)} flow alerts")
        print()
        print("📊 Recent Flow Alerts:")
        for i, alert in enumerate(flow_alerts[:5], 1):
            ticker = alert.get("ticker", "N/A")
            alert_type = alert.get("alert_type", "N/A")
            premium = alert.get("cost_basis", alert.get("premium", 0))
            sentiment = alert.get("sentiment", "N/A")
            print(f"   {i}. {ticker} | Type: {alert_type} | Premium: ${float(premium):,.2f} | Sentiment: {sentiment}")
    else:
        print("⚠️  No flow alerts available")

    print()
    print("="*80)
    print("✅ TESTING COMPLETE")
    print("="*80)
    print()
    print("📋 Summary:")
    print("   - Greek Exposure: Includes Gamma, Delta, Vanna, Charm, Vega, Theta")
    print("   - Dark Pool: Large institutional block trades")
    print("   - Flow Alerts: Significant options flow across all tickers")
    print()

    adapter.close()

if __name__ == "__main__":
    main()
