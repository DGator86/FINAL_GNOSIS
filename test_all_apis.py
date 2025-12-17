"""Test all API connections for GNOSIS trading system."""

import os
import sys

# Set credentials programmatically
os.environ["ALPACA_API_KEY"] = "PKDGAH5CJM4G3RZ2NP5WQNH22U"
os.environ["ALPACA_SECRET_KEY"] = "EfW43tDsmhWgvJkucKhJL3bsXmKyu5Kt1B3WxTFcuHEq"
os.environ["UNUSUAL_WHALES_API_TOKEN"] = "8932cd23-72b3-4f74-9848-13f9103b9df5"

print("=" * 60)
print("TESTING ALL API CONNECTIONS")
print("=" * 60)

# Test 1: Alpaca API
print("\n[1] ALPACA API (Paper Trading)")
print("-" * 40)
try:
    from alpaca.trading.client import TradingClient

    trading = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)
    account = trading.get_account()
    print(f"  Status: ✅ CONNECTED")
    print(f"  Account Status: {account.status}")
    print(f"  Buying Power: ${float(account.buying_power):,.2f}")
    print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"  Cash: ${float(account.cash):,.2f}")
except Exception as e:
    print(f"  Status: ❌ FAILED - {e}")

# Test 2: Alpaca Historical Data
print("\n[2] ALPACA HISTORICAL DATA")
print("-" * 40)
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from datetime import datetime, timedelta

    data_client = StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    )

    request = StockBarsRequest(
        symbol_or_symbols=["SPY"],
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=5),
        end=datetime.now(),
    )
    bars = data_client.get_stock_bars(request)
    bar_count = len(bars["SPY"]) if "SPY" in bars else 0
    print(f"  Status: ✅ CONNECTED")
    print(f"  Retrieved {bar_count} bars for SPY")
    if bar_count > 0:
        latest = bars["SPY"][-1]
        print(f"  Latest Close: ${latest.close:.2f}")
except Exception as e:
    print(f"  Status: ❌ FAILED - {e}")

# Test 3: Unusual Whales API
print("\n[3] UNUSUAL WHALES API")
print("-" * 40)
try:
    import httpx

    token = os.getenv("UNUSUAL_WHALES_API_TOKEN")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    # Try options contracts endpoint
    r = httpx.get(
        "https://api.unusualwhales.com/api/stock/SPY/option-contracts", headers=headers, timeout=15
    )

    if r.status_code == 200:
        data = r.json()
        contracts = data.get("data", [])
        print(f"  Status: ✅ CONNECTED")
        print(f"  Retrieved {len(contracts)} option contracts")
    elif r.status_code == 403:
        print(f"  Status: ⚠️ AUTH ERROR (403)")
        print(f"  Check subscription tier at unusualwhales.com")
    elif r.status_code == 404:
        print(f"  Status: ⚠️ Endpoint not found (404)")
        print(f"  Token may be valid but endpoint unavailable")
    else:
        print(f"  Status: ⚠️ HTTP {r.status_code}")
        print(f"  Response: {r.text[:100]}")

except Exception as e:
    print(f"  Status: ❌ FAILED - {e}")

# Test 4: Unusual Whales Flow Alerts
print("\n[4] UNUSUAL WHALES FLOW ALERTS")
print("-" * 40)
try:
    r = httpx.get(
        "https://api.unusualwhales.com/api/option-trades/flow-alerts", headers=headers, timeout=15
    )

    if r.status_code == 200:
        data = r.json()
        alerts = data.get("data", [])
        print(f"  Status: ✅ CONNECTED")
        print(f"  Retrieved {len(alerts)} flow alerts")
    elif r.status_code == 403:
        print(f"  Status: ⚠️ Requires higher subscription tier")
    else:
        print(f"  Status: ⚠️ HTTP {r.status_code}")

except Exception as e:
    print(f"  Status: ❌ FAILED - {e}")

# Test 5: Massive.com API
print("\n[5] MASSIVE.COM API")
print("-" * 40)
try:
    api_key = "Jm_fqc_gtSTSXG78P67dpBpO3LX_4P6D"

    # Massive uses Polygon-compatible API
    r = httpx.get(
        f"https://api.polygon.io/v2/aggs/ticker/SPY/prev", params={"apiKey": api_key}, timeout=15
    )

    if r.status_code == 200:
        data = r.json()
        print(f"  Status: ✅ CONNECTED (via Polygon)")
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            print(f"  SPY Previous Close: ${result.get('c', 'N/A')}")
    elif r.status_code == 401:
        print(f"  Status: ⚠️ API Key invalid or not Polygon-compatible")
        print(f"  Massive.com may use different endpoint format")
    else:
        print(f"  Status: ⚠️ HTTP {r.status_code}")

except Exception as e:
    print(f"  Status: ❌ FAILED - {e}")

print("\n" + "=" * 60)
print("API TEST SUMMARY")
print("=" * 60)
print("""
✅ = Working and returning data
⚠️ = Connected but limited/no data (subscription tier)
❌ = Connection failed

For trading, Alpaca is the primary requirement.
Unusual Whales enhances options flow analysis.
Massive.com provides additional market data.
""")
