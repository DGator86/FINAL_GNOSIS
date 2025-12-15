#!/usr/bin/env python3
"""Debug script to inspect Unusual Whales API responses.

Tests multiple endpoints to find which ones work with your API key.
"""

import os
import json
from datetime import datetime

import httpx
from dotenv import load_dotenv

load_dotenv()

# Import the credentials module that the main system uses
from config.credentials import get_unusual_whales_token


def test_endpoint(client, base_url: str, path: str, params: dict = None):
    """Test a single endpoint and report results."""
    url = f"{base_url}{path}"
    print(f"\n{'='*70}")
    print(f"GET {path}")
    print('='*70)

    try:
        response = client.get(url, params=params)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS!")

            if isinstance(data, dict):
                print(f"   Keys: {list(data.keys())}")

                # Look for data arrays
                for key in ["data", "contracts", "options", "chains", "results"]:
                    if key in data and isinstance(data[key], list):
                        print(f"   {key} count: {len(data[key])}")
                        if data[key]:
                            print(f"   First item keys: {list(data[key][0].keys())}")
                            print(f"\n   📄 Sample item:")
                            print(json.dumps(data[key][0], indent=2)[:1500])
                        break
                else:
                    # No list found, print structure
                    print(f"\n   📄 Response preview:")
                    print(json.dumps(data, indent=2)[:1500])
            else:
                print(f"   Response: {str(data)[:500]}")

            return True, data
        else:
            print(f"❌ Error {response.status_code}: {response.text[:300]}")
            return False, None

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False, None


def test_unusual_whales():
    """Test Unusual Whales API endpoints."""

    # Get token using same method as main system
    token = get_unusual_whales_token()

    if not token:
        print("❌ No Unusual Whales token available")
        return

    # Check source
    env_token = os.getenv("UNUSUAL_WHALES_API_TOKEN") or os.getenv("UNUSUAL_WHALES_TOKEN") or os.getenv("UNUSUAL_WHALES_API_KEY")
    if env_token:
        print(f"✅ Token from .env: {token[:8]}...{token[-4:]}")
    else:
        print(f"⚠️  Using HARDCODED default token: {token[:8]}...{token[-4:]}")
        print("   (This token may have limited access)")

    base_url = "https://api.unusualwhales.com"
    headers = {
        "Accept": "application/json, text/plain",
        "Authorization": f"Bearer {token}",
    }

    symbol = "SPY"
    working_endpoints = []

    with httpx.Client(headers=headers, timeout=30.0) as client:

        # ===== ENDPOINTS FROM UW SUPPORT EMAIL =====
        print("\n" + "="*70)
        print("  TESTING ENDPOINTS FROM UNUSUAL WHALES SUPPORT EMAIL")
        print("="*70)

        # Flow alerts (from support email)
        success, _ = test_endpoint(client, base_url, "/api/option-trades/flow-alerts", {
            "limit": "5",
            "is_call": "true",
        })
        if success:
            working_endpoints.append("/api/option-trades/flow-alerts")

        # Market tide (from support email)
        success, _ = test_endpoint(client, base_url, "/api/market/market-tide", {
            "interval_5m": "false"
        })
        if success:
            working_endpoints.append("/api/market/market-tide")

        # Dark pool (from support email)
        success, _ = test_endpoint(client, base_url, f"/api/darkpool/{symbol}", {
            "limit": "3"
        })
        if success:
            working_endpoints.append(f"/api/darkpool/{symbol}")

        # ===== LIKELY OPTIONS CHAIN ENDPOINTS =====
        print("\n" + "="*70)
        print("  TESTING LIKELY OPTIONS CHAIN ENDPOINTS")
        print("="*70)

        # Try various possible options chain endpoints
        options_endpoints = [
            f"/api/stock/{symbol}/option-chains",
            f"/api/stock/{symbol}/options",
            f"/api/stock/{symbol}/option-contracts",
            f"/api/options/{symbol}/chain",
            f"/api/options/chain/{symbol}",
            f"/api/stock/{symbol}/greeks",
            f"/api/stock/{symbol}/greek-exposure",
            f"/api/stock/{symbol}/max-pain",
            f"/api/stock/{symbol}/iv-rank",
            f"/api/stock/{symbol}/overview",
            f"/api/stock/{symbol}",
        ]

        for endpoint in options_endpoints:
            success, data = test_endpoint(client, base_url, endpoint)
            if success:
                working_endpoints.append(endpoint)

        # ===== OPTIONS FLOW ENDPOINTS =====
        print("\n" + "="*70)
        print("  TESTING OPTIONS FLOW ENDPOINTS")
        print("="*70)

        flow_endpoints = [
            f"/api/stock/{symbol}/flow",
            f"/api/stock/{symbol}/flow-alerts",
            f"/api/stock/{symbol}/options-flow",
            f"/api/option-trades/{symbol}",
        ]

        for endpoint in flow_endpoints:
            success, _ = test_endpoint(client, base_url, endpoint, {"limit": "5"})
            if success:
                working_endpoints.append(endpoint)

    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("  SUMMARY - WORKING ENDPOINTS")
    print("="*70)

    if working_endpoints:
        print("\n✅ These endpoints work with your API key:\n")
        for ep in working_endpoints:
            print(f"   • {ep}")
    else:
        print("\n❌ No endpoints returned 200. Check your API key subscription level.")

    print("\n" + "="*70)


if __name__ == "__main__":
    test_unusual_whales()
