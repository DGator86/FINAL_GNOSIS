#!/usr/bin/env python3
"""Debug script to inspect Unusual Whales API responses."""

import os
import json
from datetime import datetime

import httpx
from dotenv import load_dotenv

load_dotenv()

def test_unusual_whales():
    """Test Unusual Whales API and show raw response."""

    # Get token
    token = os.getenv("UNUSUAL_WHALES_API_KEY") or os.getenv("UNUSUAL_WHALES_TOKEN")

    if not token:
        print("❌ No UNUSUAL_WHALES_API_KEY found in .env")
        return

    print(f"✅ Token found: {token[:8]}...{token[-4:]}")

    base_url = "https://api.unusualwhales.com"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }

    symbol = "SPY"

    # Test 1: Option contracts endpoint
    print(f"\n{'='*60}")
    print(f"Testing: GET /api/stock/{symbol}/option-contracts")
    print('='*60)

    url = f"{base_url}/api/stock/{symbol}/option-contracts"
    params = {"limit": 10}  # Just get 10 for testing

    try:
        with httpx.Client(headers=headers, timeout=30.0) as client:
            response = client.get(url, params=params)

            print(f"Status: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")

            if response.status_code == 200:
                data = response.json()
                print(f"\n📦 Response structure:")
                print(f"   Keys: {list(data.keys())}")

                contracts = data.get("data", []) or data.get("contracts", [])
                print(f"   Contracts count: {len(contracts)}")

                if contracts:
                    print(f"\n📄 First contract (raw):")
                    print(json.dumps(contracts[0], indent=2))

                    print(f"\n📄 Contract keys:")
                    print(f"   {list(contracts[0].keys())}")
                else:
                    print("\n❌ No contracts in response!")
                    print(f"Full response: {json.dumps(data, indent=2)[:2000]}")
            else:
                print(f"\n❌ Error response:")
                print(response.text[:1000])

    except Exception as e:
        print(f"❌ Request failed: {e}")

    # Test 2: Flow endpoint
    print(f"\n{'='*60}")
    print(f"Testing: GET /api/stock/{symbol}/flow")
    print('='*60)

    url = f"{base_url}/api/stock/{symbol}/flow"
    today = datetime.now().strftime("%Y-%m-%d")
    params = {"start": today, "end": today}

    try:
        with httpx.Client(headers=headers, timeout=30.0) as client:
            response = client.get(url, params=params)

            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"Response: {json.dumps(data, indent=2)[:1500]}")
            else:
                print(f"Error: {response.text[:500]}")

    except Exception as e:
        print(f"❌ Request failed: {e}")

    # Test 3: Check available endpoints
    print(f"\n{'='*60}")
    print("Testing: GET /api/stock/SPY (basic stock info)")
    print('='*60)

    url = f"{base_url}/api/stock/{symbol}"

    try:
        with httpx.Client(headers=headers, timeout=30.0) as client:
            response = client.get(url)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            else:
                print(f"Error: {response.text[:500]}")
    except Exception as e:
        print(f"❌ Request failed: {e}")


if __name__ == "__main__":
    test_unusual_whales()
