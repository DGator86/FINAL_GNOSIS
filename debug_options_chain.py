#!/usr/bin/env python3
"""Debug script to see raw API response for options chains"""

import os
import httpx
import json

# Get token from environment
token = os.getenv("UNUSUAL_WHALES_API_TOKEN") or "8932cd23-72b3-4f74-9848-13f9103b9df5"

print("="*80)
print("🔍 OPTIONS CHAIN API RESPONSE DEBUG")
print("="*80)
print(f"Token: {token[:20]}...")
print()

headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {token}",
}

# Test SPY options chain
url = "https://api.unusualwhales.com/api/stock/SPY/option-contracts"
params = {"limit": 5}

print(f"URL: {url}")
print(f"Params: {params}")
print()

try:
    response = httpx.get(url, headers=headers, params=params, timeout=30.0)
    print(f"Status Code: {response.status_code}")
    print()

    if response.status_code == 200:
        data = response.json()
        print("✅ Response received")
        print()
        print("Response keys:", list(data.keys()))
        print()

        # Check for data
        if "data" in data:
            contracts = data["data"]
            print(f"Found 'data' key with {len(contracts)} contracts")
            if contracts:
                print()
                print("First contract structure:")
                print(json.dumps(contracts[0], indent=2))
            else:
                print("⚠️  'data' array is EMPTY")
        elif "contracts" in data:
            contracts = data["contracts"]
            print(f"Found 'contracts' key with {len(contracts)} contracts")
            if contracts:
                print()
                print("First contract structure:")
                print(json.dumps(contracts[0], indent=2))
            else:
                print("⚠️  'contracts' array is EMPTY")
        else:
            print("❌ No 'data' or 'contracts' key found")
            print()
            print("Full response:")
            print(json.dumps(data, indent=2))

    elif response.status_code == 401:
        print("❌ 401 Unauthorized - Token invalid or expired")
    elif response.status_code == 403:
        print("❌ 403 Forbidden - Token doesn't have access to this endpoint")
    elif response.status_code == 404:
        print("❌ 404 Not Found - Endpoint doesn't exist or symbol not found")
    else:
        print(f"❌ Unexpected status: {response.status_code}")
        print(response.text[:500])

except httpx.HTTPError as e:
    print(f"❌ HTTP Error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

print()
print("="*80)
