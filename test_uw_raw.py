#!/usr/bin/env python3
"""Raw API test to see exact response."""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("UNUSUAL_WHALES_TOKEN") or os.getenv("UNUSUAL_WHALES_API_KEY")

if not token:
    print("No token found!")
    exit(1)

print(f"Token: {token[:30]}...")
print()

# Test different endpoints
endpoints = [
    "https://api.unusualwhales.com/api/options/contracts/SPY",
    "https://api.unusualwhales.com/api/stock/SPY",
    "https://api.unusualwhales.com/api/options/SPY",
]

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/json"
}

for url in endpoints:
    print(f"Testing: {url}")
    print("-" * 80)
    try:
        response = httpx.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:500]}")
    except Exception as e:
        print(f"Error: {e}")
    print()
