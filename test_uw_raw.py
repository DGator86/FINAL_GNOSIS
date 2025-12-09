#!/usr/bin/env python3
"""Raw API test to inspect Unusual Whales responses."""

import os

import pytest

# Only run this test when explicitly requested
if not os.getenv("RUN_UW_LIVE"):
    pytest.skip(
        "Raw Unusual Whales API test disabled; set RUN_UW_LIVE=1 to run against live API",
        allow_module_level=True,
    )

from dotenv import load_dotenv

# Load environment variables from the repository root
load_dotenv(dotenv_path=".env")

import httpx

TOKEN = (
    os.getenv("UNUSUAL_WHALES_API_TOKEN")
    or os.getenv("UNUSUAL_WHALES_TOKEN")
    or os.getenv("UNUSUAL_WHALES_API_KEY")
)

ENDPOINTS = [
    "https://api.unusualwhales.com/api/options/contracts/SPY",
    "https://api.unusualwhales.com/api/stock/SPY",
    "https://api.unusualwhales.com/api/options/SPY",
]


@pytest.mark.parametrize("url", ENDPOINTS)
def test_raw_unusual_whales_response(url: str) -> None:
    """Hit a handful of endpoints and report the raw status code/response."""

    headers = {"Accept": "application/json"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"

    response = httpx.get(url, headers=headers, timeout=10)

    expected_status = 200 if TOKEN else 401
    assert (
        response.status_code == expected_status
    ), f"Unexpected status {response.status_code} for {url}"

    print(f"URL: {url}")
    print(f"Status: {response.status_code}")
    print(f"Response (first 500 chars): {response.text[:500]}")
