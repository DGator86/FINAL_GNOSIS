"""Unusual Whales API adapter with corrected v2 endpoints and Bearer auth."""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import httpx
from loguru import logger

from engines.inputs.options_chain_adapter import OptionContract, OptionsChainAdapter


class UnusualWhalesOptionsAdapter(OptionsChainAdapter):
    """Options chain adapter for the Unusual Whales v2 API.

    Key details (verified Nov 2025):
    - Base URL: https://api.unusualwhales.com
    - Authentication: Bearer token via Authorization header
    - Endpoint: /v2/options/contracts/{symbol}?expiration_date=YYYY-MM-DD (optional)
    """

    BASE_URL = "https://api.unusualwhales.com"

    def __init__(self, *, token: Optional[str] = None):
        """Initialize the adapter with Bearer authentication.

        Args:
            token: Optional explicit token; otherwise read from env vars.
        """

        self.api_token = token or os.getenv("UNUSUAL_WHALES_TOKEN") or os.getenv("UNUSUAL_WHALES_API_KEY")
        self.use_stub = False

        if not self.api_token:
            logger.warning("⚠️  UNUSUAL_WHALES_TOKEN not set → using stub data fallback")
            self.client = None
            self.use_stub = True
        else:
            self.headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_token}",
            }
            self.client = httpx.Client(headers=self.headers, timeout=30.0)
            logger.info("UnusualWhalesOptionsAdapter initialized with API token")

    def get_chain(self, symbol: str, timestamp: datetime, expiration: Optional[str] = None) -> List[OptionContract]:
        """Get options chain for a symbol using the v2 contracts endpoint.

        Args:
            symbol: Underlying symbol.
            timestamp: Data timestamp (used for stub fallback default expirations).
            expiration: Optional expiration date (YYYY-MM-DD).

        Returns:
            List of :class:`OptionContract` entries with greeks populated when available.
        """

        if not self.client or not self.api_token or self.use_stub:
            return self._get_stub_chain(symbol, timestamp)

        url = f"{self.BASE_URL}/v2/options/contracts/{symbol}"
        params = {"expiration_date": expiration} if expiration else {}

        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()
            contracts_data = response.json().get("contracts", [])

            if not contracts_data:
                logger.warning(f"No options chain data for {symbol} - using stub")
                return self._get_stub_chain(symbol, timestamp)

            contracts: List[OptionContract] = []
            for option in contracts_data:
                try:
                    exp_str = option.get("expiration_date")
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d") if exp_str else timestamp

                    option_type = (option.get("type") or "").lower()
                    if option_type not in {"call", "put"}:
                        continue

                    greeks = option.get("greeks", {}) or {}
                    contracts.append(
                        OptionContract(
                            symbol=option.get("symbol") or f"{symbol}_{exp_date.strftime('%Y%m%d')}{option_type[0].upper()}{option.get('strike', 0)}",
                            strike=float(option.get("strike", 0) or 0),
                            expiration=exp_date,
                            option_type=option_type,
                            bid=float(option.get("bid", 0) or 0),
                            ask=float(option.get("ask", 0) or 0),
                            last=float(option.get("last", 0) or 0),
                            volume=float(option.get("volume", 0) or 0),
                            open_interest=float(option.get("open_interest", 0) or 0),
                            implied_volatility=float(option.get("implied_volatility", 0) or 0),
                            delta=float(greeks.get("delta", 0) or 0),
                            gamma=float(greeks.get("gamma", 0) or 0),
                            theta=float(greeks.get("theta", 0) or 0),
                            vega=float(greeks.get("vega", 0) or 0),
                            rho=float(greeks.get("rho", 0) or 0),
                        )
                    )
                except (ValueError, KeyError) as error:
                    logger.debug(f"Error parsing option contract: {error}")
                    continue

            if contracts:
                logger.info(f"✅ Retrieved {len(contracts)} option contracts for {symbol}")
                return contracts

            logger.warning(f"No valid contracts parsed for {symbol} - using stub")
            return self._get_stub_chain(symbol, timestamp)

        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            if status_code in {401, 403}:
                logger.error("❌ Invalid token or subscription missing API access - switching to stub mode")
            elif status_code == 404:
                logger.warning("⚠️  Unusual Whales endpoint returned 404 - verify symbol or subscription")
            else:
                logger.error(f"❌ Unusual Whales HTTP error {status_code}: {error}")
            self.use_stub = True
            return self._get_stub_chain(symbol, timestamp)
        except httpx.HTTPError as error:
            logger.error(f"HTTP error getting options chain for {symbol}: {error}")
            return self._get_stub_chain(symbol, timestamp)
        except Exception as error:
            logger.error(f"Error getting options chain for {symbol}: {error}")
            return self._get_stub_chain(symbol, timestamp)

    def _get_stub_chain(self, symbol: str, timestamp: datetime) -> List[OptionContract]:
        """Return deterministic stub data when the API is unavailable."""

        logger.info(f"📊 Using stub options chain for {symbol}")
        try:
            from engines.inputs.stub_adapters import StaticOptionsAdapter

            return StaticOptionsAdapter().get_chain(symbol, timestamp)
        except Exception as error:  # pragma: no cover - defensive fallback
            logger.warning(f"Stub adapter error for {symbol}: {error} - returning empty chain")
            return []

    # The flow/IV helpers remain for compatibility with existing callers.
    def get_unusual_activity(self, symbol: Optional[str] = None) -> List[dict]:
        if not self.client or not self.api_token:
            logger.debug("No API token - skipping unusual activity")
            return []

        try:
            endpoints = [
                f"{self.BASE_URL}/api/activity",
                f"{self.BASE_URL}/api/options/activity",
            ]

            params = {"ticker": symbol} if symbol else {}

            for url in endpoints:
                try:
                    response = self.client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    activity = data.get("data", []) or data.get("activity", [])
                    if activity:
                        logger.info(f"Retrieved {len(activity)} unusual activity records")
                        return activity
                except httpx.HTTPStatusError:
                    continue

            logger.debug("No unusual activity endpoint working")
            return []
        except Exception as error:
            logger.error(f"Error getting unusual activity: {error}")
            return []

    def get_flow_summary(self, symbol: str) -> dict:
        if not self.client or not self.api_token:
            return {}

        try:
            endpoints = [
                f"{self.BASE_URL}/api/flow/{symbol}",
                f"{self.BASE_URL}/api/options/flow/{symbol}",
            ]

            for url in endpoints:
                try:
                    response = self.client.get(url)
                    response.raise_for_status()
                    data = response.json()
                    return data.get("data", {})
                except httpx.HTTPStatusError:
                    continue

            return {}
        except Exception as error:
            logger.error(f"Error getting flow summary for {symbol}: {error}")
            return {}

    def get_implied_volatility(self, symbol: str) -> Optional[float]:
        if not self.client or not self.api_token:
            return None

        try:
            endpoints = [
                f"{self.BASE_URL}/api/stock/{symbol}/iv",
                f"{self.BASE_URL}/api/options/{symbol}/iv",
            ]

            for url in endpoints:
                try:
                    response = self.client.get(url)
                    response.raise_for_status()
                    data = response.json()
                    iv = data.get("data", {}).get("iv")
                    if iv is not None:
                        return float(iv)
                except httpx.HTTPStatusError:
                    continue

            return None
        except Exception as error:
            logger.error(f"Error getting IV for {symbol}: {error}")
            return None

    def close(self) -> None:
        if self.client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Backwards compatibility for existing imports
UnusualWhalesAdapter = UnusualWhalesOptionsAdapter
