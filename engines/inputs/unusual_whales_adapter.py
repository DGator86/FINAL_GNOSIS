"""Unusual Whales API adapter with corrected endpoints and Bearer auth."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from loguru import logger

from gnosis.utils.option_utils import OptionUtils
from engines.inputs.options_chain_adapter import OptionContract, OptionsChainAdapter


class UnusualWhalesOptionsAdapter(OptionsChainAdapter):
    """Options chain adapter for the Unusual Whales API.

    Key details (verified Nov 2025):
    - Base URL: https://api.unusualwhales.com
    - Authentication: Bearer token via Authorization header
    - Endpoint: /api/stock/{symbol}/option-contracts (full OCC symbols)
    """

    BASE_URL = "https://api.unusualwhales.com"

    def __init__(self, *, token: Optional[str] = None, client: Optional[httpx.Client] = None):
        """Initialize the adapter with Bearer authentication."""

        self.api_token = token or os.getenv("UNUSUAL_WHALES_TOKEN") or os.getenv("UNUSUAL_WHALES_API_KEY")
        self.use_stub = False
        self._warning_cache: Dict[str, datetime] = {}

        if os.getenv("UNUSUAL_WHALES_DISABLED", "false").lower() in {"1", "true", "yes"}:
            logger.info("Unusual Whales disabled via UNUSUAL_WHALES_DISABLED – using stub data")
            self.client = None
            self.use_stub = True
            return

        if not self.api_token:
            logger.warning("⚠️  UNUSUAL_WHALES_TOKEN not set → using stub data fallback")
            self.client = None
            self.use_stub = True
        else:
            self.headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_token}",
            }
            self.client = client or httpx.Client(headers=self.headers, timeout=30.0)
            logger.info("UnusualWhalesOptionsAdapter initialized with API token")

    def get_chain(self, symbol: str, timestamp: datetime, expiration: Optional[str] = None) -> List[OptionContract]:
        """Get options chain for a symbol using the public contracts endpoint."""

        if not self.client or not self.api_token or self.use_stub:
            return self._get_stub_chain(symbol, timestamp)

        url = f"{self.BASE_URL}/api/stock/{symbol}/option-contracts"
        params = {"expiration_date": expiration} if expiration else {}

        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
            contracts_data = payload.get("data", []) or payload.get("contracts", [])

            if not contracts_data:
                logger.warning(f"No options chain data for {symbol} - using stub")
                return self._get_stub_chain(symbol, timestamp)

            contracts: List[OptionContract] = []
            for option in contracts_data:
                try:
                    symbol_str = option.get("symbol") or option.get("occ_symbol")
                    if not symbol_str:
                        continue

                    parsed = OptionUtils.parse_occ_symbol(symbol_str.replace(" ", ""))
                    exp_date = parsed["expiration"]
                    option_type = parsed["option_type"]
                    strike = float(parsed["strike"])

                    bid = float(option.get("nbbo_bid", option.get("bid", 0)) or 0)
                    ask = float(option.get("nbbo_ask", option.get("ask", 0)) or 0)
                    last = float(option.get("last_price", option.get("last", 0)) or 0)
                    volume = float(option.get("volume", 0) or 0)
                    oi = float(option.get("open_interest", option.get("open_interest", 0)) or 0)
                    iv = float(option.get("implied_volatility", option.get("iv", 0)) or 0)

                    contracts.append(
                        OptionContract(
                            symbol=symbol_str.strip(),
                            strike=strike,
                            expiration=exp_date,
                            option_type=option_type,
                            bid=bid,
                            ask=ask,
                            last=last,
                            volume=volume,
                            open_interest=oi,
                            implied_volatility=iv,
                            delta=float(option.get("delta", 0) or 0),
                            gamma=float(option.get("gamma", 0) or 0),
                            theta=float(option.get("theta", 0) or 0),
                            vega=float(option.get("vega", 0) or 0),
                            rho=float(option.get("rho", 0) or 0),
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
            detail = error.response.text if error.response else ""
            self._log_once(symbol, url, params, status_code, detail)
            if status_code in {401, 403}:
                logger.error(
                    "❌ Unusual Whales auth/subscription error %s: %s - switching to stub mode",
                    status_code,
                    detail,
                )
                self.use_stub = True
            else:
                logger.error(f"❌ Unusual Whales HTTP error {status_code}: {error} | detail={detail}")
            return self._get_stub_chain(symbol, timestamp)
        except httpx.HTTPError as error:
            logger.error(f"HTTP error getting options chain for {symbol}: {error}")
            return self._get_stub_chain(symbol, timestamp)
        except Exception as error:
            logger.error(f"Error getting options chain for {symbol}: {error}")
            return self._get_stub_chain(symbol, timestamp)

    def _log_once(self, symbol: str, url: str, params: dict, status_code: int, detail: str) -> None:
        """De-duplicate noisy warnings per symbol/status pair."""

        cache_key = f"{symbol}:{status_code}:{url}"
        last = self._warning_cache.get(cache_key)
        now = datetime.utcnow()
        if not last or (now - last).total_seconds() > 900:  # 15 minutes
            if status_code == 404:
                logger.warning(
                    "⚠️  Unusual Whales returned 404 for {symbol} | url={url} | params={params} | detail={detail}",
                    symbol=symbol,
                    url=url,
                    params=params,
                    detail=(detail[:200] if detail else ""),
                )
            else:
                logger.warning(
                    "Unusual Whales non-2xx response for {symbol}: {status} | url={url} | params={params} | detail={detail}",
                    symbol=symbol,
                    status=status_code,
                    url=url,
                    params=params,
                    detail=(detail[:200] if detail else ""),
                )
            self._warning_cache[cache_key] = now

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
