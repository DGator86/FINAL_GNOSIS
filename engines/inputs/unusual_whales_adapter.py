"""Unusual Whales API adapter with corrected endpoints and Bearer auth."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from loguru import logger

from config.credentials import get_unusual_whales_token
from engines.inputs.options_chain_adapter import OptionContract, OptionsChainAdapter
from gnosis.utils.option_utils import OptionUtils


@dataclass
class UnusualWhalesConfig:
    """Runtime configuration for the Unusual Whales adapter."""

    base_url: str
    timeout: float
    token: str
    use_stub: bool

    @classmethod
    def from_env(cls, token: Optional[str] = None) -> "UnusualWhalesConfig":
        """Build configuration using environment variables and optional override."""

        api_token = get_unusual_whales_token(token)

        if not api_token:
            raise ValueError("Unusual Whales API token is required for historical data")

        base_url = os.getenv("UNUSUAL_WHALES_BASE_URL", "https://api.unusualwhales.com").rstrip("/")
        timeout = float(os.getenv("UNUSUAL_WHALES_TIMEOUT", "30.0"))

        # Backtesting must use real data, so stubs are never allowed
        return cls(base_url=base_url, timeout=timeout, token=api_token, use_stub=False)


class UnusualWhalesOptionsAdapter(OptionsChainAdapter):
    """Options chain adapter for the Unusual Whales API.

    Key details (verified Nov 2025):
    - Base URL: https://api.unusualwhales.com
    - Authentication: Bearer token via Authorization header
    - Endpoint: /api/stock/{symbol}/option-contracts (full OCC symbols)
    """

    def __init__(self, *, token: Optional[str] = None, client: Optional[httpx.Client] = None):
        """Initialize the adapter with Bearer authentication."""

        self.config = UnusualWhalesConfig.from_env(token)
        self.api_token = self.config.token
        self.use_stub = self.config.use_stub
        self.base_url = self.config.base_url
        self.timeout = self.config.timeout
        self._warning_cache: Dict[str, datetime] = {}
        self._disabled_reason: Optional[str] = None

        if self.use_stub:
            raise RuntimeError("Unusual Whales stubs are disabled – real data required")

        if not self.api_token:
            raise RuntimeError("Unusual Whales API token is required for real data")

        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }
        self.client = client or httpx.Client(headers=self.headers, timeout=self.timeout)
        logger.info(
            "UnusualWhalesOptionsAdapter initialized (token from env, base_url=%s, timeout=%.1fs)",
            self.base_url,
            self.timeout,
        )

    def _fetch_greeks(self, symbol: str) -> Dict[str, Dict[str, float]]:
        """Fetch Greeks from the separate /greeks endpoint and index by option_symbol."""
        greeks_map: Dict[str, Dict[str, float]] = {}

        try:
            url = f"{self.base_url}/api/stock/{symbol}/greeks"
            response = self.client.get(url)
            response.raise_for_status()
            payload = response.json()
            greeks_data = payload.get("data", [])

            for row in greeks_data:
                # Each row has call and put Greeks for a strike/expiry combo
                call_symbol = row.get("call_option_symbol", "")
                put_symbol = row.get("put_option_symbol", "")

                if call_symbol:
                    greeks_map[call_symbol] = {
                        "delta": float(row.get("call_delta", 0) or 0),
                        "gamma": float(row.get("call_gamma", 0) or 0),
                        "theta": float(row.get("call_theta", 0) or 0),
                        "vega": float(row.get("call_vega", 0) or 0),
                        "rho": float(row.get("call_rho", 0) or 0),
                        "charm": float(row.get("call_charm", 0) or 0),
                        "vanna": float(row.get("call_vanna", 0) or 0),
                    }

                if put_symbol:
                    greeks_map[put_symbol] = {
                        "delta": float(row.get("put_delta", 0) or 0),
                        "gamma": float(row.get("put_gamma", 0) or 0),
                        "theta": float(row.get("put_theta", 0) or 0),
                        "vega": float(row.get("put_vega", 0) or 0),
                        "rho": float(row.get("put_rho", 0) or 0),
                        "charm": float(row.get("put_charm", 0) or 0),
                        "vanna": float(row.get("put_vanna", 0) or 0),
                    }

            logger.debug(f"Fetched Greeks for {len(greeks_map)} option symbols")
            return greeks_map

        except Exception as error:
            logger.warning(f"Could not fetch Greeks for {symbol}: {error}")
            return {}

    def get_chain(self, symbol: str, timestamp: datetime, expiration: Optional[str] = None) -> List[OptionContract]:
        """Get options chain for a symbol, merging contracts with Greeks from separate endpoints."""

        if not self.client:
            raise RuntimeError("Unusual Whales client not initialized; real data is required")

        # Step 1: Fetch contracts from /option-contracts
        url = f"{self.base_url}/api/stock/{symbol}/option-contracts"
        params = {"limit": 500}
        if expiration:
            params["expiration_date"] = expiration

        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
            contracts_data = payload.get("data", []) or payload.get("contracts", [])

            if not contracts_data:
                logger.info(
                    "⏭️  Unusual Whales returned no contracts for %s - skipping symbol",
                    symbol,
                )
                return []

            # Step 2: Fetch Greeks from /greeks endpoint (separate API call)
            greeks_map = self._fetch_greeks(symbol)

            # Step 3: Parse contracts and merge with Greeks
            contracts: List[OptionContract] = []
            for option in contracts_data:
                try:
                    # The API returns "option_symbol" not "symbol" or "occ_symbol"
                    symbol_str = option.get("option_symbol") or option.get("symbol") or option.get("occ_symbol")
                    if not symbol_str:
                        continue

                    symbol_str = symbol_str.strip()
                    parsed = OptionUtils.parse_occ_symbol(symbol_str.replace(" ", ""))
                    exp_date = parsed["expiration"]
                    option_type = parsed["option_type"]
                    strike = float(parsed["strike"])

                    bid = float(option.get("nbbo_bid", option.get("bid", 0)) or 0)
                    ask = float(option.get("nbbo_ask", option.get("ask", 0)) or 0)
                    last = float(option.get("last_price", option.get("last", 0)) or 0)
                    volume = float(option.get("volume", 0) or 0)
                    oi = float(option.get("open_interest", 0) or 0)
                    iv = float(option.get("implied_volatility", option.get("iv", 0)) or 0)

                    # Look up Greeks by option_symbol
                    greeks = greeks_map.get(symbol_str, {})

                    contracts.append(
                        OptionContract(
                            symbol=symbol_str,
                            strike=strike,
                            expiration=exp_date,
                            option_type=option_type,
                            bid=bid,
                            ask=ask,
                            last=last,
                            volume=volume,
                            open_interest=oi,
                            implied_volatility=iv,
                            delta=greeks.get("delta", 0.0),
                            gamma=greeks.get("gamma", 0.0),
                            theta=greeks.get("theta", 0.0),
                            vega=greeks.get("vega", 0.0),
                            rho=greeks.get("rho", 0.0),
                        )
                    )
                except (ValueError, KeyError) as error:
                    logger.debug(f"Error parsing option contract: {error}")
                    continue

            if contracts:
                n_with_greeks = sum(1 for c in contracts if c.delta != 0 or c.gamma != 0)
                logger.info(f"✅ Retrieved {len(contracts)} option contracts for {symbol} ({n_with_greeks} with Greeks)")
                return contracts

            logger.warning(
                f"No valid contracts parsed for {symbol} from Unusual Whales response"
            )
            return []

        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            detail = self._extract_detail(error.response)
            self._log_once(symbol, url, params, status_code, detail)

            if status_code in {401, 403}:
                logger.error(
                    "❌ Unusual Whales authentication/subscription error %s - real data required",
                    status_code,
                )
                raise RuntimeError("Unusual Whales auth/subscription error") from error

            if status_code == 404:
                raise RuntimeError(f"Unusual Whales has no data for {symbol} (404)")

            if status_code in {400, 422}:
                raise RuntimeError(
                    f"Unusual Whales rejected request for {symbol} | status={status_code} | detail={detail}"
                )

            if status_code == 429 or status_code >= 500:
                raise RuntimeError(
                    f"Unusual Whales transient/unavailable for {symbol} | status={status_code} | detail={detail}"
                )

            raise RuntimeError(
                f"Unexpected Unusual Whales response for {symbol} | status={status_code} | detail={detail}"
            )
        except httpx.HTTPError as error:
            logger.error(f"HTTP error getting options chain for {symbol}: {error}")
            raise
        except Exception as error:
            logger.error(f"Error getting options chain for {symbol}: {error}")
            raise

    def get_flow_snapshot(self, symbol: str, timestamp: datetime) -> Dict[str, float]:
        """Retrieve aggregated options flow for sentiment scoring."""

        if not self.client:
            raise RuntimeError("Unusual Whales client not initialized")

        params = {"start": timestamp.strftime("%Y-%m-%d"), "end": timestamp.strftime("%Y-%m-%d")}
        url = f"{self.base_url}/api/stock/{symbol}/flow"

        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()
            payload = response.json() or {}
            data = payload.get("data", {}) or payload

            return {
                "call_volume": float(data.get("call_volume", 0) or 0),
                "put_volume": float(data.get("put_volume", 0) or 0),
                "call_premium": float(data.get("call_premium", 0) or 0),
                "put_premium": float(data.get("put_premium", 0) or 0),
                "sweep_ratio": float(data.get("sweep_ratio", data.get("sweep_percentage", 0)) or 0),
            }
        except httpx.HTTPError as error:
            logger.error(f"Error fetching Unusual Whales flow for {symbol}: {error}")
            return {}

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

    @staticmethod
    def _extract_detail(response: Optional[httpx.Response]) -> str:
        if not response:
            return ""

        try:
            payload = response.json()
            if isinstance(payload, dict):
                for key in ("detail", "message", "error"):
                    if payload.get(key):
                        return str(payload[key])
            return str(payload)[:500]
        except Exception:
            return response.text[:500] if response.text else ""

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
        """Fetch real flow alerts instead of the deprecated placeholder endpoints."""

        if not self.client or not self.api_token:
            logger.debug("No API token - skipping unusual activity")
            return []

        try:
            urls = [f"{self.base_url}/api/option-trades/flow-alerts"]
            if symbol:
                urls.insert(0, f"{self.base_url}/api/stock/{symbol}/flow-alerts")

            params = {"limit": 50}

            for url in urls:
                try:
                    response = self.client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    activity = data.get("data", []) or data.get("alerts", [])
                    if activity:
                        logger.info("Retrieved %s flow alerts", len(activity))
                        return activity
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 404:
                        logger.debug("Flow alerts endpoint not available: %s", url)
                        continue
                    raise

            logger.debug("No flow alerts returned from Unusual Whales")
            return []
        except Exception as error:
            logger.error(f"Error getting unusual activity: {error}")
            return []

    def get_flow_summary(self, symbol: str) -> dict:
        """Return the most recent flow items for the ticker."""

        if not self.client or not self.api_token:
            return {}

        try:
            url = f"{self.base_url}/api/stock/{symbol}/flow-recent"
            params = {"limit": 50}
            response = self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("data", {})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info("No recent flow for %s (404)", symbol)
                return {}
            logger.error(
                "Flow summary request for %s failed | status=%s | detail=%s",
                symbol,
                exc.response.status_code,
                self._extract_detail(exc.response),
            )
            return {}
        except Exception as error:
            logger.error(f"Error getting flow summary for {symbol}: {error}")
            return {}

    def get_implied_volatility(self, symbol: str) -> Optional[float]:
        """Pull the 30-day implied volatility from the realized volatility endpoint."""

        if not self.client or not self.api_token:
            return None

        try:
            url = f"{self.base_url}/api/stock/{symbol}/volatility/realized"
            response = self.client.get(url, params={"timeframe": "30d"})
            response.raise_for_status()
            data = response.json().get("data")

            if isinstance(data, list) and data:
                first = data[0]
                iv_value = first.get("implied_volatility")
                if iv_value is not None:
                    return float(iv_value)

            return None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info("Volatility endpoint missing for %s (404)", symbol)
                return None
            logger.error(
                "Failed to fetch implied volatility for %s | status=%s | detail=%s",
                symbol,
                exc.response.status_code,
                self._extract_detail(exc.response),
            )
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
