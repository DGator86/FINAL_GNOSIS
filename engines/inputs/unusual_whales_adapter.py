"""Unusual Whales API adapter with corrected endpoints and Bearer auth."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import httpx
from loguru import logger

from gnosis.utils.option_utils import OptionUtils
from engines.inputs.options_chain_adapter import OptionContract, OptionsChainAdapter


@dataclass
class UnusualWhalesConfig:
    """Runtime configuration for the Unusual Whales adapter."""

    base_url: str
    timeout: float
    token: Optional[str]
    use_stub: bool

    @classmethod
    def from_env(cls, token: Optional[str] = None) -> "UnusualWhalesConfig":
        """Build configuration using environment variables and optional override."""

        api_token = token or os.getenv("UNUSUAL_WHALES_API_TOKEN")
        if not api_token:
            api_token = os.getenv("UNUSUAL_WHALES_TOKEN") or os.getenv("UNUSUAL_WHALES_API_KEY")

        base_url = os.getenv("UNUSUAL_WHALES_BASE_URL", "https://api.unusualwhales.com").rstrip("/")
        timeout = float(os.getenv("UNUSUAL_WHALES_TIMEOUT", "30.0"))
        use_stub = os.getenv("UNUSUAL_WHALES_DISABLED", "false").lower() in {"1", "true", "yes"}

        return cls(base_url=base_url, timeout=timeout, token=api_token, use_stub=use_stub)


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
            self.client = None
            self._disabled_reason = "disabled-via-env"
            logger.info(
                "Unusual Whales disabled via UNUSUAL_WHALES_DISABLED – using stub data"
            )
            return

        if not self.api_token:
            self.client = None
            self._disabled_reason = "missing-token"
            logger.error(
                "🚫 UNUSUAL_WHALES_API_TOKEN is not set – Unusual Whales data will be skipped"
            )
            return

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

    def get_chain(self, symbol: str, timestamp: datetime, expiration: Optional[str] = None) -> List[OptionContract]:
        """Get options chain for a symbol using the public contracts endpoint."""

        if not self.client:
            if self.use_stub:
                return self._get_stub_chain(symbol, timestamp)

            logger.info(
                "⏭️  Skipping Unusual Whales for %s (adapter disabled: %s)",
                symbol,
                self._disabled_reason or "unknown",
            )
            return []

        url = f"{self.base_url}/api/stock/{symbol}/option-contracts"
        params = {"expiration_date": expiration, "limit": 500} if expiration else {"limit": 500}

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

            contracts: List[OptionContract] = []
            for option in contracts_data:
                try:
                    # Dec 2025: API returns "option_symbol" field (not "symbol" or "occ_symbol")
                    symbol_str = option.get("option_symbol") or option.get("symbol") or option.get("occ_symbol")
                    if not symbol_str:
                        continue

                    parsed = OptionUtils.parse_occ_symbol(symbol_str.replace(" ", ""))
                    exp_date = parsed["expiration"]
                    option_type = parsed["option_type"]
                    strike = float(parsed["strike"])

                    # Dec 2025: API uses nbbo_bid/nbbo_ask (prioritize these over bid/ask)
                    bid = float(option.get("nbbo_bid") or option.get("bid") or 0)
                    ask = float(option.get("nbbo_ask") or option.get("ask") or 0)
                    last = float(option.get("last_price") or option.get("last") or 0)
                    volume = float(option.get("volume") or 0)
                    oi = float(option.get("open_interest") or 0)
                    iv = float(option.get("implied_volatility") or option.get("iv") or 0)

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

            logger.warning(
                f"No valid contracts parsed for {symbol} from Unusual Whales response"
            )
            return []

        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            detail = self._extract_detail(error.response)
            self._log_once(symbol, url, params, status_code, detail)

            if status_code in {401, 403}:
                self._disabled_reason = f"auth-{status_code}"
                self.client = None
                logger.error(
                    "❌ Unusual Whales authentication/subscription error %s - check UNUSUAL_WHALES_API_TOKEN",
                    status_code,
                )
                return []

            if status_code == 404:
                logger.info(
                    "⏭️  Unusual Whales has no data for %s (404) - skipping without stub",
                    symbol,
                )
                return []

            if status_code in {400, 422}:
                logger.error(
                    "❌ Unusual Whales rejected request for %s | status=%s | detail=%s",
                    symbol,
                    status_code,
                    detail,
                )
                return []

            if status_code == 429 or status_code >= 500:
                logger.warning(
                    "⚠️  Unusual Whales transient error for %s | status=%s | detail=%s",
                    symbol,
                    status_code,
                    detail,
                )
                return self._get_stub_chain(symbol, timestamp)

            logger.error(
                "❌ Unexpected Unusual Whales response for %s | status=%s | detail=%s",
                symbol,
                status_code,
                detail,
            )
            return []
        except httpx.HTTPError as error:
            logger.warning(f"HTTP error getting options chain for {symbol}: {error}")
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

    def get_greek_exposure(self, symbol: str, date: Optional[str] = None) -> dict:
        """Get Greek Exposure including Gamma, Delta, Vanna, and Charm.

        Returns aggregate greek exposure that market makers are exposed to.
        Includes higher-order greeks like Vanna and Charm for sophisticated analysis.

        Response includes:
        - call_gamma, put_gamma: Gamma exposure
        - call_delta, put_delta: Delta exposure
        - call_vanna, put_vanna: Vanna exposure (dDelta/dIV)
        - call_charm, put_charm: Charm exposure (dDelta/dTime)
        - call_vega, put_vega: Vega exposure
        - call_theta, put_theta: Theta exposure
        """
        if not self.client or not self.api_token:
            logger.debug("No API token - skipping greek exposure for %s", symbol)
            return {}

        try:
            url = f"{self.base_url}/api/stock/{symbol}/greek-exposure"
            params = {}
            if date:
                params["date"] = date

            response = self.client.get(url, params=params if params else None)
            response.raise_for_status()
            data = response.json()

            # Extract the latest data point
            exposure_data = data.get("data", [])
            if exposure_data and isinstance(exposure_data, list):
                latest = exposure_data[0]
                logger.info(
                    "✅ Retrieved greek exposure for %s: GEX(calls)=%.2f, VEX=%.2f, Charm=%.2f",
                    symbol,
                    float(latest.get("call_gamma", 0)),
                    float(latest.get("call_vanna", 0)),
                    float(latest.get("call_charm", 0)),
                )
                return latest

            return {}
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info("Greek exposure not available for %s (404)", symbol)
                return {}
            logger.error(
                "Failed to fetch greek exposure for %s | status=%s | detail=%s",
                symbol,
                exc.response.status_code,
                self._extract_detail(exc.response),
            )
            return {}
        except Exception as error:
            logger.error(f"Error getting greek exposure for {symbol}: {error}")
            return {}

    def get_dark_pool(self, symbol: str, min_premium: Optional[float] = None, limit: int = 100) -> List[dict]:
        """Get dark pool trades for a ticker.

        Returns recent dark pool (off-exchange) trades, useful for identifying
        institutional block trades and large player positioning.

        Args:
            symbol: Stock ticker
            min_premium: Minimum trade value (e.g., 1000000 for $1M+ trades)
            limit: Max number of trades to return (default 100, max 500)

        Returns:
            List of dark pool trades with size, price, premium, timestamp
        """
        if not self.client or not self.api_token:
            logger.debug("No API token - skipping dark pool for %s", symbol)
            return []

        try:
            url = f"{self.base_url}/api/darkpool/{symbol}"
            params = {"limit": min(limit, 500)}
            if min_premium:
                params["min_premium"] = min_premium

            response = self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            trades = data.get("data", [])
            if trades:
                total_premium = sum(float(t.get("premium", 0)) for t in trades)
                logger.info(
                    "✅ Retrieved %d dark pool trades for %s (total premium: $%.2fM)",
                    len(trades),
                    symbol,
                    total_premium / 1_000_000,
                )
                return trades

            logger.debug("No dark pool trades for %s", symbol)
            return []
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.info("No dark pool data for %s (404)", symbol)
                return []
            logger.error(
                "Failed to fetch dark pool for %s | status=%s | detail=%s",
                symbol,
                exc.response.status_code,
                self._extract_detail(exc.response),
            )
            return []
        except Exception as error:
            logger.error(f"Error getting dark pool for {symbol}: {error}")
            return []

    def close(self) -> None:
        if self.client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Backwards compatibility for existing imports
UnusualWhalesAdapter = UnusualWhalesOptionsAdapter
