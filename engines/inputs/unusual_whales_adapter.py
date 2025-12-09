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

    def close(self) -> None:
        if self.client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Backwards compatibility for existing imports
UnusualWhalesAdapter = UnusualWhalesOptionsAdapter
