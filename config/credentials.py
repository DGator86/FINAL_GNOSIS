"""Central credential resolver with hardcoded defaults and env overrides.

This module provides a single source for API credentials so adapters don't
need to duplicate hardcoded fallbacks. Environment variables still take
precedence, but the provided defaults align with the repository's configured
paper/testing keys.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

# Hardcoded defaults (paper/testing credentials)
ALPACA_DEFAULT_API_KEY = "PKDGAH5CJM4G3RZ2NP5WQNH22U"
ALPACA_DEFAULT_SECRET_KEY = "EfW43tDsmhWgvJkucKhJL3bsXmKyu5Kt1B3WxTFcuHEq"
ALPACA_DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"

UNUSUAL_WHALES_DEFAULT_TOKEN = "8932cd23-72b3-4f74-9848-13f9103b9df5"

MASSIVE_DEFAULT_PRIMARY_KEY = "Jm_fqc_gtSTSXG78P67dpBpO3LX_4P6D"
MASSIVE_DEFAULT_SECONDARY_KEY = "22265906-ec01-4a42-928a-0037ccadbde3"


@dataclass(frozen=True)
class AlpacaCredentials:
    api_key: str
    secret_key: str
    base_url: str


def get_alpaca_credentials(
    api_key: Optional[str] = None, secret_key: Optional[str] = None, base_url: Optional[str] = None
) -> AlpacaCredentials:
    """Return Alpaca credentials preferring explicit args, then env, then defaults."""

    resolved_api_key = api_key or os.getenv("ALPACA_API_KEY") or ALPACA_DEFAULT_API_KEY
    resolved_secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY") or ALPACA_DEFAULT_SECRET_KEY
    resolved_base_url = base_url or os.getenv("ALPACA_BASE_URL") or ALPACA_DEFAULT_BASE_URL
    return AlpacaCredentials(
        api_key=resolved_api_key,
        secret_key=resolved_secret_key,
        base_url=resolved_base_url,
    )


def get_unusual_whales_token(token: Optional[str] = None) -> str:
    """Return Unusual Whales token preferring explicit arg, then env, then default."""

    return (
        token
        or os.getenv("UNUSUAL_WHALES_API_TOKEN")
        or os.getenv("UNUSUAL_WHALES_TOKEN")
        or os.getenv("UNUSUAL_WHALES_API_KEY")
        or UNUSUAL_WHALES_DEFAULT_TOKEN
    )


def get_massive_api_keys(primary: Optional[str] = None, secondary: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Return MASSIVE primary/secondary API keys preferring args, then env, then defaults."""

    resolved_primary = primary or os.getenv("MASSIVE_API_KEY") or MASSIVE_DEFAULT_PRIMARY_KEY
    resolved_secondary = secondary or os.getenv("MASSIVE_API_KEY_SECONDARY") or MASSIVE_DEFAULT_SECONDARY_KEY
    return resolved_primary, resolved_secondary


def massive_api_enabled(default: bool = True) -> bool:
    """Return whether MASSIVE API usage is enabled (env override with default)."""

    return os.getenv("MASSIVE_API_ENABLED", str(default).lower()).lower() == "true"
