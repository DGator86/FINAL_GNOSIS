"""Shared helpers for configuring Alpaca connections.

These helpers centralize how the application determines whether to use
paper or live trading and which base URL to target. They rely on
environment variables so CLI tools, the dashboard, and automated loops
all stay in sync.
"""

from __future__ import annotations

import os


def _env_flag(name: str, default: bool = True) -> bool:
    """Parse a boolean environment flag.

    Args:
        name: Environment variable to read.
        default: Default value when the variable is unset.

    Returns:
        Boolean value for the flag.
    """

    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_alpaca_paper_setting(default: bool = True) -> bool:
    """Determine whether Alpaca should run in paper mode.

    The ALPACA_PAPER environment variable overrides the default. Any of
    the following values are treated as **true** (case-insensitive):
    ``1``, ``true``, ``yes``, ``y``, ``on``.
    """

    return _env_flag("ALPACA_PAPER", default)


def get_alpaca_base_url(paper: bool) -> str:
    """Return the correct Alpaca base URL for the requested mode."""

    override = os.getenv("ALPACA_BASE_URL")
    if override:
        return override

    return "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"

