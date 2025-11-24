"""Utilities for loading and persisting the active trading universe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from loguru import logger

WATCHLIST_PATH = Path("data/universe/current_watchlist.json")
_FALLBACK_SYMBOLS = ["SPY"]


def _generate_dynamic_watchlist(max_names: int | None = None) -> List[str]:
    """Generate and persist a dynamic top-N watchlist using the scanner config."""

    try:
        from config import load_config
        from engines.scanner import get_dynamic_universe

        config = load_config()
        top_n = max_names or config.scanner.default_top_n

        logger.info(f"Generating dynamic universe (top {top_n}) because no watchlist was found")
        symbols = get_dynamic_universe(config.scanner.model_dump(), top_n=top_n)
        symbols = symbols[:max_names] if max_names is not None else symbols

        save_active_watchlist(symbols)
        logger.info(f"Saved dynamic watchlist with {len(symbols)} symbols")
        return symbols
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(f"Failed to generate dynamic watchlist: {exc}")
        return _FALLBACK_SYMBOLS[:max_names] if max_names is not None else _FALLBACK_SYMBOLS


def load_active_watchlist(max_names: int | None = None) -> List[str]:
    """
    Load the current active universe of symbols.

    Returns a sorted, deduped, upper-case list. If the file does not
    exist or is empty, automatically seeds it using the dynamic
    universe ranker so multi-symbol trading can start immediately.
    """
    if not WATCHLIST_PATH.exists():
        return _generate_dynamic_watchlist(max_names=max_names)

    try:
        with WATCHLIST_PATH.open() as f:
            symbols = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(f"Failed to read watchlist ({WATCHLIST_PATH}): {exc}")
        return _generate_dynamic_watchlist(max_names=max_names)

    symbols = [s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()]
    symbols = sorted(set(symbols))

    if not symbols:
        return _generate_dynamic_watchlist(max_names=max_names)

    if max_names is not None:
        symbols = symbols[:max_names]

    return symbols


def save_active_watchlist(symbols: List[str]) -> None:
    """Persist the current active watchlist to disk."""
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with WATCHLIST_PATH.open("w") as f:
        json.dump(symbols, f)
