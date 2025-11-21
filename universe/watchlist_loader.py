"""Utilities for loading and persisting the active trading universe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

WATCHLIST_PATH = Path("data/universe/current_watchlist.json")


def load_active_watchlist(max_names: int | None = None) -> List[str]:
    """
    Load the current active universe of symbols.

    Returns a sorted, deduped, upper-case list. If the file does not
    exist, fall back to ["SPY"] so the system still runs.
    """
    if not WATCHLIST_PATH.exists():
        return ["SPY"]

    with WATCHLIST_PATH.open() as f:
        symbols = json.load(f)

    symbols = [s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()]
    symbols = sorted(set(symbols))

    if max_names is not None:
        symbols = symbols[:max_names]

    return symbols


def save_active_watchlist(symbols: List[str]) -> None:
    """Persist the current active watchlist to disk."""
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with WATCHLIST_PATH.open("w") as f:
        json.dump(symbols, f)
