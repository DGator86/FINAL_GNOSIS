"""Helpers for the SaaS control plane."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import load_config
from main import build_pipeline
from universe.watchlist_loader import load_active_watchlist

LEDGER_PATH = Path("data/ledger.jsonl")


def load_recent_ledger_entries(limit: int = 20) -> List[Dict[str, Any]]:
    """Load the most recent ledger entries from disk."""

    if not LEDGER_PATH.exists():
        return []

    entries: List[Dict[str, Any]] = []

    with open(LEDGER_PATH, "r") as handle:
        for line in handle:
            try:
                raw = json.loads(line)
                timestamp = raw.get("timestamp")
                if timestamp:
                    raw["timestamp"] = str(timestamp)
                entries.append(raw)
            except json.JSONDecodeError:
                continue

    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return entries[:limit]


def watchlist_overview() -> Dict[str, Any]:
    """Return the active watchlist with a friendly summary."""

    try:
        symbols = load_active_watchlist()
        return {
            "symbols": sorted(symbols),
            "count": len(symbols),
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - defensive guard for UI
        return {"symbols": [], "count": 0, "error": str(exc)}


def get_health_snapshot() -> Dict[str, Any]:
    """Summarize the system configuration, ledger, and watchlist health."""

    config_ok = True
    config_error: Optional[str] = None

    try:
        config = load_config()
    except Exception as exc:  # pragma: no cover - defensive guard for UI
        config_ok = False
        config_error = str(exc)
        config = None

    ledger_entries = load_recent_ledger_entries(limit=200)
    latest_timestamp = ledger_entries[0]["timestamp"] if ledger_entries else None

    watchlist = watchlist_overview()

    return {
        "config_loaded": config_ok,
        "config_error": config_error,
        "ledger_entries": len(ledger_entries),
        "latest_run": latest_timestamp,
        "watchlist_size": watchlist.get("count", 0),
        "watchlist_error": watchlist.get("error"),
        "paper_trading": getattr(getattr(config, "settings", None), "paper_trading", None)
        if config
        else None,
    }


def run_pipeline_once(symbol: str) -> Dict[str, Any]:
    """Run a single pipeline iteration and return a simplified summary."""

    try:
        config = load_config()
    except Exception as exc:
        return {"ok": False, "error": f"Config load failed: {exc}"}

    try:
        runner = build_pipeline(symbol, config)
    except Exception as exc:
        return {"ok": False, "error": f"Pipeline build failed: {exc}"}

    try:
        result = runner.run_once(datetime.now(timezone.utc))
    except Exception as exc:  # pragma: no cover - pipeline exceptions should be surfaced in UI
        return {"ok": False, "error": f"Pipeline execution failed: {exc}"}

    return {
        "ok": True,
        "symbol": symbol.upper(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hedge": result.hedge_snapshot.model_dump() if result.hedge_snapshot else None,
        "liquidity": result.liquidity_snapshot.model_dump() if result.liquidity_snapshot else None,
        "sentiment": result.sentiment_snapshot.model_dump() if result.sentiment_snapshot else None,
        "elasticity": result.elasticity_snapshot.model_dump() if result.elasticity_snapshot else None,
        "consensus": result.consensus,
        "trade_ideas": [idea.model_dump() for idea in result.trade_ideas],
    }
