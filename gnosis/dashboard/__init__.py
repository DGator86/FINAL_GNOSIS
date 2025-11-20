"""Minimal FastAPI dashboard stubs for live trading.

These endpoints expose the latest state pushed from the live trading bot so the
existing launcher scripts can run without missing imports.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Gnosis Trading Dashboard")

_state: Dict[str, Any] = {
    "positions": {},
    "trades": [],
    "agent_votes": {},
    "regime": None,
    "bars": [],
    "portfolio": {},
}


def update_positions(positions: Dict[str, Any]) -> None:
    _state["positions"] = positions


def add_trade(trade: Dict[str, Any]) -> None:
    _state.setdefault("trades", []).append(trade)


def update_agent_votes(votes: Dict[str, Any]) -> None:
    _state["agent_votes"] = votes


def update_regime(regime: Dict[str, Any]) -> None:
    _state["regime"] = regime


def update_bar(bar: Dict[str, Any]) -> None:
    _state.setdefault("bars", []).append(bar)
    _state["bars"] = _state["bars"][-200:]


def update_portfolio_stats(stats: Dict[str, Any]) -> None:
    _state["portfolio"] = stats


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return "<h1>Gnosis Trading Dashboard</h1><p>Data stream active.</p>"


@app.get("/api/state", response_class=JSONResponse)
async def get_state() -> Dict[str, Any]:
    return _state


__all__ = [
    "app",
    "update_positions",
    "add_trade",
    "update_agent_votes",
    "update_regime",
    "update_bar",
    "update_portfolio_stats",
]
