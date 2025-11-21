"""Alpaca REST client wrapper with symbol-aware order submission."""

from __future__ import annotations

import os
from typing import Any

from alpaca_trade_api import REST


class AlpacaClient:
    """Lightweight Alpaca client that submits orders from trade objects."""

    def __init__(self, key_id: str, secret_key: str, base_url: str):
        self.api = REST(key_id, secret_key, base_url)

    @classmethod
    def from_env(cls) -> "AlpacaClient":
        """Instantiate using environment variables."""
        return cls(
            key_id=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_SECRET_KEY"],
            base_url=os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        )

    def submit_order_from_trade(self, trade: Any) -> None:
        """Submit an order based on a ProposedTrade-like object."""
        self.api.submit_order(
            symbol=trade.symbol,
            qty=trade.qty,
            side=trade.side,
            type=trade.order_type,
            time_in_force=trade.time_in_force,
            limit_price=getattr(trade, "limit_price", None),
            stop_price=getattr(trade, "stop_price", None),
        )
