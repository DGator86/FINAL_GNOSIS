"""Alpaca REST client wrapper with symbol-aware order submission."""

from __future__ import annotations

import os
from typing import Any

import time
from typing import Any, Dict, List, Optional, Union

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionSnapshotRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, OptionLegRequest
from alpaca_trade_api import REST


class AlpacaClient:
    """Lightweight Alpaca client that submits orders from trade objects."""

    def __init__(self, key_id: str, secret_key: str, base_url: str):
        self.base_url = base_url
        self.api = REST(key_id, secret_key, base_url)

    @classmethod
    def from_env(cls, mode: str = "paper") -> "AlpacaClient":
        """Instantiate using environment variables.

        Args:
            mode: "paper" (default) or "live". If ALPACA_BASE_URL is set it
                takes precedence over the mode default.
        """

        base_url = os.environ.get(
            "ALPACA_BASE_URL",
            "https://paper-api.alpaca.markets" if mode == "paper" else "https://api.alpaca.markets",
        )

        return cls(
            key_id=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_SECRET_KEY"],
            base_url=base_url,
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

    def place_multi_leg_option_order(
        self,
        legs: List[Dict[str, Any]],
        order_class: Optional[Union[str, OrderClass]] = None,
        quantity: int = 1,
        time_in_force: Union[str, TimeInForce] = "day",
        extended_hours: bool = False,
    ) -> Dict[str, Any]:
        """Place a multi-leg option market order.

        Args:
            legs: List of option legs, each containing:
                - symbol (str): Option contract symbol
                - side (str): 'buy' or 'sell'
                - ratio_qty (int): Quantity ratio for the leg
            order_class: Order class ('simple', 'mleg', etc.)
            quantity: Base quantity for the order
            time_in_force: Time in force (only 'day' supported for options)
            extended_hours: Allow extended hours execution

        Returns:
            Order response dictionary

        Example:
            # Bull call spread
            legs = [
                {"symbol": "AAPL230616C00150000", "side": "buy", "ratio_qty": 1},
                {"symbol": "AAPL230616C00160000", "side": "sell", "ratio_qty": 1}
            ]
            client.place_multi_leg_option_order(legs)
        """
        # Initialize TradingClient if not already done
        if not hasattr(self, "_trading_client"):
            self._trading_client = TradingClient(
                api_key=os.environ["ALPACA_API_KEY"],
                secret_key=os.environ["ALPACA_SECRET_KEY"],
                paper=self.base_url == "https://paper-api.alpaca.markets",
            )

        # Validate inputs
        if not legs:
            raise ValueError("No option legs provided")
        if len(legs) > 4:
            raise ValueError("Maximum of 4 legs allowed for option orders")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        # Convert time_in_force to enum
        if isinstance(time_in_force, str):
            if time_in_force.lower() != "day":
                raise ValueError("Only 'day' time_in_force is supported for options")
            time_in_force_enum = TimeInForce.DAY
        elif isinstance(time_in_force, TimeInForce):
            if time_in_force != TimeInForce.DAY:
                raise ValueError("Only DAY time_in_force is supported for options")
            time_in_force_enum = time_in_force
        else:
            raise ValueError("Invalid time_in_force type")

        # Convert order class
        if order_class is None:
            order_class = OrderClass.MLEG if len(legs) > 1 else OrderClass.SIMPLE
        elif isinstance(order_class, str):
            class_mapping = {
                "simple": OrderClass.SIMPLE,
                "mleg": OrderClass.MLEG,
                "bracket": OrderClass.BRACKET,
                "oco": OrderClass.OCO,
                "oto": OrderClass.OTO,
            }
            order_class = class_mapping.get(order_class.lower())
            if not order_class:
                raise ValueError("Invalid order class")

        # Process legs
        order_legs: List[OptionLegRequest] = []
        for leg in legs:
            if not isinstance(leg["ratio_qty"], int) or leg["ratio_qty"] <= 0:
                raise ValueError(f"Invalid ratio_qty for leg {leg['symbol']}")

            side = OrderSide.BUY if leg["side"].lower() == "buy" else OrderSide.SELL
            order_legs.append(
                OptionLegRequest(symbol=leg["symbol"], side=side, ratio_qty=leg["ratio_qty"])
            )

        # Create order request
        if order_class == OrderClass.MLEG:
            order_data = MarketOrderRequest(
                qty=quantity,
                order_class=order_class,
                time_in_force=time_in_force_enum,
                extended_hours=extended_hours,
                client_order_id=f"gnosis_opt_{int(time.time())}",
                type=OrderType.MARKET,
                legs=order_legs,
            )
        else:
            # Single-leg order
            order_data = MarketOrderRequest(
                symbol=order_legs[0].symbol,
                qty=quantity,
                side=order_legs[0].side,
                order_class=order_class,
                time_in_force=time_in_force_enum,
                extended_hours=extended_hours,
                client_order_id=f"gnosis_opt_{int(time.time())}",
                type=OrderType.MARKET,
            )

        # Submit order
        order = self._trading_client.submit_order(order_data)

        # Convert to dict for return
        return {
            "id": str(order.id),
            "client_order_id": order.client_order_id,
            "status": str(order.status),
            "order_class": str(order.order_class),
            "qty": order.qty,
            "legs": [
                {
                    "symbol": leg.symbol,
                    "side": str(leg.side),
                    "ratio_qty": leg.ratio_qty,
                }
                for leg in (order.legs if hasattr(order, "legs") and order.legs else [])
            ],
        }

    def get_option_snapshot(self, symbol_or_symbols: Union[str, List[str]]) -> Dict[str, Any]:
        """Get option snapshot with Greeks and underlying data.

        Args:
            symbol_or_symbols: Option contract symbol(s)

        Returns:
            Dictionary containing latest quote, Greeks, and underlying info
        """
        # Initialize OptionHistoricalDataClient if not already done
        if not hasattr(self, "_option_data_client"):
            self._option_data_client = OptionHistoricalDataClient(
                api_key=os.environ["ALPACA_API_KEY"],
                secret_key=os.environ["ALPACA_SECRET_KEY"],
            )

        # Create request
        request = OptionSnapshotRequest(symbol_or_symbols=symbol_or_symbols)

        # Get snapshot
        snapshots = self._option_data_client.get_option_snapshot(request)

        # Convert to dict
        result = {}
        for symbol, snapshot in snapshots.items():
            result[symbol] = {
                "latest_quote": {
                    "bid_price": snapshot.latest_quote.bid_price if snapshot.latest_quote else None,
                    "ask_price": snapshot.latest_quote.ask_price if snapshot.latest_quote else None,
                    "bid_size": snapshot.latest_quote.bid_size if snapshot.latest_quote else None,
                    "ask_size": snapshot.latest_quote.ask_size if snapshot.latest_quote else None,
                },
                "greeks": {
                    "delta": snapshot.greeks.delta if snapshot.greeks else None,
                    "gamma": snapshot.greeks.gamma if snapshot.greeks else None,
                    "theta": snapshot.greeks.theta if snapshot.greeks else None,
                    "vega": snapshot.greeks.vega if snapshot.greeks else None,
                    "rho": snapshot.greeks.rho if snapshot.greeks else None,
                },
                "implied_volatility": snapshot.implied_volatility
                if hasattr(snapshot, "implied_volatility")
                else None,
            }

        return result
