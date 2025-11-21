"""High-level Alpaca trading helper with environment-based credentials.

This module exposes a small convenience wrapper around the official
``alpaca-py`` SDK so GNOSIS components can access account details,
positions, market data, and basic order flows without hard-coding API
keys. It intentionally avoids the deprecated ``alpaca-trade-api``
package and relies on environment variables for secrets.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from execution.broker_adapters.settings import (
    get_alpaca_paper_setting,
    get_required_options_level,
)


@dataclass
class AlpacaConfig:
    """Configuration for Alpaca connections."""

    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    paper: bool = field(default_factory=lambda: get_alpaca_paper_setting(True))

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("ALPACA_API_KEY")
        if self.api_secret is None:
            self.api_secret = os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca credentials not found. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY or pass explicit credentials to AlpacaConfig."
            )


class AlpacaTrader:
    """Alpaca trading interface for GNOSIS using alpaca-py."""

    def __init__(self, config: Optional[AlpacaConfig] = None):
        self.config = config or AlpacaConfig()
        self.logger = self._setup_logger()

        self.trading_client = TradingClient(
            api_key=self.config.api_key,
            secret_key=self.config.api_secret,
            paper=self.config.paper,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=self.config.api_key,
            secret_key=self.config.api_secret,
        )

        self._verify_connection()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("gnosis.alpaca_trader")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _verify_connection(self) -> None:
        try:
            account = self.trading_client.get_account()
            self.logger.info(
                "Connected to Alpaca - Account ID: %s - Status: %s",
                account.id,
                account.status,
            )
            self._enforce_options_permissions(account)
        except APIError as exc:
            self.logger.error("Failed to connect to Alpaca: %s", exc)
            raise

    @staticmethod
    def _map_time_in_force(value: str) -> TimeInForce:
        mapping = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
        }
        return mapping.get(value.lower(), TimeInForce.GTC)

    def get_account(self) -> Dict[str, Any]:
        try:
            account = self.trading_client.get_account()
            return {
                "account_id": str(account.id),
                "status": account.status,
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "transfers_blocked": account.transfers_blocked,
                "account_blocked": account.account_blocked,
                "options_trading_level": getattr(account, "options_trading_level", None),
                "options_approved_level": getattr(account, "options_approved_level", None),
                "options_buying_power": float(account.options_buying_power)
                if getattr(account, "options_buying_power", None) is not None
                else None,
            }
        except APIError as exc:
            self.logger.error("Failed to get account info: %s", exc)
            return {}

    def get_positions(self) -> List[Dict[str, Any]]:
        try:
            positions = self.trading_client.get_all_positions()
            result: List[Dict[str, Any]] = []
            for pos in positions:
                result.append(
                    {
                        "symbol": pos.symbol,
                        "qty": float(pos.qty),
                        "side": pos.side.value,
                        "market_value": float(pos.market_value),
                        "avg_entry_price": float(pos.avg_entry_price),
                        "current_price": float(pos.current_price),
                        "unrealized_pl": float(pos.unrealized_pl),
                        "unrealized_plpc": float(pos.unrealized_plpc),
                        "change_today": float(pos.change_today),
                    }
                )
            return result
        except APIError as exc:
            self.logger.error("Failed to get positions: %s", exc)
            return []

    def get_market_data(
        self, symbol: str, timeframe: TimeFrame = TimeFrame.Hour, limit: int = 100
    ) -> pd.DataFrame:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                limit=limit,
            )
            bars = self.data_client.get_stock_bars(request).df
            if bars.empty:
                self.logger.warning("No data received for %s", symbol)
                return pd.DataFrame()

            bars = bars.reset_index()
            bars = bars.rename(
                columns={
                    "timestamp": "timestamp",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
            )
            return bars
        except APIError as exc:
            self.logger.error("Failed to get market data for %s: %s", symbol, exc)
            return pd.DataFrame()

    def place_market_order(
        self, symbol: str, qty: float, side: str, time_in_force: str = "gtc"
    ) -> Optional[Dict[str, Any]]:
        try:
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                time_in_force=self._map_time_in_force(time_in_force),
            )
            order = self.trading_client.submit_order(request)
            self.logger.info("Market order placed: %s %s %s", side.upper(), qty, symbol)
            return self._format_order(order)
        except APIError as exc:
            self.logger.error("Failed to place market order: %s", exc)
            return None

    def place_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        time_in_force: str = "gtc",
    ) -> Optional[Dict[str, Any]]:
        try:
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                limit_price=limit_price,
                time_in_force=self._map_time_in_force(time_in_force),
            )
            order = self.trading_client.submit_order(request)
            self.logger.info(
                "Limit order placed: %s %s %s @ %s", side.upper(), qty, symbol, limit_price
            )
            return self._format_order(order)
        except APIError as exc:
            self.logger.error("Failed to place limit order: %s", exc)
            return None

    def place_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        take_profit_price: float,
        stop_loss_price: float,
    ) -> Optional[Dict[str, Any]]:
        try:
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=take_profit_price),
                stop_loss=StopLossRequest(stop_price=stop_loss_price),
                time_in_force=TimeInForce.GTC,
            )
            order = self.trading_client.submit_order(request)
            self.logger.info(
                "Bracket order placed: %s %s %s TP: %s SL: %s",
                side.upper(),
                qty,
                symbol,
                take_profit_price,
                stop_loss_price,
            )
            return self._format_order(order)
        except APIError as exc:
            self.logger.error("Failed to place bracket order: %s", exc)
            return None

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.trading_client.cancel_order_by_id(order_id)
            self.logger.info("Order cancelled: %s", order_id)
            return True
        except APIError as exc:
            self.logger.error("Failed to cancel order: %s", exc)
            return False

    def cancel_all_orders(self) -> bool:
        try:
            self.trading_client.cancel_orders()
            self.logger.info("All orders cancelled")
            return True
        except APIError as exc:
            self.logger.error("Failed to cancel all orders: %s", exc)
            return False

    def close_position(self, symbol: str) -> bool:
        try:
            self.trading_client.close_position(symbol)
            self.logger.info("Position closed: %s", symbol)
            return True
        except APIError as exc:
            self.logger.error("Failed to close position: %s", exc)
            return False

    def close_all_positions(self) -> bool:
        try:
            self.trading_client.close_all_positions(cancel_orders=True)
            self.logger.info("All positions closed")
            return True
        except APIError as exc:
            self.logger.error("Failed to close all positions: %s", exc)
            return False

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return self._format_order(order)
        except APIError as exc:
            self.logger.error("Failed to get order status: %s", exc)
            return None

    def is_market_open(self) -> bool:
        try:
            clock = self.trading_client.get_clock()
            return bool(clock.is_open)
        except APIError as exc:
            self.logger.error("Failed to check market status: %s", exc)
            return False

    def get_next_market_open(self) -> Optional[datetime]:
        try:
            clock = self.trading_client.get_clock()
            return clock.next_open
        except APIError as exc:
            self.logger.error("Failed to get next market open: %s", exc)
            return None

    def _enforce_options_permissions(self, account: object) -> None:
        """Ensure the account exposes the required options trading tier."""

        required_level = get_required_options_level()
        active_level = getattr(account, "options_trading_level", None)
        approved_level = getattr(account, "options_approved_level", None)

        if active_level is None:
            self.logger.warning(
                "Alpaca account does not report an options trading level; cannot validate permissions"
            )
            return

        self.logger.info(
            "Options trading level detected: %s (approved: %s, required: %s)",
            active_level,
            approved_level,
            required_level,
        )

        if active_level < required_level:
            raise ValueError(
                "Alpaca options trading level is insufficient for configured strategies. "
                f"Detected level {active_level} but require {required_level}. "
                "Upgrade to level 3 in the Alpaca dashboard to enable advanced options execution."
            )

    def _format_order(self, order: Any) -> Dict[str, Any]:
        return {
            "order_id": str(order.id),
            "symbol": order.symbol,
            "qty": float(order.qty),
            "side": order.side.value if hasattr(order.side, "value") else order.side,
            "type": order.type.value if hasattr(order.type, "value") else order.type,
            "status": order.status.value
            if hasattr(order.status, "value")
            else order.status,
            "submitted_at": order.submitted_at,
            "filled_avg_price": float(order.filled_avg_price)
            if order.filled_avg_price
            else None,
            "limit_price": float(order.limit_price) if getattr(order, "limit_price", None) else None,
        }


__all__ = ["AlpacaConfig", "AlpacaTrader"]
