"""Alpaca broker adapter - Real implementation with Alpaca API."""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import (
    OptionSnapshotRequest,
    StockLatestQuoteRequest,
)
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
)
from loguru import logger
from pydantic import BaseModel

from config.credentials import get_alpaca_credentials
from execution.broker_adapters.settings import (
    get_alpaca_base_url,
    get_alpaca_paper_setting,
    get_required_options_level,
)
from execution.risk_utils import (
    assert_within_max,
    calculate_order_value,
    is_option_symbol,
)
from gnosis.utils.option_utils import OptionUtils


class Account(BaseModel):
    """Account information."""
    
    account_id: str
    cash: float
    buying_power: float
    portfolio_value: float
    equity: float
    last_equity: float
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    options_trading_level: Optional[int] = None
    options_approved_level: Optional[int] = None
    options_buying_power: Optional[float] = None


class Position(BaseModel):
    """Position information."""
    
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    side: str


class AlpacaBrokerAdapter:
    """Alpaca broker adapter for paper/live trading."""

    def __init__(
        self,
        paper: Optional[bool] = None,
        *,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        """
        Initialize Alpaca adapter.

        Args:
            paper: Whether to use paper trading (default: True)
        """
        # Allow explicit override or fall back to environment flag
        self.paper = get_alpaca_paper_setting() if paper is None else paper
        creds = get_alpaca_credentials(api_key=api_key, secret_key=secret_key, base_url=get_alpaca_base_url(self.paper))
        self.api_key = creds.api_key
        self.secret_key = creds.secret_key
        self.base_url = creds.base_url

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca credentials not found in environment. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")

        # Initialize trading client
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
        )

        # Initialize data client
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
        )
        self.option_data_client: Optional[OptionHistoricalDataClient] = None

        # Risk management settings from environment
        self.max_position_size_pct = float(os.getenv("MAX_POSITION_SIZE_PCT", "2.0")) / 100.0
        self.max_daily_loss_usd = float(os.getenv("MAX_DAILY_LOSS_USD", "5000.0"))
        self.max_portfolio_leverage = float(os.getenv("MAX_PORTFOLIO_LEVERAGE", "1.0"))

        # Track daily P&L for circuit breaker
        self.session_start_equity: float | None = None
        self._checked_permissions = False

        logger.info(
            "AlpacaBrokerAdapter initialized (paper=%s, base_url=%s)",
            self.paper,
            self.base_url,
        )
        logger.info(
            "Risk Limits - Max Position: %.1f%%, Max Daily Loss: $%s",
            self.max_position_size_pct * 100,
            f"{self.max_daily_loss_usd:,.2f}",
        )
    
    def get_account(self) -> Account:
        """Get account information."""
        try:
            account = self.trading_client.get_account()

            result = Account(
                account_id=str(account.id),  # Convert UUID to string
                cash=float(account.cash),
                buying_power=float(account.buying_power),
                portfolio_value=float(account.portfolio_value),
                equity=float(account.equity),
                last_equity=float(account.last_equity),
                pattern_day_trader=account.pattern_day_trader,
                trading_blocked=account.trading_blocked,
                options_trading_level=getattr(account, "options_trading_level", None),
                options_approved_level=getattr(account, "options_approved_level", None),
                options_buying_power=(
                    float(account.options_buying_power)
                    if getattr(account, "options_buying_power", None) is not None
                    else None
                ),
            )

            if self.session_start_equity is None:
                self.session_start_equity = result.equity
            if not self._checked_permissions:
                self._verify_options_permissions(account)
                self._checked_permissions = True

            self.equity = result.equity
            self.cash = result.cash
            self.buying_power = result.buying_power

            return result
        except APIError as e:
            logger.error(f"Error getting account info: {e}")
            raise

    def _validate_position_size(self, symbol: str, quantity: float, current_price: Optional[float] = None) -> None:
        """
        Validate that the order doesn't exceed position size limits.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            current_price: Current price (will fetch if not provided)

        Raises:
            ValueError: If position size exceeds limits
        """
        # Get current account value
        account = self.get_account()
        portfolio_value = account.portfolio_value
        max_position_value = portfolio_value * self.max_position_size_pct

        # Get current price if not provided
        if current_price is None:
            if is_option_symbol(symbol):
                current_price = self._fetch_option_price(symbol)
            else:
                try:
                    quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                    quote = self.data_client.get_stock_latest_quote(quote_request)
                    current_price = float(quote[symbol].ask_price)
                except Exception as e:
                    logger.warning(f"Could not fetch price for {symbol}, skipping order sizing: {e}")
                    raise ValueError(f"Cannot size order for {symbol} without a price") from e

        if current_price is None or current_price <= 0:
            raise ValueError(f"Cannot size order for {symbol}: invalid price {current_price}")

        order_value = calculate_order_value(symbol, quantity, current_price)

        assert_within_max(symbol, order_value, portfolio_value, self.max_position_size_pct)

        logger.debug(f"Position size validation passed: ${order_value:,.2f} <= ${max_position_value:,.2f}")

    def _fetch_option_price(self, symbol: str) -> Optional[float]:
        try:
            if self.option_data_client is None:
                self.option_data_client = OptionHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                )

            request = OptionSnapshotRequest(symbol_or_symbols=symbol)
            snapshot = self.option_data_client.get_option_snapshot(request)
            snap = snapshot.get(symbol) if hasattr(snapshot, "get") else None
            if snap:
                bid = getattr(snap.latest_quote, "bid_price", None) if snap.latest_quote else None
                ask = getattr(snap.latest_quote, "ask_price", None) if snap.latest_quote else None
                last = getattr(snap, "last_price", None)
                prices = [p for p in [bid, ask, last] if p is not None and p > 0]
                if prices:
                    return float(sum(prices) / len(prices))
        except Exception as error:  # pragma: no cover - network dependency
            logger.warning(
                "Could not fetch option price for %s: %s (symbol fields=%s)",
                symbol,
                error,
                OptionUtils.parse_occ_symbol(symbol) if len(symbol) >= 15 else {},
            )
        return None

    def _check_daily_loss_limit(self) -> None:
        """
        Check if daily loss limit has been exceeded (circuit breaker).

        Raises:
            ValueError: If daily loss limit exceeded
        """
        if self.session_start_equity is None:
            logger.warning("Session start equity not set, cannot check daily loss limit")
            return

        # Get current equity
        account = self.get_account()
        current_equity = account.equity

        # Calculate session P&L
        session_pnl = current_equity - self.session_start_equity

        # Check if loss exceeds limit
        if session_pnl < -self.max_daily_loss_usd:
            raise ValueError(
                f"CIRCUIT BREAKER TRIGGERED: Daily loss of ${-session_pnl:,.2f} exceeds limit of "
                f"${self.max_daily_loss_usd:,.2f}. Trading halted for this session."
            )

        logger.debug(f"Daily loss check passed: P&L = ${session_pnl:+,.2f}")

    def _verify_options_permissions(self, account: object) -> None:
        """Ensure the account has the required options trading level."""

        required_level = get_required_options_level()
        approved_level = getattr(account, "options_approved_level", None)
        active_level = getattr(account, "options_trading_level", None)

        if active_level is None:
            logger.warning(
                "Alpaca account does not report an options trading level; cannot validate permissions"
            )
            return

        logger.info(
            "Options trading level detected: %s (approved: %s, required: %s)",
            active_level,
            approved_level,
            required_level,
        )

        if active_level < required_level:
            raise ValueError(
                "Alpaca options trading level is insufficient. "
                f"Detected level {active_level} but level {required_level} is required for multi-leg strategies. "
                "Visit the Alpaca dashboard to request the highest options tier."
            )
    
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        try:
            positions = self.trading_client.get_all_positions()
            
            result = []
            for pos in positions:
                result.append(Position(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    avg_entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                    cost_basis=float(pos.cost_basis),
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_pct=float(pos.unrealized_plpc),
                    side=pos.side.value,
                ))
            
            return result
        except APIError as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        try:
            pos = self.trading_client.get_open_position(symbol)
            
            return Position(
                symbol=pos.symbol,
                quantity=float(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                cost_basis=float(pos.cost_basis),
                unrealized_pnl=float(pos.unrealized_pl),
                unrealized_pnl_pct=float(pos.unrealized_plpc),
                side=pos.side.value,
            )
        except APIError:
            return None
    
    def place_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Optional[str]:
        """
        Place an order.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity (fractional shares supported)
            side: "buy" or "sell"
            order_type: "market", "limit", "stop", "stop_limit"
            time_in_force: "day", "gtc", "ioc", "fok"
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Risk management checks before placing order
            if side.lower() == "buy":
                # Check daily loss limit (circuit breaker)
                self._check_daily_loss_limit()

                # Validate position size
                self._validate_position_size(symbol, quantity, limit_price if limit_price else None)

            # Convert string enums
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            order_type_lower = order_type.lower()
            
            # Convert time in force
            tif_map = {
                "day": TimeInForce.DAY,
                "gtc": TimeInForce.GTC,
                "ioc": TimeInForce.IOC,
                "fok": TimeInForce.FOK,
            }
            tif = tif_map.get(time_in_force.lower(), TimeInForce.DAY)
            
            # Create appropriate order request based on type
            if order_type_lower == "market":
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=tif,
                )
                logger.info(f"Submitting MARKET order: {side.upper()} {quantity} {symbol}")
                
            elif order_type_lower == "limit":
                if limit_price is None:
                    logger.error("Limit price required for limit orders")
                    return None
                order_data = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=tif,
                    limit_price=limit_price,
                )
                logger.info(f"Submitting LIMIT order: {side.upper()} {quantity} {symbol} @ ${limit_price}")
                
            elif order_type_lower == "stop":
                if stop_price is None:
                    logger.error("Stop price required for stop orders")
                    return None
                order_data = StopOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=tif,
                    stop_price=stop_price,
                )
                logger.info(f"Submitting STOP order: {side.upper()} {quantity} {symbol} @ stop ${stop_price}")
                
            elif order_type_lower == "stop_limit":
                if limit_price is None or stop_price is None:
                    logger.error("Both limit and stop prices required for stop-limit orders")
                    return None
                order_data = StopLimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=tif,
                    limit_price=limit_price,
                    stop_price=stop_price,
                )
                logger.info(
                    f"Submitting STOP-LIMIT order: {side.upper()} {quantity} {symbol} "
                    f"@ stop ${stop_price}, limit ${limit_price}"
                )
                
            else:
                logger.error(f"Unsupported order type: {order_type}. Supported: market, limit, stop, stop_limit")
                return None
            
            # Submit order
            order = self.trading_client.submit_order(order_data)
            
            logger.info(f"Order submitted successfully - Order ID: {order.id}")
            
            return str(order.id)
            
        except APIError as e:
            logger.error(f"Error placing order: {e}")
            return None

    def place_bracket_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        take_profit_price: float,
        stop_loss_price: float,
        time_in_force: str = "gtc",
    ) -> Optional[str]:
        """
        Place a bracket order with stop loss and take profit.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: "buy" or "sell"
            take_profit_price: Take profit limit price
            stop_loss_price: Stop loss price
            time_in_force: "day", "gtc", "ioc", "fok"

        Returns:
            Order ID if successful, None otherwise
        """
        try:
            # Risk management checks before placing order
            if side.lower() == "buy":
                # Check daily loss limit (circuit breaker)
                self._check_daily_loss_limit()

                # Validate position size
                self._validate_position_size(symbol, quantity, None)

            # Convert string enums
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            # Convert time in force
            tif_map = {
                "day": TimeInForce.DAY,
                "gtc": TimeInForce.GTC,
                "ioc": TimeInForce.IOC,
                "fok": TimeInForce.FOK,
            }
            tif = tif_map.get(time_in_force.lower(), TimeInForce.GTC)

            # Create bracket order request
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=tif,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=take_profit_price),
                stop_loss=StopLossRequest(stop_price=stop_loss_price),
            )

            logger.info(
                f"Submitting BRACKET order: {side.upper()} {quantity} {symbol} "
                f"| TP: ${take_profit_price:.2f} | SL: ${stop_loss_price:.2f}"
            )

            # Submit order
            order = self.trading_client.submit_order(order_data)

            logger.info(f"Bracket order submitted successfully - Order ID: {order.id}")

            return str(order.id)

        except APIError as e:
            logger.error(f"Error placing bracket order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except APIError as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def close_position(self, symbol: str, qty: Optional[float] = None) -> bool:
        """
        Close a position (all or partial).
        
        Args:
            symbol: Trading symbol
            qty: Quantity to close (None = close all)
            
        Returns:
            True if successful
        """
        try:
            if qty is None:
                # Close entire position
                self.trading_client.close_position(symbol)
                logger.info(f"Closed entire position in {symbol}")
            else:
                # Close partial position
                pos = self.get_position(symbol)
                if not pos:
                    logger.warning(f"No position found for {symbol}")
                    return False
                
                # Determine side for closing order
                side = "sell" if float(pos.quantity) > 0 else "buy"
                self.place_order(symbol, abs(qty), side)
                logger.info(f"Closed {qty} shares of {symbol}")
            
            return True
        except APIError as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    def close_all_positions(self) -> bool:
        """Close all positions."""
        try:
            self.trading_client.close_all_positions(cancel_orders=True)
            logger.info("Closed all positions")
            return True
        except APIError as e:
            logger.error(f"Error closing all positions: {e}")
            return False
    
    def get_latest_quote(self, symbol: str) -> Optional[dict]:
        """Get latest quote for a symbol."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.data_client.get_stock_latest_quote(request)
            
            if symbol in quote:
                q = quote[symbol]
                return {
                    "symbol": symbol,
                    "bid": float(q.bid_price),
                    "ask": float(q.ask_price),
                    "bid_size": float(q.bid_size),
                    "ask_size": float(q.ask_size),
                    "timestamp": q.timestamp,
                }
            
            return None
        except APIError as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None

    def get_market_clock(self) -> Optional[dict]:
        """Return the current market clock status."""
        try:
            clock = self.trading_client.get_clock()
            return {
                "is_open": bool(clock.is_open),
                "next_open": clock.next_open,
                "next_close": clock.next_close,
            }
        except APIError as e:
            logger.error(f"Error retrieving market clock: {e}")
            return None

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        clock = self.get_market_clock()
        return bool(clock and clock.get("is_open"))

    def get_next_market_open(self) -> Optional[datetime]:
        """Get the next market open timestamp."""
        clock = self.get_market_clock()
        return clock.get("next_open") if clock else None
