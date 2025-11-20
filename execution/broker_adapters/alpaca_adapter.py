"""Alpaca broker adapter - Real implementation with Alpaca API."""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from loguru import logger
from pydantic import BaseModel

from execution.broker_adapters.settings import get_alpaca_base_url, get_alpaca_paper_setting


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
    
    def __init__(self, paper: Optional[bool] = None):
        """
        Initialize Alpaca adapter.
        
        Args:
            paper: Whether to use paper trading (default: True)
        """
        # Allow explicit override or fall back to environment flag
        self.paper = get_alpaca_paper_setting() if paper is None else paper
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = get_alpaca_base_url(self.paper)
        
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
        
        logger.info(f"AlpacaBrokerAdapter initialized (paper={self.paper}, base_url={self.base_url})")
        
        # Verify connection
        try:
            account = self.trading_client.get_account()
            logger.info(f"Connected to Alpaca - Account ID: {account.id}, Balance: ${float(account.cash):,.2f}")
        except APIError as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise
    
    def get_account(self) -> Account:
        """Get account information."""
        try:
            account = self.trading_client.get_account()
            
            return Account(
                account_id=str(account.id),  # Convert UUID to string
                cash=float(account.cash),
                buying_power=float(account.buying_power),
                portfolio_value=float(account.portfolio_value),
                equity=float(account.equity),
                last_equity=float(account.last_equity),
                pattern_day_trader=account.pattern_day_trader,
                trading_blocked=account.trading_blocked,
            )
        except APIError as e:
            logger.error(f"Error getting account info: {e}")
            raise
    
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
            # Convert string enums
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            
            # For now, only support market orders
            # TODO: Add support for limit/stop orders
            if order_type.lower() != "market":
                logger.warning(f"Order type {order_type} not yet supported, using market order")
            
            # Create market order request
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC,
            )
            
            # Submit order
            order = self.trading_client.submit_order(order_data)
            
            logger.info(f"Order submitted: {side.upper()} {quantity} {symbol} - Order ID: {order.id}")
            
            return order.id
            
        except APIError as e:
            logger.error(f"Error placing order: {e}")
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
