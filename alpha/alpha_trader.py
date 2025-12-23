"""
Gnosis Alpha - Alpaca Trading Integration

Simple stock trading execution using Alpaca API.
PDT-compliant, directional only, no options.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Alpaca imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopOrderRequest,
        StopLimitOrderRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import (
        OrderSide,
        TimeInForce,
        OrderStatus,
        QueryOrderStatus,
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
    ALPACA_AVAILABLE = True
except ImportError:
    logger.warning("Alpaca SDK not available")
    ALPACA_AVAILABLE = False

from alpha.alpha_config import AlphaConfig
from alpha.pdt_tracker import PDTTracker
from alpha.signal_generator import AlphaSignal, SignalDirection


@dataclass
class AlphaPosition:
    """A position held in Alpha."""
    symbol: str
    quantity: int
    side: str  # "long" or "short"
    entry_price: float
    entry_date: date
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def is_day_trade_eligible(self) -> bool:
        """Check if closing this position today would be a day trade."""
        return self.entry_date == date.today()
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "side": self.side,
            "entry_price": self.entry_price,
            "entry_date": self.entry_date.isoformat(),
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "is_day_trade_eligible": self.is_day_trade_eligible,
        }


@dataclass
class AlphaOrder:
    """An order placed through Alpha."""
    order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: str
    status: str
    submitted_at: datetime
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_qty: int = 0
    
    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "status": self.status,
            "submitted_at": self.submitted_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_price": self.filled_price,
            "filled_qty": self.filled_qty,
        }


class AlphaTrader:
    """
    Alpaca trading integration for Gnosis Alpha.
    
    Executes simple stock orders based on Alpha signals.
    Tracks PDT compliance automatically.
    """
    
    def __init__(
        self,
        config: Optional[AlphaConfig] = None,
        pdt_tracker: Optional[PDTTracker] = None,
        paper: bool = True,
    ):
        """
        Initialize Alpha Trader.
        
        Args:
            config: Alpha configuration
            pdt_tracker: PDT compliance tracker
            paper: Use paper trading (default True)
        """
        self.config = config or AlphaConfig.from_env()
        self.pdt_tracker = pdt_tracker or PDTTracker(
            max_day_trades=self.config.max_day_trades,
            lookback_days=self.config.pdt_lookback_days,
        )
        self.paper = paper
        
        # Initialize Alpaca clients
        self.trading_client: Optional[TradingClient] = None
        self.data_client: Optional[StockHistoricalDataClient] = None
        
        if ALPACA_AVAILABLE:
            self._init_alpaca()
        else:
            logger.error("Alpaca SDK not available - trading disabled")
        
        # Position tracking
        self._positions: Dict[str, AlphaPosition] = {}
        self._pending_orders: Dict[str, AlphaOrder] = {}
    
    def _init_alpaca(self) -> None:
        """Initialize Alpaca API clients."""
        try:
            self.trading_client = TradingClient(
                api_key=self.config.alpaca_api_key,
                secret_key=self.config.alpaca_secret_key,
                paper=self.paper,
            )
            
            self.data_client = StockHistoricalDataClient(
                api_key=self.config.alpaca_api_key,
                secret_key=self.config.alpaca_secret_key,
            )
            
            # Get account info and update PDT tracker
            account = self.trading_client.get_account()
            self.pdt_tracker.update_account_value(float(account.equity))
            
            logger.info(
                f"Alpaca connected | Account: ${float(account.equity):,.2f} | "
                f"PDT Restricted: {self.pdt_tracker.is_pdt_restricted}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca: {e}")
            self.trading_client = None
            self.data_client = None
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information."""
        if not self.trading_client:
            return {"error": "Alpaca not connected"}
        
        try:
            account = self.trading_client.get_account()
            
            return {
                "account_number": account.account_number,
                "status": str(account.status),
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "day_trading_buying_power": float(account.daytrading_buying_power) if account.daytrading_buying_power else 0,
                "pdt_status": {
                    "is_restricted": self.pdt_tracker.is_pdt_restricted,
                    "day_trades_used": self.pdt_tracker.day_trades_used,
                    "day_trades_remaining": self.pdt_tracker.day_trades_remaining,
                },
                "pattern_day_trader": account.pattern_day_trader,
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {"error": str(e)}
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        if not self.data_client:
            return None
        
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if symbol in quotes:
                quote = quotes[symbol]
                # Use mid price
                mid = (float(quote.bid_price) + float(quote.ask_price)) / 2
                return mid
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
        
        return None
    
    def get_positions(self) -> List[AlphaPosition]:
        """Get current open positions."""
        if not self.trading_client:
            return []
        
        positions = []
        
        try:
            alpaca_positions = self.trading_client.get_all_positions()
            
            for pos in alpaca_positions:
                current_price = float(pos.current_price)
                entry_price = float(pos.avg_entry_price)
                qty = int(pos.qty)
                
                # Calculate P&L
                if pos.side == "long":
                    pnl = (current_price - entry_price) * qty
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - current_price) * qty
                    pnl_pct = (entry_price - current_price) / entry_price
                
                position = AlphaPosition(
                    symbol=pos.symbol,
                    quantity=qty,
                    side="long" if pos.side == "long" else "short",
                    entry_price=entry_price,
                    entry_date=date.today(),  # Alpaca doesn't provide this directly
                    current_price=current_price,
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_pct=float(pos.unrealized_plpc),
                )
                positions.append(position)
                
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
        
        return positions
    
    def calculate_position_size(
        self,
        symbol: str,
        signal: AlphaSignal,
    ) -> int:
        """
        Calculate position size based on config and account.
        
        Uses percentage of portfolio with max position limit.
        """
        if not self.trading_client:
            return 0
        
        try:
            account = self.trading_client.get_account()
            equity = float(account.equity)
            
            # Get current price
            price = signal.entry_price or self.get_current_price(symbol)
            if not price:
                logger.warning(f"Cannot get price for {symbol}")
                return 0
            
            # Calculate max position value
            max_position_value = equity * self.config.max_position_pct
            
            # Calculate shares
            shares = int(max_position_value / price)
            
            # Ensure at least 1 share if we have buying power
            if shares == 0 and max_position_value >= price:
                shares = 1
            
            logger.info(
                f"Position size for {symbol}: {shares} shares @ ${price:.2f} = "
                f"${shares * price:,.2f} ({self.config.max_position_pct:.0%} of portfolio)"
            )
            
            return shares
            
        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            return 0
    
    def execute_signal(
        self,
        signal: AlphaSignal,
        force: bool = False,
    ) -> Optional[AlphaOrder]:
        """
        Execute a trading signal.
        
        Args:
            signal: The signal to execute
            force: Force execution even if confidence is low
            
        Returns:
            AlphaOrder if successful, None otherwise
        """
        if not self.trading_client:
            logger.error("Alpaca not connected")
            return None
        
        # Validate signal
        if signal.direction == SignalDirection.HOLD:
            logger.info(f"Signal is HOLD for {signal.symbol} - no action")
            return None
        
        if not force and signal.confidence < self.config.min_confidence:
            logger.info(
                f"Signal confidence ({signal.confidence:.0%}) below threshold "
                f"({self.config.min_confidence:.0%}) - skipping"
            )
            return None
        
        # Check PDT compliance for potential day trade
        existing_positions = self.get_positions()
        has_position = any(p.symbol == signal.symbol for p in existing_positions)
        
        if has_position and signal.direction == SignalDirection.SELL:
            # This would close a position - check if it's a day trade
            position = next(p for p in existing_positions if p.symbol == signal.symbol)
            can_close, reason = self.pdt_tracker.can_close_position(
                signal.symbol, 
                position.entry_date
            )
            if not can_close:
                logger.warning(f"PDT restriction: {reason}")
                return None
        
        # Determine order side
        if signal.direction == SignalDirection.BUY:
            side = OrderSide.BUY
        else:
            side = OrderSide.SELL
        
        # Calculate quantity
        qty = self.calculate_position_size(signal.symbol, signal)
        if qty <= 0:
            logger.warning(f"Invalid quantity for {signal.symbol}")
            return None
        
        try:
            # Use market order for simplicity
            order_request = MarketOrderRequest(
                symbol=signal.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            
            order = self.trading_client.submit_order(order_request)
            
            alpha_order = AlphaOrder(
                order_id=str(order.id),
                symbol=signal.symbol,
                side=side.value,
                quantity=qty,
                order_type="market",
                status=str(order.status),
                submitted_at=datetime.now(timezone.utc),
            )
            
            logger.info(
                f"Order submitted: {side.value} {qty} {signal.symbol} | "
                f"Order ID: {order.id}"
            )
            
            return alpha_order
            
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            return None
    
    def execute_signals(
        self,
        signals: List[AlphaSignal],
        max_orders: int = 3,
    ) -> List[AlphaOrder]:
        """
        Execute multiple signals with position limits.
        
        Args:
            signals: List of signals to execute
            max_orders: Maximum number of orders to place
            
        Returns:
            List of executed orders
        """
        orders = []
        
        # Get current positions
        current_positions = self.get_positions()
        position_symbols = {p.symbol for p in current_positions}
        
        # Check position limits
        if len(current_positions) >= self.config.max_positions:
            logger.warning(
                f"Position limit reached ({len(current_positions)}/{self.config.max_positions})"
            )
            # Only allow SELL signals
            signals = [s for s in signals if s.direction == SignalDirection.SELL]
        
        # Process signals
        for signal in signals[:max_orders]:
            # Skip if already have position and signal is BUY
            if signal.symbol in position_symbols and signal.direction == SignalDirection.BUY:
                logger.info(f"Already holding {signal.symbol} - skipping BUY")
                continue
            
            order = self.execute_signal(signal)
            if order:
                orders.append(order)
        
        logger.info(f"Executed {len(orders)} of {len(signals)} signals")
        
        return orders
    
    def get_orders(self, status: str = "open") -> List[AlphaOrder]:
        """Get orders by status."""
        if not self.trading_client:
            return []
        
        try:
            query_status = {
                "open": QueryOrderStatus.OPEN,
                "closed": QueryOrderStatus.CLOSED,
                "all": QueryOrderStatus.ALL,
            }.get(status, QueryOrderStatus.OPEN)
            
            request = GetOrdersRequest(status=query_status)
            orders = self.trading_client.get_orders(request)
            
            return [
                AlphaOrder(
                    order_id=str(o.id),
                    symbol=o.symbol,
                    side=str(o.side),
                    quantity=int(o.qty),
                    order_type=str(o.type),
                    status=str(o.status),
                    submitted_at=o.submitted_at,
                    filled_at=o.filled_at,
                    filled_price=float(o.filled_avg_price) if o.filled_avg_price else None,
                    filled_qty=int(o.filled_qty) if o.filled_qty else 0,
                )
                for o in orders
            ]
            
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not self.trading_client:
            return False
        
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def close_position(self, symbol: str) -> Optional[AlphaOrder]:
        """Close a position entirely."""
        if not self.trading_client:
            return None
        
        try:
            # Check PDT
            positions = self.get_positions()
            position = next((p for p in positions if p.symbol == symbol), None)
            
            if not position:
                logger.warning(f"No position found for {symbol}")
                return None
            
            can_close, reason = self.pdt_tracker.can_close_position(
                symbol,
                position.entry_date
            )
            
            if not can_close:
                logger.warning(f"PDT restriction: {reason}")
                return None
            
            # Close position
            order = self.trading_client.close_position(symbol)
            
            alpha_order = AlphaOrder(
                order_id=str(order.id),
                symbol=symbol,
                side=str(order.side),
                quantity=int(order.qty),
                order_type=str(order.type),
                status=str(order.status),
                submitted_at=datetime.now(timezone.utc),
            )
            
            # Record as day trade if applicable
            if position.is_day_trade_eligible:
                self.pdt_tracker.record_day_trade(
                    symbol=symbol,
                    buy_time=datetime.now(timezone.utc),
                    sell_time=datetime.now(timezone.utc),
                    buy_price=position.entry_price,
                    sell_price=position.current_price or position.entry_price,
                    quantity=position.quantity,
                )
            
            logger.info(f"Position closed: {symbol}")
            return alpha_order
            
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return None
    
    def close_all_positions(self) -> List[AlphaOrder]:
        """Close all open positions."""
        if not self.trading_client:
            return []
        
        orders = []
        positions = self.get_positions()
        
        for position in positions:
            order = self.close_position(position.symbol)
            if order:
                orders.append(order)
        
        logger.info(f"Closed {len(orders)} positions")
        return orders
    
    def get_status(self) -> Dict[str, Any]:
        """Get full trading status."""
        return {
            "account": self.get_account_info(),
            "positions": [p.to_dict() for p in self.get_positions()],
            "open_orders": [o.to_dict() for o in self.get_orders("open")],
            "pdt_status": self.pdt_tracker.get_status(),
            "config": self.config.to_dict(),
        }
