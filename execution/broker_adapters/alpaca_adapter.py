"""Alpaca broker adapter - stub implementation."""

from __future__ import annotations

import os
from typing import Any, List, Optional

from loguru import logger
from pydantic import BaseModel


class Account(BaseModel):
    """Account information."""
    
    account_id: str = "paper"
    cash: float = 100000.0
    buying_power: float = 100000.0
    portfolio_value: float = 100000.0


class Position(BaseModel):
    """Position information."""
    
    symbol: str
    quantity: float
    avg_entry_price: float
    unrealized_pnl: float


class AlpacaBrokerAdapter:
    """Alpaca broker adapter for paper/live trading."""
    
    def __init__(self, paper: bool = True):
        """
        Initialize Alpaca adapter.
        
        Args:
            paper: Whether to use paper trading (default: True)
        """
        self.paper = paper
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca credentials not found in environment")
        
        logger.info(f"AlpacaBrokerAdapter initialized (paper={paper})")
    
    def get_account(self) -> Account:
        """Get account information."""
        # Stub implementation
        return Account()
    
    def get_positions(self) -> List[Position]:
        """Get current positions."""
        # Stub implementation
        return []
    
    def place_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str = "market",
    ) -> Optional[str]:
        """
        Place an order.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: "buy" or "sell"
            order_type: Order type (default: "market")
            
        Returns:
            Order ID if successful
        """
        # Stub implementation
        logger.info(f"Placing {side} order: {quantity} {symbol} ({order_type})")
        return "stub_order_id"
