"""Market data adapter protocol."""

from __future__ import annotations

from datetime import datetime
from typing import List, Protocol

from pydantic import BaseModel


class OHLCV(BaseModel):
    """OHLCV bar data."""
    
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class Quote(BaseModel):
    """Real-time quote."""
    
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last: float
    last_size: float


class MarketDataAdapter(Protocol):
    """Protocol for market data providers."""
    
    def get_bars(
        self, 
        symbol: str, 
        start: datetime, 
        end: datetime,
        timeframe: str = "1Day"
    ) -> List[OHLCV]:
        """
        Get historical OHLCV bars.
        
        Args:
            symbol: Trading symbol
            start: Start timestamp
            end: End timestamp
            timeframe: Bar timeframe (e.g., "1Min", "1Hour", "1Day")
            
        Returns:
            List of OHLCV bars
        """
        ...
    
    def get_quote(self, symbol: str) -> Quote:
        """
        Get current quote.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current quote
        """
        ...
