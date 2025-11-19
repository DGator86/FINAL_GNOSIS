"""Options chain data adapter protocol."""

from __future__ import annotations

from datetime import datetime
from typing import List, Protocol

from pydantic import BaseModel


class OptionContract(BaseModel):
    """Single option contract."""
    
    symbol: str
    strike: float
    expiration: datetime
    option_type: str  # "call" or "put"
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: float = 0.0
    open_interest: float = 0.0
    implied_volatility: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


class OptionsChainAdapter(Protocol):
    """Protocol for options chain data providers."""
    
    def get_chain(self, symbol: str, timestamp: datetime) -> List[OptionContract]:
        """
        Get options chain for a symbol.
        
        Args:
            symbol: Underlying symbol
            timestamp: Data timestamp
            
        Returns:
            List of option contracts
        """
        ...
