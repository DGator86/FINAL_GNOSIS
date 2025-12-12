"""Base engine protocol."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from schemas.core_schemas import EngineSnapshot


class Engine(Protocol):
    """Protocol for all engines."""
    
    def run(self, symbol: str, timestamp: datetime) -> EngineSnapshot:
        """
        Run the engine for a symbol at a specific timestamp.
        
        Args:
            symbol: Trading symbol
            timestamp: Evaluation timestamp
            
        Returns:
            Engine-specific snapshot
        """
        ...
