"""Dynamic Universe Manager - manages top N trading opportunities based on scanner rankings."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import List, Set, Optional
from loguru import logger

from engines.scanner import OpportunityScanner, Opportunity


@dataclass
class UniverseUpdate:
    """Represents changes to the active universe."""
    added: List[str]  # Symbols entering universe
    removed: List[str]  # Symbols leaving universe
    current: List[str]  # Current full universe
    timestamp: datetime
    opportunities: List[Opportunity]  # Full ranked list


class DynamicUniverseManager:
    """Manages a dynamic universe of top trading opportunities.
    
    Periodically rescans the market to identify top N symbols based on
    opportunity score, and notifies when symbols enter/exit the universe.
    """
    
    def __init__(
        self,
        scanner: OpportunityScanner,
        top_n: int = 25,
        refresh_interval_seconds: int = 900,  # 15 minutes default
        min_score_threshold: float = 0.5,  # Minimum opportunity score
    ):
        """Initialize universe manager.
        
        Args:
            scanner: OpportunityScanner instance for identifying opportunities
            top_n: Number of top symbols to maintain in universe
            refresh_interval_seconds: How often to rescan (default 15 min)
            min_score_threshold: Minimum score for inclusion
        """
        self.scanner = scanner
        self.top_n = top_n
        self.refresh_interval = refresh_interval_seconds
        self.min_score_threshold = min_score_threshold
        
        # Current state
        self.active_universe: List[str] = []
        self.last_scan_results: List[Opportunity] = []
        self.last_scan_time: Optional[datetime] = None
        
        logger.info(
            f"DynamicUniverseManager initialized | top_n={top_n} | "
            f"refresh={refresh_interval_seconds}s | min_score={min_score_threshold}"
        )
    
    async def refresh_universe(self, candidate_symbols: Optional[List[str]] = None) -> UniverseUpdate:
        """Scan market and update universe.
        
        Args:
            candidate_symbols: Optional list of symbols to scan. If None, will
                             use scanner's default universe.
        
        Returns:
            UniverseUpdate with symbols added/removed and current universe
        """
        logger.info(f"Scanning universe for top {self.top_n} opportunities...")
        
        # Run scanner
        scan_result = self.scanner.scan(
            symbols=candidate_symbols,
            top_n=self.top_n
        )
        
        # Filter by minimum score
        qualified_opps = [
            opp for opp in scan_result.opportunities
            if opp.score >= self.min_score_threshold
        ]
        
        # Take top N
        top_opportunities = qualified_opps[:self.top_n]
        new_universe = [opp.symbol for opp in top_opportunities]
        
        # Calculate changes
        added = list(set(new_universe) - set(self.active_universe))
        removed = list(set(self.active_universe) - set(new_universe))
        
        # Update state
        old_universe = self.active_universe.copy()
        self.active_universe = new_universe
        self.last_scan_results = top_opportunities
        self.last_scan_time = datetime.now()
        
        logger.info(
            f"Universe updated: {len(new_universe)} symbols | "
            f"+{len(added)} added | -{len(removed)} removed"
        )
        
        if added:
            logger.info(f"  Added: {', '.join(added)}")
        if removed:
            logger.info(f"  Removed: {', '.join(removed)}")
        
        return UniverseUpdate(
            added=added,
            removed=removed,
            current=new_universe,
            timestamp=datetime.now(),
            opportunities=top_opportunities
        )
    
    def get_opportunity_for_symbol(self, symbol: str) -> Optional[Opportunity]:
        """Get the Opportunity data for a symbol in the universe."""
        for opp in self.last_scan_results:
            if opp.symbol == symbol:
                return opp
        return None
    
    def is_in_universe(self, symbol: str) -> bool:
        """Check if symbol is currently in active universe."""
        return symbol in self.active_universe
    
    def get_rank(self, symbol: str) -> Optional[int]:
        """Get current rank of symbol (1-indexed), or None if not in universe."""
        try:
            return self.active_universe.index(symbol) + 1
        except ValueError:
            return None


__all__ = ["DynamicUniverseManager", "UniverseUpdate"]
