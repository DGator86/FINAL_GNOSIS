"""Dynamic Universe Manager - manages top N trading opportunities based on scanner rankings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import math

from loguru import logger

from engines.scanner import Opportunity, OpportunityScanner


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
            f"refresh={refresh_interval_seconds}s | "
            f"min_score={min_score_threshold}"
        )

    async def refresh_universe(
        self, candidate_symbols: Optional[List[str]] = None
    ) -> UniverseUpdate:
        """Scan market and update universe.

        Args:
            candidate_symbols: Optional list of symbols to scan. If None,
                             will use scanner's default universe.

        Returns:
            UniverseUpdate with symbols added/removed and current universe
        """
        logger.info(f"Scanning universe for top {self.top_n} opportunities...")

        # Run scanner - provide empty list if None to satisfy type checker
        scan_result = self.scanner.scan(
            symbols=candidate_symbols if candidate_symbols is not None else [], top_n=self.top_n
        )

        initial_scan = self.last_scan_time is None

        # Filter by minimum score and drop NaNs explicitly
        qualified_opps = [
            opp
            for opp in scan_result.opportunities
            if opp.score is not None
            and not math.isnan(opp.score)
            and opp.score >= self.min_score_threshold
        ]

        # Take top N
        top_opportunities = qualified_opps[: self.top_n]
        new_universe = [opp.symbol for opp in top_opportunities]

        # If nothing cleared the threshold but we have scores, fall back to best available
        if not top_opportunities and scan_result.opportunities:
            logger.warning(
                "All symbols scored below min_score={threshold:.2f}; using best available {count} opportunities",
                threshold=self.min_score_threshold,
                count=min(len(scan_result.opportunities), self.top_n),
            )
            top_opportunities = scan_result.opportunities[: self.top_n]
            new_universe = [opp.symbol for opp in top_opportunities]

        # Calculate changes
        added = list(set(new_universe) - set(self.active_universe))
        removed = list(set(self.active_universe) - set(new_universe))

        # Update state, but avoid dropping to zero on transient data gaps
        if not top_opportunities and self.active_universe:
            logger.warning(
                "Universe scan returned no qualifying symbols (scanned={scanned}, above_threshold={qualified}). "
                "Keeping existing universe of {current_count} symbols.",
                scanned=scan_result.symbols_scanned,
                qualified=len(qualified_opps),
                current_count=len(self.active_universe),
            )
            return UniverseUpdate(
                added=[],
                removed=[],
                current=self.active_universe,
                timestamp=datetime.now(),
                opportunities=self.last_scan_results,
            )

        if not top_opportunities and not self.active_universe:
            if not scan_result.opportunities:
                logger.warning(
                    "Universe scan returned no opportunities (scanned={scanned}). "
                    "Initial universe is empty – no trades will be placed until data becomes available. "
                    "Check market data credentials and scanners (ALPACA_API_KEY/SECRET, ALPACA_DATA_FEED, Unusual Whales token).",
                    scanned=scan_result.symbols_scanned,
                )
            else:
                logger.warning(
                    "Universe scan returned no qualifying symbols and no active universe exists (scanned={scanned}, above_threshold={qualified}).",
                    scanned=scan_result.symbols_scanned,
                    qualified=len(qualified_opps),
                )
            if initial_scan:
                logger.info(
                    "Initial universe size is 0. System will continue scanning but trading will remain idle until opportunities qualify."
                )

        # Update state
        self.active_universe = new_universe
        self.last_scan_results = top_opportunities
        self.last_scan_time = datetime.now()

        logger.info(
            f"Universe updated: {len(new_universe)} symbols | "
            f"+{len(added)} added | -{len(removed)} removed "
            f"(scanned={scan_result.symbols_scanned}, above_threshold={len(qualified_opps)})"
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
            opportunities=top_opportunities,
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
