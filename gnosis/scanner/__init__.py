"""Multi-timeframe scanner with full OpportunityScanner integration.

Provides multi-timeframe opportunity scanning with:
- Dynamic universe ranking
- Full engine integration (Hedge, Liquidity, Sentiment, Elasticity)
- Async and sync scanning modes
- Priority-based filtering

Author: Super Gnosis Elite Trading System
Version: 2.0.0 - Full scanning implementation
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from engines.scanner import OpportunityScanner, Opportunity, ScanResult, get_dynamic_universe
from engines.engine_factory import EngineFactory


@dataclass
class ScannerResult:
    """Result from a multi-timeframe scan."""
    scan_time: datetime
    opportunities: List[Opportunity] = field(default_factory=list)
    symbols_scanned: int = 0
    scan_duration_seconds: float = 0.0
    priority_symbols: List[str] = field(default_factory=list)
    
    @property
    def top_opportunity(self) -> Optional[Opportunity]:
        """Get the highest-ranked opportunity."""
        return self.opportunities[0] if self.opportunities else None
    
    @property
    def long_opportunities(self) -> List[Opportunity]:
        """Get opportunities with long direction."""
        return [o for o in self.opportunities if o.direction == "long"]
    
    @property
    def short_opportunities(self) -> List[Opportunity]:
        """Get opportunities with short direction."""
        return [o for o in self.opportunities if o.direction == "short"]


class MultiTimeframeScanner:
    """
    Multi-timeframe opportunity scanner with full engine integration.
    
    Features:
    - Dynamic universe selection
    - Full DHPE engine integration
    - Async scanning support
    - Priority-based filtering
    - Configurable scan parameters
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        **engine_kwargs: Any,
    ) -> None:
        """
        Initialize MultiTimeframeScanner.
        
        Args:
            config: Scanner configuration
            **engine_kwargs: Direct engine instances (hedge_engine, liquidity_engine, etc.)
        """
        self.config = config or {}
        self.engine_kwargs = engine_kwargs
        self.scanner: Optional[OpportunityScanner] = None
        self._priority_symbols: List[str] = []
        
        # Try to initialize with provided engines
        if engine_kwargs:
            try:
                self.scanner = OpportunityScanner(**engine_kwargs)
                logger.info("MultiTimeframeScanner initialized with provided engines")
            except Exception as e:
                logger.warning(f"Could not initialize with provided engines: {e}")
        
        # If no engines provided, try to create via factory
        if self.scanner is None and self.config.get("auto_create_engines", True):
            try:
                self._create_scanner_from_factory()
            except Exception as e:
                logger.warning(f"Could not create scanner from factory: {e}")
        
        # Load priority symbols from config
        self._priority_symbols = self.config.get("priority_symbols", [
            "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META"
        ])
        
        logger.info(
            f"MultiTimeframeScanner ready | "
            f"scanner={'enabled' if self.scanner else 'disabled'} | "
            f"priority_symbols={len(self._priority_symbols)}"
        )
    
    def _create_scanner_from_factory(self) -> None:
        """Create scanner using EngineFactory."""
        factory = EngineFactory(self.config)
        self.scanner = factory.create_scanner()
        logger.info("Scanner created via EngineFactory")

    async def scan_all(
        self,
        priority_only: bool = False,
        top_n: int = 25,
    ) -> List[Opportunity]:
        """
        Async scan of all symbols in the universe.
        
        Args:
            priority_only: If True, only scan priority symbols
            top_n: Maximum opportunities to return
            
        Returns:
            List of ranked opportunities
        """
        # Get symbols to scan
        if priority_only:
            symbols = self._priority_symbols
            logger.info(f"Scanning {len(symbols)} priority symbols")
        else:
            symbols = get_dynamic_universe(self.config, top_n=top_n * 2)
            logger.info(f"Scanning {len(symbols)} symbols from dynamic universe")
        
        # Run scan (sync scanner, but wrapped for async interface)
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._run_scan(symbols, top_n),
        )
        
        return result.opportunities
    
    def scan_priority(self, top_n: int = 10) -> ScannerResult:
        """
        Scan only priority symbols for quick opportunities.
        
        Args:
            top_n: Maximum opportunities to return
            
        Returns:
            ScannerResult with opportunities
        """
        return self._run_scan(self._priority_symbols, top_n)
    
    def scan_universe(self, top_n: int = 25) -> ScannerResult:
        """
        Scan the full dynamic universe.
        
        Args:
            top_n: Maximum opportunities to return
            
        Returns:
            ScannerResult with opportunities
        """
        symbols = get_dynamic_universe(self.config, top_n=top_n * 2)
        return self._run_scan(symbols, top_n)

    def run(
        self,
        symbols: Optional[List[str]] = None,
        top_n: int = 10,
    ) -> ScannerResult:
        """
        Run a scan on specified symbols or dynamic universe.
        
        Args:
            symbols: List of symbols to scan (None for dynamic universe)
            top_n: Maximum opportunities to return
            
        Returns:
            ScannerResult with ranked opportunities
        """
        if symbols is None:
            symbols = get_dynamic_universe(self.config, top_n=top_n * 2)
        
        return self._run_scan(symbols, top_n)
    
    def _run_scan(self, symbols: List[str], top_n: int) -> ScannerResult:
        """
        Internal scan execution.
        
        Args:
            symbols: Symbols to scan
            top_n: Maximum opportunities
            
        Returns:
            ScannerResult
        """
        if not self.scanner:
            logger.warning("No scanner available - returning empty result")
            return ScannerResult(
                scan_time=datetime.now(),
                opportunities=[],
                symbols_scanned=0,
                priority_symbols=self._priority_symbols,
            )
        
        try:
            result: ScanResult = self.scanner.scan(symbols, top_n=top_n)
            
            return ScannerResult(
                scan_time=result.scan_timestamp,
                opportunities=result.opportunities,
                symbols_scanned=result.symbols_scanned,
                scan_duration_seconds=result.scan_duration_seconds,
                priority_symbols=self._priority_symbols,
            )
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return ScannerResult(
                scan_time=datetime.now(),
                opportunities=[],
                symbols_scanned=len(symbols),
                priority_symbols=self._priority_symbols,
            )
    
    def get_priority_symbols(self) -> List[str]:
        """Get current priority symbols list."""
        return self._priority_symbols.copy()
    
    def set_priority_symbols(self, symbols: List[str]) -> None:
        """Set priority symbols for focused scanning."""
        self._priority_symbols = symbols
        logger.info(f"Updated priority symbols: {symbols}")
    
    def add_priority_symbol(self, symbol: str) -> None:
        """Add a symbol to priority list."""
        if symbol not in self._priority_symbols:
            self._priority_symbols.append(symbol)
            logger.debug(f"Added {symbol} to priority symbols")
    
    def remove_priority_symbol(self, symbol: str) -> None:
        """Remove a symbol from priority list."""
        if symbol in self._priority_symbols:
            self._priority_symbols.remove(symbol)
            logger.debug(f"Removed {symbol} from priority symbols")


# Factory function for easy creation
def create_scanner(config: Optional[Dict[str, Any]] = None) -> MultiTimeframeScanner:
    """
    Create a MultiTimeframeScanner with optional configuration.
    
    Args:
        config: Scanner configuration
        
    Returns:
        Configured MultiTimeframeScanner
    """
    return MultiTimeframeScanner(config=config)


__all__ = [
    "MultiTimeframeScanner",
    "ScannerResult",
    "create_scanner",
]
