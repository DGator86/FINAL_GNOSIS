"""Minimal multi-timeframe scanner stub.

This shim keeps the public launcher scripts functional while delegating
opportunity scanning to the existing engines where available.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from loguru import logger

from engines.scanner import OpportunityScanner, get_dynamic_universe


@dataclass
class ScannerResult:
    scan_time: datetime
    opportunities: List[Any]


class MultiTimeframeScanner:
    """Wrapper around :class:`engines.scanner.OpportunityScanner`."""

    def __init__(self, config: Dict[str, Any] = None, **engine_kwargs: Any) -> None:
        self.config = config or {}
        self.engine_kwargs = engine_kwargs
        try:
            if engine_kwargs:
                self.scanner = OpportunityScanner(**engine_kwargs)
            else:
                self.scanner = None
        except Exception as e:
            logger.warning(f"Could not initialize OpportunityScanner: {e}")
            self.scanner = None
        logger.info("MultiTimeframeScanner ready")

    async def scan_all(self, priority_only: bool = False) -> List[Any]:
        """
        Scan all symbols.
        
        Args:
            priority_only: If True, only scan priority symbols.
            
        Returns:
            List of scan results.
        """
        # TODO: Implement full scanning logic or integrate with OpportunityScanner
        # For now, return empty list to allow system to run without crashing
        return []

    def run(self, symbols: List[str] | None = None, top_n: int = 10) -> ScannerResult:
        if symbols is None:
            symbols = get_dynamic_universe(self.config, top_n=top_n)
        
        if self.scanner:
            result = self.scanner.scan(symbols, top_n=top_n)
            return ScannerResult(scan_time=result.scan_timestamp, opportunities=result.opportunities)
        
        return ScannerResult(scan_time=datetime.now(), opportunities=[])


__all__ = ["MultiTimeframeScanner", "ScannerResult"]
