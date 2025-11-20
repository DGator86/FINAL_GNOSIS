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

    def __init__(self, config: Dict[str, Any], **engine_kwargs: Any) -> None:
        self.config = config
        self.engine_kwargs = engine_kwargs
        self.scanner = OpportunityScanner(**engine_kwargs)
        logger.info("MultiTimeframeScanner ready")

    def run(self, symbols: List[str] | None = None, top_n: int = 10) -> ScannerResult:
        if symbols is None:
            symbols = get_dynamic_universe(self.config, top_n=top_n)
        result = self.scanner.scan(symbols, top_n=top_n)
        return ScannerResult(scan_time=result.scan_timestamp, opportunities=result.opportunities)


__all__ = ["MultiTimeframeScanner", "ScannerResult"]
