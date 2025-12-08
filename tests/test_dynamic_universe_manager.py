import asyncio
from datetime import datetime
import asyncio

import pytest

pytest.importorskip("loguru")

from gnosis.dynamic_universe_manager import DynamicUniverseManager
from engines.scanner import Opportunity, ScanResult


class StubScanner:
    def __init__(self, opportunities):
        self._opps = opportunities

    def scan(self, symbols, top_n=25):
        return ScanResult(
            scan_timestamp=datetime.now(),
            symbols_scanned=len(symbols),
            scan_duration_seconds=0.1,
            opportunities=self._opps,
        )


def opp(symbol: str, score: float) -> Opportunity:
    return Opportunity(
        rank=0,
        symbol=symbol,
        score=score,
        opportunity_type="breakout",
        direction="long",
        confidence=0.8,
        energy_asymmetry=0.6,
        movement_energy=0.5,
        liquidity_score=0.7,
        options_score=0.5,
        reasoning="test",
    )


def test_universe_retains_previous_when_empty_refresh():
    scanner = StubScanner([opp("AAPL", 0.8), opp("MSFT", 0.7)])
    manager = DynamicUniverseManager(scanner=scanner, top_n=2, min_score_threshold=0.5)

    first = asyncio.run(manager.refresh_universe(["AAPL", "MSFT"]))
    assert len(first.current) == 2

    # Now simulate empty scan result
    scanner._opps = []
    second = asyncio.run(manager.refresh_universe(["AAPL", "MSFT"]))

    assert len(second.current) == 2  # kept previous


def test_universe_filters_scores_below_threshold():
    scanner = StubScanner([opp("AAPL", 0.8), opp("LOW", 0.2)])
    manager = DynamicUniverseManager(scanner=scanner, top_n=5, min_score_threshold=0.5)

    update = asyncio.run(manager.refresh_universe(["AAPL", "LOW"]))

    assert update.current == ["AAPL"]


def test_universe_falls_back_when_all_below_threshold():
    scanner = StubScanner([opp("AAPL", 0.2), opp("MSFT", 0.1), opp("JPM", 0.05)])
    manager = DynamicUniverseManager(scanner=scanner, top_n=2, min_score_threshold=0.5)

    update = asyncio.run(manager.refresh_universe(["AAPL", "MSFT", "JPM"]))

    assert set(update.current) <= {"AAPL", "MSFT", "JPM"}
    assert len(update.current) == 2
