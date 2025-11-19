"""Opportunity scanner for identifying trading opportunities across multiple symbols."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List

from loguru import logger
from pydantic import BaseModel


# Legacy default universe - kept for backwards compatibility
# BUT: New behavior uses dynamic_universe.py to get current top N
DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",  # Indices
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",  # Tech
    "JPM", "BAC", "GS", "MS",  # Finance
    "XLE", "XLF", "XLK", "XLV",  # Sectors
]


def get_dynamic_universe(config: Dict[str, Any], top_n: int = 25) -> List[str]:
    """
    Get current top N options underlyings using dynamic ranking.
    
    Args:
        config: Scanner configuration
        top_n: Number of symbols to return
        
    Returns:
        List of top N symbols
    """
    from engines.dynamic_universe import DynamicUniverseRanker
    
    ranker = DynamicUniverseRanker(config)
    return ranker.get_top_n(top_n)


class Opportunity(BaseModel):
    """Single trading opportunity."""
    
    rank: int
    symbol: str
    score: float
    opportunity_type: str
    direction: str
    confidence: float
    energy_asymmetry: float
    movement_energy: float
    liquidity_score: float
    options_score: float = 0.5
    reasoning: str


class ScanResult(BaseModel):
    """Result from an opportunity scan."""
    
    scan_timestamp: datetime
    symbols_scanned: int
    scan_duration_seconds: float
    opportunities: List[Opportunity]


class OpportunityScanner:
    """Scans multiple symbols to identify trading opportunities."""
    
    def __init__(
        self,
        hedge_engine: Any,
        liquidity_engine: Any,
        sentiment_engine: Any,
        elasticity_engine: Any,
        options_adapter: Any,
        market_adapter: Any,
    ):
        """
        Initialize scanner.
        
        Args:
            hedge_engine: Hedge engine instance
            liquidity_engine: Liquidity engine instance
            sentiment_engine: Sentiment engine instance
            elasticity_engine: Elasticity engine instance
            options_adapter: Options data adapter
            market_adapter: Market data adapter
        """
        self.hedge_engine = hedge_engine
        self.liquidity_engine = liquidity_engine
        self.sentiment_engine = sentiment_engine
        self.elasticity_engine = elasticity_engine
        self.options_adapter = options_adapter
        self.market_adapter = market_adapter
        logger.info("OpportunityScanner initialized")
    
    def scan(self, symbols: List[str], top_n: int = 25) -> ScanResult:
        """
        Scan symbols for opportunities.
        
        Args:
            symbols: List of symbols to scan
            top_n: Number of top opportunities to return
            
        Returns:
            ScanResult with ranked opportunities
        """
        start_time = time.time()
        timestamp = datetime.now()
        
        logger.info(f"Scanning {len(symbols)} symbols for opportunities...")
        
        opportunities = []
        
        for symbol in symbols:
            try:
                opp = self._evaluate_symbol(symbol, timestamp)
                if opp:
                    opportunities.append(opp)
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        # Sort by score (descending)
        opportunities.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for i, opp in enumerate(opportunities[:top_n], 1):
            opp.rank = i
        
        duration = time.time() - start_time
        
        return ScanResult(
            scan_timestamp=timestamp,
            symbols_scanned=len(symbols),
            scan_duration_seconds=duration,
            opportunities=opportunities[:top_n],
        )
    
    def _evaluate_symbol(self, symbol: str, timestamp: datetime) -> Opportunity:
        """Evaluate a single symbol for opportunity quality."""
        # Run engines
        hedge_snap = self.hedge_engine.run(symbol, timestamp)
        liquidity_snap = self.liquidity_engine.run(symbol, timestamp)
        sentiment_snap = self.sentiment_engine.run(symbol, timestamp)
        elasticity_snap = self.elasticity_engine.run(symbol, timestamp)
        
        # Calculate opportunity score (0-1)
        # Factors: energy asymmetry, liquidity, sentiment alignment, volatility
        energy_score = min(1.0, abs(hedge_snap.energy_asymmetry))
        liquidity_score = liquidity_snap.liquidity_score
        sentiment_score = abs(sentiment_snap.sentiment_score)
        volatility_score = min(1.0, elasticity_snap.volatility / 0.5)  # Normalize to 50%
        
        # Weighted average
        score = (
            energy_score * 0.35 +
            liquidity_score * 0.25 +
            sentiment_score * 0.25 +
            volatility_score * 0.15
        )
        
        # Determine opportunity type
        if volatility_score > 0.6 and energy_score > 0.5:
            opp_type = "breakout"
        elif energy_score > 0.7:
            opp_type = "directional"
        elif liquidity_score > 0.7:
            opp_type = "scalp"
        else:
            opp_type = "swing"
        
        # Determine direction
        if hedge_snap.energy_asymmetry > 0.2:
            direction = "long"
        elif hedge_snap.energy_asymmetry < -0.2:
            direction = "short"
        else:
            direction = "neutral"
        
        # Confidence from hedge engine
        confidence = hedge_snap.confidence
        
        # Reasoning
        reasoning = (
            f"{opp_type.capitalize()} opportunity with "
            f"{abs(hedge_snap.energy_asymmetry):.2f} energy asymmetry, "
            f"{liquidity_snap.liquidity_score:.2f} liquidity, "
            f"{elasticity_snap.volatility:.1%} volatility"
        )
        
        return Opportunity(
            rank=0,  # Will be assigned after sorting
            symbol=symbol,
            score=score,
            opportunity_type=opp_type,
            direction=direction,
            confidence=confidence,
            energy_asymmetry=hedge_snap.energy_asymmetry,
            movement_energy=hedge_snap.movement_energy,
            liquidity_score=liquidity_snap.liquidity_score,
            options_score=0.5,  # Placeholder
            reasoning=reasoning,
        )
