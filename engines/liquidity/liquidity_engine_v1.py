"""Liquidity Engine v1 - Market liquidity analysis."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

from loguru import logger

from engines.inputs.market_data_adapter import MarketDataAdapter
from schemas.core_schemas import LiquiditySnapshot


class LiquidityEngineV1:
    """
    Liquidity Engine v1 for market liquidity analysis.
    
    Analyzes bid-ask spreads, volume, depth, and impact costs.
    """
    
    def __init__(self, market_adapter: MarketDataAdapter, config: Dict[str, Any]):
        """
        Initialize Liquidity Engine.
        
        Args:
            market_adapter: Market data provider
            config: Engine configuration
        """
        self.market_adapter = market_adapter
        self.config = config
        logger.info("LiquidityEngineV1 initialized")
    
    def run(self, symbol: str, timestamp: datetime) -> LiquiditySnapshot:
        """
        Run liquidity analysis for a symbol.
        
        Args:
            symbol: Trading symbol
            timestamp: Analysis timestamp
            
        Returns:
            LiquiditySnapshot with liquidity metrics
        """
        logger.debug(f"Running LiquidityEngineV1 for {symbol} at {timestamp}")
        
        try:
            # Get current quote
            quote = self.market_adapter.get_quote(symbol)
            
            # Calculate bid-ask spread
            mid_price = (quote.bid + quote.ask) / 2
            spread_pct = ((quote.ask - quote.bid) / mid_price) * 100 if mid_price > 0 else 0.0
            
            # Get recent volume data
            bars = self.market_adapter.get_bars(
                symbol,
                timestamp - timedelta(days=5),
                timestamp,
                timeframe="1Day"
            )
            
            avg_volume = sum(bar.volume for bar in bars) / len(bars) if bars else 0.0
            
            # Calculate depth (sum of bid and ask sizes)
            depth = quote.bid_size + quote.ask_size
            
            # Calculate impact cost (simplified)
            impact_cost = spread_pct * 0.5  # Assume crossing half the spread
            
            # Calculate liquidity score (0-1 scale)
            # Higher volume and tighter spreads = higher score
            volume_score = min(1.0, avg_volume / 10_000_000)  # Normalize to 10M
            spread_score = max(0.0, 1.0 - (spread_pct / 1.0))  # 1% spread = 0 score
            liquidity_score = (volume_score * 0.7 + spread_score * 0.3)
            
            return LiquiditySnapshot(
                timestamp=timestamp,
                symbol=symbol,
                liquidity_score=liquidity_score,
                bid_ask_spread=spread_pct,
                volume=avg_volume,
                depth=depth,
                impact_cost=impact_cost,
            )
        
        except Exception as e:
            logger.error(f"Error in LiquidityEngineV1 for {symbol}: {e}")
            return LiquiditySnapshot(
                timestamp=timestamp,
                symbol=symbol,
            )
