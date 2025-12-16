"""Elasticity Engine v1 - Volatility and regime analysis."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

from loguru import logger

from adapters.market_data_adapter import MarketDataAdapter
from schemas.core_schemas import ElasticitySnapshot


class ElasticityEngineV1:
    """
    Elasticity Engine v1 for volatility and regime analysis.
    
    Analyzes market volatility and trend characteristics.
    """
    
    def __init__(self, market_adapter: MarketDataAdapter, config: Dict[str, Any]):
        """
        Initialize Elasticity Engine.
        
        Args:
            market_adapter: Market data provider
            config: Engine configuration
        """
        self.market_adapter = market_adapter
        self.config = config
        logger.info("ElasticityEngineV1 initialized")
    
    def run(self, symbol: str, timestamp: datetime) -> ElasticitySnapshot:
        """
        Run elasticity analysis for a symbol.
        
        Args:
            symbol: Trading symbol
            timestamp: Analysis timestamp
            
        Returns:
            ElasticitySnapshot with volatility metrics
        """
        logger.debug(f"Running ElasticityEngineV1 for {symbol} at {timestamp}")
        
        try:
            window = self.config.get("volatility_window", 20)
            bars = self.market_adapter.get_bars(
                symbol,
                timestamp - timedelta(days=window),
                timestamp,
                timeframe="1Day"
            )
            
            if len(bars) < 2:
                return ElasticitySnapshot(
                    timestamp=timestamp,
                    symbol=symbol,
                )
            
            # Calculate returns
            returns = []
            for i in range(1, len(bars)):
                if bars[i-1].close > 0:
                    ret = (bars[i].close - bars[i-1].close) / bars[i-1].close
                    returns.append(ret)
            
            if not returns:
                return ElasticitySnapshot(
                    timestamp=timestamp,
                    symbol=symbol,
                )
            
            # Calculate volatility (annualized standard deviation)
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = (variance ** 0.5) * (252 ** 0.5)  # Annualized
            
            # Classify volatility regime
            if volatility < 0.15:
                regime = "low"
            elif volatility > 0.30:
                regime = "high"
            else:
                regime = "moderate"
            
            # Calculate trend strength
            prices = [bar.close for bar in bars]
            price_range = max(prices) - min(prices)
            recent_move = abs(prices[-1] - prices[0])
            trend_strength = recent_move / price_range if price_range > 0 else 0.0
            
            return ElasticitySnapshot(
                timestamp=timestamp,
                symbol=symbol,
                volatility=volatility,
                volatility_regime=regime,
                trend_strength=trend_strength,
            )
        
        except Exception as e:
            logger.error(f"Error in ElasticityEngineV1 for {symbol}: {e}")
            return ElasticitySnapshot(
                timestamp=timestamp,
                symbol=symbol,
            )
