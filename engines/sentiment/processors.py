"""Sentiment processors for news, flow, and technical analysis."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

from loguru import logger

from engines.inputs.market_data_adapter import MarketDataAdapter
from engines.inputs.news_adapter import NewsAdapter


class NewsSentimentProcessor:
    """Processes news sentiment."""
    
    def __init__(self, news_adapter: NewsAdapter, config: Dict[str, Any]):
        self.news_adapter = news_adapter
        self.config = config
    
    def process(self, symbol: str, timestamp: datetime) -> float:
        """Calculate news sentiment score."""
        try:
            articles = self.news_adapter.get_news(
                symbol,
                timestamp - timedelta(days=7),
                timestamp
            )
            
            if not articles:
                return 0.0
            
            # Weight recent news more heavily
            total_weight = 0.0
            weighted_sentiment = 0.0
            
            for article in articles:
                age_days = (timestamp - article.timestamp).days
                weight = max(0.1, 1.0 - (age_days / 7.0))
                weighted_sentiment += article.sentiment * weight
                total_weight += weight
            
            return weighted_sentiment / total_weight if total_weight > 0 else 0.0
        
        except Exception as e:
            logger.error(f"Error in NewsSentimentProcessor: {e}")
            return 0.0


class FlowSentimentProcessor:
    """Processes order flow sentiment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def process(self, symbol: str, timestamp: datetime) -> float:
        """Calculate flow sentiment score."""
        # Stub implementation - would need tick data
        return 0.0


class TechnicalSentimentProcessor:
    """Processes technical analysis sentiment."""
    
    def __init__(self, market_adapter: MarketDataAdapter, config: Dict[str, Any]):
        self.market_adapter = market_adapter
        self.config = config
    
    def process(self, symbol: str, timestamp: datetime) -> float:
        """Calculate technical sentiment score."""
        try:
            bars = self.market_adapter.get_bars(
                symbol,
                timestamp - timedelta(days=20),
                timestamp,
                timeframe="1Day"
            )
            
            if len(bars) < 2:
                return 0.0
            
            # Simple momentum: compare recent price to 20-day average
            recent_price = bars[-1].close
            avg_price = sum(bar.close for bar in bars) / len(bars)
            
            momentum = (recent_price - avg_price) / avg_price if avg_price > 0 else 0.0
            
            # Normalize to [-1, 1]
            return max(-1.0, min(1.0, momentum * 10))
        
        except Exception as e:
            logger.error(f"Error in TechnicalSentimentProcessor: {e}")
            return 0.0
