"""Sentiment processors for news, flow, and technical analysis."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

from loguru import logger

from adapters.market_data_adapter import MarketDataAdapter
from adapters.news_adapter import NewsAdapter


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
    """Processes order flow sentiment using Unusual Whales when available."""

    def __init__(self, config: Dict[str, Any], flow_adapter: Any | None = None):
        self.config = config
        self.flow_adapter = flow_adapter

    def process(self, symbol: str, timestamp: datetime) -> float:
        """Calculate flow sentiment score from options flow intensity and skew."""

        if not self.flow_adapter:
            return 0.0

        try:
            snapshot = self.flow_adapter.get_flow_snapshot(symbol, timestamp)
            if not snapshot:
                return 0.0

            call_pressure = snapshot.get("call_volume", 0) + snapshot.get("call_premium", 0)
            put_pressure = snapshot.get("put_volume", 0) + snapshot.get("put_premium", 0)
            total = call_pressure + put_pressure
            if total == 0:
                return 0.0

            skew = (call_pressure - put_pressure) / total
            intensity = snapshot.get("sweep_ratio", 0)
            score = (skew * 0.7) + (intensity * 0.3)
            return max(-1.0, min(1.0, score))

        except Exception as exc:
            logger.error(f"Error in FlowSentimentProcessor: {exc}")
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
