"""Sentiment Engine v1 - Multi-source sentiment analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from loguru import logger

from schemas.core_schemas import SentimentSnapshot


class SentimentEngineV1:
    """
    Sentiment Engine v1 for multi-source sentiment analysis.
    
    Combines news, flow, and technical sentiment.
    """
    
    def __init__(self, processors: List[Any], config: Dict[str, Any]):
        """
        Initialize Sentiment Engine.
        
        Args:
            processors: List of sentiment processors
            config: Engine configuration
        """
        self.processors = processors
        self.config = config
        logger.info("SentimentEngineV1 initialized")
    
    def run(self, symbol: str, timestamp: datetime) -> SentimentSnapshot:
        """
        Run sentiment analysis for a symbol.
        
        Args:
            symbol: Trading symbol
            timestamp: Analysis timestamp
            
        Returns:
            SentimentSnapshot with combined sentiment scores
        """
        logger.debug(f"Running SentimentEngineV1 for {symbol} at {timestamp}")
        
        news_weight = self.config.get("news_weight", 0.4)
        flow_weight = self.config.get("flow_weight", 0.3)
        technical_weight = self.config.get("technical_weight", 0.3)
        
        news_sentiment = 0.0
        flow_sentiment = 0.0
        technical_sentiment = 0.0
        
        for processor in self.processors:
            processor_name = processor.__class__.__name__
            try:
                sentiment = processor.process(symbol, timestamp)
                
                if "News" in processor_name:
                    news_sentiment = sentiment
                elif "Flow" in processor_name:
                    flow_sentiment = sentiment
                elif "Technical" in processor_name:
                    technical_sentiment = sentiment
            except Exception as e:
                logger.error(f"Error in {processor_name}: {e}")
        
        # Calculate weighted sentiment
        sentiment_score = (
            news_sentiment * news_weight +
            flow_sentiment * flow_weight +
            technical_sentiment * technical_weight
        )
        
        # Confidence based on agreement between sources
        sentiments = [news_sentiment, flow_sentiment, technical_sentiment]
        avg_abs_diff = sum(abs(s - sentiment_score) for s in sentiments) / len(sentiments)
        confidence = max(0.1, 1.0 - avg_abs_diff)
        
        return SentimentSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            sentiment_score=sentiment_score,
            news_sentiment=news_sentiment,
            flow_sentiment=flow_sentiment,
            technical_sentiment=technical_sentiment,
            confidence=confidence,
        )
