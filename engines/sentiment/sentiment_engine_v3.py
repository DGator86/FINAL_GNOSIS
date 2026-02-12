"""Sentiment Engine v3 - Enhanced sentiment analysis with Unusual Whales and Social Media.

New V3.1 Features:
- Unusual Whales flow sentiment integration
- Social Media sentiment (Twitter/X, Reddit)
- Retail sentiment aggregation from r/wallstreetbets, r/stocks
- Enhanced confidence scoring with multi-source agreement

Author: Super Gnosis Elite Trading System
Version: 3.1.0
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from engines.inputs.unusual_whales_adapter import UnusualWhalesAdapter
from engines.sentiment.social_media_adapter import (
    SocialMediaSentimentAggregator,
    create_social_media_aggregator,
)
from schemas.core_schemas import SentimentSnapshot


class SentimentEngineV3:
    """
    Sentiment Engine v3 for multi-source sentiment analysis.

    V3.1 Features:
    - Unusual Whales flow sentiment integration
    - Social Media sentiment (Twitter/X, Reddit)
    - Retail sentiment aggregation
    - Enhanced confidence scoring
    """

    def __init__(
        self,
        processors: List[Any],
        unusual_whales_adapter: Optional[UnusualWhalesAdapter] = None,
        config: Optional[Dict[str, Any]] = None,
        social_media_aggregator: Optional[SocialMediaSentimentAggregator] = None,
    ):
        """
        Initialize Sentiment Engine V3.

        Args:
            processors: List of sentiment processors
            unusual_whales_adapter: Unusual Whales data provider
            config: Engine configuration
            social_media_aggregator: Social media sentiment aggregator (Twitter/Reddit)
        """
        self.processors = processors
        self.unusual_whales_adapter = unusual_whales_adapter
        self.config = config or {}
        
        # V3.1: Social media integration
        self.social_media_aggregator = social_media_aggregator
        if not self.social_media_aggregator and self.config.get("enable_social_media", True):
            # Create default aggregator (simulation mode if no API keys)
            self.social_media_aggregator = create_social_media_aggregator(
                twitter_bearer_token=self.config.get("twitter_bearer_token"),
                reddit_client_id=self.config.get("reddit_client_id"),
                reddit_client_secret=self.config.get("reddit_client_secret"),
                config=self.config.get("social_media_config"),
            )
        
        logger.info(
            f"SentimentEngineV3 initialized | "
            f"processors={len(processors)} | "
            f"unusual_whales={'enabled' if unusual_whales_adapter else 'disabled'} | "
            f"social_media={'enabled' if self.social_media_aggregator else 'disabled'}"
        )

    def run(self, symbol: str, timestamp: datetime) -> SentimentSnapshot:
        """
        Run sentiment analysis for a symbol.

        Args:
            symbol: Trading symbol
            timestamp: Analysis timestamp

        Returns:
            SentimentSnapshot with combined sentiment scores
        """
        logger.debug(f"Running SentimentEngineV3 for {symbol} at {timestamp}")

        # Configurable weights (V3.1: Added social media weight)
        news_weight = self.config.get("news_weight", 0.25)
        flow_weight = self.config.get("flow_weight", 0.35)  # Options flow
        technical_weight = self.config.get("technical_weight", 0.25)
        social_weight = self.config.get("social_weight", 0.15)  # V3.1: Social media

        news_sentiment = 0.0
        flow_sentiment = 0.0
        technical_sentiment = 0.0
        social_sentiment = 0.0

        # Process standard sentiment sources
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

        # V3: Analyze Unusual Whales flow
        unusual_flow_sentiment = self._analyze_unusual_flow(symbol)

        # V3: Blend standard flow with unusual flow
        if unusual_flow_sentiment is not None:
            flow_sentiment = flow_sentiment * 0.5 + unusual_flow_sentiment * 0.5

        # V3.1: Analyze social media sentiment
        social_sentiment, social_confidence = self._analyze_social_media(symbol)
        
        # Calculate weighted sentiment (V3.1: Includes social media)
        if self.social_media_aggregator and social_confidence > 0.3:
            # Include social media if confidence is reasonable
            sentiment_score = (
                news_sentiment * news_weight
                + flow_sentiment * flow_weight
                + technical_sentiment * technical_weight
                + social_sentiment * social_weight
            )
            # Normalize weights
            total_weight = news_weight + flow_weight + technical_weight + social_weight
            sentiment_score /= total_weight
        else:
            # Fallback to standard weights without social
            sentiment_score = (
                news_sentiment * news_weight
                + flow_sentiment * flow_weight
                + technical_sentiment * technical_weight
            )
            # Normalize to account for missing social weight
            total_weight = news_weight + flow_weight + technical_weight
            sentiment_score /= total_weight

        # V3.1: Enhanced confidence based on agreement across all sources
        sentiments = [news_sentiment, flow_sentiment, technical_sentiment]
        if self.social_media_aggregator and social_confidence > 0.3:
            sentiments.append(social_sentiment)
        
        avg_abs_diff = sum(abs(s - sentiment_score) for s in sentiments) / len(sentiments)
        base_confidence = max(0.1, 1.0 - avg_abs_diff)

        # Boost confidence if unusual flow confirms
        if unusual_flow_sentiment is not None and abs(unusual_flow_sentiment) > 0.5:
            if (unusual_flow_sentiment > 0 and sentiment_score > 0) or (
                unusual_flow_sentiment < 0 and sentiment_score < 0
            ):
                base_confidence = min(1.0, base_confidence * 1.15)
        
        # V3.1: Boost confidence if social media confirms
        if social_confidence > 0.5 and abs(social_sentiment) > 0.3:
            if (social_sentiment > 0 and sentiment_score > 0) or (
                social_sentiment < 0 and sentiment_score < 0
            ):
                base_confidence = min(1.0, base_confidence * 1.1)

        return SentimentSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            sentiment_score=sentiment_score,
            news_sentiment=news_sentiment,
            flow_sentiment=flow_sentiment,
            technical_sentiment=technical_sentiment,
            confidence=base_confidence,
            # V3.1: Extended data in metadata
            metadata={
                "social_sentiment": social_sentiment,
                "social_confidence": social_confidence,
                "unusual_flow_sentiment": unusual_flow_sentiment,
            },
        )

    def _analyze_unusual_flow(self, symbol: str) -> Optional[float]:
        """
        Analyze Unusual Whales flow data.

        Returns:
            Sentiment score from -1 to 1, or None if no data
        """
        if not self.unusual_whales_adapter:
            return None

        try:
            # Get unusual activity
            activity = self.unusual_whales_adapter.get_unusual_activity(symbol)

            if not activity:
                return None

            # Analyze flow direction
            bullish_flow = 0
            bearish_flow = 0

            for trade in activity:
                # Assume activity has 'sentiment' or 'type' field
                trade_type = trade.get("type", "").lower()
                premium = trade.get("premium", 0)

                if "call" in trade_type or "bullish" in trade_type:
                    bullish_flow += premium
                elif "put" in trade_type or "bearish" in trade_type:
                    bearish_flow += premium

            total_flow = bullish_flow + bearish_flow

            if total_flow == 0:
                return 0.0

            # Calculate sentiment (-1 to 1)
            flow_sentiment = (bullish_flow - bearish_flow) / total_flow

            logger.debug(
                f"Unusual flow for {symbol}: "
                f"bullish=${bullish_flow:.0f}, bearish=${bearish_flow:.0f}, "
                f"sentiment={flow_sentiment:.2f}"
            )

            return flow_sentiment

        except Exception as e:
            logger.debug(f"Error analyzing unusual flow for {symbol}: {e}")
            return None

    def _analyze_social_media(self, symbol: str) -> tuple[float, float]:
        """
        Analyze social media sentiment from Twitter and Reddit.
        
        V3.1: Full implementation using SocialMediaSentimentAggregator.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (sentiment_score, confidence) where:
            - sentiment_score: -1.0 to 1.0
            - confidence: 0.0 to 1.0
        """
        if not self.social_media_aggregator:
            return 0.0, 0.0
        
        try:
            # Get aggregated social sentiment
            lookback_hours = self.config.get("social_lookback_hours", 24)
            result = self.social_media_aggregator.get_sentiment(symbol, lookback_hours)
            
            logger.debug(
                f"Social media sentiment for {symbol}: "
                f"overall={result.overall_sentiment:.2f}, "
                f"twitter={result.twitter_sentiment}, reddit={result.reddit_sentiment}, "
                f"posts={result.post_count}, confidence={result.confidence:.2f}, "
                f"trending={result.trending}"
            )
            
            return result.overall_sentiment, result.confidence
            
        except Exception as e:
            logger.warning(f"Error analyzing social media for {symbol}: {e}")
            return 0.0, 0.0
    
    def _calculate_retail_sentiment(self, symbol: str) -> float:
        """
        Calculate retail sentiment from social media.
        
        Legacy method - delegates to _analyze_social_media for backward compatibility.

        Returns:
            Retail sentiment score from -1 to 1
        """
        sentiment, _ = self._analyze_social_media(symbol)
        return sentiment
    
    def get_social_media_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed social media sentiment breakdown.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with detailed social media analysis or None if unavailable
        """
        if not self.social_media_aggregator:
            return None
        
        try:
            lookback_hours = self.config.get("social_lookback_hours", 24)
            result = self.social_media_aggregator.get_sentiment(symbol, lookback_hours)
            
            return {
                "symbol": result.symbol,
                "timestamp": result.timestamp.isoformat(),
                "overall_sentiment": result.overall_sentiment,
                "twitter_sentiment": result.twitter_sentiment,
                "reddit_sentiment": result.reddit_sentiment,
                "post_count": result.post_count,
                "bullish_count": result.bullish_count,
                "bearish_count": result.bearish_count,
                "neutral_count": result.neutral_count,
                "engagement_weighted_sentiment": result.engagement_weighted_sentiment,
                "confidence": result.confidence,
                "trending": result.trending,
                "sample_posts": [
                    {
                        "platform": p.platform,
                        "content": p.content[:200],
                        "sentiment": p.sentiment,
                        "engagement": p.engagement,
                    }
                    for p in result.posts[:10]
                ],
            }
        except Exception as e:
            logger.error(f"Error getting social media details for {symbol}: {e}")
            return None
