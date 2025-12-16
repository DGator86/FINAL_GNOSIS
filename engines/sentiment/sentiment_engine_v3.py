"""Sentiment Engine v3 - Enhanced sentiment analysis with Unusual Whales integration."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from adapters.unusual_whales_adapter import UnusualWhalesAdapter
from schemas.core_schemas import SentimentSnapshot


class SentimentEngineV3:
    """
    Sentiment Engine v3 for multi-source sentiment analysis.

    New V3 Features:
    - Unusual Whales flow sentiment integration
    - Retail sentiment aggregation
    - Enhanced confidence scoring
    """

    def __init__(
        self,
        processors: List[Any],
        unusual_whales_adapter: Optional[UnusualWhalesAdapter],
        config: Dict[str, Any],
    ):
        """
        Initialize Sentiment Engine V3.

        Args:
            processors: List of sentiment processors
            unusual_whales_adapter: Unusual Whales data provider
            config: Engine configuration
        """
        self.processors = processors
        self.unusual_whales_adapter = unusual_whales_adapter
        self.config = config
        logger.info("SentimentEngineV3 initialized")

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

        news_weight = self.config.get("news_weight", 0.3)
        flow_weight = self.config.get("flow_weight", 0.4)  # V3: Increased flow weight
        technical_weight = self.config.get("technical_weight", 0.3)

        news_sentiment = 0.0
        flow_sentiment = 0.0
        technical_sentiment = 0.0

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

        # Calculate weighted sentiment
        sentiment_score = (
            news_sentiment * news_weight
            + flow_sentiment * flow_weight
            + technical_sentiment * technical_weight
        )

        # V3: Enhanced confidence based on agreement and unusual flow conviction
        sentiments = [news_sentiment, flow_sentiment, technical_sentiment]
        avg_abs_diff = sum(abs(s - sentiment_score) for s in sentiments) / len(sentiments)
        base_confidence = max(0.1, 1.0 - avg_abs_diff)

        # Boost confidence if unusual flow confirms
        if unusual_flow_sentiment is not None and abs(unusual_flow_sentiment) > 0.5:
            if (unusual_flow_sentiment > 0 and sentiment_score > 0) or (
                unusual_flow_sentiment < 0 and sentiment_score < 0
            ):
                base_confidence = min(1.0, base_confidence * 1.2)

        return SentimentSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            sentiment_score=sentiment_score,
            news_sentiment=news_sentiment,
            flow_sentiment=flow_sentiment,
            technical_sentiment=technical_sentiment,
            confidence=base_confidence,
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

    def _calculate_retail_sentiment(self, symbol: str) -> float:
        """
        Calculate retail sentiment from social media.

        Placeholder for future implementation.

        Returns:
            Retail sentiment score from -1 to 1
        """
        # TODO: Integrate with social media APIs (Twitter, Reddit, etc.)
        return 0.0
