"""Multi-Timeframe Sentiment Agent v3 - Enhanced with flow conviction scoring."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from agents.confidence_builder import TimeframeSignal
from schemas.core_schemas import AgentSuggestion, DirectionEnum, PipelineResult


class SentimentAgentV3:
    """
    Multi-timeframe sentiment agent with trend detection.

    New V3 Features:
    - Flow conviction scoring from Unusual Whales
    - Enhanced divergence detection
    - Improved confidence weighting
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_confidence = config.get("min_confidence", 0.5)
        self.sentiment_threshold = config.get("sentiment_threshold", 0.2)
        logger.info("SentimentAgentV3 (multi-timeframe + flow) initialized")

    def suggest(
        self, pipeline_result: PipelineResult, timestamp: datetime
    ) -> Optional[AgentSuggestion]:
        """Generate suggestion based on sentiment snapshot (backward compatibility)."""
        if not pipeline_result.sentiment_snapshot:
            return None

        snapshot = pipeline_result.sentiment_snapshot

        if snapshot.confidence < self.min_confidence:
            return None

        # Determine direction from sentiment score
        if snapshot.sentiment_score > self.sentiment_threshold:
            direction = DirectionEnum.LONG
            reasoning = f"Positive sentiment ({snapshot.sentiment_score:.2f})"
        elif snapshot.sentiment_score < -self.sentiment_threshold:
            direction = DirectionEnum.SHORT
            reasoning = f"Negative sentiment ({snapshot.sentiment_score:.2f})"
        else:
            direction = DirectionEnum.NEUTRAL
            reasoning = "Neutral sentiment"

        return AgentSuggestion(
            agent_name="sentiment_agent_v3",
            timestamp=timestamp,
            symbol=pipeline_result.symbol,
            direction=direction,
            confidence=snapshot.confidence,
            reasoning=reasoning,
            target_allocation=0.0,
        )

    def suggest_multiframe(
        self, sentiment_snapshots: Dict[str, Any], symbol: str, timestamp: datetime
    ) -> List[TimeframeSignal]:
        """Generate timeframe signals from multi-timeframe sentiment data."""
        signals = []

        for timeframe, snapshot in sentiment_snapshots.items():
            if snapshot is None or snapshot.confidence < self.min_confidence:
                continue

            # Determine direction from sentiment score
            if snapshot.sentiment_score > self.sentiment_threshold:
                direction = 1.0  # Bullish sentiment
                reasoning = f"Positive sentiment ({snapshot.sentiment_score:.2f})"
            elif snapshot.sentiment_score < -self.sentiment_threshold:
                direction = -1.0  # Bearish sentiment
                reasoning = f"Negative sentiment ({snapshot.sentiment_score:.2f})"
            else:
                direction = 0.0  # Neutral
                reasoning = "Neutral sentiment"

            # Calculate strength from absolute sentiment score
            strength = min(1.0, abs(snapshot.sentiment_score))

            # Confidence from snapshot
            confidence = snapshot.confidence

            signal = TimeframeSignal(
                timeframe=timeframe,
                direction=direction,
                strength=strength,
                confidence=confidence,
                reasoning=f"{timeframe}: {reasoning}",
            )

            signals.append(signal)
            logger.debug(f"SentimentAgent {timeframe} signal: {direction:+.2f} @ {confidence:.2f}")

        return signals

    def get_flow_conviction(self, sentiment_snapshot: Any, strategy_source: str) -> float:
        """
        V3: Get flow conviction score.

        For "cheap_call" strategies, we want high flow conviction.

        Args:
            sentiment_snapshot: Current sentiment data
            strategy_source: "0dte", "cheap_call", or "standard"

        Returns:
            Flow conviction score from 0 to 1
        """
        if not sentiment_snapshot:
            return 0.0

        # Flow sentiment is already integrated in V3 engine
        flow_sentiment = getattr(sentiment_snapshot, "flow_sentiment", 0.0)

        # Conviction = absolute strength of flow sentiment
        conviction = abs(flow_sentiment)

        # For cheap calls, we want strong directional flow
        if strategy_source == "cheap_call":
            # Boost conviction if flow is strongly directional
            if conviction > 0.6:
                logger.info(f"High flow conviction for cheap call: {conviction:.2f}")
                return min(1.0, conviction * 1.2)

        return conviction

    def detect_divergences(self, sentiment_snapshots: Dict[str, Any]) -> Dict[str, Any]:
        """Detect sentiment divergences across timeframes."""
        if not sentiment_snapshots:
            return {"has_divergence": False, "type": None}

        # Get sentiment scores by timeframe
        scores = {}
        for tf in ["1Min", "5Min", "15Min", "30Min", "1Hour", "4Hour", "1Day"]:
            if tf in sentiment_snapshots and sentiment_snapshots[tf] is not None:
                scores[tf] = sentiment_snapshots[tf].sentiment_score

        if len(scores) < 3:
            return {"has_divergence": False, "type": None}

        # Check for bullish divergence (short-term bearish, long-term bullish)
        short_term_bearish = all(
            scores.get(tf, 0) < -self.sentiment_threshold
            for tf in ["1Min", "5Min", "15Min"]
            if tf in scores
        )
        long_term_bullish = all(
            scores.get(tf, 0) > self.sentiment_threshold
            for tf in ["1Hour", "4Hour", "1Day"]
            if tf in scores
        )

        if short_term_bearish and long_term_bullish:
            return {
                "has_divergence": True,
                "type": "bullish",
                "confidence": 0.7,
                "reasoning": "Short-term bearish, long-term bullish - potential reversal up",
            }

        # Check for bearish divergence (short-term bullish, long-term bearish)
        short_term_bullish = all(
            scores.get(tf, 0) > self.sentiment_threshold
            for tf in ["1Min", "5Min", "15Min"]
            if tf in scores
        )
        long_term_bearish = all(
            scores.get(tf, 0) < -self.sentiment_threshold
            for tf in ["1Hour", "4Hour", "1Day"]
            if tf in scores
        )

        if short_term_bullish and long_term_bearish:
            return {
                "has_divergence": True,
                "type": "bearish",
                "confidence": 0.7,
                "reasoning": "Short-term bullish, long-term bearish - potential reversal down",
            }

        return {"has_divergence": False, "type": None}


__all__ = ["SentimentAgentV3"]
