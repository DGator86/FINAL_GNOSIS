"""Sentiment Agent v1 with PPF (Past/Present/Future) analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from schemas.core_schemas import (
    AgentSuggestion,
    DirectionEnum,
    PipelineResult,
    PPFAnalysis,
    PastAnalysis,
    PresentAnalysis,
    FutureAnalysis,
)


class SentimentAgentV1:
    """Sentiment Agent v1 for sentiment-based suggestions with PPF analysis."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("SentimentAgentV1 initialized")

    def suggest(self, pipeline_result: PipelineResult, timestamp: datetime) -> Optional[AgentSuggestion]:
        """Generate suggestion based on sentiment snapshot with PPF analysis."""
        if not pipeline_result.sentiment_snapshot:
            return None

        snapshot = pipeline_result.sentiment_snapshot
        min_threshold = self.config.get("min_sentiment_threshold", 0.2)

        if abs(snapshot.sentiment_score) < min_threshold:
            return None

        # Determine direction from sentiment
        if snapshot.sentiment_score > 0:
            direction = DirectionEnum.LONG
            reasoning = f"Positive sentiment {snapshot.sentiment_score:.2f}"
        else:
            direction = DirectionEnum.SHORT
            reasoning = f"Negative sentiment {snapshot.sentiment_score:.2f}"

        confidence = min(1.0, abs(snapshot.sentiment_score) * snapshot.confidence)

        # Build PPF analysis for sentiment domain
        ppf = self._build_ppf_analysis(pipeline_result, timestamp, direction, confidence)

        return AgentSuggestion(
            agent_name="sentiment_agent_v1",
            timestamp=timestamp,
            symbol=pipeline_result.symbol,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            target_allocation=0.0,
            ppf_analysis=ppf,
        )

    def _build_ppf_analysis(
        self,
        pipeline_result: PipelineResult,
        timestamp: datetime,
        direction: DirectionEnum,
        confidence: float,
    ) -> PPFAnalysis:
        """
        Build Past/Present/Future analysis for the sentiment domain.

        PAST: Historical sentiment patterns, seasonality
        PRESENT: Current news, flow, technical sentiment
        FUTURE: Momentum projections, upcoming events impact
        """
        snapshot = pipeline_result.sentiment_snapshot
        elasticity = pipeline_result.elasticity_snapshot
        ml_snapshot = pipeline_result.ml_snapshot

        # === PAST ANALYSIS ===
        past = PastAnalysis(
            # Could be enhanced with historical sentiment trends
            seasonality_bias=0.0,  # Would need historical data
        )

        # === PRESENT ANALYSIS ===
        present = PresentAnalysis(
            news_sentiment=snapshot.news_sentiment,
            news_intensity=snapshot.intensity if snapshot.intensity else 0.0,
        )

        # Add technical indicators from elasticity snapshot if available
        if elasticity:
            present.volatility = elasticity.volatility

        # MTF score if available
        if snapshot.mtf_score:
            present.rsi = snapshot.mtf_score * 50 + 50  # Convert -1 to +1 to RSI-like 0-100

        # === FUTURE ANALYSIS ===
        future = FutureAnalysis()

        # Use ML projections if available
        if ml_snapshot and ml_snapshot.forecast:
            forecast = ml_snapshot.forecast
            metadata = forecast.metadata or {}

            predictions_pct = metadata.get("predictions_pct", {})
            future.projected_move_1m = predictions_pct.get(1, 0.0)
            future.projected_move_5m = predictions_pct.get(5, 0.0)
            future.projected_move_15m = predictions_pct.get(15, 0.0)
            future.projected_move_60m = predictions_pct.get(60, 0.0)
            future.move_confidence = forecast.confidence
            future.projected_direction = metadata.get("direction", "neutral")

        # Directional bias from synthesis
        dir_value = 0.0
        if direction == DirectionEnum.LONG:
            dir_value = confidence
        elif direction == DirectionEnum.SHORT:
            dir_value = -confidence

        return PPFAnalysis(
            timestamp=timestamp,
            symbol=pipeline_result.symbol,
            domain="sentiment",
            past=past,
            present=present,
            future=future,
            directional_bias=dir_value,
            confidence=confidence,
            time_horizon="intraday",
            reasoning=f"Sentiment PPF: News={snapshot.news_sentiment:.2f}, Flow={snapshot.flow_sentiment:.2f}, Tech={snapshot.technical_sentiment:.2f}",
        )
