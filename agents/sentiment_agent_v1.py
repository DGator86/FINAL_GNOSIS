"""Sentiment Agent v1."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from schemas.core_schemas import AgentSuggestion, DirectionEnum, PipelineResult


class SentimentAgentV1:
    """Sentiment Agent v1 for sentiment-based suggestions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("SentimentAgentV1 initialized")
    
    def suggest(self, pipeline_result: PipelineResult, timestamp: datetime) -> Optional[AgentSuggestion]:
        """Generate suggestion based on sentiment snapshot."""
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
        
        return AgentSuggestion(
            agent_name="sentiment_agent_v1",
            timestamp=timestamp,
            symbol=pipeline_result.symbol,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            target_allocation=0.0,
        )
