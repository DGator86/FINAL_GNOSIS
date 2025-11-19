"""Liquidity Agent v1."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from schemas.core_schemas import AgentSuggestion, DirectionEnum, PipelineResult


class LiquidityAgentV1:
    """Liquidity Agent v1 for tradability assessment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("LiquidityAgentV1 initialized")
    
    def suggest(self, pipeline_result: PipelineResult, timestamp: datetime) -> Optional[AgentSuggestion]:
        """Generate suggestion based on liquidity snapshot."""
        if not pipeline_result.liquidity_snapshot:
            return None
        
        snapshot = pipeline_result.liquidity_snapshot
        min_score = self.config.get("min_liquidity_score", 0.3)
        
        if snapshot.liquidity_score < min_score:
            return None
        
        # Liquidity agent doesn't provide directional bias
        # It only confirms tradability
        confidence = snapshot.liquidity_score
        reasoning = f"Liquidity score {snapshot.liquidity_score:.2f}, spread {snapshot.bid_ask_spread:.4f}%"
        
        return AgentSuggestion(
            agent_name="liquidity_agent_v1",
            timestamp=timestamp,
            symbol=pipeline_result.symbol,
            direction=DirectionEnum.NEUTRAL,
            confidence=confidence,
            reasoning=reasoning,
            target_allocation=0.0,
        )
