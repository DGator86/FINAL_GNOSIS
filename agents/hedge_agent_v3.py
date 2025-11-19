"""Hedge Agent v3 - Energy-aware interpretation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from schemas.core_schemas import AgentSuggestion, DirectionEnum, PipelineResult


class HedgeAgentV3:
    """Hedge Agent v3 with energy-aware interpretation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("HedgeAgentV3 initialized")
    
    def suggest(self, pipeline_result: PipelineResult, timestamp: datetime) -> Optional[AgentSuggestion]:
        """Generate suggestion based on hedge snapshot."""
        if not pipeline_result.hedge_snapshot:
            return None
        
        snapshot = pipeline_result.hedge_snapshot
        min_confidence = self.config.get("min_confidence", 0.5)
        
        if snapshot.confidence < min_confidence:
            return None
        
        # Determine direction from energy asymmetry
        if snapshot.energy_asymmetry > 0.3:
            direction = DirectionEnum.LONG
            reasoning = f"Positive energy asymmetry ({snapshot.energy_asymmetry:.2f}), upward bias"
        elif snapshot.energy_asymmetry < -0.3:
            direction = DirectionEnum.SHORT
            reasoning = f"Negative energy asymmetry ({snapshot.energy_asymmetry:.2f}), downward bias"
        else:
            direction = DirectionEnum.NEUTRAL
            reasoning = "Energy asymmetry neutral, no clear directional bias"
        
        # Adjust confidence based on movement energy
        confidence = snapshot.confidence * (1.0 + min(0.5, snapshot.movement_energy / 100.0))
        confidence = min(1.0, confidence)
        
        return AgentSuggestion(
            agent_name="hedge_agent_v3",
            timestamp=timestamp,
            symbol=pipeline_result.symbol,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            target_allocation=0.0,
        )
