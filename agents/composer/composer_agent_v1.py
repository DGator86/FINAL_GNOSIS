"""Composer Agent v1 - Consensus building."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from loguru import logger

from schemas.core_schemas import AgentSuggestion, DirectionEnum


class ComposerAgentV1:
    """Composer Agent v1 for building consensus from agent suggestions."""
    
    def __init__(self, weights: Any, config: Dict[str, Any]):
        self.weights = weights if hasattr(weights, '__dict__') else type('obj', (object,), weights)
        self.config = config
        logger.info("ComposerAgentV1 initialized")
    
    def compose(self, suggestions: List[AgentSuggestion], timestamp: datetime) -> Dict[str, Any]:
        """
        Build consensus from agent suggestions.
        
        Args:
            suggestions: List of agent suggestions
            timestamp: Composition timestamp
            
        Returns:
            Consensus dictionary
        """
        if not suggestions:
            return {
                "direction": DirectionEnum.NEUTRAL.value,
                "confidence": 0.0,
                "reasoning": "No suggestions available",
            }
        
        # Weight suggestions by agent type and confidence
        weighted_direction = 0.0
        total_weight = 0.0
        
        for suggestion in suggestions:
            agent_weight = self._get_agent_weight(suggestion.agent_name)
            weight = agent_weight * suggestion.confidence
            
            # Convert direction to numeric (-1, 0, 1)
            direction_value = {
                DirectionEnum.SHORT: -1.0,
                DirectionEnum.NEUTRAL: 0.0,
                DirectionEnum.LONG: 1.0,
            }.get(suggestion.direction, 0.0)
            
            weighted_direction += direction_value * weight
            total_weight += weight
        
        if total_weight == 0:
            return {
                "direction": DirectionEnum.NEUTRAL.value,
                "confidence": 0.0,
                "reasoning": "No weighted suggestions",
            }
        
        # Calculate consensus direction
        consensus_value = weighted_direction / total_weight
        
        if consensus_value > 0.3:
            direction = DirectionEnum.LONG
        elif consensus_value < -0.3:
            direction = DirectionEnum.SHORT
        else:
            direction = DirectionEnum.NEUTRAL
        
        # Calculate confidence as agreement level
        confidence = min(1.0, abs(consensus_value))
        
        reasoning = f"Consensus from {len(suggestions)} agents: {consensus_value:.2f}"
        
        return {
            "direction": direction.value,
            "confidence": confidence,
            "consensus_value": consensus_value,
            "num_agents": len(suggestions),
            "reasoning": reasoning,
        }
    
    def _get_agent_weight(self, agent_name: str) -> float:
        """Get weight for an agent."""
        if "hedge" in agent_name.lower():
            return getattr(self.weights, 'hedge', 0.4)
        elif "liquidity" in agent_name.lower():
            return getattr(self.weights, 'liquidity', 0.2)
        elif "sentiment" in agent_name.lower():
            return getattr(self.weights, 'sentiment', 0.4)
        return 0.33
