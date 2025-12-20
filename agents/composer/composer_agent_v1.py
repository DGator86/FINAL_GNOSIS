"""Composer Agent v1 - Consensus building."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Union

from loguru import logger

from schemas.core_schemas import AgentSignal, AgentSuggestion, DirectionEnum


class ComposerAgentV1:
    """Composer Agent v1 for building consensus from agent suggestions.
    
    Supports two input formats:
    1. List[AgentSuggestion] with timestamp - for pipeline integration
    2. Dict[str, AgentSignal] - for direct signal composition (used in tests)
    """
    
    def __init__(self, weights: Any, config: Dict[str, Any]):
        self.weights = weights if hasattr(weights, '__dict__') else type('obj', (object,), weights)
        self.config = config
        self.min_consensus_score = config.get("min_consensus_score", 0.5)
        logger.info("ComposerAgentV1 initialized")
    
    def compose(
        self, 
        suggestions: Union[List[AgentSuggestion], Dict[str, AgentSignal]], 
        timestamp: datetime = None
    ) -> Union[Dict[str, Any], AgentSignal]:
        """
        Build consensus from agent suggestions or signals.
        
        Args:
            suggestions: List of AgentSuggestion or Dict of AgentSignal by agent name
            timestamp: Composition timestamp (optional for dict input)
            
        Returns:
            Consensus dictionary or AgentSignal depending on input type
        """
        # Handle dictionary input (AgentSignal-based, used in tests)
        if isinstance(suggestions, dict):
            return self._compose_from_signals(suggestions)
        
        # Handle list input (AgentSuggestion-based, used in pipeline)
        return self._compose_from_suggestions(suggestions, timestamp)
    
    def _compose_from_signals(self, signals: Dict[str, AgentSignal]) -> AgentSignal:
        """Compose consensus from dictionary of AgentSignal objects."""
        if not signals:
            return AgentSignal(
                timestamp=datetime.now(timezone.utc),
                symbol="UNKNOWN",
                signal="neutral",
                confidence=0.0,
                reasoning="No signals available",
            )
        
        # Extract symbol from first signal
        first_signal = next(iter(signals.values()))
        symbol = first_signal.symbol
        timestamp = first_signal.timestamp
        
        # Calculate weighted consensus
        weighted_direction = 0.0
        total_weight = 0.0
        confidence_sum = 0.0
        reasoning_parts = []
        
        for agent_name, signal in signals.items():
            agent_weight = self._get_agent_weight(agent_name)
            weight = agent_weight * signal.confidence
            
            # Convert signal string to numeric (-1, 0, 1)
            signal_str = signal.signal.lower()
            direction_value = {
                "bearish": -1.0,
                "neutral": 0.0,
                "bullish": 1.0,
            }.get(signal_str, 0.0)
            
            weighted_direction += direction_value * weight
            total_weight += weight
            confidence_sum += signal.confidence * agent_weight
            reasoning_parts.append(f"{agent_name}: {signal_str} ({signal.confidence:.2f})")
        
        if total_weight == 0:
            return AgentSignal(
                timestamp=timestamp,
                symbol=symbol,
                signal="neutral",
                confidence=0.0,
                reasoning="No weighted signals",
            )
        
        # Calculate consensus metrics
        consensus_value = weighted_direction / total_weight
        consensus_score = confidence_sum / sum(self._get_agent_weight(n) for n in signals.keys())
        
        # Determine consensus signal
        if consensus_value > 0.3:
            consensus_signal = "bullish"
        elif consensus_value < -0.3:
            consensus_signal = "bearish"
        else:
            consensus_signal = "neutral"
        
        # Calculate confidence based on agreement and average confidence
        strength = abs(consensus_value)
        confidence = min(1.0, consensus_score * (0.5 + 0.5 * strength))
        
        return AgentSignal(
            timestamp=timestamp,
            symbol=symbol,
            signal=consensus_signal,
            confidence=confidence,
            strength=strength,
            consensus_score=consensus_score,
            reasoning=f"Consensus: {consensus_signal} | " + " | ".join(reasoning_parts),
        )
    
    def _compose_from_suggestions(
        self, 
        suggestions: List[AgentSuggestion], 
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Compose consensus from list of AgentSuggestion objects (pipeline format)."""
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
