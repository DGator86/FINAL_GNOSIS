"""Trade Agent v1 - Trade idea generation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from schemas.core_schemas import DirectionEnum, PipelineResult, StrategyType, TradeIdea


class TradeAgentV1:
    """Trade Agent v1 for generating trade ideas from consensus."""
    
    def __init__(self, options_adapter: Any, config: Dict[str, Any]):
        self.options_adapter = options_adapter
        self.config = config
        logger.info("TradeAgentV1 initialized")
    
    def generate_ideas(
        self, 
        pipeline_result: PipelineResult, 
        timestamp: datetime
    ) -> List[TradeIdea]:
        """
        Generate trade ideas from pipeline results.
        
        Args:
            pipeline_result: Complete pipeline result
            timestamp: Generation timestamp
            
        Returns:
            List of trade ideas
        """
        if not pipeline_result.consensus:
            return []
        
        consensus = pipeline_result.consensus
        direction_str = consensus.get("direction", "neutral")
        confidence = consensus.get("confidence", 0.0)
        
        # Convert string to enum
        direction = DirectionEnum(direction_str) if direction_str else DirectionEnum.NEUTRAL
        
        if direction == DirectionEnum.NEUTRAL or confidence < 0.5:
            return []
        
        # Determine strategy type based on engine signals
        strategy_type = self._select_strategy(pipeline_result)
        
        # Calculate position size
        max_size = self.config.get("max_position_size", 10000.0)
        risk_per_trade = self.config.get("risk_per_trade", 0.02)
        size = max_size * risk_per_trade * confidence
        
        reasoning = f"{strategy_type.value} strategy based on consensus ({confidence:.2f})"
        
        trade_idea = TradeIdea(
            timestamp=timestamp,
            symbol=pipeline_result.symbol,
            strategy_type=strategy_type,
            direction=direction,
            confidence=confidence,
            size=size,
            reasoning=reasoning,
        )
        
        return [trade_idea]
    
    def _select_strategy(self, pipeline_result: PipelineResult) -> StrategyType:
        """Select appropriate strategy based on pipeline results."""
        # Simple heuristic: high volatility = breakout, low = mean reversion
        if pipeline_result.elasticity_snapshot:
            volatility = pipeline_result.elasticity_snapshot.volatility
            if volatility > 0.3:
                return StrategyType.BREAKOUT
            elif volatility < 0.15:
                return StrategyType.MEAN_REVERSION
        
        # Default to directional
        return StrategyType.DIRECTIONAL
