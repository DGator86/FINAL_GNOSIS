"""Multi-Timeframe Liquidity Agent v2.

Analyzes order book depth, imbalances, and support/resistance across
multiple timeframes to generate confidence-weighted signals.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from agents.confidence_builder import TimeframeSignal
from schemas.core_schemas import AgentSuggestion, DirectionEnum, PipelineResult


class LiquidityAgentV2:
    """Multi-timeframe liquidity agent with depth analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_confidence = config.get("min_confidence", 0.5)
        self.imbalance_threshold = config.get("imbalance_threshold", 0.15)
        logger.info("LiquidityAgentV2 (multi-timeframe) initialized")
    
    def suggest(
        self,
        pipeline_result: PipelineResult,
        timestamp: datetime
    ) -> Optional[AgentSuggestion]:
        """Generate suggestion based on liquidity snapshot (backward compatibility)."""
        if not pipeline_result.liquidity_snapshot:
            return None
        
        snapshot = pipeline_result.liquidity_snapshot
        
        if snapshot.confidence < self.min_confidence:
            return None
        
        # Determine direction from bid/ask imbalance
        if snapshot.bid_ask_imbalance > self.imbalance_threshold:
            direction = DirectionEnum.LONG
            reasoning = f"Strong bid pressure ({snapshot.bid_ask_imbalance:.2f})"
        elif snapshot.bid_ask_imbalance < -self.imbalance_threshold:
            direction = DirectionEnum.SHORT
            reasoning = f"Strong ask pressure ({snapshot.bid_ask_imbalance:.2f})"
        else:
            direction = DirectionEnum.NEUTRAL
            reasoning = "Balanced order book"
        
        return AgentSuggestion(
            agent_name="liquidity_agent_v2",
            timestamp=timestamp,
            symbol=pipeline_result.symbol,
            direction=direction,
            confidence=snapshot.confidence,
            reasoning=reasoning,
            target_allocation=0.0,
        )
    
    def suggest_multiframe(
        self,
        liquidity_snapshots: Dict[str, Any],  # Dict[timeframe, LiquiditySnapshot]
        symbol: str,
        timestamp: datetime
    ) -> List[TimeframeSignal]:
        """Generate timeframe signals from multi-timeframe liquidity data.
        
        Args:
            liquidity_snapshots: Dictionary mapping timeframe to LiquiditySnapshot
            symbol: Trading symbol
            timestamp: Analysis timestamp
        
        Returns:
            List of TimeframeSignal objects for ConfidenceBuilder
        """
        signals = []
        
        for timeframe, snapshot in liquidity_snapshots.items():
            if snapshot is None or snapshot.confidence < self.min_confidence:
                continue
            
            # Determine direction from bid/ask imbalance
            if snapshot.bid_ask_imbalance > self.imbalance_threshold:
                direction = 1.0  # Bullish (more bids)
                reasoning = f"Bid pressure ({snapshot.bid_ask_imbalance:.2f})"
            elif snapshot.bid_ask_imbalance < -self.imbalance_threshold:
                direction = -1.0  # Bearish (more asks)
                reasoning = f"Ask pressure ({snapshot.bid_ask_imbalance:.2f})"
            else:
                direction = 0.0  # Neutral
                reasoning = "Balanced order book"
            
            # Calculate strength from depth
            # Higher total depth = more reliable signal
            strength = min(1.0, getattr(snapshot,'total_depth', 100) / 1000.0)
            
            # Confidence from snapshot
            confidence = snapshot.confidence
            
            signal = TimeframeSignal(
                timeframe=timeframe,
                direction=direction,
                strength=strength,
                confidence=confidence,
                reasoning=f"{timeframe}: {reasoning} | depth={strength:.2f}"
            )
            
            signals.append(signal)
            logger.debug(f"LiquidityAgent {timeframe} signal: {direction:+.2f} @ {confidence:.2f}")
        
        return signals
    
    def detect_support_resistance(
        self,
        liquidity_snapshots: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        """Detect support/resistance levels across timeframes.
        
        Args:
            liquidity_snapshots: Multi-timeframe liquidity data
            current_price: Current market price
        
        Returns:
            Dictionary with support/resistance analysis
        """
        support_levels = []
        resistance_levels = []
        
        for timeframe, snapshot in liquidity_snapshots.items():
            if snapshot is None:
                continue
            
            # Support = strong bid depth below current price
            if hasattr(snapshot, 'bid_depth') and snapshot.bid_depth > 0:
                support_levels.append({
                    'timeframe': timeframe,
                    'price': current_price * 0.98,  # Approximate
                    'strength': snapshot.bid_depth
                })
            
            # Resistance = strong ask depth above current price
            if hasattr(snapshot, 'ask_depth') and snapshot.ask_depth > 0:
                resistance_levels.append({
                    'timeframe': timeframe,
                    'price': current_price * 1.02,  # Approximate
                    'strength': snapshot.ask_depth
                })
        
        return {
            'support_levels': sorted(support_levels, key=lambda x: x['strength'], reverse=True),
            'resistance_levels': sorted(resistance_levels, key=lambda x: x['strength'], reverse=True),
            'has_strong_support': len(support_levels) >= 3,
            'has_strong_resistance': len(resistance_levels) >= 3
        }


__all__ = ['LiquidityAgentV2']
