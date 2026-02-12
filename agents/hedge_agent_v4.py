"""Multi-Timeframe Hedge Agent v4.

Analyzes hedge engine outputs (options flow) across multiple timeframes
to detect confluence and generate confidence-weighted signals.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from agents.confidence_builder import TimeframeSignal
from schemas.core_schemas import AgentSuggestion, DirectionEnum, PipelineResult


class HedgeAgentV4:
    """Multi-timeframe hedge agent with options flow confluence detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_confidence = config.get("min_confidence", 0.5)
        self.min_energy_threshold = config.get("min_energy_threshold", 0.3)
        logger.info("HedgeAgentV4 (multi-timeframe) initialized")
    
    def suggest(
        self,
        pipeline_result: PipelineResult,
        timestamp: datetime
    ) -> Optional[AgentSuggestion]:
        """Generate suggestion based on hedge snapshot (backward compatibility).
        
        This is the single-timeframe interface for backward compatibility.
        Use suggest_multiframe() for full multi-timeframe analysis.
        """
        if not pipeline_result.hedge_snapshot:
            return None
        
        snapshot = pipeline_result.hedge_snapshot
        
        if snapshot.confidence < self.min_confidence:
            return None
        
        # Determine direction from energy asymmetry
        if snapshot.energy_asymmetry > self.min_energy_threshold:
            direction = DirectionEnum.LONG
            reasoning = f"Positive energy asymmetry ({snapshot.energy_asymmetry:.2f})"
        elif snapshot.energy_asymmetry < -self.min_energy_threshold:
            direction = DirectionEnum.SHORT
            reasoning = f"Negative energy asymmetry ({snapshot.energy_asymmetry:.2f})"
        else:
            direction = DirectionEnum.NEUTRAL
            reasoning = "Energy asymmetry neutral"
        
        # Adjust confidence based on movement energy
        confidence = snapshot.confidence * (1.0 + min(0.5, snapshot.movement_energy / 100.0))
        confidence = min(1.0, confidence)
        
        return AgentSuggestion(
            agent_name="hedge_agent_v4",
            timestamp=timestamp,
            symbol=pipeline_result.symbol,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            target_allocation=0.0,
        )
    
    def suggest_multiframe(
        self,
        hedge_snapshots: Dict[str, Any],  # Dict[timeframe, HedgeSnapshot]
        symbol: str,
        timestamp: datetime
    ) -> List[TimeframeSignal]:
        """Generate timeframe signals from multi-timeframe hedge data.
        
        Args:
            hedge_snapshots: Dictionary mapping timeframe to HedgeSnapshot
            symbol: Trading symbol
            timestamp: Analysis timestamp
        
        Returns:
            List of TimeframeSignal objects for ConfidenceBuilder
        """
        signals = []
        
        for timeframe, snapshot in hedge_snapshots.items():
            if snapshot is None or snapshot.confidence < self.min_confidence:
                continue
            
            # Determine direction from energy asymmetry
            if snapshot.energy_asymmetry > self.min_energy_threshold:
                direction = 1.0  # Bullish
                reasoning = f"Positive options flow energy ({snapshot.energy_asymmetry:.2f})"
            elif snapshot.energy_asymmetry < -self.min_energy_threshold:
                direction = -1.0  # Bearish
                reasoning = f"Negative options flow energy ({snapshot.energy_asymmetry:.2f})"
            else:
                direction = 0.0  # Neutral
                reasoning = "Neutral options flow"
            
            # Calculate strength from movement energy
            # Higher movement energy = stronger signal
            strength = min(1.0, snapshot.movement_energy / 100.0)
            
            # Confidence from snapshot
            confidence = snapshot.confidence
            
            signal = TimeframeSignal(
                timeframe=timeframe,
                direction=direction,
                strength=strength,
                confidence=confidence,
                reasoning=f"{timeframe}: {reasoning} | strength={strength:.2f}"
            )
            
            signals.append(signal)
            logger.debug(f"HedgeAgent {timeframe} signal: {direction:+.2f} @ {confidence:.2f}")
        
        return signals
    
    def detect_confluence(
        self,
        hedge_snapshots: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect options flow confluence across timeframes.
        
        Confluence occurs when multiple timeframes show similar
        energy asymmetry (all bullish or all bearish).
        
        Returns:
            Dictionary with confluence analysis
        """
        if not hedge_snapshots:
            return {"has_confluence": False, "confidence": 0.0}
        
        # Get energy asymmetries from each timeframe
        asymmetries = {
            tf: snapshot.energy_asymmetry
            for tf, snapshot in hedge_snapshots.items()
            if snapshot is not None
        }
        
        if not asymmetries:
            return {"has_confluence": False, "confidence": 0.0}
        
        # Check if all timeframes agree on direction
        bullish_count = sum(1 for a in asymmetries.values() if a > self.min_energy_threshold)
        bearish_count = sum(1 for a in asymmetries.values() if a < -self.min_energy_threshold)
        total_count = len(asymmetries)
        
        # Strong confluence if 80%+ agree
        confluence_threshold = 0.8
        has_confluence = (
            (bullish_count / total_count >= confluence_threshold) or
            (bearish_count / total_count >= confluence_threshold)
        )
        
        direction = "bullish" if bullish_count > bearish_count else \
                   "bearish" if bearish_count > bullish_count else "neutral"
        
        # Confluence strength
        agreement_ratio = max(bullish_count, bearish_count) / total_count
        
        return {
            "has_confluence": has_confluence,
            "direction": direction,
            "agreement_ratio": agreement_ratio,
            "bullish_timeframes": bullish_count,
            "bearish_timeframes": bearish_count,
            "neutral_timeframes": total_count - bullish_count - bearish_count,
            "confidence": agreement_ratio if has_confluence else 0.0
        }


__all__ = ['HedgeAgentV4']
