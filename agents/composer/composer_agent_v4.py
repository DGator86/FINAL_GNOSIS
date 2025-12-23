"""
Composer Agent V4 - Unified Signal Composition with GNOSIS Architecture.

This composer properly integrates all primary agents:
- HedgeAgent (from HedgeEngine)
- SentimentAgent (from SentimentEngine)
- LiquidityAgent (from LiquidityEngine with PENTA methodology)

The composer:
1. Receives signals from all three primary agents
2. Applies configurable weights
3. Builds consensus direction and confidence
4. Provides structured output for Trade Agents

Architecture Position:
    Primary Agents → Composer Agent V4 → Trade Agents
    
Author: GNOSIS Trading System
Version: 4.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from loguru import logger

# Import schemas
try:
    from schemas.core_schemas import (
        AgentSignal,
        AgentSuggestion,
        DirectionEnum,
        PipelineResult,
    )
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    AgentSignal = None
    AgentSuggestion = None
    DirectionEnum = None
    PipelineResult = None


class ComposerMode(str, Enum):
    """Composer operating mode."""
    STANDARD = "standard"  # All agents equal weight
    EXPRESS_0DTE = "0dte"  # Prioritize liquidity for fast execution
    EXPRESS_CHEAP_CALL = "cheap_call"  # Prioritize sentiment for flow conviction
    PENTA_CONFLUENCE = "penta_confluence"  # Use PENTA methodology bonus


@dataclass
class ComposerOutput:
    """Output from Composer Agent V4."""
    
    # Core consensus
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float  # 0.0 to 1.0
    consensus_score: float  # How aligned the agents are
    
    # Agent contributions
    hedge_contribution: float = 0.0
    sentiment_contribution: float = 0.0
    liquidity_contribution: float = 0.0
    
    # PENTA methodology
    penta_confluence: Optional[str] = None  # "PENTA", "QUAD", "TRIPLE", "DOUBLE"
    penta_confidence_bonus: float = 0.0
    
    # Analysis
    reasoning: str = ""
    agent_signals: Dict[str, str] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    mode: str = "standard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "consensus_score": round(self.consensus_score, 4),
            "hedge_contribution": round(self.hedge_contribution, 4),
            "sentiment_contribution": round(self.sentiment_contribution, 4),
            "liquidity_contribution": round(self.liquidity_contribution, 4),
            "penta_confluence": self.penta_confluence,
            "penta_confidence_bonus": round(self.penta_confidence_bonus, 4),
            "reasoning": self.reasoning,
            "agent_signals": self.agent_signals,
            "risk_factors": self.risk_factors,
            "timestamp": self.timestamp.isoformat(),
            "mode": self.mode,
        }


class ComposerAgentV4:
    """
    Composer Agent V4 - Unified consensus builder for GNOSIS architecture.
    
    This composer integrates all primary agents and builds consensus:
    - HedgeAgent: Dealer flow, gamma, energy asymmetry (40% default weight)
    - SentimentAgent: News, social, technical sentiment (40% default weight)  
    - LiquidityAgent: Market quality + PENTA methodology (20% default weight)
    
    The composer supports multiple modes:
    - Standard: Balanced weighting across all agents
    - 0DTE: Prioritize liquidity for fast options execution
    - Cheap Call: Prioritize sentiment for flow conviction
    - PENTA Confluence: Apply methodology bonus for high-confluence setups
    
    Architecture:
        HedgeAgent  ─┐
        SentimentAgent ─┼→ ComposerAgentV4 → Trade Agents
        LiquidityAgent ─┘
    """
    
    VERSION = "4.0.0"
    
    # Default agent weights
    DEFAULT_WEIGHTS = {
        "hedge": 0.40,
        "sentiment": 0.40,
        "liquidity": 0.20,
    }
    
    # Express lane weights
    EXPRESS_WEIGHTS = {
        "0dte": {
            "hedge": 0.30,
            "liquidity": 0.50,  # Critical for 0DTE
            "sentiment": 0.20,
        },
        "cheap_call": {
            "hedge": 0.20,
            "liquidity": 0.20,
            "sentiment": 0.60,  # Flow conviction critical
        },
    }
    
    # PENTA confluence bonuses
    PENTA_BONUSES = {
        "PENTA": 0.30,  # All 5 methodologies agree
        "QUAD": 0.25,   # 4 methodologies agree
        "TRIPLE": 0.20, # 3 methodologies agree
        "DOUBLE": 0.10, # 2 methodologies agree
    }
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Composer Agent V4.
        
        Args:
            weights: Agent weights (hedge, sentiment, liquidity)
            config: Configuration options
        """
        self.config = config or {}
        
        # Set weights
        self.agent_weights = weights or self.DEFAULT_WEIGHTS.copy()
        
        # Normalize weights to sum to 1.0
        total = sum(self.agent_weights.values())
        if total > 0:
            self.agent_weights = {k: v / total for k, v in self.agent_weights.items()}
        
        # Configuration
        self.min_confidence = self.config.get("min_confidence", 0.5)
        self.bullish_threshold = self.config.get("bullish_threshold", 0.3)
        self.bearish_threshold = self.config.get("bearish_threshold", -0.3)
        
        logger.info(
            f"ComposerAgentV4 v{self.VERSION} initialized | "
            f"weights: hedge={self.agent_weights.get('hedge', 0):.0%}, "
            f"sentiment={self.agent_weights.get('sentiment', 0):.0%}, "
            f"liquidity={self.agent_weights.get('liquidity', 0):.0%}"
        )
    
    def compose(
        self,
        hedge_signal: Optional[Dict[str, Any]] = None,
        sentiment_signal: Optional[Dict[str, Any]] = None,
        liquidity_signal: Optional[Dict[str, Any]] = None,
        penta_confluence: Optional[str] = None,
        mode: ComposerMode = ComposerMode.STANDARD,
    ) -> ComposerOutput:
        """
        Compose signals from all primary agents into unified consensus.
        
        Args:
            hedge_signal: Signal from HedgeAgent
                {"direction": "bullish"|"bearish"|"neutral", "confidence": 0.0-1.0, ...}
            sentiment_signal: Signal from SentimentAgent
                {"direction": "bullish"|"bearish"|"neutral", "confidence": 0.0-1.0, ...}
            liquidity_signal: Signal from LiquidityAgent
                {"direction": "bullish"|"bearish"|"neutral", "confidence": 0.0-1.0, ...}
            penta_confluence: PENTA confluence level from LiquidityAgentV5
            mode: Operating mode (standard, 0dte, cheap_call)
            
        Returns:
            ComposerOutput with unified direction and confidence
        """
        # Get appropriate weights for mode
        if mode.value in self.EXPRESS_WEIGHTS:
            weights = self.EXPRESS_WEIGHTS[mode.value]
        else:
            weights = self.agent_weights
        
        # Normalize and extract signals
        signals = []
        agent_signals = {}
        
        # Process hedge signal
        if hedge_signal:
            direction = self._normalize_direction(hedge_signal.get("direction", "neutral"))
            confidence = hedge_signal.get("confidence", 0.5)
            signals.append({
                "agent": "hedge",
                "direction": direction,
                "confidence": confidence,
                "weight": weights.get("hedge", 0.33),
            })
            agent_signals["hedge"] = f"{direction} ({confidence:.0%})"
        
        # Process sentiment signal
        if sentiment_signal:
            direction = self._normalize_direction(sentiment_signal.get("direction", "neutral"))
            confidence = sentiment_signal.get("confidence", 0.5)
            signals.append({
                "agent": "sentiment",
                "direction": direction,
                "confidence": confidence,
                "weight": weights.get("sentiment", 0.33),
            })
            agent_signals["sentiment"] = f"{direction} ({confidence:.0%})"
        
        # Process liquidity signal
        if liquidity_signal:
            direction = self._normalize_direction(liquidity_signal.get("direction", "neutral"))
            confidence = liquidity_signal.get("confidence", 0.5)
            signals.append({
                "agent": "liquidity",
                "direction": direction,
                "confidence": confidence,
                "weight": weights.get("liquidity", 0.33),
            })
            agent_signals["liquidity"] = f"{direction} ({confidence:.0%})"
        
        # Handle no signals
        if not signals:
            return ComposerOutput(
                direction="NEUTRAL",
                confidence=0.0,
                consensus_score=0.0,
                reasoning="No signals received from agents",
                mode=mode.value,
            )
        
        # Calculate weighted direction
        weighted_direction = 0.0
        total_weight = 0.0
        contributions = {}
        
        for signal in signals:
            direction_value = self._direction_to_numeric(signal["direction"])
            weight = signal["weight"] * signal["confidence"]
            weighted_direction += direction_value * weight
            total_weight += weight
            contributions[signal["agent"]] = direction_value * weight
        
        # Calculate consensus direction
        if total_weight > 0:
            consensus_value = weighted_direction / total_weight
        else:
            consensus_value = 0.0
        
        # Determine direction
        if consensus_value > self.bullish_threshold:
            direction = "LONG"
        elif consensus_value < self.bearish_threshold:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"
        
        # Calculate consensus score (how aligned agents are)
        alignment_count = sum(
            1 for s in signals 
            if self._direction_to_numeric(s["direction"]) * consensus_value > 0
        )
        consensus_score = alignment_count / len(signals) if signals else 0.0
        
        # Calculate base confidence
        avg_confidence = sum(s["confidence"] for s in signals) / len(signals)
        base_confidence = avg_confidence * (0.5 + 0.5 * abs(consensus_value))
        
        # Apply PENTA confluence bonus
        penta_bonus = 0.0
        if penta_confluence and penta_confluence in self.PENTA_BONUSES:
            penta_bonus = self.PENTA_BONUSES[penta_confluence]
            base_confidence = min(1.0, base_confidence * (1 + penta_bonus))
        
        # Calculate final confidence
        final_confidence = base_confidence * consensus_score
        
        # Clamp confidence
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Identify risk factors
        risk_factors = []
        if consensus_score < 0.5:
            risk_factors.append("Low agent alignment")
        if avg_confidence < 0.5:
            risk_factors.append("Low average confidence")
        if direction == "NEUTRAL":
            risk_factors.append("No clear directional bias")
        
        # Build reasoning
        reasoning_parts = []
        if direction != "NEUTRAL":
            reasoning_parts.append(f"{direction} consensus ({final_confidence:.0%} confidence)")
        else:
            reasoning_parts.append("NEUTRAL - no clear consensus")
        
        if penta_confluence:
            reasoning_parts.append(f"{penta_confluence} confluence (+{penta_bonus:.0%} bonus)")
        
        if mode != ComposerMode.STANDARD:
            reasoning_parts.append(f"Express mode: {mode.value}")
        
        reasoning = " | ".join(reasoning_parts)
        
        return ComposerOutput(
            direction=direction,
            confidence=final_confidence,
            consensus_score=consensus_score,
            hedge_contribution=contributions.get("hedge", 0.0),
            sentiment_contribution=contributions.get("sentiment", 0.0),
            liquidity_contribution=contributions.get("liquidity", 0.0),
            penta_confluence=penta_confluence,
            penta_confidence_bonus=penta_bonus,
            reasoning=reasoning,
            agent_signals=agent_signals,
            risk_factors=risk_factors,
            mode=mode.value,
        )
    
    def compose_from_pipeline(
        self,
        pipeline_result: Any,
        timestamp: datetime,
        mode: ComposerMode = ComposerMode.STANDARD,
    ) -> ComposerOutput:
        """
        Compose signals from a PipelineResult object.
        
        Args:
            pipeline_result: PipelineResult with engine snapshots
            timestamp: Current timestamp
            mode: Operating mode
            
        Returns:
            ComposerOutput with unified direction and confidence
        """
        hedge_signal = None
        sentiment_signal = None
        liquidity_signal = None
        penta_confluence = None
        
        # Extract hedge signal from snapshot
        if hasattr(pipeline_result, 'hedge_snapshot') and pipeline_result.hedge_snapshot:
            snapshot = pipeline_result.hedge_snapshot
            energy = getattr(snapshot, 'energy_asymmetry', 0)
            confidence = getattr(snapshot, 'confidence', 0.5)
            
            if energy > 0.3:
                direction = "bullish"
            elif energy < -0.3:
                direction = "bearish"
            else:
                direction = "neutral"
            
            hedge_signal = {"direction": direction, "confidence": confidence}
        
        # Extract sentiment signal from snapshot
        if hasattr(pipeline_result, 'sentiment_snapshot') and pipeline_result.sentiment_snapshot:
            snapshot = pipeline_result.sentiment_snapshot
            score = getattr(snapshot, 'sentiment_score', 0)
            confidence = getattr(snapshot, 'confidence', 0.5)
            
            if score > 0.2:
                direction = "bullish"
            elif score < -0.2:
                direction = "bearish"
            else:
                direction = "neutral"
            
            sentiment_signal = {"direction": direction, "confidence": confidence}
        
        # Extract liquidity signal from snapshot
        if hasattr(pipeline_result, 'liquidity_snapshot') and pipeline_result.liquidity_snapshot:
            snapshot = pipeline_result.liquidity_snapshot
            score = getattr(snapshot, 'liquidity_score', 0.5)
            
            # Check for PENTA state
            if hasattr(snapshot, 'penta_state') and snapshot.penta_state:
                penta_state = snapshot.penta_state
                # Calculate confluence from PENTA
                agreeing = sum([
                    1 if getattr(penta_state, f'{m}_signal', 'neutral') != 'neutral' else 0
                    for m in ['wyckoff', 'ict', 'order_flow', 'supply_demand', 'liquidity_concepts']
                ])
                
                if agreeing >= 5:
                    penta_confluence = "PENTA"
                elif agreeing >= 4:
                    penta_confluence = "QUAD"
                elif agreeing >= 3:
                    penta_confluence = "TRIPLE"
                elif agreeing >= 2:
                    penta_confluence = "DOUBLE"
            
            liquidity_signal = {"direction": "neutral", "confidence": score}
        
        return self.compose(
            hedge_signal=hedge_signal,
            sentiment_signal=sentiment_signal,
            liquidity_signal=liquidity_signal,
            penta_confluence=penta_confluence,
            mode=mode,
        )
    
    def compose_from_suggestions(
        self,
        suggestions: List[Any],
        timestamp: datetime,
    ) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility with V1/V2 interface.
        
        Args:
            suggestions: List of AgentSuggestion objects
            timestamp: Composition timestamp
            
        Returns:
            Dictionary with direction, confidence, reasoning
        """
        if not suggestions:
            return {
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "reasoning": "No suggestions available",
            }
        
        # Convert suggestions to signals
        hedge_signal = None
        sentiment_signal = None
        liquidity_signal = None
        
        for suggestion in suggestions:
            agent_name = getattr(suggestion, 'agent_name', '').lower()
            direction = getattr(suggestion, 'direction', None)
            confidence = getattr(suggestion, 'confidence', 0.5)
            
            # Map direction enum to string
            if hasattr(direction, 'value'):
                if direction.value in ['LONG', 'UP', 'long', 'up']:
                    dir_str = 'bullish'
                elif direction.value in ['SHORT', 'DOWN', 'short', 'down']:
                    dir_str = 'bearish'
                else:
                    dir_str = 'neutral'
            else:
                dir_str = str(direction).lower() if direction else 'neutral'
            
            signal = {"direction": dir_str, "confidence": confidence}
            
            if 'hedge' in agent_name:
                hedge_signal = signal
            elif 'sentiment' in agent_name:
                sentiment_signal = signal
            elif 'liquidity' in agent_name:
                liquidity_signal = signal
        
        output = self.compose(
            hedge_signal=hedge_signal,
            sentiment_signal=sentiment_signal,
            liquidity_signal=liquidity_signal,
        )
        
        return {
            "direction": output.direction,
            "confidence": output.confidence,
            "consensus_score": output.consensus_score,
            "reasoning": output.reasoning,
        }
    
    def _normalize_direction(self, direction: Any) -> str:
        """Normalize direction to standard format."""
        if direction is None:
            return "neutral"
        
        # Handle enum
        if hasattr(direction, 'value'):
            direction = direction.value
        
        direction_str = str(direction).lower()
        
        if direction_str in ['bullish', 'long', 'up', 'buy']:
            return 'bullish'
        elif direction_str in ['bearish', 'short', 'down', 'sell']:
            return 'bearish'
        else:
            return 'neutral'
    
    def _direction_to_numeric(self, direction: str) -> float:
        """Convert direction string to numeric value."""
        return {
            'bullish': 1.0,
            'bearish': -1.0,
            'neutral': 0.0,
        }.get(direction.lower(), 0.0)


# Factory function
def create_composer_v4(
    weights: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ComposerAgentV4:
    """Create a ComposerAgentV4 instance."""
    return ComposerAgentV4(weights=weights, config=config)


__all__ = [
    "ComposerAgentV4",
    "ComposerOutput",
    "ComposerMode",
    "create_composer_v4",
]
