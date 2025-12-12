"""Enhanced Composer Agent v2 with Multi-Timeframe Analysis.

Performs backward analysis (trend confirmation), forward analysis (prediction),
and integrates ConfidenceBuilder for go/no-go decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from agents.confidence_builder import ConfidenceBuilder, ConfidenceScore, TimeframeSignal
from schemas.core_schemas import AgentSuggestion, DirectionEnum


@dataclass
class ComposerDecision:
    """Decision output from Composer Agent."""

    timestamp: datetime
    symbol: str
    go_signal: bool
    predicted_direction: str  # 'LONG', 'SHORT', 'NEUTRAL'
    confidence: float
    predicted_timeframe: str  # Recommended trade duration
    risk_reward_ratio: float
    reasoning: str
    confidence_score: Optional[ConfidenceScore] = None
    backward_analysis: Optional[Dict[str, Any]] = None
    forward_analysis: Optional[Dict[str, Any]] = None


class ComposerAgentV2:
    """Enhanced Composer Agent with multi-timeframe intelligence.

    Responsibilities:
    1. Aggregate signals from all primary agents
    2. Backward analysis - confirm trends across timeframes
    3. Forward analysis - predict likely scenarios
    4. Calculate compound confidence
    5. Make go/no-go decision for Trade Agent
    """

    def __init__(
        self, weights: Optional[Dict[str, float]] = None, config: Optional[Dict[str, Any]] = None
    ):
        """Initialize Composer Agent.

        Args:
            weights: Agent weights (hedge, liquidity, sentiment)
            config: Configuration dictionary
        """
        self.config = config or {}

        # Agent weights
        default_weights = {"hedge": 0.4, "liquidity": 0.2, "sentiment": 0.4}
        self.agent_weights = weights or default_weights

        # Initialize ConfidenceBuilder
        confidence_config = self.config.get("confidence_builder", {})
        self.confidence_builder = ConfidenceBuilder(
            weights=confidence_config.get("timeframe_weights"),
            min_confidence_threshold=confidence_config.get("min_confidence_threshold", 0.70),
            min_alignment_threshold=confidence_config.get("min_alignment_threshold", 0.60),
        )

        logger.info(f"ComposerAgentV2 initialized | agent_weights={self.agent_weights}")

    def compose(self, suggestions: List[AgentSuggestion], timestamp: datetime) -> Dict[str, Any]:
        """Legacy compose method for backward compatibility.

        Use compose_multiframe() for full multi-timeframe capabilities.
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

    def compose_multiframe(
        self,
        all_timeframe_signals: List[TimeframeSignal],
        symbol: str,
        timestamp: datetime,
        current_price: float,
    ) -> ComposerDecision:
        """Compose multi-timeframe signals into actionable decision.

        Args:
            all_timeframe_signals: Signals from all agents across all timeframes
            symbol: Trading symbol
            timestamp: Decision timestamp
            current_price: Current market price

        Returns:
            ComposerDecision with go/no-go and strategy parameters
        """
        # Step 1: Calculate confidence using ConfidenceBuilder
        confidence_score = self.confidence_builder.calculate_confidence(all_timeframe_signals)

        logger.info(
            f"Composer analyzing {symbol} | "
            f"confidence={confidence_score.overall_confidence:.2%} | "
            f"alignment={confidence_score.alignment_score:.2%}"
        )

        # Step 2: Backward Analysis (trend confirmation)
        backward_analysis = self._backward_analysis(all_timeframe_signals, confidence_score)

        # Step 3: Forward Analysis (prediction)
        forward_analysis = self._forward_analysis(
            all_timeframe_signals, confidence_score, current_price
        )

        # Step 4: Make go/no-go decision
        go_signal = self.confidence_builder.meets_threshold(confidence_score)

        # Step 5: Determine direction
        if confidence_score.weighted_direction > 0.3:
            predicted_direction = "LONG"
        elif confidence_score.weighted_direction < -0.3:
            predicted_direction = "SHORT"
        else:
            predicted_direction = "NEUTRAL"
            go_signal = False  # Don't trade neutral signals

        # Step 6: Get recommended timeframe
        predicted_timeframe = self.confidence_builder.get_recommended_timeframe(confidence_score)

        # Step 7: Calculate risk/reward ratio
        risk_reward_ratio = self._calculate_risk_reward(
            predicted_direction, predicted_timeframe, forward_analysis
        )

        # Step 8: Build reasoning
        reasoning = self._build_reasoning(
            confidence_score, backward_analysis, forward_analysis, go_signal
        )

        decision = ComposerDecision(
            timestamp=timestamp,
            symbol=symbol,
            go_signal=go_signal,
            predicted_direction=predicted_direction,
            confidence=confidence_score.overall_confidence,
            predicted_timeframe=predicted_timeframe,
            risk_reward_ratio=risk_reward_ratio,
            reasoning=reasoning,
            confidence_score=confidence_score,
            backward_analysis=backward_analysis,
            forward_analysis=forward_analysis,
        )

        logger.info(
            f"Composer decision: {symbol} | "
            f"GO={go_signal} | {predicted_direction} | "
            f"conf={confidence_score.overall_confidence:.2%} | "
            f"TF={predicted_timeframe}"
        )

        return decision

    def _backward_analysis(
        self, signals: List[TimeframeSignal], confidence_score: ConfidenceScore
    ) -> Dict[str, Any]:
        """Backward analysis: Confirm trend across timeframes.

        Looks at historical alignment to validate current trend.
        """
        # Group signals by timeframe
        tf_signals: Dict[str, List[TimeframeSignal]] = {}
        for signal in signals:
            if signal.timeframe not in tf_signals:
                tf_signals[signal.timeframe] = []
            tf_signals[signal.timeframe].append(signal)

        # Check trend consistency
        timeframe_order = ["1Min", "5Min", "15Min", "30Min", "1Hour", "4Hour", "1Day"]
        trend_consistent = True
        prev_direction = None

        for tf in timeframe_order:
            if tf in tf_signals:
                # Average direction for this timeframe
                avg_direction = sum(s.direction for s in tf_signals[tf]) / len(tf_signals[tf])

                if prev_direction is not None:
                    # Check if direction flipped (trend inconsistency)
                    if (prev_direction > 0.1 and avg_direction < -0.1) or (
                        prev_direction < -0.1 and avg_direction > 0.1
                    ):
                        trend_consistent = False

                prev_direction = avg_direction

        return {
            "trend_consistent": trend_consistent,
            "alignment_score": confidence_score.alignment_score,
            "timeframes_analyzed": len(tf_signals),
            "dominant_trend": "bullish"
            if confidence_score.weighted_direction > 0
            else "bearish"
            if confidence_score.weighted_direction < 0
            else "neutral",
        }

    def _forward_analysis(
        self,
        signals: List[TimeframeSignal],
        confidence_score: ConfidenceScore,
        current_price: float,
    ) -> Dict[str, Any]:
        """Forward analysis: Predict likely price movement.

        Uses timeframe momentum and confidence to predict scenarios.
        """
        # Calculate expected move based on dominant timeframe
        dominant_tf = confidence_score.dominant_timeframe

        # Base expected move on timeframe (longer TF = bigger move)
        expected_move_pct = {
            "1Min": 0.005,  # 0.5%
            "5Min": 0.01,  # 1%
            "15Min": 0.015,  # 1.5%
            "30Min": 0.02,  # 2%
            "1Hour": 0.03,  # 3%
            "4Hour": 0.05,  # 5%
            "1Day": 0.08,  # 8%
        }.get(dominant_tf, 0.02)

        # Adjust by confidence (higher confidence = bigger expected move)
        expected_move_pct *= confidence_score.overall_confidence

        # Calculate target prices
        direction_multiplier = 1 if confidence_score.weighted_direction > 0 else -1
        target_price = current_price * (1 + (expected_move_pct * direction_multiplier))
        stop_price = current_price * (1 - (expected_move_pct * 0.4 * abs(direction_multiplier)))

        return {
            "expected_move_pct": expected_move_pct,
            "target_price": target_price,
            "stop_price": stop_price,
            "dominant_timeframe": dominant_tf,
            "prediction_confidence": confidence_score.overall_confidence,
        }

    def _calculate_risk_reward(
        self, direction: str, timeframe: str, forward_analysis: Dict[str, Any]
    ) -> float:
        """Calculate risk/reward ratio for the trade."""
        if direction == "NEUTRAL":
            return 0.0

        expected_move = float(forward_analysis.get("expected_move_pct", 0.02))

        # Risk is typically 30-50% of expected reward
        risk_pct = expected_move * 0.4
        reward_pct = expected_move

        if risk_pct == 0:
            return 0.0

        return float(reward_pct / risk_pct)

    def _build_reasoning(
        self,
        confidence_score: ConfidenceScore,
        backward_analysis: Dict[str, Any],
        forward_analysis: Dict[str, Any],
        go_signal: bool,
    ) -> str:
        """Build human-readable reasoning for the decision."""
        parts = []

        # Confidence summary
        parts.append(confidence_score.reasoning)

        # Backward analysis
        if backward_analysis["trend_consistent"]:
            parts.append("✓ Trend confirmed across timeframes")
        else:
            parts.append("⚠ Trend inconsistency detected")

        # Forward analysis
        expected_move = forward_analysis["expected_move_pct"] * 100
        parts.append(f"Expected move: {expected_move:.1f}%")

        # Decision
        if go_signal:
            parts.append("→ GO for trade execution")
        else:
            parts.append("⊗ Insufficient confidence - waiting")

        return " | ".join(parts)

    def _get_agent_weight(self, agent_name: str) -> float:
        """Get weight for an agent (legacy compatibility)."""
        if "hedge" in agent_name.lower():
            return self.agent_weights.get("hedge", 0.4)
        elif "liquidity" in agent_name.lower():
            return self.agent_weights.get("liquidity", 0.2)
        elif "sentiment" in agent_name.lower():
            return self.agent_weights.get("sentiment", 0.4)
        return 0.33


__all__ = ["ComposerAgentV2", "ComposerDecision"]
