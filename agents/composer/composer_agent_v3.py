"""Enhanced Composer Agent v3 with Express Lane Support.

Performs backward analysis (trend confirmation), forward analysis (prediction),
and integrates ConfidenceBuilder for go/no-go decisions.

New V3 Features:
- Express Lane signal routing (0DTE, Cheap Calls)
- Strategy-specific agent weighting
- Enhanced risk/reward calculation
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
    strategy_source: str = "standard"  # V3: "standard", "0dte", "cheap_call"
    confidence_score: Optional[ConfidenceScore] = None
    backward_analysis: Optional[Dict] = None
    forward_analysis: Optional[Dict] = None


class ComposerAgentV3:
    """
    Enhanced Composer Agent with multi-timeframe intelligence and Express Lane support.

    Responsibilities:
    1. Aggregate signals from all primary agents
    2. Backward analysis - confirm trends across timeframes
    3. Forward analysis - predict likely scenarios
    4. Calculate compound confidence
    5. Make go/no-go decision for Trade Agent
    6. V3: Route Express Lane signals with strategy-specific weighting
    """

    def __init__(
        self, weights: Optional[Dict[str, float]] = None, config: Optional[Dict[str, Any]] = None
    ):
        """Initialize Composer Agent V3."""
        self.config = config or {}

        # Agent weights (standard)
        default_weights = {"hedge": 0.4, "liquidity": 0.2, "sentiment": 0.4}
        self.agent_weights = weights or default_weights

        # V3: Express Lane weights
        self.express_weights = {
            "0dte": {
                "hedge": 0.3,
                "liquidity": 0.5,  # Liquidity is critical for 0DTE
                "sentiment": 0.2,
            },
            "cheap_call": {
                "hedge": 0.2,
                "liquidity": 0.2,
                "sentiment": 0.6,  # Flow conviction is critical for cheap calls
            },
        }

        # Initialize ConfidenceBuilder
        confidence_config = self.config.get("confidence_builder", {})
        self.confidence_builder = ConfidenceBuilder(
            weights=confidence_config.get("timeframe_weights"),
            min_confidence_threshold=confidence_config.get("min_confidence_threshold", 0.70),
            min_alignment_threshold=confidence_config.get("min_alignment_threshold", 0.60),
        )

        logger.info(f"ComposerAgentV3 initialized | agent_weights={self.agent_weights}")

    def compose(self, suggestions: List[AgentSuggestion], timestamp: datetime) -> Dict[str, Any]:
        """Legacy compose method for backward compatibility."""
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
        strategy_source: str = "standard",  # V3: New parameter
    ) -> ComposerDecision:
        """
        Compose multi-timeframe signals into actionable decision.

        V3: Supports Express Lane routing with strategy-specific weighting.
        """
        # V3: Get strategy-specific weights
        if strategy_source in self.express_weights:
            logger.info(f"Using Express Lane weights for {strategy_source}")
            # TODO: Apply express weights to signals (would require signal metadata)

        # Step 1: Calculate confidence using ConfidenceBuilder
        confidence_score = self.confidence_builder.calculate_confidence(all_timeframe_signals)

        logger.info(
            f"Composer analyzing {symbol} ({strategy_source}) | "
            f"confidence={confidence_score.overall_confidence:.2%} | "
            f"alignment={confidence_score.alignment_score:.2%}"
        )

        # Step 2: Backward Analysis (trend confirmation)
        backward_analysis = self._backward_analysis(all_timeframe_signals, confidence_score)

        # Step 3: Forward Analysis (prediction)
        forward_analysis = self._forward_analysis(
            all_timeframe_signals,
            confidence_score,
            current_price,
            strategy_source,  # V3: Pass strategy source
        )

        # Step 4: Make go/no-go decision
        go_signal = self.confidence_builder.meets_threshold(confidence_score)

        # V3: Adjust thresholds for Express Lane
        if strategy_source == "0dte":
            # 0DTE requires higher confidence due to speed requirement
            if confidence_score.overall_confidence < 0.75:
                go_signal = False
                logger.info("0DTE signal rejected: confidence too low")
        elif strategy_source == "cheap_call":
            # Cheap calls can tolerate slightly lower confidence if flow is strong
            if confidence_score.overall_confidence >= 0.65:
                go_signal = True

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

        # V3: Override timeframe for Express Lane
        if strategy_source == "0dte":
            predicted_timeframe = "1Min"  # Force 1-minute for 0DTE
        elif strategy_source == "cheap_call":
            # Cheap calls are typically held longer
            if predicted_timeframe in ["1Min", "5Min"]:
                predicted_timeframe = "15Min"

        # Step 7: Calculate risk/reward ratio
        risk_reward_ratio = self._calculate_risk_reward(
            predicted_direction,
            predicted_timeframe,
            forward_analysis,
            strategy_source,  # V3: Pass strategy source
        )

        # Step 8: Build reasoning
        reasoning = self._build_reasoning(
            confidence_score,
            backward_analysis,
            forward_analysis,
            go_signal,
            strategy_source,  # V3: Pass strategy source
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
            strategy_source=strategy_source,  # V3: Include strategy source
            confidence_score=confidence_score,
            backward_analysis=backward_analysis,
            forward_analysis=forward_analysis,
        )

        logger.info(
            f"Composer decision: {symbol} | "
            f"GO={go_signal} | {predicted_direction} | "
            f"conf={confidence_score.overall_confidence:.2%} | "
            f"TF={predicted_timeframe} | source={strategy_source}"
        )

        return decision

    def _backward_analysis(
        self, signals: List[TimeframeSignal], confidence_score: ConfidenceScore
    ) -> Dict[str, Any]:
        """Backward analysis: Confirm trend across timeframes."""
        # Group signals by timeframe
        tf_signals = {}
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
        strategy_source: str = "standard",  # V3: New parameter
    ) -> Dict[str, Any]:
        """Forward analysis: Predict likely price movement."""
        # Calculate expected move based on dominant timeframe
        dominant_tf = confidence_score.dominant_timeframe

        # Base expected move on timeframe
        expected_move_pct = {
            "1Min": 0.005,  # 0.5%
            "5Min": 0.01,  # 1%
            "15Min": 0.015,  # 1.5%
            "30Min": 0.02,  # 2%
            "1Hour": 0.03,  # 3%
            "4Hour": 0.05,  # 5%
            "1Day": 0.08,  # 8%
        }.get(dominant_tf, 0.02)

        # V3: Adjust for strategy source
        if strategy_source == "0dte":
            # 0DTE targets smaller, faster moves
            expected_move_pct *= 0.5
        elif strategy_source == "cheap_call":
            # Cheap calls target larger moves
            expected_move_pct *= 2.0

        # Adjust by confidence
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
        self,
        direction: str,
        timeframe: str,
        forward_analysis: Dict,
        strategy_source: str = "standard",  # V3: New parameter
    ) -> float:
        """Calculate risk/reward ratio for the trade."""
        if direction == "NEUTRAL":
            return 0.0

        expected_move = forward_analysis.get("expected_move_pct", 0.02)

        # V3: Strategy-specific R:R adjustments
        if strategy_source == "0dte":
            # 0DTE: Tighter risk, smaller reward
            risk_pct = expected_move * 0.75  # 15% stop for 20% target
            reward_pct = expected_move
        elif strategy_source == "cheap_call":
            # Cheap calls: Wider risk, larger reward
            risk_pct = expected_move * 0.3  # Let it ride
            reward_pct = expected_move
        else:
            # Standard
            risk_pct = expected_move * 0.4
            reward_pct = expected_move

        if risk_pct == 0:
            return 0.0

        return reward_pct / risk_pct

    def _build_reasoning(
        self,
        confidence_score: ConfidenceScore,
        backward_analysis: Dict,
        forward_analysis: Dict,
        go_signal: bool,
        strategy_source: str = "standard",  # V3: New parameter
    ) -> str:
        """Build human-readable reasoning for the decision."""
        parts = []

        # V3: Strategy source indicator
        if strategy_source != "standard":
            parts.append(f"🚀 Express Lane: {strategy_source.upper()}")

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

    def _get_agent_weight(self, agent_name: str, strategy_source: str = "standard") -> float:
        """Get weight for an agent (with V3 Express Lane support)."""
        # V3: Use express weights if applicable
        if strategy_source in self.express_weights:
            weights = self.express_weights[strategy_source]
        else:
            weights = self.agent_weights

        if "hedge" in agent_name.lower():
            return weights.get("hedge", 0.4)
        elif "liquidity" in agent_name.lower():
            return weights.get("liquidity", 0.2)
        elif "sentiment" in agent_name.lower():
            return weights.get("sentiment", 0.4)
        return 0.33


__all__ = ["ComposerAgentV3", "ComposerDecision"]
