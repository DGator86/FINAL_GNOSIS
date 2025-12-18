"""Composer Agent v1 - Consensus building with MTF and PPF integration."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from schemas.core_schemas import (
    AgentSuggestion,
    DirectionEnum,
    MTFAnalysis,
    PPFAnalysis,
    PastAnalysis,
    PresentAnalysis,
    FutureAnalysis,
)


class ComposerAgentV1:
    """
    Composer Agent v1 for building consensus from agent suggestions.

    Integrates:
    - Agent PPF analyses (Past/Present/Future from each domain)
    - MTF analysis (Multi-timeframe alignment)
    - Weighted consensus direction

    Outputs a comprehensive consensus that the Trade Agent uses for strategy selection.
    """

    def __init__(self, weights: Any, config: Dict[str, Any]):
        self.weights = weights if hasattr(weights, '__dict__') else type('obj', (object,), weights)
        self.config = config
        logger.info("ComposerAgentV1 initialized")

    def compose(
        self,
        suggestions: List[AgentSuggestion],
        timestamp: datetime,
        mtf_analysis: Optional[MTFAnalysis] = None,
    ) -> Dict[str, Any]:
        """
        Build consensus from agent suggestions with MTF and PPF integration.

        Args:
            suggestions: List of agent suggestions (each with PPF analysis)
            timestamp: Composition timestamp
            mtf_analysis: Multi-timeframe analysis (optional)

        Returns:
            Comprehensive consensus dictionary for Trade Agent
        """
        if not suggestions:
            return {
                "direction": DirectionEnum.NEUTRAL.value,
                "confidence": 0.0,
                "reasoning": "No suggestions available",
                "mtf_alignment": 0.0,
                "aggregated_ppf": None,
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
                "mtf_alignment": 0.0,
                "aggregated_ppf": None,
            }

        # Calculate base consensus direction
        consensus_value = weighted_direction / total_weight

        if consensus_value > 0.3:
            direction = DirectionEnum.LONG
        elif consensus_value < -0.3:
            direction = DirectionEnum.SHORT
        else:
            direction = DirectionEnum.NEUTRAL

        # Base confidence
        confidence = min(1.0, abs(consensus_value))

        # === MTF INTEGRATION ===
        mtf_alignment = 0.0
        mtf_boost = 1.0
        strongest_timeframe = ""
        mtf_direction = "neutral"

        if mtf_analysis:
            mtf_alignment = mtf_analysis.alignment_score
            strongest_timeframe = mtf_analysis.dominant_timeframe
            mtf_direction = mtf_analysis.overall_direction

            # Boost or penalize confidence based on MTF alignment
            if mtf_alignment > 0.7:
                # Strong alignment - boost confidence
                mtf_boost = 1.2
            elif mtf_alignment < 0.3:
                # Conflicting signals - reduce confidence
                mtf_boost = 0.7

            # Check if MTF agrees with agent consensus
            mtf_agrees = (
                (mtf_direction == "long" and direction == DirectionEnum.LONG) or
                (mtf_direction == "short" and direction == DirectionEnum.SHORT) or
                (mtf_direction == "neutral" and direction == DirectionEnum.NEUTRAL)
            )

            if mtf_agrees:
                mtf_boost *= 1.1  # Extra boost for agreement
            elif mtf_direction != "neutral" and direction != DirectionEnum.NEUTRAL:
                # MTF disagrees - significant confidence penalty
                mtf_boost *= 0.6

        # Apply MTF boost to confidence
        adjusted_confidence = min(1.0, confidence * mtf_boost)

        # === PPF AGGREGATION ===
        aggregated_ppf = self._aggregate_ppf_analyses(suggestions, timestamp)

        # === STRATEGY HINTS FROM MTF ===
        best_strategy = ""
        best_expiry = ""
        if mtf_analysis and mtf_analysis.signals:
            # Find the strongest signal's strategy recommendation
            strongest_signal = max(
                mtf_analysis.signals,
                key=lambda s: abs(s.strength) * s.confidence,
                default=None
            )
            if strongest_signal:
                best_strategy = strongest_signal.strategy
                best_expiry = strongest_signal.strategy_details

        reasoning = (
            f"Consensus from {len(suggestions)} agents: {consensus_value:.2f}. "
            f"MTF alignment: {mtf_alignment:.0%} ({mtf_direction}). "
            f"Strongest TF: {strongest_timeframe}"
        )

        return {
            "direction": direction.value,
            "confidence": adjusted_confidence,
            "consensus_value": consensus_value,
            "num_agents": len(suggestions),
            "reasoning": reasoning,

            # MTF data for Trade Agent
            "mtf_alignment": mtf_alignment,
            "mtf_direction": mtf_direction,
            "strongest_timeframe": strongest_timeframe,
            "mtf_agrees_with_consensus": mtf_agrees if mtf_analysis else False,

            # Strategy hints from MTF
            "suggested_strategy": best_strategy,
            "suggested_expiry": best_expiry,

            # Aggregated PPF for Trade Agent
            "aggregated_ppf": aggregated_ppf.model_dump() if aggregated_ppf else None,

            # Raw agent PPF for detailed analysis
            "agent_ppf_summaries": [
                {
                    "agent": s.agent_name,
                    "direction": s.direction.value,
                    "confidence": s.confidence,
                    "ppf_domain": s.ppf_analysis.domain if s.ppf_analysis else None,
                    "ppf_bias": s.ppf_analysis.directional_bias if s.ppf_analysis else 0.0,
                }
                for s in suggestions
            ],
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

    def _aggregate_ppf_analyses(
        self,
        suggestions: List[AgentSuggestion],
        timestamp: datetime,
    ) -> Optional[PPFAnalysis]:
        """
        Aggregate PPF analyses from all agents into a unified view.

        Combines Past, Present, Future from all domains into a single PPFAnalysis.
        """
        ppf_analyses = [s.ppf_analysis for s in suggestions if s.ppf_analysis]

        if not ppf_analyses:
            return None

        # Aggregate PAST
        all_support = []
        all_resistance = []
        all_regimes = []
        total_hist_vol = 0.0

        for ppf in ppf_analyses:
            all_support.extend(ppf.past.support_levels)
            all_resistance.extend(ppf.past.resistance_levels)
            all_regimes.extend(ppf.past.regime_history)
            total_hist_vol += ppf.past.historical_volatility

        aggregated_past = PastAnalysis(
            support_levels=sorted(set(all_support))[-5:] if all_support else [],  # Top 5 support
            resistance_levels=sorted(set(all_resistance))[:5] if all_resistance else [],  # Top 5 resistance
            regime_history=list(dict.fromkeys(all_regimes))[:5],  # Unique recent regimes
            historical_volatility=total_hist_vol / len(ppf_analyses) if ppf_analyses else 0.0,
        )

        # Aggregate PRESENT (average where applicable)
        n = len(ppf_analyses)
        aggregated_present = PresentAnalysis(
            volatility=sum(p.present.volatility for p in ppf_analyses) / n,
            gamma_exposure=sum(p.present.gamma_exposure for p in ppf_analyses) / n,
            vanna_exposure=sum(p.present.vanna_exposure for p in ppf_analyses) / n,
            charm_exposure=sum(p.present.charm_exposure for p in ppf_analyses) / n,
            liquidity_score=sum(p.present.liquidity_score for p in ppf_analyses) / n,
            bid_ask_spread=sum(p.present.bid_ask_spread for p in ppf_analyses) / n,
            news_sentiment=sum(p.present.news_sentiment for p in ppf_analyses) / n,
        )

        # Aggregate FUTURE (average projections, take max for events)
        aggregated_future = FutureAnalysis(
            projected_move_1m=sum(p.future.projected_move_1m for p in ppf_analyses) / n,
            projected_move_5m=sum(p.future.projected_move_5m for p in ppf_analyses) / n,
            projected_move_15m=sum(p.future.projected_move_15m for p in ppf_analyses) / n,
            projected_move_60m=sum(p.future.projected_move_60m for p in ppf_analyses) / n,
            move_confidence=sum(p.future.move_confidence for p in ppf_analyses) / n,
            charm_decay_impact=sum(p.future.charm_decay_impact for p in ppf_analyses) / n,
        )

        # Aggregate directional bias
        total_bias = sum(p.directional_bias for p in ppf_analyses)
        avg_bias = total_bias / n
        avg_confidence = sum(p.confidence for p in ppf_analyses) / n

        # Get symbol from first analysis
        symbol = ppf_analyses[0].symbol if ppf_analyses else ""

        return PPFAnalysis(
            timestamp=timestamp,
            symbol=symbol,
            domain="aggregated",
            past=aggregated_past,
            present=aggregated_present,
            future=aggregated_future,
            directional_bias=avg_bias,
            confidence=avg_confidence,
            time_horizon="intraday",
            reasoning=f"Aggregated PPF from {n} agents",
        )
