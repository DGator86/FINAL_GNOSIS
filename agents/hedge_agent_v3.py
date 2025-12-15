"""Hedge Agent v3 - Energy-aware interpretation with LSTM lookahead integration."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from schemas.core_schemas import AgentSuggestion, DirectionEnum, PipelineResult


class HedgeAgentV3:
    """Hedge Agent v3 with energy-aware interpretation and LSTM lookahead predictions."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_lstm = config.get("use_lstm", True)
        self.lstm_weight = config.get("lstm_weight", 0.4)  # Weight for LSTM vs energy signals
        self.var_confidence = config.get("var_confidence", 0.99)
        self.risk_floor = config.get("risk_floor", 0.2)
        logger.info(f"HedgeAgentV3 initialized (LSTM enabled: {self.use_lstm})")

    def suggest(self, pipeline_result: PipelineResult, timestamp: datetime) -> Optional[AgentSuggestion]:
        """Generate suggestion based on hedge snapshot and LSTM predictions."""
        if not pipeline_result.hedge_snapshot:
            return None

        snapshot = pipeline_result.hedge_snapshot
        min_confidence = self.config.get("min_confidence", 0.5)

        if snapshot.confidence < min_confidence:
            return None

        # Determine direction from energy asymmetry & probabilistic regimes
        energy_direction, energy_reasoning = self._get_energy_direction(snapshot)
        energy_reasoning = self._augment_reasoning_with_regime(snapshot, energy_reasoning)

        # Incorporate LSTM predictions if available
        if self.use_lstm and pipeline_result.ml_snapshot and pipeline_result.ml_snapshot.forecast:
            direction, confidence, reasoning = self._combine_lstm_and_energy(
                energy_direction=energy_direction,
                energy_reasoning=energy_reasoning,
                snapshot=snapshot,
                ml_forecast=pipeline_result.ml_snapshot.forecast,
            )
        else:
            direction = energy_direction
            reasoning = energy_reasoning
            # Adjust confidence based on movement energy
            confidence = self._risk_adjust_confidence(snapshot)

        return AgentSuggestion(
            agent_name="hedge_agent_v3",
            timestamp=timestamp,
            symbol=pipeline_result.symbol,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            target_allocation=0.0,
        )

    def _get_energy_direction(self, snapshot) -> tuple[DirectionEnum, str]:
        """Determine direction from energy asymmetry"""
        if snapshot.energy_asymmetry > 0.3:
            return DirectionEnum.LONG, f"Positive energy asymmetry ({snapshot.energy_asymmetry:.2f}), upward bias"
        elif snapshot.energy_asymmetry < -0.3:
            return DirectionEnum.SHORT, f"Negative energy asymmetry ({snapshot.energy_asymmetry:.2f}), downward bias"
        else:
            return DirectionEnum.NEUTRAL, "Energy asymmetry neutral, no clear directional bias"

    def _augment_reasoning_with_regime(self, snapshot, reasoning: str) -> str:
        if not snapshot.regime_probabilities:
            return reasoning
        top_regime = max(snapshot.regime_probabilities, key=snapshot.regime_probabilities.get)
        prob = snapshot.regime_probabilities.get(top_regime, 0.0)
        return f"{reasoning} | Regime: {top_regime} (p={prob:.2f})"

    def _combine_lstm_and_energy(
        self,
        energy_direction: DirectionEnum,
        energy_reasoning: str,
        snapshot,
        ml_forecast,
    ) -> tuple[DirectionEnum, float, str]:
        """
        Combine LSTM predictions with energy-based signals

        Returns:
            (direction, confidence, reasoning)
        """
        # Extract LSTM metadata
        lstm_metadata = ml_forecast.metadata or {}
        lstm_direction_str = lstm_metadata.get("direction", "neutral")
        lstm_confidence = ml_forecast.confidence or 0.0

        # Convert LSTM direction string to enum
        lstm_direction_map = {
            "up": DirectionEnum.LONG,
            "down": DirectionEnum.SHORT,
            "neutral": DirectionEnum.NEUTRAL,
        }
        lstm_direction = lstm_direction_map.get(lstm_direction_str, DirectionEnum.NEUTRAL)

        # Get short-term prediction (1min or 5min horizon)
        predictions_pct = lstm_metadata.get("predictions_pct", {})
        short_term_forecast = predictions_pct.get(1, predictions_pct.get(5, 0.0))

        # Calculate base confidence from movement energy
        energy_confidence = self._risk_adjust_confidence(snapshot)

        # Check agreement between LSTM and energy signals
        signals_agree = (lstm_direction == energy_direction)

        combined_direction, combined_confidence, reasoning = self._bayesian_blend(
            energy_direction,
            energy_reasoning,
            energy_confidence,
            lstm_direction,
            lstm_confidence,
            short_term_forecast,
        )

        return combined_direction, combined_confidence, reasoning

    def _bayesian_blend(
        self,
        energy_direction: DirectionEnum,
        energy_reasoning: str,
        energy_confidence: float,
        lstm_direction: DirectionEnum,
        lstm_confidence: float,
        short_term_forecast: float,
    ) -> tuple[DirectionEnum, float, str]:
        """Blend LSTM and energy signals using a Bayesian-style weighting."""
        signals_agree = lstm_direction == energy_direction
        evidence_weight = self.lstm_weight if lstm_direction != DirectionEnum.NEUTRAL else 0.2
        prior = energy_confidence
        likelihood = lstm_confidence * evidence_weight
        posterior = min(1.0, prior + likelihood - prior * likelihood)

        if signals_agree and lstm_direction != DirectionEnum.NEUTRAL:
            direction = lstm_direction
            reasoning = (
                f"LSTM & Energy agree: {lstm_direction.value.upper()}. "
                f"Energy: {energy_reasoning}. "
                f"LSTM predicts {short_term_forecast:+.2f}% (conf: {lstm_confidence:.2f})"
            )

        elif not signals_agree and lstm_confidence > 0.7:
            # LSTM has high confidence but disagrees - trust LSTM more
            direction = lstm_direction
            posterior = max(posterior, lstm_confidence * 0.9)
            reasoning = (
                f"LSTM override (high conf: {lstm_confidence:.2f}): {lstm_direction.value.upper()}. "
                f"Predicts {short_term_forecast:+.2f}% vs energy {energy_direction.value}"
            )

        elif not signals_agree and energy_confidence > 0.7:
            # Energy has high confidence but LSTM disagrees - trust energy more
            direction = energy_direction
            posterior = max(posterior, energy_confidence * 0.9)
            reasoning = (
                f"Energy override (high conf: {energy_confidence:.2f}): {energy_direction.value.upper()}. "
                f"{energy_reasoning} vs LSTM {lstm_direction.value}"
            )

        else:
            # Disagreement with low confidence - go neutral or weighted average
            direction = DirectionEnum.NEUTRAL
            posterior = (energy_confidence + lstm_confidence) / 2 * 0.7  # Penalty for uncertainty

            reasoning = (
                f"Mixed signals (low confidence). Energy: {energy_direction.value}, "
                f"LSTM: {lstm_direction.value} ({short_term_forecast:+.2f}%). Staying neutral"
            )

        return direction, min(1.0, posterior), reasoning

    def _risk_adjust_confidence(self, snapshot) -> float:
        """Apply simple VaR-inspired haircut using movement energy and liquidity friction."""
        base = snapshot.confidence * (1.0 + min(0.5, snapshot.movement_energy / 100.0))
        var_haircut = max(0.01, 1 - self.var_confidence)
        liquidity_penalty = max(0.0, snapshot.liquidity_friction - 0.2)
        jump_penalty = snapshot.jump_intensity
        adjusted = base * (1 - var_haircut) * (1 - 0.5 * liquidity_penalty) * (1 - 0.3 * jump_penalty)
        return max(self.risk_floor, min(1.0, adjusted))
