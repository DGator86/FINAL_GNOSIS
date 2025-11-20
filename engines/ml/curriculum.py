"""Curriculum-driven reinforcement-style evaluator inspired by Rogue and Deep RL tutorials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class PolicySignal:
    action: str
    risk_multiplier: float
    rationale: str
    curriculum_stage: str


class CurriculumRLEvaluator:
    """Heuristic policy coordinator that mimics curriculum-based RL progression."""

    def __init__(self, stages: Optional[List[str]] = None) -> None:
        self.stages = stages or ["warmup", "stabilize", "expand", "explore"]
        self.current_stage_index = 0

    @property
    def current_stage(self) -> str:
        return self.stages[self.current_stage_index]

    def advance_stage(self) -> None:
        if self.current_stage_index < len(self.stages) - 1:
            self.current_stage_index += 1
            logger.info(f"Advancing curriculum to stage {self.current_stage}")

    def recommend(
        self,
        forecast: Dict[str, Any],
        similarity: Dict[str, Any],
        anomaly: Dict[str, Any],
        sentiment_score: float,
    ) -> PolicySignal:
        """Generate a policy suggestion given model diagnostics."""

        anomaly_flagged = anomaly.get("flagged", False)
        similarity_score = float(similarity.get("similarity", 0.0))
        forecast_values: List[float] = forecast.get("forecast", []) if forecast else []
        forecast_conf = float(forecast.get("confidence", 0.0)) if forecast else 0.0
        trend = 0.0
        if len(forecast_values) >= 2:
            trend = forecast_values[-1] - forecast_values[0]

        action = "hold"
        risk_multiplier = 1.0
        rationale_parts = []

        if anomaly_flagged:
            action = "pause"
            risk_multiplier = 0.0
            rationale_parts.append("anomaly flagged by isolation forest")
        elif forecast_conf > 0.6 and trend != 0.0:
            if trend > 0 and sentiment_score >= 0:
                action = "long_bias"
                rationale_parts.append("forecast uptrend aligns with sentiment")
            elif trend < 0 and sentiment_score <= 0:
                action = "short_bias"
                rationale_parts.append("forecast downtrend aligns with sentiment")
            risk_multiplier = min(1.5, 1.0 + similarity_score)
        else:
            rationale_parts.append("insufficient confidence; staying neutral")

        if similarity_score > 0.8 and not anomaly_flagged:
            self.advance_stage()
        elif similarity_score < 0.4 and self.current_stage_index > 0:
            self.current_stage_index -= 1
            logger.info(f"Regressing curriculum to stage {self.current_stage}")

        rationale = "; ".join(rationale_parts) if rationale_parts else "baseline policy"
        return PolicySignal(
            action=action,
            risk_multiplier=risk_multiplier,
            rationale=rationale,
            curriculum_stage=self.current_stage,
        )
