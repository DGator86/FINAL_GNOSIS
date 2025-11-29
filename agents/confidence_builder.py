"""Confidence Building System.

Calculates compound confidence across multiple timeframes
based on signal alignment and timeframe weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class TimeframeSignal:
    """Signal for a specific timeframe."""
    timeframe: str
    direction: float  # -1.0 (bearish) to +1.0 (bullish)
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str


@dataclass
class ConfidenceScore:
    """Compound confidence score across timeframes."""
    overall_confidence: float  # 0.0 to 1.0
    weighted_direction: float  # -1.0 to +1.0
    alignment_score: float  # 0.0 to 1.0 (how aligned are timeframes)
    dominant_timeframe: str
    timeframe_breakdown: Dict[str, float]
    reasoning: str


class ConfidenceBuilder:
    """Builds compound confidence across timeframes.
    
    Higher timeframes carry more weight as they represent stronger trends.
    Confidence increases when multiple timeframes align.
    """
    
    # Default weights favor longer timeframes
    DEFAULT_WEIGHTS = {
        '1Min': 0.05,
        '5Min': 0.10,
        '15Min': 0.15,
        '30Min': 0.15,
        '1Hour': 0.20,
        '4Hour': 0.20,
        '1Day': 0.15
    }
    
    def __init__(
        self, 
        weights: Optional[Dict[str, float]] = None,
        min_confidence_threshold: float = 0.70,
        min_alignment_threshold: float = 0.60
    ):
        """Initialize ConfidenceBuilder.
        
        Args:
            weights: Custom timeframe weights (defaults to DEFAULT_WEIGHTS)
            min_confidence_threshold: Minimum confidence to trigger action
            min_alignment_threshold: Minimum alignment for high confidence
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.min_confidence_threshold = min_confidence_threshold
        self.min_alignment_threshold = min_alignment_threshold
        
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        logger.info(
            f"ConfidenceBuilder initialized | "
            f"min_confidence={min_confidence_threshold} | "
            f"weights={self.weights}"
        )
    
    def calculate_confidence(
        self,
        timeframe_signals: List[TimeframeSignal]
    ) -> ConfidenceScore:
        """Calculate compound confidence from timeframe signals.
        
        Algorithm:
        1. Weight each signal by timeframe importance
        2. Calculate weighted direction (-1 to +1)
        3. Measure alignment (how much timeframes agree)
        4. Compute overall confidence
        
        Args:
            timeframe_signals: List of signals from different timeframes
        
        Returns:
            ConfidenceScore with overall assessment
        """
        if not timeframe_signals:
            return ConfidenceScore(
                overall_confidence=0.0,
                weighted_direction=0.0,
                alignment_score=0.0,
                dominant_timeframe='None',
                timeframe_breakdown={},
                reasoning="No timeframe signals available"
            )
        
        # Calculate weighted direction
        weighted_direction = 0.0
        total_weight = 0.0
        timeframe_breakdown = {}
        
        for signal in timeframe_signals:
            weight = self.weights.get(signal.timeframe, 0.0)
            contribution = signal.direction * signal.strength * signal.confidence * weight
            weighted_direction += contribution
            total_weight += weight
            timeframe_breakdown[signal.timeframe] = contribution
        
        if total_weight > 0:
            weighted_direction /= total_weight
        
        # Calculate alignment score (how much timeframes agree)
        alignment_score = self._calculate_alignment(timeframe_signals)
        
        # Determine dominant timeframe (highest weighted contribution)
        dominant_timeframe = max(
            timeframe_breakdown.items(),
            key=lambda x: abs(x[1])
        )[0] if timeframe_breakdown else 'None'
        
        # Calculate overall confidence
        # Factors:
        # 1. Absolute weighted direction (stronger = higher confidence)
        # 2. Alignment score (more alignment = higher confidence)
        # 3. Number of agreeing timeframes
        
        base_confidence = abs(weighted_direction)
        alignment_bonus = alignment_score * 0.3
        overall_confidence = min(1.0, base_confidence + alignment_bonus)
        
        # Generate reasoning
        direction_label = "Bullish" if weighted_direction > 0.3 else \
                         "Bearish" if weighted_direction < -0.3 else "Neutral"
        alignment_label = "Strong" if alignment_score > 0.7 else \
                         "Moderate" if alignment_score > 0.5 else "Weak"
        
        reasoning = (
            f"{direction_label} bias (direction={weighted_direction:.2f}) | "
            f"{alignment_label} alignment ({alignment_score:.1%}) | "
            f"Dominant: {dominant_timeframe} | "
            f"{len(timeframe_signals)} timeframes analyzed"
        )
        
        return ConfidenceScore(
            overall_confidence=overall_confidence,
            weighted_direction=weighted_direction,
            alignment_score=alignment_score,
            dominant_timeframe=dominant_timeframe,
            timeframe_breakdown=timeframe_breakdown,
            reasoning=reasoning
        )
    
    def _calculate_alignment(self, signals: List[TimeframeSignal]) -> float:
        """Calculate how aligned timeframes are.
        
        Returns:
            Alignment score from 0.0 (conflicting) to 1.0 (perfect alignment)
        """
        if len(signals) < 2:
            return 1.0
        
        # Get directions (ignoring neutral signals)
        directions = [s.direction for s in signals if abs(s.direction) > 0.1]
        
        if not directions:
            return 0.5  # All neutral
        
        # Calculate variance in directions
        # Low variance = high alignment
        mean_direction = sum(directions) / len(directions)
        variance = sum((d - mean_direction) ** 2 for d in directions) / len(directions)
        
        # Convert variance to alignment score
        # variance ranges from 0 (perfect) to 4 (maximum disagreement)
        alignment = max(0.0, 1.0 - (variance / 4.0))
        
        # Bonus for all timeframes agreeing on direction
        all_bullish = all(d > 0.1 for d in directions)
        all_bearish = all(d < -0.1 for d in directions)
        if all_bullish or all_bearish:
            alignment = min(1.0, alignment + 0.2)
        
        return alignment
    
    def meets_threshold(self, score: ConfidenceScore) -> bool:
        """Check if confidence score meets minimum thresholds.
        
        Returns:
            True if confidence and alignment are sufficient for action
        """
        return (
            score.overall_confidence >= self.min_confidence_threshold and
            score.alignment_score >= self.min_alignment_threshold
        )
    
    def get_recommended_timeframe(self, score: ConfidenceScore) -> str:
        """Get recommended trade timeframe based on dominant signals.
        
        Args:
            score: ConfidenceScore to analyze
        
        Returns:
            Recommended timeframe for trade duration (e.g., '4Hour', '1Day')
        """
        # Use dominant timeframe as recommended duration
        return score.dominant_timeframe


__all__ = ['ConfidenceBuilder', 'TimeframeSignal', 'ConfidenceScore']
