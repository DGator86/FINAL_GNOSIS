"""Liquidity Agent v1 with PPF (Past/Present/Future) analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from schemas.core_schemas import (
    AgentSuggestion,
    DirectionEnum,
    PipelineResult,
    PPFAnalysis,
    PastAnalysis,
    PresentAnalysis,
    FutureAnalysis,
)


class LiquidityAgentV1:
    """Liquidity Agent v1 for tradability assessment with PPF analysis."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        logger.info("LiquidityAgentV1 initialized")

    def suggest(self, pipeline_result: PipelineResult, timestamp: datetime) -> Optional[AgentSuggestion]:
        """Generate suggestion based on liquidity snapshot with PPF analysis."""
        if not pipeline_result.liquidity_snapshot:
            return None

        snapshot = pipeline_result.liquidity_snapshot
        min_score = self.config.get("min_liquidity_score", 0.3)

        if snapshot.liquidity_score < min_score:
            return None

        # Liquidity agent doesn't provide directional bias
        # It only confirms tradability
        confidence = snapshot.liquidity_score
        reasoning = f"Liquidity score {snapshot.liquidity_score:.2f}, spread {snapshot.bid_ask_spread:.4f}%"

        # Build PPF analysis for liquidity domain
        ppf = self._build_ppf_analysis(pipeline_result, timestamp, confidence)

        return AgentSuggestion(
            agent_name="liquidity_agent_v1",
            timestamp=timestamp,
            symbol=pipeline_result.symbol,
            direction=DirectionEnum.NEUTRAL,
            confidence=confidence,
            reasoning=reasoning,
            target_allocation=0.0,
            ppf_analysis=ppf,
        )

    def _build_ppf_analysis(
        self,
        pipeline_result: PipelineResult,
        timestamp: datetime,
        confidence: float,
    ) -> PPFAnalysis:
        """
        Build Past/Present/Future analysis for the liquidity domain.

        PAST: Historical volume patterns, liquidity trends
        PRESENT: Current liquidity score, bid-ask spread, volume, depth
        FUTURE: Expected volume changes, forecast depth
        """
        snapshot = pipeline_result.liquidity_snapshot

        # === PAST ANALYSIS ===
        past = PastAnalysis(
            # Liquidity patterns from historical data
            historical_volatility=0.0,  # Could be enhanced with historical liquidity volatility
        )

        # === PRESENT ANALYSIS ===
        present = PresentAnalysis(
            liquidity_score=snapshot.liquidity_score,
            bid_ask_spread=snapshot.bid_ask_spread,
            volume_ratio=snapshot.volume / 1_000_000 if snapshot.volume else 0.0,  # Normalize to millions
        )

        # Add percentile score if available
        if snapshot.percentile_score:
            present.iv_percentile = snapshot.percentile_score  # Reusing field for liquidity percentile

        # === FUTURE ANALYSIS ===
        future = FutureAnalysis()

        # Forecast depth if available
        if snapshot.forecast_depth:
            # Average of forecasted depth values indicates expected liquidity
            avg_forecast = sum(snapshot.forecast_depth) / len(snapshot.forecast_depth) if snapshot.forecast_depth else 0
            future.expected_volume_change = avg_forecast - snapshot.depth if snapshot.depth else 0.0

        # Liquidity friction indicates expected trading difficulty
        if snapshot.liquidity_friction:
            # Higher friction = expect worse liquidity conditions
            future.future_oi_trend = "decreasing" if snapshot.liquidity_friction > 0.5 else "stable"

        return PPFAnalysis(
            timestamp=timestamp,
            symbol=pipeline_result.symbol,
            domain="liquidity",
            past=past,
            present=present,
            future=future,
            directional_bias=0.0,  # Liquidity is non-directional
            confidence=confidence,
            time_horizon="intraday",
            reasoning=f"Liquidity PPF: Score={snapshot.liquidity_score:.2f}, Spread={snapshot.bid_ask_spread:.4f}, Vol={snapshot.volume:,.0f}",
        )
