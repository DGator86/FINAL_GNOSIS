"""Multi-Timeframe Liquidity Agent v3 - Enhanced with 0DTE scalp assessment."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from agents.confidence_builder import TimeframeSignal
from schemas.core_schemas import AgentSuggestion, DirectionEnum, PipelineResult


class LiquidityAgentV3:
    """
    Multi-timeframe liquidity agent with depth analysis.

    New V3 Features:
    - 0DTE scalp feasibility assessment
    - Enhanced gamma squeeze awareness
    - Improved depth scoring
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_confidence = config.get("min_confidence", 0.5)
        self.imbalance_threshold = config.get("imbalance_threshold", 0.15)
        logger.info("LiquidityAgentV3 (multi-timeframe + 0DTE) initialized")

    def suggest(
        self, pipeline_result: PipelineResult, timestamp: datetime
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
            agent_name="liquidity_agent_v3",
            timestamp=timestamp,
            symbol=pipeline_result.symbol,
            direction=direction,
            confidence=snapshot.confidence,
            reasoning=reasoning,
            target_allocation=0.0,
        )

    def suggest_multiframe(
        self, liquidity_snapshots: Dict[str, Any], symbol: str, timestamp: datetime
    ) -> List[TimeframeSignal]:
        """Generate timeframe signals from multi-timeframe liquidity data."""
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
            strength = min(1.0, getattr(snapshot, "total_depth", 100) / 1000.0)

            # Confidence from snapshot
            confidence = snapshot.confidence

            signal = TimeframeSignal(
                timeframe=timeframe,
                direction=direction,
                strength=strength,
                confidence=confidence,
                reasoning=f"{timeframe}: {reasoning} | depth={strength:.2f}",
            )

            signals.append(signal)
            logger.debug(f"LiquidityAgent {timeframe} signal: {direction:+.2f} @ {confidence:.2f}")

        return signals

    def assess_scalp_feasibility(self, liquidity_snapshot: Any, strategy_source: str) -> bool:
        """
        V3: Assess if 0DTE scalping is feasible.

        Args:
            liquidity_snapshot: Current liquidity data
            strategy_source: "0dte", "cheap_call", or "standard"

        Returns:
            True if scalping is feasible
        """
        if strategy_source != "0dte":
            return True  # Not a scalp, no special check needed

        if not liquidity_snapshot:
            return False

        # Check spread (must be tight for scalping)
        if liquidity_snapshot.bid_ask_spread > 0.3:  # 0.3% max spread
            logger.warning(
                f"Spread too wide for 0DTE scalp: {liquidity_snapshot.bid_ask_spread:.2%}"
            )
            return False

        # Check depth (must have sufficient depth)
        min_depth = 500  # Minimum depth for scalping
        total_depth = getattr(liquidity_snapshot, "total_depth", 0)

        if total_depth < min_depth:
            logger.warning(f"Insufficient depth for 0DTE scalp: {total_depth}")
            return False

        # Check liquidity score
        if liquidity_snapshot.liquidity_score < 0.7:
            logger.warning(
                f"Low liquidity score for 0DTE scalp: {liquidity_snapshot.liquidity_score:.2f}"
            )
            return False

        logger.info(
            f"0DTE scalp feasible: spread={liquidity_snapshot.bid_ask_spread:.2%}, depth={total_depth}"
        )
        return True

    def detect_support_resistance(
        self, liquidity_snapshots: Dict[str, Any], current_price: float
    ) -> Dict[str, Any]:
        """Detect support/resistance levels across timeframes."""
        support_levels = []
        resistance_levels = []

        for timeframe, snapshot in liquidity_snapshots.items():
            if snapshot is None:
                continue

            # Support = strong bid depth below current price
            if hasattr(snapshot, "bid_depth") and snapshot.bid_depth > 0:
                support_levels.append(
                    {
                        "timeframe": timeframe,
                        "price": current_price * 0.98,
                        "strength": snapshot.bid_depth,
                    }
                )

            # Resistance = strong ask depth above current price
            if hasattr(snapshot, "ask_depth") and snapshot.ask_depth > 0:
                resistance_levels.append(
                    {
                        "timeframe": timeframe,
                        "price": current_price * 1.02,
                        "strength": snapshot.ask_depth,
                    }
                )

        return {
            "support_levels": sorted(support_levels, key=lambda x: x["strength"], reverse=True),
            "resistance_levels": sorted(
                resistance_levels, key=lambda x: x["strength"], reverse=True
            ),
            "has_strong_support": len(support_levels) >= 3,
            "has_strong_resistance": len(resistance_levels) >= 3,
        }


__all__ = ["LiquidityAgentV3"]
