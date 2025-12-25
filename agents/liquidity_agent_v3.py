"""Multi-Timeframe Liquidity Agent v3 - Wyckoff Enhanced.

Analyzes liquidity with full Wyckoff methodology integration:
- Volume Spread Analysis (VSA) signals
- Wyckoff phase tracking
- Seven logical events detection
- Spring/Upthrust entry signals
- Accumulation/Distribution structure awareness

Author: Super Gnosis Elite Trading System
Version: 3.0.0 - Wyckoff Integration
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from agents.confidence_builder import TimeframeSignal
from schemas.core_schemas import AgentSuggestion, DirectionEnum, PipelineResult

# Import Wyckoff components from liquidity engine
try:
    from engines.liquidity import (
        LiquidityEngineV4,
        WyckoffPhase,
        WyckoffEvent,
        VSASignal,
        MarketStructure,
        WyckoffState,
    )
    WYCKOFF_AVAILABLE = True
except ImportError:
    WYCKOFF_AVAILABLE = False
    logger.warning("Wyckoff components not available, using basic liquidity analysis")


class LiquidityAgentV3:
    """Multi-timeframe liquidity agent with Wyckoff methodology integration.
    
    This agent combines traditional order book analysis with Wyckoff's
    structural market analysis for enhanced signal generation.
    
    Features:
    - VSA-based signal confirmation
    - Phase-aware trading recommendations
    - Event-based entry/exit signals
    - Structure recognition for trend bias
    - Support/resistance from Wyckoff levels
    """
    
    # Wyckoff phase trading weights
    PHASE_WEIGHTS = {
        "phase_c": 1.3,      # Best opportunity (Spring/UTAD test)
        "phase_d": 1.2,      # Strong trend development
        "phase_e": 1.1,      # Trend continuation
        "phase_b": 0.9,      # Building cause - be cautious
        "phase_a": 0.8,      # Early phase - wait for development
        "unknown": 1.0,
        "trending_up": 1.0,
        "trending_down": 1.0,
    }
    
    # High-value events for entry signals
    ENTRY_EVENTS = {
        "spring": ("long", 0.9),
        "upthrust_after_distribution": ("short", 0.9),
        "sign_of_strength": ("long", 0.8),
        "sign_of_weakness": ("short", 0.8),
        "last_point_of_support": ("long", 0.85),
        "last_point_of_supply": ("short", 0.85),
    }
    
    # VSA confirmation signals
    VSA_BULLISH = {"no_supply", "confirmation"}
    VSA_BEARISH = {"no_demand", "confirmation"}
    VSA_WARNING = {"divergence_warning", "absorption", "stopping_volume", "climactic_action"}
    
    def __init__(
        self,
        config: Dict[str, Any],
        liquidity_engine: Optional[LiquidityEngineV4] = None
    ):
        """Initialize LiquidityAgentV3.
        
        Args:
            config: Agent configuration
            liquidity_engine: Optional LiquidityEngineV4 instance for Wyckoff state access
        """
        self.config = config
        self.liquidity_engine = liquidity_engine
        self.min_confidence = config.get("min_confidence", 0.5)
        self.imbalance_threshold = config.get("imbalance_threshold", 0.15)
        self.wyckoff_weight = config.get("wyckoff_weight", 0.4)  # Weight for Wyckoff signals
        self.enable_wyckoff = config.get("enable_wyckoff", True) and WYCKOFF_AVAILABLE
        
        logger.info(
            f"LiquidityAgentV3 initialized | "
            f"wyckoff_enabled={self.enable_wyckoff} | "
            f"wyckoff_weight={self.wyckoff_weight}"
        )
    
    def suggest(
        self,
        pipeline_result: PipelineResult,
        timestamp: datetime
    ) -> Optional[AgentSuggestion]:
        """Generate suggestion based on liquidity + Wyckoff analysis.
        
        Args:
            pipeline_result: Pipeline result with liquidity snapshot
            timestamp: Analysis timestamp
            
        Returns:
            AgentSuggestion or None
        """
        if not pipeline_result.liquidity_snapshot:
            return None

        snapshot = pipeline_result.liquidity_snapshot
        symbol = pipeline_result.symbol
        
        # Base liquidity analysis
        base_confidence = self._risk_adjust_confidence(snapshot)
        
        # Get base direction from order book
        base_direction, base_reasoning = self._analyze_order_book(snapshot)
        
        # Get Wyckoff overlay if available
        wyckoff_signal = None
        if self.enable_wyckoff and self.liquidity_engine:
            wyckoff_signal = self._get_wyckoff_signal(symbol)
        
        # Combine signals
        final_direction, final_confidence, reasoning = self._combine_signals(
            base_direction=base_direction,
            base_confidence=base_confidence,
            base_reasoning=base_reasoning,
            wyckoff_signal=wyckoff_signal,
            snapshot=snapshot
        )
        
        if final_confidence < self.min_confidence:
            return None
        
        return AgentSuggestion(
            agent_name="liquidity_agent_v3",
            timestamp=timestamp,
            symbol=symbol,
            direction=final_direction,
            confidence=final_confidence,
            reasoning=reasoning,
            target_allocation=0.0,
        )
    
    def suggest_multiframe(
        self,
        liquidity_snapshots: Dict[str, Any],
        symbol: str,
        timestamp: datetime
    ) -> List[TimeframeSignal]:
        """Generate timeframe signals with Wyckoff enhancement.
        
        Args:
            liquidity_snapshots: Dict[timeframe, LiquiditySnapshot]
            symbol: Trading symbol
            timestamp: Analysis timestamp
            
        Returns:
            List of TimeframeSignal for ConfidenceBuilder
        """
        signals = []
        
        # Get Wyckoff state for phase adjustment
        wyckoff_state = None
        if self.enable_wyckoff and self.liquidity_engine:
            wyckoff_state = self.liquidity_engine.get_wyckoff_state(symbol)
        
        for timeframe, snapshot in liquidity_snapshots.items():
            if snapshot is None:
                continue
            
            # Skip low confidence snapshots
            if hasattr(snapshot, 'confidence') and snapshot.confidence < self.min_confidence:
                continue
            
            # Base direction from order book
            direction, reasoning = self._get_timeframe_direction(snapshot)
            
            # Apply Wyckoff phase adjustment
            phase_mult = 1.0
            if wyckoff_state:
                phase_mult = self.PHASE_WEIGHTS.get(wyckoff_state.phase.value, 1.0)
                
                # Add Wyckoff context to reasoning
                reasoning = f"{reasoning} | Phase={wyckoff_state.phase.value}"
                
                # Align direction with structure
                if wyckoff_state.structure.value in ["accumulation", "re_accumulation"]:
                    if direction < 0:  # Bearish signal in accumulation
                        direction *= 0.5  # Reduce bearish confidence
                        reasoning += " (acc bias)"
                elif wyckoff_state.structure.value in ["distribution", "re_distribution"]:
                    if direction > 0:  # Bullish signal in distribution
                        direction *= 0.5  # Reduce bullish confidence
                        reasoning += " (dist bias)"
            
            # Calculate confidence
            base_conf = getattr(snapshot, 'confidence', 0.7)
            confidence = base_conf * phase_mult
            
            # Strength from depth analysis
            strength = self._calculate_strength(snapshot)
            
            signal = TimeframeSignal(
                timeframe=timeframe,
                direction=direction,
                strength=strength * phase_mult,
                confidence=min(1.0, confidence),
                reasoning=f"{timeframe}: {reasoning}"
            )
            
            signals.append(signal)
            logger.debug(
                f"LiquidityAgentV3 {timeframe}: dir={direction:+.2f} "
                f"conf={confidence:.2f} phase_mult={phase_mult:.2f}"
            )
        
        return signals
    
    def get_wyckoff_entry_signal(
        self,
        symbol: str,
        timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """Get explicit Wyckoff-based entry signal.
        
        Args:
            symbol: Trading symbol
            timestamp: Current timestamp
            
        Returns:
            Entry signal dict or None
        """
        if not self.enable_wyckoff or not self.liquidity_engine:
            return None
        
        state = self.liquidity_engine.get_wyckoff_state(symbol)
        if not state:
            return None
        
        # Check for high-value events
        current_event = state.current_event.value
        if current_event in self.ENTRY_EVENTS:
            direction, base_conf = self.ENTRY_EVENTS[current_event]
            
            # Adjust confidence based on phase
            phase_mult = self.PHASE_WEIGHTS.get(state.phase.value, 1.0)
            confidence = base_conf * phase_mult
            
            return {
                "signal": direction,
                "event": current_event,
                "phase": state.phase.value,
                "structure": state.structure.value,
                "confidence": confidence,
                "timestamp": timestamp,
                "reasoning": f"Wyckoff {current_event} in {state.phase.value}/{state.structure.value}"
            }
        
        return None
    
    def get_support_resistance(
        self,
        symbol: str
    ) -> Dict[str, List[float]]:
        """Get Wyckoff-derived support and resistance levels.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with support and resistance levels
        """
        result = {"support": [], "resistance": []}
        
        if not self.enable_wyckoff or not self.liquidity_engine:
            return result
        
        state = self.liquidity_engine.get_wyckoff_state(symbol)
        if not state or not state.range_structure:
            return result
        
        range_struct = state.range_structure
        
        # Add range boundaries
        result["support"].append(range_struct.lower_boundary)
        result["resistance"].append(range_struct.upper_boundary)
        
        # Add creek/ice levels
        if range_struct.creek_level:
            result["resistance"].append(range_struct.creek_level)
        if range_struct.ice_level:
            result["support"].append(range_struct.ice_level)
        
        return result
    
    def analyze_vsa(
        self,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Get Volume Spread Analysis for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            VSA analysis dict or None
        """
        if not self.enable_wyckoff or not self.liquidity_engine:
            return None
        
        state = self.liquidity_engine.get_wyckoff_state(symbol)
        if not state or not state.vsa_analysis:
            return None
        
        vsa = state.vsa_analysis
        
        # Determine VSA implication
        signal_value = vsa.signal.value
        if signal_value in self.VSA_BULLISH:
            implication = "bullish"
        elif signal_value in self.VSA_BEARISH:
            implication = "bearish"
        elif signal_value in self.VSA_WARNING:
            implication = "warning"
        else:
            implication = "neutral"
        
        return {
            "signal": signal_value,
            "range_type": vsa.range_type,
            "volume_type": vsa.volume_type,
            "harmony": vsa.effort_result_harmony,
            "subsequent_shift": vsa.subsequent_shift,
            "implication": implication,
            "reasoning": vsa.reasoning
        }
    
    def _get_wyckoff_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get Wyckoff signal for combining with base analysis."""
        if not self.liquidity_engine:
            return None
        
        state = self.liquidity_engine.get_wyckoff_state(symbol)
        if not state:
            return None
        
        # Get trading recommendation
        signal, confidence = self.liquidity_engine.get_trading_signal(symbol)
        
        # Convert to direction value
        if signal == "long":
            direction = 1.0
        elif signal == "short":
            direction = -1.0
        else:
            direction = 0.0
        
        return {
            "direction": direction,
            "confidence": confidence,
            "phase": state.phase.value,
            "structure": state.structure.value,
            "event": state.current_event.value,
            "bias": state.trading_bias,
            "composite_score": state.composite_score
        }
    
    def _analyze_order_book(self, snapshot) -> Tuple[DirectionEnum, str]:
        """Analyze order book for direction."""
        imbalance = getattr(snapshot, 'bid_ask_imbalance', 0.0)
        
        if imbalance > self.imbalance_threshold:
            return DirectionEnum.LONG, f"Bid pressure ({imbalance:.2f})"
        elif imbalance < -self.imbalance_threshold:
            return DirectionEnum.SHORT, f"Ask pressure ({imbalance:.2f})"
        else:
            return DirectionEnum.NEUTRAL, "Balanced book"
    
    def _get_timeframe_direction(self, snapshot) -> Tuple[float, str]:
        """Get direction value from snapshot."""
        imbalance = getattr(snapshot, 'bid_ask_imbalance', 0.0)
        
        if imbalance > self.imbalance_threshold:
            return 1.0, f"Bid pressure ({imbalance:.2f})"
        elif imbalance < -self.imbalance_threshold:
            return -1.0, f"Ask pressure ({imbalance:.2f})"
        else:
            return 0.0, "Balanced"
    
    def _calculate_strength(self, snapshot) -> float:
        """Calculate signal strength from depth."""
        total_depth = getattr(snapshot, 'depth', 100)
        return min(1.0, total_depth / 1000.0)
    
    def _combine_signals(
        self,
        base_direction: DirectionEnum,
        base_confidence: float,
        base_reasoning: str,
        wyckoff_signal: Optional[Dict[str, Any]],
        snapshot
    ) -> Tuple[DirectionEnum, float, str]:
        """Combine base liquidity and Wyckoff signals."""
        
        if not wyckoff_signal:
            # No Wyckoff data - use base only
            liquidity_asymmetry = self._liquidity_asymmetry(snapshot)
            final_conf = self._bayesian_blend(base_confidence, liquidity_asymmetry)
            reasoning = f"{base_reasoning} | asym={liquidity_asymmetry:.2f}"
            return base_direction, final_conf, reasoning
        
        # Weight the signals
        base_weight = 1.0 - self.wyckoff_weight
        wyckoff_weight = self.wyckoff_weight
        
        # Convert base direction to value
        if base_direction == DirectionEnum.LONG:
            base_dir_val = 1.0
        elif base_direction == DirectionEnum.SHORT:
            base_dir_val = -1.0
        else:
            base_dir_val = 0.0
        
        # Combine directions
        wyckoff_dir = wyckoff_signal["direction"]
        combined_dir = (base_dir_val * base_weight) + (wyckoff_dir * wyckoff_weight)
        
        # Determine final direction
        if combined_dir > 0.3:
            final_direction = DirectionEnum.LONG
        elif combined_dir < -0.3:
            final_direction = DirectionEnum.SHORT
        else:
            final_direction = DirectionEnum.NEUTRAL
        
        # Combine confidence
        wyckoff_conf = wyckoff_signal["confidence"]
        combined_conf = (base_confidence * base_weight) + (wyckoff_conf * wyckoff_weight)
        
        # Apply phase multiplier
        phase = wyckoff_signal["phase"]
        phase_mult = self.PHASE_WEIGHTS.get(phase, 1.0)
        final_conf = combined_conf * phase_mult
        
        # Build reasoning
        event = wyckoff_signal["event"]
        structure = wyckoff_signal["structure"]
        reasoning = (
            f"{base_reasoning} | Wyckoff: {phase}/{structure}"
            f"{f' ({event})' if event != 'none' else ''}"
            f" | combined_dir={combined_dir:+.2f}"
        )
        
        return final_direction, min(1.0, final_conf), reasoning
    
    def _liquidity_asymmetry(self, snapshot) -> float:
        """Calculate liquidity asymmetry."""
        total_depth = max(getattr(snapshot, 'depth', 1.0), 1e-6)
        up_depth = getattr(snapshot, "bid_depth", total_depth * 0.5)
        down_depth = getattr(snapshot, "ask_depth", total_depth * 0.5)
        return (up_depth - down_depth) / total_depth
    
    def _bayesian_blend(self, agent_conf: float, asymmetry: float) -> float:
        """Bayesian confidence blending."""
        prior = 0.6
        ml_conf = min(1.0, abs(asymmetry))
        return (agent_conf * prior + ml_conf * (1 - prior)) / 2
    
    def _risk_adjust_confidence(self, snapshot) -> float:
        """Risk-adjusted confidence calculation."""
        hist_returns = getattr(snapshot, "historical_returns", [0.0])
        position_size = getattr(snapshot, "position_size", 1.0)
        var_factor = np.std(hist_returns) * position_size if hist_returns else 0.0
        base_conf = getattr(snapshot, 'confidence', 0.7)
        return max(0.0, base_conf * (1 - var_factor))


__all__ = ['LiquidityAgentV3']
