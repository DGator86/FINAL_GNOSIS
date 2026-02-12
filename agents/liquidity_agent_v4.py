"""Unified Liquidity Agent V4 - Wyckoff + ICT Combined.

This agent combines both Wyckoff and ICT methodologies for comprehensive
market structure analysis and high-confluence trading signals.

Wyckoff Analysis:
- Volume Spread Analysis (VSA)
- Phase tracking (A-E)
- Event detection (Spring, Upthrust, SOS, SOW, etc.)
- Accumulation/Distribution structure

ICT Analysis:
- Swing points & liquidity levels
- Fair Value Gaps (FVG)
- Order Blocks
- Premium/Discount zones & OTE
- Daily bias
- Liquidity sweeps

Signal Generation:
- Confluence scoring from both methodologies
- Phase-aligned ICT entries
- Structure-confirmed FVG/OB trades

Author: Super Gnosis Elite Trading System
Version: 4.0.0 - Unified Wyckoff + ICT
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from agents.confidence_builder import TimeframeSignal
from schemas.core_schemas import AgentSuggestion, DirectionEnum, PipelineResult

# Import Wyckoff components
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
    logger.warning("Wyckoff components not available")

# Import ICT components
try:
    from engines.liquidity import (
        ICTEngine,
        ICTSnapshot,
        FairValueGap,
        OrderBlock,
        DailyBias,
        ZoneType,
        FVGType,
        FVGStatus,
        LiquidityType,
    )
    ICT_AVAILABLE = True
except ImportError:
    ICT_AVAILABLE = False
    logger.warning("ICT components not available")


class LiquidityAgentV4:
    """Unified liquidity agent combining Wyckoff and ICT methodologies.
    
    This agent provides the most comprehensive market structure analysis
    by combining classic Wyckoff concepts with modern ICT trading strategies.
    
    Signal Confluence Factors:
    - Wyckoff Phase (25%): Phase C/D entries preferred
    - Wyckoff VSA (15%): Volume confirmation
    - ICT Daily Bias (15%): Directional alignment
    - ICT Zone (15%): Premium/Discount positioning
    - ICT FVG (15%): Fair Value Gap proximity
    - ICT Order Block (10%): Institutional order flow
    - ICT Liquidity Sweep (5%): Stop hunt reversals
    """
    
    # Wyckoff phase trading multipliers
    WYCKOFF_PHASE_MULT = {
        "phase_c": 1.3,      # Best entry phase (Spring/UTAD)
        "phase_d": 1.2,      # Trend development
        "phase_e": 1.1,      # Trend continuation
        "phase_b": 0.85,     # Building cause - be patient
        "phase_a": 0.7,      # Early phase - wait
        "unknown": 1.0,
        "trending_up": 1.0,
        "trending_down": 1.0,
    }
    
    # Wyckoff events with high entry value
    WYCKOFF_ENTRY_EVENTS = {
        "spring": ("long", 0.9),
        "upthrust_after_distribution": ("short", 0.9),
        "sign_of_strength": ("long", 0.8),
        "sign_of_weakness": ("short", 0.8),
        "last_point_of_support": ("long", 0.85),
        "last_point_of_supply": ("short", 0.85),
    }
    
    # ICT entry events
    ICT_ENTRY_EVENTS = {
        "fvg_entry": 0.15,
        "order_block_entry": 0.10,
        "ote_entry": 0.15,
        "liquidity_sweep_reversal": 0.10,
    }
    
    def __init__(
        self,
        config: Dict[str, Any],
        wyckoff_engine: Optional[LiquidityEngineV4] = None,
        ict_engine: Optional[ICTEngine] = None,
    ):
        """Initialize LiquidityAgentV4.
        
        Args:
            config: Agent configuration
            wyckoff_engine: Optional LiquidityEngineV4 for Wyckoff analysis
            ict_engine: Optional ICTEngine for ICT analysis
        """
        self.config = config
        self.wyckoff_engine = wyckoff_engine
        self.ict_engine = ict_engine
        
        # Configuration
        self.min_confidence = config.get("min_confidence", 0.5)
        self.wyckoff_weight = config.get("wyckoff_weight", 0.4)
        self.ict_weight = config.get("ict_weight", 0.4)
        self.base_weight = config.get("base_weight", 0.2)
        
        # Enable flags
        self.enable_wyckoff = config.get("enable_wyckoff", True) and WYCKOFF_AVAILABLE
        self.enable_ict = config.get("enable_ict", True) and ICT_AVAILABLE
        
        logger.info(
            f"LiquidityAgentV4 initialized | "
            f"wyckoff={self.enable_wyckoff} | "
            f"ict={self.enable_ict} | "
            f"weights: wyckoff={self.wyckoff_weight}, ict={self.ict_weight}"
        )
    
    def suggest(
        self,
        pipeline_result: PipelineResult,
        timestamp: datetime,
        ict_snapshot: Optional[ICTSnapshot] = None,
    ) -> Optional[AgentSuggestion]:
        """Generate unified suggestion combining Wyckoff and ICT.
        
        Args:
            pipeline_result: Pipeline result with liquidity snapshot
            timestamp: Analysis timestamp
            ict_snapshot: Optional pre-computed ICT snapshot
            
        Returns:
            AgentSuggestion or None
        """
        if not pipeline_result.liquidity_snapshot:
            return None
        
        symbol = pipeline_result.symbol
        snapshot = pipeline_result.liquidity_snapshot
        
        # Gather signals from all sources
        wyckoff_signal = self._get_wyckoff_signal(symbol) if self.enable_wyckoff else None
        ict_signal = self._get_ict_signal(symbol, ict_snapshot) if self.enable_ict else None
        base_signal = self._get_base_signal(snapshot)
        
        # Combine signals
        final_direction, final_confidence, reasoning = self._combine_all_signals(
            wyckoff_signal=wyckoff_signal,
            ict_signal=ict_signal,
            base_signal=base_signal,
        )
        
        if final_confidence < self.min_confidence:
            logger.debug(f"LiquidityAgentV4 {symbol}: confidence {final_confidence:.0%} below threshold")
            return None
        
        return AgentSuggestion(
            agent_name="liquidity_agent_v4",
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
        timestamp: datetime,
        ict_snapshot: Optional[ICTSnapshot] = None,
    ) -> List[TimeframeSignal]:
        """Generate multi-timeframe signals with Wyckoff + ICT.
        
        Args:
            liquidity_snapshots: Dict[timeframe, LiquiditySnapshot]
            symbol: Trading symbol
            timestamp: Analysis timestamp
            ict_snapshot: Optional ICT analysis snapshot
            
        Returns:
            List of TimeframeSignal for ConfidenceBuilder
        """
        signals = []
        
        # Get Wyckoff state for phase context
        wyckoff_state = None
        if self.enable_wyckoff and self.wyckoff_engine:
            wyckoff_state = self.wyckoff_engine.get_wyckoff_state(symbol)
        
        # Get ICT state for confluence
        ict_state = None
        if self.enable_ict and self.ict_engine:
            ict_state = self.ict_engine.get_state(symbol)
        elif ict_snapshot:
            ict_state = ict_snapshot
        
        for timeframe, snapshot in liquidity_snapshots.items():
            if snapshot is None:
                continue
            
            # Calculate combined direction and confidence
            direction, strength, confidence, reasoning = self._analyze_timeframe(
                timeframe=timeframe,
                snapshot=snapshot,
                wyckoff_state=wyckoff_state,
                ict_state=ict_state,
            )
            
            if confidence < self.min_confidence:
                continue
            
            signal = TimeframeSignal(
                timeframe=timeframe,
                direction=direction,
                strength=strength,
                confidence=confidence,
                reasoning=f"{timeframe}: {reasoning}"
            )
            
            signals.append(signal)
            logger.debug(
                f"LiquidityAgentV4 {timeframe}: dir={direction:+.2f} "
                f"conf={confidence:.2f}"
            )
        
        return signals
    
    def get_confluence_analysis(
        self,
        symbol: str,
        ict_snapshot: Optional[ICTSnapshot] = None,
    ) -> Dict[str, Any]:
        """Get detailed confluence analysis from both methodologies.
        
        Args:
            symbol: Trading symbol
            ict_snapshot: Optional ICT snapshot
            
        Returns:
            Detailed confluence analysis
        """
        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "wyckoff": {},
            "ict": {},
            "confluence": {},
            "recommendation": None,
        }
        
        # Wyckoff Analysis
        if self.enable_wyckoff and self.wyckoff_engine:
            wyckoff_state = self.wyckoff_engine.get_wyckoff_state(symbol)
            if wyckoff_state:
                analysis["wyckoff"] = {
                    "phase": wyckoff_state.phase.value,
                    "structure": wyckoff_state.structure.value,
                    "current_event": wyckoff_state.current_event.value,
                    "trading_bias": wyckoff_state.trading_bias,
                    "confidence": wyckoff_state.confidence,
                    "composite_score": wyckoff_state.composite_score,
                    "vsa_signal": wyckoff_state.vsa_analysis.signal.value if wyckoff_state.vsa_analysis else None,
                }
        
        # ICT Analysis
        ict_state = ict_snapshot
        if not ict_state and self.enable_ict and self.ict_engine:
            ict_state = self.ict_engine.get_state(symbol)
        
        if ict_state:
            analysis["ict"] = {
                "current_zone": ict_state.current_zone.value if ict_state.current_zone else None,
                "in_ote": ict_state.in_ote,
                "daily_bias": ict_state.daily_bias.bias.value if ict_state.daily_bias else None,
                "daily_bias_confidence": ict_state.daily_bias.confidence if ict_state.daily_bias else 0,
                "bullish_fvgs": len(ict_state.bullish_fvgs),
                "bearish_fvgs": len(ict_state.bearish_fvgs),
                "unfilled_fvgs": len([f for f in ict_state.bullish_fvgs + ict_state.bearish_fvgs 
                                      if f.status == FVGStatus.UNFILLED]),
                "bullish_order_blocks": len(ict_state.bullish_order_blocks),
                "bearish_order_blocks": len(ict_state.bearish_order_blocks),
                "recent_sweeps": len(ict_state.recent_sweeps),
                "entry_signal": ict_state.entry_signal,
                "entry_confidence": ict_state.entry_confidence,
            }
        
        # Confluence Analysis
        confluence = self._calculate_confluence(analysis["wyckoff"], analysis["ict"])
        analysis["confluence"] = confluence
        
        # Generate recommendation
        if confluence["score"] >= 0.6:
            analysis["recommendation"] = {
                "action": confluence["direction"],
                "confidence": confluence["score"],
                "reasoning": confluence["reasoning"],
                "entry_type": confluence["best_entry_type"],
            }
        
        return analysis
    
    def get_entry_setups(
        self,
        symbol: str,
        ict_snapshot: Optional[ICTSnapshot] = None,
    ) -> List[Dict[str, Any]]:
        """Get specific entry setups from combined analysis.
        
        Args:
            symbol: Trading symbol
            ict_snapshot: Optional ICT snapshot
            
        Returns:
            List of entry setup dictionaries
        """
        setups = []
        
        # Get states
        wyckoff_state = None
        if self.enable_wyckoff and self.wyckoff_engine:
            wyckoff_state = self.wyckoff_engine.get_wyckoff_state(symbol)
        
        ict_state = ict_snapshot
        if not ict_state and self.enable_ict and self.ict_engine:
            ict_state = self.ict_engine.get_state(symbol)
        
        # 1. Wyckoff Event Entries
        if wyckoff_state and wyckoff_state.current_event.value in self.WYCKOFF_ENTRY_EVENTS:
            direction, base_conf = self.WYCKOFF_ENTRY_EVENTS[wyckoff_state.current_event.value]
            phase_mult = self.WYCKOFF_PHASE_MULT.get(wyckoff_state.phase.value, 1.0)
            
            setup = {
                "type": "wyckoff_event",
                "direction": direction,
                "confidence": base_conf * phase_mult,
                "event": wyckoff_state.current_event.value,
                "phase": wyckoff_state.phase.value,
                "structure": wyckoff_state.structure.value,
                "reasoning": f"Wyckoff {wyckoff_state.current_event.value} in {wyckoff_state.phase.value}",
            }
            
            # Boost if ICT confirms
            if ict_state:
                if direction == "long" and ict_state.current_zone == ZoneType.DISCOUNT:
                    setup["confidence"] *= 1.15
                    setup["reasoning"] += " + ICT discount zone"
                elif direction == "short" and ict_state.current_zone == ZoneType.PREMIUM:
                    setup["confidence"] *= 1.15
                    setup["reasoning"] += " + ICT premium zone"
            
            setups.append(setup)
        
        # 2. ICT FVG Entries
        if ict_state:
            # Check for price at unfilled FVG
            for fvg in ict_state.bullish_fvgs:
                if fvg.status == FVGStatus.UNFILLED:
                    setup = {
                        "type": "ict_fvg",
                        "direction": "long",
                        "confidence": 0.65,
                        "fvg_type": "BISI",
                        "fvg_range": f"{fvg.low:.2f} - {fvg.high:.2f}",
                        "consequent_encroachment": fvg.consequent_encroachment,
                        "reasoning": f"Bullish FVG at {fvg.low:.2f}-{fvg.high:.2f}",
                    }
                    
                    # Boost if Wyckoff confirms
                    if wyckoff_state:
                        if wyckoff_state.structure.value in ["accumulation", "re_accumulation"]:
                            setup["confidence"] *= 1.2
                            setup["reasoning"] += f" + Wyckoff {wyckoff_state.structure.value}"
                    
                    setups.append(setup)
            
            for fvg in ict_state.bearish_fvgs:
                if fvg.status == FVGStatus.UNFILLED:
                    setup = {
                        "type": "ict_fvg",
                        "direction": "short",
                        "confidence": 0.65,
                        "fvg_type": "SIBI",
                        "fvg_range": f"{fvg.low:.2f} - {fvg.high:.2f}",
                        "consequent_encroachment": fvg.consequent_encroachment,
                        "reasoning": f"Bearish FVG at {fvg.low:.2f}-{fvg.high:.2f}",
                    }
                    
                    if wyckoff_state:
                        if wyckoff_state.structure.value in ["distribution", "re_distribution"]:
                            setup["confidence"] *= 1.2
                            setup["reasoning"] += f" + Wyckoff {wyckoff_state.structure.value}"
                    
                    setups.append(setup)
        
        # 3. ICT Order Block Entries
        if ict_state:
            for ob in ict_state.bullish_order_blocks:
                if not ob.is_mitigated:
                    setup = {
                        "type": "ict_order_block",
                        "direction": "long",
                        "confidence": 0.7 if ob.is_high_probability else 0.55,
                        "ob_type": ob.type.value,
                        "ob_range": f"{ob.low:.2f} - {ob.high:.2f}",
                        "mean_threshold": ob.mean_threshold,
                        "reasoning": f"Bullish OB ({ob.type.value}) at {ob.low:.2f}-{ob.high:.2f}",
                    }
                    setups.append(setup)
            
            for ob in ict_state.bearish_order_blocks:
                if not ob.is_mitigated:
                    setup = {
                        "type": "ict_order_block",
                        "direction": "short",
                        "confidence": 0.7 if ob.is_high_probability else 0.55,
                        "ob_type": ob.type.value,
                        "ob_range": f"{ob.low:.2f} - {ob.high:.2f}",
                        "mean_threshold": ob.mean_threshold,
                        "reasoning": f"Bearish OB ({ob.type.value}) at {ob.low:.2f}-{ob.high:.2f}",
                    }
                    setups.append(setup)
        
        # 4. ICT Liquidity Sweep Reversals
        if ict_state:
            for sweep in ict_state.recent_sweeps:
                if sweep.reversal_detected:
                    direction = "long" if sweep.liquidity_level.type == LiquidityType.SELL_SIDE else "short"
                    setup = {
                        "type": "ict_liquidity_sweep",
                        "direction": direction,
                        "confidence": 0.75,
                        "sweep_type": sweep.liquidity_level.type.value,
                        "sweep_price": sweep.sweep_price,
                        "reasoning": f"Liquidity sweep reversal at {sweep.sweep_price:.2f}",
                    }
                    
                    # Major boost if aligns with Wyckoff Spring/Upthrust
                    if wyckoff_state:
                        if direction == "long" and wyckoff_state.current_event.value == "spring":
                            setup["confidence"] = 0.9
                            setup["reasoning"] += " + Wyckoff Spring CONFIRMED"
                        elif direction == "short" and wyckoff_state.current_event.value in ["upthrust", "upthrust_after_distribution"]:
                            setup["confidence"] = 0.9
                            setup["reasoning"] += " + Wyckoff Upthrust CONFIRMED"
                    
                    setups.append(setup)
        
        # 5. OTE Zone Entry
        if ict_state and ict_state.in_ote:
            direction = "long" if ict_state.current_zone == ZoneType.DISCOUNT else "short"
            setup = {
                "type": "ict_ote",
                "direction": direction,
                "confidence": 0.65,
                "zone": ict_state.current_zone.value if ict_state.current_zone else "unknown",
                "reasoning": f"Price in OTE zone ({ict_state.current_zone.value if ict_state.current_zone else 'N/A'})",
            }
            
            # Boost with daily bias alignment
            if ict_state.daily_bias:
                if (direction == "long" and ict_state.daily_bias.bias == DailyBias.BULLISH) or \
                   (direction == "short" and ict_state.daily_bias.bias == DailyBias.BEARISH):
                    setup["confidence"] *= 1.2
                    setup["reasoning"] += f" + Daily bias {ict_state.daily_bias.bias.value}"
            
            setups.append(setup)
        
        # Sort by confidence
        setups.sort(key=lambda x: x["confidence"], reverse=True)
        
        return setups
    
    def _get_wyckoff_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get Wyckoff signal for a symbol."""
        if not self.wyckoff_engine:
            return None
        
        state = self.wyckoff_engine.get_wyckoff_state(symbol)
        if not state:
            return None
        
        # Convert trading bias to direction
        if state.trading_bias == "long":
            direction = 1.0
        elif state.trading_bias == "short":
            direction = -1.0
        else:
            direction = 0.0
        
        return {
            "direction": direction,
            "confidence": state.confidence,
            "phase": state.phase.value,
            "structure": state.structure.value,
            "event": state.current_event.value,
            "composite_score": state.composite_score,
            "vsa": state.vsa_analysis.signal.value if state.vsa_analysis else None,
        }
    
    def _get_ict_signal(
        self,
        symbol: str,
        ict_snapshot: Optional[ICTSnapshot] = None
    ) -> Optional[Dict[str, Any]]:
        """Get ICT signal for a symbol."""
        state = ict_snapshot
        if not state and self.ict_engine:
            state = self.ict_engine.get_state(symbol)
        
        if not state:
            return None
        
        # Convert entry signal to direction
        if state.entry_signal == "long":
            direction = 1.0
        elif state.entry_signal == "short":
            direction = -1.0
        else:
            direction = 0.0
        
        return {
            "direction": direction,
            "confidence": state.entry_confidence,
            "zone": state.current_zone.value if state.current_zone else None,
            "in_ote": state.in_ote,
            "daily_bias": state.daily_bias.bias.value if state.daily_bias else None,
            "bullish_fvgs": len([f for f in state.bullish_fvgs if f.status == FVGStatus.UNFILLED]),
            "bearish_fvgs": len([f for f in state.bearish_fvgs if f.status == FVGStatus.UNFILLED]),
            "sweeps_with_reversal": len([s for s in state.recent_sweeps if s.reversal_detected]),
        }
    
    def _get_base_signal(self, snapshot) -> Dict[str, Any]:
        """Get base liquidity signal from snapshot."""
        imbalance = getattr(snapshot, 'bid_ask_imbalance', 0.0)
        
        if imbalance > 0.15:
            direction = 1.0
        elif imbalance < -0.15:
            direction = -1.0
        else:
            direction = 0.0
        
        return {
            "direction": direction,
            "confidence": getattr(snapshot, 'confidence', 0.5),
            "imbalance": imbalance,
            "depth": getattr(snapshot, 'depth', 0),
        }
    
    def _combine_all_signals(
        self,
        wyckoff_signal: Optional[Dict[str, Any]],
        ict_signal: Optional[Dict[str, Any]],
        base_signal: Dict[str, Any],
    ) -> Tuple[DirectionEnum, float, str]:
        """Combine all signals into final direction and confidence."""
        
        total_weight = 0.0
        weighted_direction = 0.0
        weighted_confidence = 0.0
        reasoning_parts = []
        
        # Base signal
        if base_signal:
            weight = self.base_weight
            total_weight += weight
            weighted_direction += base_signal["direction"] * weight
            weighted_confidence += base_signal["confidence"] * weight
            if base_signal["direction"] != 0:
                reasoning_parts.append(f"Base: {'bullish' if base_signal['direction'] > 0 else 'bearish'}")
        
        # Wyckoff signal
        if wyckoff_signal:
            weight = self.wyckoff_weight
            total_weight += weight
            
            # Apply phase multiplier
            phase_mult = self.WYCKOFF_PHASE_MULT.get(wyckoff_signal["phase"], 1.0)
            adj_conf = wyckoff_signal["confidence"] * phase_mult
            
            weighted_direction += wyckoff_signal["direction"] * weight * phase_mult
            weighted_confidence += adj_conf * weight
            
            event_str = f" ({wyckoff_signal['event']})" if wyckoff_signal['event'] != 'none' else ''
            reasoning_parts.append(
                f"Wyckoff: {wyckoff_signal['phase']}/{wyckoff_signal['structure']}{event_str}"
            )
        
        # ICT signal
        if ict_signal:
            weight = self.ict_weight
            total_weight += weight
            
            # Apply zone bonus
            zone_mult = 1.0
            if ict_signal["in_ote"]:
                zone_mult = 1.15
            
            adj_conf = ict_signal["confidence"] * zone_mult
            weighted_direction += ict_signal["direction"] * weight * zone_mult
            weighted_confidence += adj_conf * weight
            
            ict_parts = []
            if ict_signal["zone"]:
                ict_parts.append(ict_signal["zone"])
            if ict_signal["in_ote"]:
                ict_parts.append("OTE")
            if ict_signal["daily_bias"]:
                ict_parts.append(f"bias:{ict_signal['daily_bias']}")
            
            reasoning_parts.append(f"ICT: {'/'.join(ict_parts) if ict_parts else 'analyzing'}")
        
        # Normalize
        if total_weight > 0:
            weighted_direction /= total_weight
            weighted_confidence /= total_weight
        
        # Determine final direction
        if weighted_direction > 0.25:
            final_direction = DirectionEnum.LONG
        elif weighted_direction < -0.25:
            final_direction = DirectionEnum.SHORT
        else:
            final_direction = DirectionEnum.NEUTRAL
        
        # Final confidence with confluence bonus
        confluence_bonus = 0.0
        active_signals = sum(1 for s in [wyckoff_signal, ict_signal, base_signal] if s and s.get("direction", 0) != 0)
        if active_signals >= 2:
            # Check if signals agree
            directions = [s.get("direction", 0) for s in [wyckoff_signal, ict_signal, base_signal] if s]
            if all(d > 0 for d in directions if d != 0) or all(d < 0 for d in directions if d != 0):
                confluence_bonus = 0.1 * (active_signals - 1)
                reasoning_parts.append(f"Confluence: {active_signals} signals aligned")
        
        final_confidence = min(1.0, weighted_confidence + confluence_bonus)
        reasoning = " | ".join(reasoning_parts)
        
        return final_direction, final_confidence, reasoning
    
    def _analyze_timeframe(
        self,
        timeframe: str,
        snapshot: Any,
        wyckoff_state: Optional[WyckoffState],
        ict_state: Optional[ICTSnapshot],
    ) -> Tuple[float, float, float, str]:
        """Analyze a single timeframe with combined methodology."""
        
        # Base direction from snapshot
        imbalance = getattr(snapshot, 'bid_ask_imbalance', 0.0)
        if imbalance > 0.15:
            base_dir = 1.0
        elif imbalance < -0.15:
            base_dir = -1.0
        else:
            base_dir = 0.0
        
        direction = base_dir
        strength = 0.5
        confidence = getattr(snapshot, 'confidence', 0.5)
        reasoning_parts = []
        
        # Apply Wyckoff context
        if wyckoff_state:
            phase_mult = self.WYCKOFF_PHASE_MULT.get(wyckoff_state.phase.value, 1.0)
            confidence *= phase_mult
            
            # Align with structure
            if wyckoff_state.structure.value in ["accumulation", "re_accumulation"]:
                if direction < 0:
                    direction *= 0.5  # Reduce bearish in accumulation
                reasoning_parts.append(f"Wyckoff:{wyckoff_state.phase.value[:3]}")
            elif wyckoff_state.structure.value in ["distribution", "re_distribution"]:
                if direction > 0:
                    direction *= 0.5  # Reduce bullish in distribution
                reasoning_parts.append(f"Wyckoff:{wyckoff_state.phase.value[:3]}")
        
        # Apply ICT context
        if ict_state:
            # Zone alignment
            if ict_state.current_zone == ZoneType.DISCOUNT and direction > 0:
                confidence *= 1.1
                reasoning_parts.append("discount")
            elif ict_state.current_zone == ZoneType.PREMIUM and direction < 0:
                confidence *= 1.1
                reasoning_parts.append("premium")
            
            # OTE bonus
            if ict_state.in_ote:
                confidence *= 1.1
                strength *= 1.1
                reasoning_parts.append("OTE")
            
            # Daily bias alignment
            if ict_state.daily_bias:
                if (direction > 0 and ict_state.daily_bias.bias == DailyBias.BULLISH) or \
                   (direction < 0 and ict_state.daily_bias.bias == DailyBias.BEARISH):
                    confidence *= 1.05
        
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "base"
        
        return direction, min(1.0, strength), min(1.0, confidence), reasoning
    
    def _calculate_confluence(
        self,
        wyckoff_data: Dict[str, Any],
        ict_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate confluence score between methodologies."""
        
        score = 0.0
        direction = 0.0
        reasoning_parts = []
        best_entry = None
        
        # Wyckoff contribution
        if wyckoff_data:
            wyckoff_dir = 1.0 if wyckoff_data.get("trading_bias") == "long" else \
                         -1.0 if wyckoff_data.get("trading_bias") == "short" else 0.0
            wyckoff_conf = wyckoff_data.get("confidence", 0)
            
            phase = wyckoff_data.get("phase", "unknown")
            phase_mult = self.WYCKOFF_PHASE_MULT.get(phase, 1.0)
            
            score += 0.4 * wyckoff_conf * phase_mult
            direction += wyckoff_dir * 0.4 * phase_mult
            
            if wyckoff_conf > 0.5:
                reasoning_parts.append(f"Wyckoff {phase} ({wyckoff_conf:.0%})")
                if phase in ["phase_c", "phase_d"]:
                    best_entry = "wyckoff_event"
        
        # ICT contribution
        if ict_data:
            ict_dir = 1.0 if ict_data.get("entry_signal") == "long" else \
                     -1.0 if ict_data.get("entry_signal") == "short" else 0.0
            ict_conf = ict_data.get("entry_confidence", 0)
            
            # Zone bonus
            zone = ict_data.get("zone")
            zone_mult = 1.1 if ict_data.get("in_ote") else 1.0
            
            score += 0.4 * ict_conf * zone_mult
            direction += ict_dir * 0.4 * zone_mult
            
            if ict_conf > 0.3:
                parts = []
                if zone:
                    parts.append(zone)
                if ict_data.get("in_ote"):
                    parts.append("OTE")
                if ict_data.get("daily_bias"):
                    parts.append(f"bias:{ict_data['daily_bias']}")
                reasoning_parts.append(f"ICT {'/'.join(parts)}")
                
                if not best_entry:
                    if ict_data.get("in_ote"):
                        best_entry = "ict_ote"
                    elif ict_data.get("unfilled_fvgs", 0) > 0:
                        best_entry = "ict_fvg"
        
        # Confluence bonus
        if wyckoff_data and ict_data:
            wyckoff_dir = 1.0 if wyckoff_data.get("trading_bias") == "long" else \
                         -1.0 if wyckoff_data.get("trading_bias") == "short" else 0.0
            ict_dir = 1.0 if ict_data.get("entry_signal") == "long" else \
                     -1.0 if ict_data.get("entry_signal") == "short" else 0.0
            
            if wyckoff_dir != 0 and ict_dir != 0 and wyckoff_dir == ict_dir:
                score += 0.2
                reasoning_parts.append("CONFLUENCE")
        
        final_direction = "long" if direction > 0.2 else "short" if direction < -0.2 else "neutral"
        
        return {
            "score": min(1.0, score),
            "direction": final_direction,
            "direction_strength": abs(direction),
            "reasoning": " | ".join(reasoning_parts),
            "best_entry_type": best_entry,
        }


__all__ = ['LiquidityAgentV4']
