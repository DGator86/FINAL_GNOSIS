"""Unified Liquidity Agent V5 - Wyckoff + ICT + Order Flow + Supply/Demand + Liquidity Concepts.

This agent combines FIVE methodologies for the most comprehensive
market structure analysis and highest-confluence trading signals.

Methodologies:
1. Wyckoff Analysis (LiquidityEngineV4):
   - Volume Spread Analysis (VSA)
   - Phase tracking (A-E)
   - Event detection (Spring, Upthrust, SOS, SOW, etc.)
   - Accumulation/Distribution structure

2. ICT Analysis (ICTEngine):
   - Swing points & liquidity levels
   - Fair Value Gaps (FVG) - BISI/SIBI
   - Order Blocks
   - Premium/Discount zones & OTE
   - Daily bias
   - Liquidity sweeps

3. Order Flow Analysis (OrderFlowEngine):
   - Footprint charts (bid/ask aggression, imbalance)
   - Cumulative Volume Delta (CVD)
   - Volume Profile (POC, Value Area, HVN/LVN)
   - Auction Market Theory

4. Supply & Demand Analysis (SupplyDemandEngine):
   - Demand Zones (lows between two highs with higher high)
   - Supply Zones (highs between two lows with lower low)
   - Zone strength validation (momentum confirmation)
   - Zone status tracking (fresh, tested, retested, broken)
   - Built-in risk management with R:R levels

5. Liquidity Concepts Analysis (LiquidityConceptsEngine):
   - Latent Liquidity Pools (buy-side/sell-side above/below highs/lows)
   - Strong/Weak Swing Classification (based on Break of Structure)
   - Liquidity Voids (areas of shallow depth - price travels easily)
   - Fractal Market Structure (smooth vs rough analysis)
   - Liquidity Inducement Detection (stop hunts, false breakouts, sweeps)

Signal Generation:
- Penta confluence scoring (5 methodologies)
- Order flow confirmed entries
- Supply/Demand zone entries with risk management
- Liquidity pool sweep reversal signals
- Volume profile aligned stops/targets
- CVD divergence detection
- Fractal structure zone validation

Author: Super Gnosis Elite Trading System
Version: 5.2.0 - Penta Methodology Integration (Wyckoff + ICT + Order Flow + S&D + Liquidity Concepts)
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
        ICTZoneType as ZoneType,  # Premium/Discount zones
        FVGType,
        FVGStatus,
        LiquidityType,
    )
    ICT_AVAILABLE = True
except ImportError:
    ICT_AVAILABLE = False
    logger.warning("ICT components not available")

# Import Order Flow components
try:
    from engines.liquidity import (
        OrderFlowEngine,
        OrderFlowState,
        OrderFlowEntry,
        OrderFlowSignal,
        FootprintPattern,
        AuctionState,
        VolumeProfile,
    )
    ORDER_FLOW_AVAILABLE = True
except ImportError:
    ORDER_FLOW_AVAILABLE = False
    logger.warning("Order Flow components not available")
    # Define placeholder enums to prevent NameError
    from enum import Enum
    class FootprintPattern(Enum):
        ABSORPTION_AT_LOW = "absorption_at_low"
        ABSORPTION_AT_HIGH = "absorption_at_high"
        INITIATIVE_BUYING = "initiative_buying"
        INITIATIVE_SELLING = "initiative_selling"
        BREAKOUT_CONFIRMATION = "breakout_confirmation"
        FAILED_AUCTION = "failed_auction"
        STOP_HUNT_COMPLETE = "stop_hunt_complete"
    class OrderFlowSignal(Enum):
        ABSORPTION = "absorption"
        INITIATION = "initiation"
        EXHAUSTION = "exhaustion"
        DIVERGENCE = "divergence"
        IMBALANCE = "imbalance"
        DELTA_FLIP = "delta_flip"
        POC_TEST = "poc_test"
        VALUE_AREA_BREAK = "value_area_break"
        HVN_REJECTION = "hvn_rejection"
        LVN_ACCELERATION = "lvn_acceleration"
        STACKED_IMBALANCE = "stacked_imbalance"
    class AuctionState:
        pass
    class VolumeProfile:
        pass
    class OrderFlowState:
        pass
    class OrderFlowEntry:
        pass

# Import Supply & Demand components
try:
    from engines.liquidity import (
        SupplyDemandEngine,
        SupplyDemandState,
        SupplyDemandZone,
        ZoneEntry,
        SDZoneType,        # Supply/Demand zone type
        ZoneStrength,
        ZoneStatus,
        MarketEquilibrium,
        SDEntrySignal,     # S&D entry signals
    )
    SUPPLY_DEMAND_AVAILABLE = True
except ImportError:
    SUPPLY_DEMAND_AVAILABLE = False
    logger.warning("Supply & Demand components not available")

# Import Liquidity Concepts components
try:
    from engines.liquidity import (
        LiquidityConceptsEngine,
        LiquidityConceptsState,
        LiquidityPool,
        LiquidityVoid,
        LiquidityInducement,
        SwingPointExtended,
        BreakOfStructure,
        FractalStructureAnalysis,
        LiquidityPoolType,
        LiquidityPoolSide,
        SwingStrength,
        MarketStructureType,
        BOSType,
        LiquidityInducementType,
    )
    LIQUIDITY_CONCEPTS_AVAILABLE = True
except ImportError:
    LIQUIDITY_CONCEPTS_AVAILABLE = False
    logger.warning("Liquidity Concepts components not available")


class LiquidityAgentV5:
    """Unified liquidity agent combining Wyckoff, ICT, Order Flow, Supply/Demand, and Liquidity Concepts.
    
    This agent provides the most comprehensive market structure analysis
    by combining FIVE powerful trading methodologies.
    
    Signal Confluence Factors (Base weights, redistributed if methods disabled):
    - Wyckoff (18%): Phase, events, VSA analysis
    - ICT (18%): Daily bias, Premium/Discount, FVGs, sweeps
    - Order Flow (18%): CVD, footprint patterns, volume profile
    - Supply/Demand (18%): Zone proximity, strength, status
    - Liquidity Concepts (18%): Pools, inducements, structure
    - Base (10%): Bid/ask imbalance
    
    Liquidity Concepts Features:
    - Latent Liquidity Pools: Major/minor pools above highs (buy-side) and below lows (sell-side)
    - Strong/Weak Highs/Lows: Classification based on Break of Structure (BOS)
    - Liquidity Voids: Areas where price travels easily (low depth)
    - Fractal Structure: Smooth vs rough market structure affects zone reliability
    - Inducement Detection: Stop hunts, false breakouts, liquidity sweeps
    
    Smart Money Insights Applied:
    - "Liquidity is fuel, not destination" - Price follows VALUE
    - Major pools create deeper liquidity than minor pools
    - Rough structure = more internal pools = zone more likely to hold
    - Inducements are traps - trade opposite direction after reversal
    
    Zone Entry Priority:
    1. Fresh S/D zones with internal liquidity pools (rough structure)
    2. Liquidity inducement reversals (stop hunt/sweep confirmation)
    3. Order Flow confirmed S/D zones
    4. Wyckoff event at S/D zone
    5. ICT FVG within S/D zone
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
    
    # Order Flow patterns with entry value
    ORDER_FLOW_ENTRY_PATTERNS = {
        FootprintPattern.ABSORPTION_AT_LOW: ("long", 0.80),
        FootprintPattern.ABSORPTION_AT_HIGH: ("short", 0.80),
        FootprintPattern.INITIATIVE_BUYING: ("long", 0.70),
        FootprintPattern.INITIATIVE_SELLING: ("short", 0.70),
        FootprintPattern.BREAKOUT_CONFIRMATION: ("long", 0.75),
        FootprintPattern.FAILED_AUCTION: ("short", 0.75),
        FootprintPattern.STOP_HUNT_COMPLETE: ("reversal", 0.85),
    }
    
    # Order Flow signals with directional value
    ORDER_FLOW_SIGNAL_WEIGHTS = {
        OrderFlowSignal.ABSORPTION: 0.15,
        OrderFlowSignal.INITIATION: 0.12,
        OrderFlowSignal.EXHAUSTION: 0.18,
        OrderFlowSignal.DIVERGENCE: 0.20,
        OrderFlowSignal.IMBALANCE: 0.10,
        OrderFlowSignal.DELTA_FLIP: 0.15,
        OrderFlowSignal.POC_TEST: 0.08,
        OrderFlowSignal.VALUE_AREA_BREAK: 0.12,
        OrderFlowSignal.HVN_REJECTION: 0.10,
        OrderFlowSignal.LVN_ACCELERATION: 0.08,
        OrderFlowSignal.STACKED_IMBALANCE: 0.15,
    }
    
    # Supply & Demand zone entry patterns
    SD_ENTRY_PATTERNS = {
        SDEntrySignal.DEMAND_ZONE_TOUCH: ("long", 0.70),
        SDEntrySignal.DEMAND_ZONE_BOUNCE: ("long", 0.80),
        SDEntrySignal.SUPPLY_ZONE_TOUCH: ("short", 0.70),
        SDEntrySignal.SUPPLY_ZONE_BOUNCE: ("short", 0.80),
    } if SUPPLY_DEMAND_AVAILABLE else {}
    
    # Zone strength multipliers
    ZONE_STRENGTH_MULT = {
        "STRONG": 1.2,
        "MODERATE": 1.0,
        "WEAK": 0.8,
        "BROKEN": 0.0,
    }
    
    # Zone status multipliers
    ZONE_STATUS_MULT = {
        "FRESH": 1.25,     # Fresh zones have highest probability
        "TESTED": 1.0,      # Tested once, still valid
        "RETESTED": 0.85,   # Multiple tests weaken the zone
        "BROKEN": 0.0,      # Broken zones are invalid
    }
    
    # Liquidity Concepts inducement entry patterns
    LC_INDUCEMENT_PATTERNS = {
        LiquidityInducementType.STOP_HUNT: 0.85,        # Quick sweep and reversal - high probability
        LiquidityInducementType.LIQUIDITY_SWEEP: 0.80,  # Major pool swept with reversal
        LiquidityInducementType.FALSE_BREAKOUT: 0.75,   # Extended move that fails
        LiquidityInducementType.INDUCEMENT_TRAP: 0.70,  # Minor pool swept to attract positions
    } if LIQUIDITY_CONCEPTS_AVAILABLE else {}
    
    # Liquidity pool type multipliers
    LC_POOL_TYPE_MULT = {
        "CLUSTERED": 1.25,      # Equal highs/lows - strongest
        "EQUAL_HIGHS": 1.20,    # Clustered highs
        "EQUAL_LOWS": 1.20,     # Clustered lows
        "MAJOR": 1.10,          # Major swing pool
        "MINOR": 0.90,          # Minor swing pool
    }
    
    # Market structure smoothness effect on zone reliability
    LC_STRUCTURE_MULT = {
        "ROUGH": 1.20,    # Rough = more internal pools = zone likely to hold
        "MIXED": 1.00,    # Neutral
        "SMOOTH": 0.85,   # Smooth = few internal pools = zone less reliable
    }
    
    def __init__(
        self,
        config: Dict[str, Any],
        liquidity_engine_v5: Optional[Any] = None,  # Unified engine (preferred)
        # Legacy individual engine parameters (backward compatibility)
        wyckoff_engine: Optional[LiquidityEngineV4] = None,
        ict_engine: Optional[ICTEngine] = None,
        order_flow_engine: Optional[OrderFlowEngine] = None,
        supply_demand_engine: Optional["SupplyDemandEngine"] = None,
        liquidity_concepts_engine: Optional["LiquidityConceptsEngine"] = None,
    ):
        """Initialize LiquidityAgentV5.
        
        This agent can be initialized in two ways:
        1. With LiquidityEngineV5 (unified engine containing all PENTA sub-engines)
        2. With individual engines (legacy, backward compatible)
        
        Architecture:
            LiquidityEngineV5 → LiquidityAgentV5 → Composer
        
        Args:
            config: Agent configuration
            liquidity_engine_v5: LiquidityEngineV5 (unified PENTA engine) - PREFERRED
            wyckoff_engine: Legacy - LiquidityEngineV4 for Wyckoff analysis
            ict_engine: Legacy - ICTEngine for ICT analysis
            order_flow_engine: Legacy - OrderFlowEngine for order flow analysis
            supply_demand_engine: Legacy - SupplyDemandEngine for S/D analysis
            liquidity_concepts_engine: Legacy - LiquidityConceptsEngine for smart money
        """
        self.config = config
        self.unified_engine = liquidity_engine_v5
        
        # If unified engine provided, extract sub-engines from it
        if liquidity_engine_v5 is not None:
            logger.info("LiquidityAgentV5: Using unified LiquidityEngineV5")
            penta_engines = liquidity_engine_v5.get_penta_engines()
            self.wyckoff_engine = penta_engines.get("wyckoff")
            self.ict_engine = penta_engines.get("ict")
            self.order_flow_engine = penta_engines.get("order_flow")
            self.supply_demand_engine = penta_engines.get("supply_demand")
            self.liquidity_concepts_engine = penta_engines.get("liquidity_concepts")
        else:
            # Legacy mode: use individual engines
            logger.info("LiquidityAgentV5: Using individual engines (legacy mode)")
            self.wyckoff_engine = wyckoff_engine
            self.ict_engine = ict_engine
            self.order_flow_engine = order_flow_engine
            self.supply_demand_engine = supply_demand_engine
            self.liquidity_concepts_engine = liquidity_concepts_engine
        
        # Configuration - adjusted weights for 5 methodologies
        self.min_confidence = config.get("min_confidence", 0.5)
        self.wyckoff_weight = config.get("wyckoff_weight", 0.18)
        self.ict_weight = config.get("ict_weight", 0.18)
        self.order_flow_weight = config.get("order_flow_weight", 0.18)
        self.supply_demand_weight = config.get("supply_demand_weight", 0.18)
        self.liquidity_concepts_weight = config.get("liquidity_concepts_weight", 0.18)
        self.base_weight = config.get("base_weight", 0.10)
        
        # Enable flags
        self.enable_wyckoff = config.get("enable_wyckoff", True) and WYCKOFF_AVAILABLE
        self.enable_ict = config.get("enable_ict", True) and ICT_AVAILABLE
        self.enable_order_flow = config.get("enable_order_flow", True) and ORDER_FLOW_AVAILABLE
        self.enable_supply_demand = config.get("enable_supply_demand", True) and SUPPLY_DEMAND_AVAILABLE
        self.enable_liquidity_concepts = config.get("enable_liquidity_concepts", True) and LIQUIDITY_CONCEPTS_AVAILABLE
        
        # Count enabled methodologies for weight redistribution
        enabled_methods = [
            ("wyckoff", self.enable_wyckoff, self.wyckoff_weight),
            ("ict", self.enable_ict, self.ict_weight),
            ("order_flow", self.enable_order_flow, self.order_flow_weight),
            ("supply_demand", self.enable_supply_demand, self.supply_demand_weight),
            ("liquidity_concepts", self.enable_liquidity_concepts, self.liquidity_concepts_weight),
        ]
        
        enabled_count = sum(1 for _, enabled, _ in enabled_methods if enabled)
        if enabled_count > 0 and enabled_count < 5:
            # Redistribute weights if some methodologies are disabled
            total_disabled_weight = sum(
                weight for _, enabled, weight in enabled_methods if not enabled
            )
            
            # Reset disabled weights to 0
            if not self.enable_wyckoff:
                self.wyckoff_weight = 0.0
            if not self.enable_ict:
                self.ict_weight = 0.0
            if not self.enable_order_flow:
                self.order_flow_weight = 0.0
            if not self.enable_supply_demand:
                self.supply_demand_weight = 0.0
            if not self.enable_liquidity_concepts:
                self.liquidity_concepts_weight = 0.0
            
            # Redistribute to enabled methodologies
            redistribution = total_disabled_weight / enabled_count
            if self.enable_wyckoff:
                self.wyckoff_weight += redistribution
            if self.enable_ict:
                self.ict_weight += redistribution
            if self.enable_order_flow:
                self.order_flow_weight += redistribution
            if self.enable_supply_demand:
                self.supply_demand_weight += redistribution
            if self.enable_liquidity_concepts:
                self.liquidity_concepts_weight += redistribution
        
        logger.info(
            f"LiquidityAgentV5 initialized | "
            f"wyckoff={self.enable_wyckoff} ({self.wyckoff_weight:.0%}) | "
            f"ict={self.enable_ict} ({self.ict_weight:.0%}) | "
            f"order_flow={self.enable_order_flow} ({self.order_flow_weight:.0%}) | "
            f"supply_demand={self.enable_supply_demand} ({self.supply_demand_weight:.0%}) | "
            f"liquidity_concepts={self.enable_liquidity_concepts} ({self.liquidity_concepts_weight:.0%})"
        )
    
    def suggest(
        self,
        pipeline_result: PipelineResult,
        timestamp: datetime,
        ict_snapshot: Optional[ICTSnapshot] = None,
        order_flow_state: Optional[OrderFlowState] = None,
        supply_demand_state: Optional["SupplyDemandState"] = None,
        liquidity_concepts_state: Optional["LiquidityConceptsState"] = None,
    ) -> Optional[AgentSuggestion]:
        """Generate unified suggestion combining all five methodologies.
        
        Args:
            pipeline_result: Pipeline result with liquidity snapshot
            timestamp: Analysis timestamp
            ict_snapshot: Optional pre-computed ICT snapshot
            order_flow_state: Optional pre-computed Order Flow state
            supply_demand_state: Optional pre-computed Supply/Demand state
            liquidity_concepts_state: Optional pre-computed Liquidity Concepts state
            
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
        order_flow_signal = self._get_order_flow_signal(symbol, order_flow_state) if self.enable_order_flow else None
        supply_demand_signal = self._get_supply_demand_signal(symbol, supply_demand_state) if self.enable_supply_demand else None
        liquidity_concepts_signal = self._get_liquidity_concepts_signal(symbol, liquidity_concepts_state) if self.enable_liquidity_concepts else None
        base_signal = self._get_base_signal(snapshot)
        
        # Combine signals
        final_direction, final_confidence, reasoning = self._combine_all_signals(
            wyckoff_signal=wyckoff_signal,
            ict_signal=ict_signal,
            order_flow_signal=order_flow_signal,
            supply_demand_signal=supply_demand_signal,
            liquidity_concepts_signal=liquidity_concepts_signal,
            base_signal=base_signal,
        )
        
        if final_confidence < self.min_confidence:
            logger.debug(f"LiquidityAgentV5 {symbol}: confidence {final_confidence:.0%} below threshold")
            return None
        
        return AgentSuggestion(
            agent_name="liquidity_agent_v5",
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
        order_flow_state: Optional[OrderFlowState] = None,
    ) -> List[TimeframeSignal]:
        """Generate multi-timeframe signals with Wyckoff + ICT + Order Flow.
        
        Args:
            liquidity_snapshots: Dict[timeframe, LiquiditySnapshot]
            symbol: Trading symbol
            timestamp: Analysis timestamp
            ict_snapshot: Optional ICT analysis snapshot
            order_flow_state: Optional Order Flow state
            
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
        
        # Get Order Flow state
        of_state = order_flow_state
        if not of_state and self.enable_order_flow and self.order_flow_engine:
            of_state = self.order_flow_engine.get_state(symbol)
        
        for timeframe, snapshot in liquidity_snapshots.items():
            if snapshot is None:
                continue
            
            # Calculate combined direction and confidence
            direction, strength, confidence, reasoning = self._analyze_timeframe(
                timeframe=timeframe,
                snapshot=snapshot,
                wyckoff_state=wyckoff_state,
                ict_state=ict_state,
                order_flow_state=of_state,
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
                f"LiquidityAgentV5 {timeframe}: dir={direction:+.2f} "
                f"conf={confidence:.2f}"
            )
        
        return signals
    
    def get_confluence_analysis(
        self,
        symbol: str,
        ict_snapshot: Optional[ICTSnapshot] = None,
        order_flow_state: Optional[OrderFlowState] = None,
        supply_demand_state: Optional["SupplyDemandState"] = None,
    ) -> Dict[str, Any]:
        """Get detailed confluence analysis from all four methodologies.
        
        Args:
            symbol: Trading symbol
            ict_snapshot: Optional ICT snapshot
            order_flow_state: Optional Order Flow state
            supply_demand_state: Optional Supply/Demand state
            
        Returns:
            Detailed confluence analysis
        """
        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "wyckoff": {},
            "ict": {},
            "order_flow": {},
            "supply_demand": {},
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
        
        # Order Flow Analysis
        of_state = order_flow_state
        if not of_state and self.enable_order_flow and self.order_flow_engine:
            of_state = self.order_flow_engine.get_state(symbol)
        
        if of_state:
            analysis["order_flow"] = {
                "auction_state": of_state.auction_state.value if of_state.auction_state else None,
                "current_cvd": of_state.current_cvd,
                "cvd_ma": of_state.cvd_ma,
                "signals": [s.name for s in of_state.signals],
                "patterns": [p.name for p in of_state.patterns],
                "signal_strength": of_state.signal_strength,
                "confluence_score": of_state.confluence_score,
            }
            
            # Add profile data if available
            if of_state.session_profile:
                profile = of_state.session_profile
                analysis["order_flow"]["poc"] = profile.poc
                analysis["order_flow"]["value_area"] = {
                    "high": profile.value_area_high,
                    "low": profile.value_area_low,
                }
        
        # Supply/Demand Analysis
        sd_state = supply_demand_state
        if not sd_state and self.enable_supply_demand and self.supply_demand_engine:
            sd_state = self.supply_demand_engine.get_state(symbol)
        
        if sd_state:
            analysis["supply_demand"] = {
                "equilibrium": sd_state.equilibrium_state.name if sd_state.equilibrium_state else None,
                "entry_signal": sd_state.entry_signal.name if sd_state.entry_signal else None,
                "signal_confidence": sd_state.signal_confidence,
                "demand_zones_count": len(sd_state.demand_zones),
                "supply_zones_count": len(sd_state.supply_zones),
                "nearest_demand": None,
                "nearest_supply": None,
            }
            
            if sd_state.nearest_demand:
                zone = sd_state.nearest_demand
                analysis["supply_demand"]["nearest_demand"] = {
                    "upper": zone.boundary.upper,
                    "lower": zone.boundary.lower,
                    "strength": zone.strength.name,
                    "status": zone.status.name,
                    "distance_pct": zone.distance_percent(sd_state.current_price),
                    "stop_loss": zone.stop_loss,
                    "take_profit_3": zone.take_profit_3,
                }
            
            if sd_state.nearest_supply:
                zone = sd_state.nearest_supply
                analysis["supply_demand"]["nearest_supply"] = {
                    "upper": zone.boundary.upper,
                    "lower": zone.boundary.lower,
                    "strength": zone.strength.name,
                    "status": zone.status.name,
                    "distance_pct": zone.distance_percent(sd_state.current_price),
                    "stop_loss": zone.stop_loss,
                    "take_profit_3": zone.take_profit_3,
                }
        
        # Build supply_demand_analysis dict for confluence calculation
        sd_analysis = {}
        if analysis["supply_demand"]:
            sd_analysis = {
                "equilibrium": analysis["supply_demand"].get("equilibrium"),
                "entry_signal": analysis["supply_demand"].get("entry_signal"),
                "signal_confidence": analysis["supply_demand"].get("signal_confidence", 0),
                "nearest_demand": analysis["supply_demand"].get("nearest_demand"),
                "nearest_supply": analysis["supply_demand"].get("nearest_supply"),
            }
        
        # Confluence Analysis - now with quad methodology
        confluence = self._calculate_quad_confluence(
            analysis["wyckoff"],
            analysis["ict"],
            analysis["order_flow"],
            sd_analysis,
        )
        analysis["confluence"] = confluence
        
        # Generate recommendation
        if confluence["score"] >= 0.6:
            analysis["recommendation"] = {
                "action": confluence["direction"],
                "confidence": confluence["score"],
                "reasoning": confluence["reasoning"],
                "entry_type": confluence["best_entry_type"],
                "order_flow_confirmation": confluence.get("order_flow_confirms", False),
                "supply_demand_confirmation": confluence.get("supply_demand_confirms", False),
            }
            
            # Add risk levels if S/D zone is the entry type
            if confluence["best_entry_type"] in ["supply_demand_fresh_zone", "supply_demand_zone"]:
                if confluence["direction"] == "long" and sd_state and sd_state.nearest_demand:
                    zone = sd_state.nearest_demand
                    analysis["recommendation"]["risk_management"] = {
                        "entry": zone.boundary.upper,
                        "stop_loss": zone.stop_loss,
                        "tp1": zone.take_profit_1,
                        "tp2": zone.take_profit_2,
                        "tp3": zone.take_profit_3,
                        "tp4": zone.take_profit_4,
                    }
                elif confluence["direction"] == "short" and sd_state and sd_state.nearest_supply:
                    zone = sd_state.nearest_supply
                    analysis["recommendation"]["risk_management"] = {
                        "entry": zone.boundary.lower,
                        "stop_loss": zone.stop_loss,
                        "tp1": zone.take_profit_1,
                        "tp2": zone.take_profit_2,
                        "tp3": zone.take_profit_3,
                        "tp4": zone.take_profit_4,
                    }
        
        return analysis
    
    def get_entry_setups(
        self,
        symbol: str,
        ict_snapshot: Optional[ICTSnapshot] = None,
        order_flow_state: Optional[OrderFlowState] = None,
        supply_demand_state: Optional["SupplyDemandState"] = None,
        liquidity_concepts_state: Optional["LiquidityConceptsState"] = None,
    ) -> List[Dict[str, Any]]:
        """Get specific entry setups from combined analysis.
        
        Args:
            symbol: Trading symbol
            ict_snapshot: Optional ICT snapshot
            order_flow_state: Optional Order Flow state
            supply_demand_state: Optional Supply/Demand state
            liquidity_concepts_state: Optional Liquidity Concepts state
            
        Returns:
            List of entry setup dictionaries with multi-methodology confirmation
        """
        setups = []
        
        # Get states
        wyckoff_state = None
        if self.enable_wyckoff and self.wyckoff_engine:
            wyckoff_state = self.wyckoff_engine.get_wyckoff_state(symbol)
        
        ict_state = ict_snapshot
        if not ict_state and self.enable_ict and self.ict_engine:
            ict_state = self.ict_engine.get_state(symbol)
        
        of_state = order_flow_state
        if not of_state and self.enable_order_flow and self.order_flow_engine:
            of_state = self.order_flow_engine.get_state(symbol)
        
        lc_state = liquidity_concepts_state
        if not lc_state and self.enable_liquidity_concepts and self.liquidity_concepts_engine:
            lc_state = self.liquidity_concepts_engine.get_state(symbol)
        
        # Get Order Flow bias for confirmation
        of_bias = None
        of_bias_conf = 0.0
        if of_state and self.order_flow_engine:
            of_bias, of_bias_conf, _ = self.order_flow_engine.get_bias(symbol)
        
        # 1. Order Flow Entry Signals (Primary)
        if of_state:
            of_entries = self.order_flow_engine.get_entry_signals(symbol) if self.order_flow_engine else []
            for entry in of_entries:
                setup = {
                    "type": f"order_flow_{entry.signal_type.name.lower()}",
                    "direction": entry.direction,
                    "confidence": entry.confidence,
                    "entry_price": entry.entry_price,
                    "stop_price": entry.stop_price,
                    "target_price": entry.target_price,
                    "pattern": entry.footprint_pattern.name if entry.footprint_pattern else None,
                    "supporting_signals": [s.name for s in entry.supporting_signals],
                    "reasoning": entry.reasoning,
                }
                
                # Boost if Wyckoff confirms
                if wyckoff_state:
                    if entry.direction == "long" and wyckoff_state.trading_bias == "long":
                        setup["confidence"] *= 1.15
                        setup["reasoning"] += f" + Wyckoff {wyckoff_state.phase.value} confirms"
                    elif entry.direction == "short" and wyckoff_state.trading_bias == "short":
                        setup["confidence"] *= 1.15
                        setup["reasoning"] += f" + Wyckoff {wyckoff_state.phase.value} confirms"
                
                # Boost if ICT confirms
                if ict_state:
                    if entry.direction == "long" and ict_state.current_zone == ZoneType.DISCOUNT:
                        setup["confidence"] *= 1.10
                        setup["reasoning"] += " + ICT discount zone"
                    elif entry.direction == "short" and ict_state.current_zone == ZoneType.PREMIUM:
                        setup["confidence"] *= 1.10
                        setup["reasoning"] += " + ICT premium zone"
                
                setups.append(setup)
        
        # 2. Wyckoff Event Entries (Order Flow Confirmed)
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
            
            # Major boost if Order Flow confirms
            if of_bias:
                if direction == "long" and of_bias == "bullish":
                    setup["confidence"] *= 1.20
                    setup["reasoning"] += f" + Order Flow bullish ({of_bias_conf:.0%})"
                    setup["order_flow_confirmed"] = True
                elif direction == "short" and of_bias == "bearish":
                    setup["confidence"] *= 1.20
                    setup["reasoning"] += f" + Order Flow bearish ({of_bias_conf:.0%})"
                    setup["order_flow_confirmed"] = True
            
            # Additional boost if ICT confirms
            if ict_state:
                if direction == "long" and ict_state.current_zone == ZoneType.DISCOUNT:
                    setup["confidence"] *= 1.10
                    setup["reasoning"] += " + ICT discount zone"
                elif direction == "short" and ict_state.current_zone == ZoneType.PREMIUM:
                    setup["confidence"] *= 1.10
                    setup["reasoning"] += " + ICT premium zone"
            
            setups.append(setup)
        
        # 3. ICT FVG Entries (Order Flow Confirmed)
        if ict_state:
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
                    
                    # Boost if Order Flow confirms buying
                    if of_bias == "bullish":
                        setup["confidence"] *= 1.20
                        setup["reasoning"] += f" + Order Flow bullish ({of_bias_conf:.0%})"
                        setup["order_flow_confirmed"] = True
                    
                    # Additional Wyckoff boost
                    if wyckoff_state:
                        if wyckoff_state.structure.value in ["accumulation", "re_accumulation"]:
                            setup["confidence"] *= 1.15
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
                    
                    # Boost if Order Flow confirms selling
                    if of_bias == "bearish":
                        setup["confidence"] *= 1.20
                        setup["reasoning"] += f" + Order Flow bearish ({of_bias_conf:.0%})"
                        setup["order_flow_confirmed"] = True
                    
                    if wyckoff_state:
                        if wyckoff_state.structure.value in ["distribution", "re_distribution"]:
                            setup["confidence"] *= 1.15
                            setup["reasoning"] += f" + Wyckoff {wyckoff_state.structure.value}"
                    
                    setups.append(setup)
        
        # 4. ICT Order Block Entries (Order Flow Confirmed)
        if ict_state:
            for ob in ict_state.bullish_order_blocks:
                if not ob.is_mitigated:
                    setup = {
                        "type": "ict_order_block",
                        "direction": "long",
                        "confidence": 0.70 if ob.is_high_probability else 0.55,
                        "ob_type": ob.type.value,
                        "ob_range": f"{ob.low:.2f} - {ob.high:.2f}",
                        "mean_threshold": ob.mean_threshold,
                        "reasoning": f"Bullish OB ({ob.type.value}) at {ob.low:.2f}-{ob.high:.2f}",
                    }
                    
                    # Boost if Order Flow shows absorption at OB
                    if of_state and FootprintPattern.ABSORPTION_AT_LOW in of_state.patterns:
                        setup["confidence"] *= 1.25
                        setup["reasoning"] += " + Absorption at lows CONFIRMED"
                        setup["order_flow_confirmed"] = True
                    
                    setups.append(setup)
            
            for ob in ict_state.bearish_order_blocks:
                if not ob.is_mitigated:
                    setup = {
                        "type": "ict_order_block",
                        "direction": "short",
                        "confidence": 0.70 if ob.is_high_probability else 0.55,
                        "ob_type": ob.type.value,
                        "ob_range": f"{ob.low:.2f} - {ob.high:.2f}",
                        "mean_threshold": ob.mean_threshold,
                        "reasoning": f"Bearish OB ({ob.type.value}) at {ob.low:.2f}-{ob.high:.2f}",
                    }
                    
                    # Boost if Order Flow shows absorption at OB
                    if of_state and FootprintPattern.ABSORPTION_AT_HIGH in of_state.patterns:
                        setup["confidence"] *= 1.25
                        setup["reasoning"] += " + Absorption at highs CONFIRMED"
                        setup["order_flow_confirmed"] = True
                    
                    setups.append(setup)
        
        # 5. ICT Liquidity Sweep + Order Flow Reversal
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
                    
                    # Major boost if Order Flow shows exhaustion/divergence
                    if of_state:
                        if OrderFlowSignal.EXHAUSTION in of_state.signals:
                            setup["confidence"] *= 1.25
                            setup["reasoning"] += " + Order Flow EXHAUSTION confirmed"
                            setup["order_flow_confirmed"] = True
                        if OrderFlowSignal.DIVERGENCE in of_state.signals:
                            setup["confidence"] *= 1.30
                            setup["reasoning"] += " + CVD DIVERGENCE confirmed"
                            setup["order_flow_confirmed"] = True
                    
                    # Wyckoff Spring/Upthrust confirmation
                    if wyckoff_state:
                        if direction == "long" and wyckoff_state.current_event.value == "spring":
                            setup["confidence"] = min(setup["confidence"] * 1.15, 0.98)
                            setup["reasoning"] += " + Wyckoff Spring"
                        elif direction == "short" and wyckoff_state.current_event.value in ["upthrust", "upthrust_after_distribution"]:
                            setup["confidence"] = min(setup["confidence"] * 1.15, 0.98)
                            setup["reasoning"] += " + Wyckoff Upthrust"
                    
                    setups.append(setup)
        
        # 6. Volume Profile Level Tests
        if of_state and of_state.session_profile:
            profile = of_state.session_profile
            current_price = of_state.current_price
            
            # POC test
            if OrderFlowSignal.POC_TEST in of_state.signals:
                setup = {
                    "type": "order_flow_poc_test",
                    "direction": of_bias if of_bias in ["bullish", "bearish"] else "neutral",
                    "confidence": 0.60,
                    "poc": profile.poc,
                    "current_price": current_price,
                    "reasoning": f"Testing POC at {profile.poc:.2f}",
                }
                
                if of_bias == "bullish":
                    setup["direction"] = "long"
                    setup["reasoning"] += f" with bullish order flow"
                elif of_bias == "bearish":
                    setup["direction"] = "short"
                    setup["reasoning"] += f" with bearish order flow"
                
                setups.append(setup)
            
            # Value Area breaks
            if OrderFlowSignal.VALUE_AREA_BREAK in of_state.signals:
                if current_price > profile.value_area_high:
                    setup = {
                        "type": "order_flow_va_breakout",
                        "direction": "long",
                        "confidence": 0.65,
                        "value_area": f"{profile.value_area_low:.2f} - {profile.value_area_high:.2f}",
                        "reasoning": f"Breaking above value area ({profile.value_area_high:.2f})",
                    }
                    if of_state.auction_state == AuctionState.INITIATING_LONG:
                        setup["confidence"] *= 1.15
                        setup["reasoning"] += " + Initiative buying"
                    setups.append(setup)
                    
                elif current_price < profile.value_area_low:
                    setup = {
                        "type": "order_flow_va_breakdown",
                        "direction": "short",
                        "confidence": 0.65,
                        "value_area": f"{profile.value_area_low:.2f} - {profile.value_area_high:.2f}",
                        "reasoning": f"Breaking below value area ({profile.value_area_low:.2f})",
                    }
                    if of_state.auction_state == AuctionState.INITIATING_SHORT:
                        setup["confidence"] *= 1.15
                        setup["reasoning"] += " + Initiative selling"
                    setups.append(setup)
        
        # 7. Supply & Demand Zone Entries (NEW - Highest Priority for Fresh Zones)
        sd_state = supply_demand_state
        if not sd_state and self.enable_supply_demand and self.supply_demand_engine:
            sd_state = self.supply_demand_engine.get_state(symbol)
        
        if sd_state and self.supply_demand_engine:
            # Get S/D entry signals
            sd_entries = self.supply_demand_engine.get_entry_signals(symbol, sd_state)
            
            for sd_entry in sd_entries:
                # Base setup from S/D zone
                setup = {
                    "type": f"supply_demand_{sd_entry.signal_type.name.lower()}",
                    "direction": sd_entry.direction,
                    "confidence": sd_entry.confidence,
                    "entry_price": sd_entry.entry_price,
                    "stop_loss": sd_entry.stop_loss,
                    "take_profit_1": sd_entry.take_profit_1,
                    "take_profit_2": sd_entry.take_profit_2,
                    "take_profit_3": sd_entry.take_profit_3,
                    "take_profit_4": sd_entry.take_profit_4,
                    "zone_strength": sd_entry.zone.strength.name,
                    "zone_status": sd_entry.zone.status.name,
                    "risk_reward": sd_entry.risk_reward,
                    "reasoning": sd_entry.reasoning,
                }
                
                # Apply zone status multiplier (fresh zones are best)
                status_mult = self.ZONE_STATUS_MULT.get(sd_entry.zone.status.name, 1.0)
                strength_mult = self.ZONE_STRENGTH_MULT.get(sd_entry.zone.strength.name, 1.0)
                setup["confidence"] *= status_mult * strength_mult
                
                # Major boost if Order Flow confirms
                if of_bias:
                    if sd_entry.direction == "long" and of_bias == "bullish":
                        setup["confidence"] *= 1.20
                        setup["reasoning"] += f" + Order Flow bullish ({of_bias_conf:.0%})"
                        setup["order_flow_confirmed"] = True
                    elif sd_entry.direction == "short" and of_bias == "bearish":
                        setup["confidence"] *= 1.20
                        setup["reasoning"] += f" + Order Flow bearish ({of_bias_conf:.0%})"
                        setup["order_flow_confirmed"] = True
                
                # Boost if Wyckoff confirms
                if wyckoff_state:
                    if sd_entry.direction == "long" and wyckoff_state.trading_bias == "long":
                        setup["confidence"] *= 1.12
                        setup["reasoning"] += f" + Wyckoff {wyckoff_state.phase.value}"
                        setup["wyckoff_confirmed"] = True
                    elif sd_entry.direction == "short" and wyckoff_state.trading_bias == "short":
                        setup["confidence"] *= 1.12
                        setup["reasoning"] += f" + Wyckoff {wyckoff_state.phase.value}"
                        setup["wyckoff_confirmed"] = True
                
                # Boost if ICT confirms
                if ict_state:
                    if sd_entry.direction == "long" and ict_state.current_zone == ZoneType.DISCOUNT:
                        setup["confidence"] *= 1.10
                        setup["reasoning"] += " + ICT discount zone"
                        setup["ict_confirmed"] = True
                    elif sd_entry.direction == "short" and ict_state.current_zone == ZoneType.PREMIUM:
                        setup["confidence"] *= 1.10
                        setup["reasoning"] += " + ICT premium zone"
                        setup["ict_confirmed"] = True
                
                # Check for QUAD CONFLUENCE
                confirmed_count = sum([
                    setup.get("order_flow_confirmed", False),
                    setup.get("wyckoff_confirmed", False),
                    setup.get("ict_confirmed", False),
                    True,  # S/D always confirmed (it's the base)
                ])
                
                if confirmed_count >= 4:
                    setup["confidence"] *= 1.15  # Extra boost for quad confluence
                    setup["reasoning"] += " [QUAD CONFLUENCE]"
                    setup["quad_confluence"] = True
                elif confirmed_count >= 3:
                    setup["reasoning"] += " [Triple Confluence]"
                    setup["triple_confluence"] = True
                
                setups.append(setup)
        
        # 8. Liquidity Concepts Entry Setups (NEW - Smart Money Analysis)
        if lc_state and self.enable_liquidity_concepts:
            # 8a. Liquidity Inducement Entries (Highest Priority - Smart Money Traps)
            if lc_state.active_inducement and lc_state.active_inducement.reversal_detected:
                inducement = lc_state.active_inducement
                ind_type = inducement.inducement_type
                base_conf = self.LC_INDUCEMENT_PATTERNS.get(ind_type, 0.70)
                
                setup = {
                    "type": f"liquidity_concepts_{ind_type.name.lower()}",
                    "direction": inducement.signal_direction,
                    "confidence": base_conf * inducement.reversal_strength,
                    "inducement_type": ind_type.name,
                    "pool_swept": inducement.pool_swept.pool_type.name,
                    "sweep_price": inducement.sweep_price,
                    "reversal_price": inducement.reversal_price,
                    "reasoning": inducement.reasoning,
                }
                
                # Major boost if Order Flow confirms reversal
                if of_bias:
                    if inducement.signal_direction == "long" and of_bias == "bullish":
                        setup["confidence"] *= 1.25
                        setup["reasoning"] += f" + Order Flow bullish ({of_bias_conf:.0%})"
                        setup["order_flow_confirmed"] = True
                    elif inducement.signal_direction == "short" and of_bias == "bearish":
                        setup["confidence"] *= 1.25
                        setup["reasoning"] += f" + Order Flow bearish ({of_bias_conf:.0%})"
                        setup["order_flow_confirmed"] = True
                
                # Boost if Wyckoff confirms (Spring/Upthrust alignment)
                if wyckoff_state:
                    if inducement.signal_direction == "long" and wyckoff_state.current_event.value == "spring":
                        setup["confidence"] *= 1.15
                        setup["reasoning"] += " + Wyckoff Spring"
                        setup["wyckoff_confirmed"] = True
                    elif inducement.signal_direction == "short" and wyckoff_state.current_event.value in ["upthrust", "upthrust_after_distribution"]:
                        setup["confidence"] *= 1.15
                        setup["reasoning"] += " + Wyckoff Upthrust"
                        setup["wyckoff_confirmed"] = True
                
                # High confidence for stop hunt with strong reversal
                if ind_type == LiquidityInducementType.STOP_HUNT and inducement.reversal_strength > 0.8:
                    setup["confidence"] *= 1.10
                    setup["reasoning"] += " [STRONG REVERSAL]"
                
                setups.append(setup)
            
            # 8b. Pool Proximity Entries (Near major liquidity pools)
            # Buy-side pool entries (when bearish near buy-side liquidity)
            if lc_state.nearest_buy_side_pool:
                pool = lc_state.nearest_buy_side_pool
                price_to_pool = (pool.price_level - lc_state.current_price) / lc_state.current_price
                
                # Only trigger if close to the pool (within 1%)
                if 0 < price_to_pool < 0.01:
                    pool_strength = pool.get_pool_strength()
                    base_conf = 0.60 * pool_strength
                    
                    setup = {
                        "type": "liquidity_concepts_buy_side_pool",
                        "direction": "short",  # Expect reversal after sweep
                        "confidence": base_conf,
                        "pool_type": pool.pool_type.name,
                        "pool_price": pool.price_level,
                        "pool_strength": pool_strength,
                        "distance_pct": price_to_pool * 100,
                        "reasoning": f"Approaching buy-side liquidity pool ({pool.pool_type.name}) at {pool.price_level:.2f}",
                    }
                    
                    # Boost if already in bearish trend (confirmation)
                    if lc_state.trend_direction == "bearish":
                        setup["confidence"] *= 1.15
                        setup["reasoning"] += " + Bearish trend continuation expected"
                    
                    # Boost for equal highs (strong liquidity)
                    if pool.pool_type in [LiquidityPoolType.EQUAL_HIGHS, LiquidityPoolType.CLUSTERED]:
                        setup["confidence"] *= 1.20
                        setup["reasoning"] += f" + {pool.pool_type.name} (strong liquidity)"
                    
                    setups.append(setup)
            
            # Sell-side pool entries (when bullish near sell-side liquidity)
            if lc_state.nearest_sell_side_pool:
                pool = lc_state.nearest_sell_side_pool
                price_to_pool = (lc_state.current_price - pool.price_level) / lc_state.current_price
                
                # Only trigger if close to the pool (within 1%)
                if 0 < price_to_pool < 0.01:
                    pool_strength = pool.get_pool_strength()
                    base_conf = 0.60 * pool_strength
                    
                    setup = {
                        "type": "liquidity_concepts_sell_side_pool",
                        "direction": "long",  # Expect reversal after sweep
                        "confidence": base_conf,
                        "pool_type": pool.pool_type.name,
                        "pool_price": pool.price_level,
                        "pool_strength": pool_strength,
                        "distance_pct": price_to_pool * 100,
                        "reasoning": f"Approaching sell-side liquidity pool ({pool.pool_type.name}) at {pool.price_level:.2f}",
                    }
                    
                    # Boost if already in bullish trend (continuation after sweep)
                    if lc_state.trend_direction == "bullish":
                        setup["confidence"] *= 1.15
                        setup["reasoning"] += " + Bullish trend continuation expected"
                    
                    # Boost for equal lows (strong liquidity)
                    if pool.pool_type in [LiquidityPoolType.EQUAL_LOWS, LiquidityPoolType.CLUSTERED]:
                        setup["confidence"] *= 1.20
                        setup["reasoning"] += f" + {pool.pool_type.name} (strong liquidity)"
                    
                    setups.append(setup)
            
            # 8c. Liquidity Void Entries (Price likely to return to fill)
            if lc_state.nearest_void_above and lc_state.trend_direction == "bullish":
                void = lc_state.nearest_void_above
                void_dist = (void.price_low - lc_state.current_price) / lc_state.current_price
                
                if void_dist < 0.03:  # Within 3%
                    setup = {
                        "type": "liquidity_concepts_void_fill_long",
                        "direction": "long",
                        "confidence": 0.55 + (0.10 * (1 - void_dist / 0.03)),  # Higher conf when closer
                        "void_high": void.price_high,
                        "void_low": void.price_low,
                        "void_size": void.size,
                        "target": void.midpoint,
                        "reasoning": f"Bullish trend + void above ({void.price_low:.2f}-{void.price_high:.2f}) likely to fill",
                    }
                    
                    if of_bias == "bullish":
                        setup["confidence"] *= 1.15
                        setup["reasoning"] += " + Order Flow bullish"
                        setup["order_flow_confirmed"] = True
                    
                    setups.append(setup)
            
            if lc_state.nearest_void_below and lc_state.trend_direction == "bearish":
                void = lc_state.nearest_void_below
                void_dist = (lc_state.current_price - void.price_high) / lc_state.current_price
                
                if void_dist < 0.03:  # Within 3%
                    setup = {
                        "type": "liquidity_concepts_void_fill_short",
                        "direction": "short",
                        "confidence": 0.55 + (0.10 * (1 - void_dist / 0.03)),
                        "void_high": void.price_high,
                        "void_low": void.price_low,
                        "void_size": void.size,
                        "target": void.midpoint,
                        "reasoning": f"Bearish trend + void below ({void.price_low:.2f}-{void.price_high:.2f}) likely to fill",
                    }
                    
                    if of_bias == "bearish":
                        setup["confidence"] *= 1.15
                        setup["reasoning"] += " + Order Flow bearish"
                        setup["order_flow_confirmed"] = True
                    
                    setups.append(setup)
            
            # 8d. Strong Swing Level Entries (Based on BOS classification)
            # Strong lows provide support for long entries
            for strong_low in lc_state.strong_lows[-3:]:  # Last 3 strong lows
                dist_to_low = (lc_state.current_price - strong_low.price) / lc_state.current_price
                if 0 < dist_to_low < 0.015:  # Within 1.5%
                    setup = {
                        "type": "liquidity_concepts_strong_low_support",
                        "direction": "long",
                        "confidence": 0.65,
                        "strong_low_price": strong_low.price,
                        "distance_pct": dist_to_low * 100,
                        "reasoning": f"Near strong low (BOS-validated) at {strong_low.price:.2f}",
                    }
                    
                    if of_bias == "bullish":
                        setup["confidence"] *= 1.15
                        setup["reasoning"] += " + Order Flow bullish"
                        setup["order_flow_confirmed"] = True
                    
                    setups.append(setup)
            
            # Strong highs provide resistance for short entries
            for strong_high in lc_state.strong_highs[-3:]:  # Last 3 strong highs
                dist_to_high = (strong_high.price - lc_state.current_price) / lc_state.current_price
                if 0 < dist_to_high < 0.015:  # Within 1.5%
                    setup = {
                        "type": "liquidity_concepts_strong_high_resistance",
                        "direction": "short",
                        "confidence": 0.65,
                        "strong_high_price": strong_high.price,
                        "distance_pct": dist_to_high * 100,
                        "reasoning": f"Near strong high (BOS-validated) at {strong_high.price:.2f}",
                    }
                    
                    if of_bias == "bearish":
                        setup["confidence"] *= 1.15
                        setup["reasoning"] += " + Order Flow bearish"
                        setup["order_flow_confirmed"] = True
                    
                    setups.append(setup)
        
        # Cap confidence at 0.95
        for setup in setups:
            setup["confidence"] = min(setup["confidence"], 0.95)
        
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
    
    def _get_order_flow_signal(
        self,
        symbol: str,
        order_flow_state: Optional[OrderFlowState] = None
    ) -> Optional[Dict[str, Any]]:
        """Get Order Flow signal for a symbol."""
        state = order_flow_state
        if not state and self.order_flow_engine:
            state = self.order_flow_engine.get_state(symbol)
        
        if not state:
            return None
        
        # Get bias from order flow engine
        bias_direction = "neutral"
        bias_confidence = 0.0
        bias_reasoning = ""
        if self.order_flow_engine:
            bias_direction, bias_confidence, bias_reasoning = self.order_flow_engine.get_bias(symbol)
        
        # Convert to numeric direction
        if bias_direction == "bullish":
            direction = 1.0
        elif bias_direction == "bearish":
            direction = -1.0
        else:
            direction = 0.0
        
        return {
            "direction": direction,
            "confidence": bias_confidence,
            "auction_state": state.auction_state.value if state.auction_state else None,
            "cvd": state.current_cvd,
            "signal_strength": state.signal_strength,
            "confluence_score": state.confluence_score,
            "signals": [s.name for s in state.signals],
            "patterns": [p.name for p in state.patterns],
            "reasoning": bias_reasoning,
        }
    
    def _get_supply_demand_signal(
        self,
        symbol: str,
        supply_demand_state: Optional["SupplyDemandState"] = None
    ) -> Optional[Dict[str, Any]]:
        """Get Supply/Demand signal for a symbol."""
        state = supply_demand_state
        if not state and self.supply_demand_engine:
            state = self.supply_demand_engine.get_state(symbol)
        
        if not state:
            return None
        
        # Get bias from supply/demand engine
        bias_direction = "neutral"
        bias_confidence = 0.0
        bias_reasoning = ""
        if self.supply_demand_engine:
            bias_direction, bias_confidence, bias_reasoning = self.supply_demand_engine.get_bias(symbol)
        
        # Convert to numeric direction
        if bias_direction == "bullish":
            direction = 1.0
        elif bias_direction == "bearish":
            direction = -1.0
        else:
            direction = 0.0
        
        # Get zone information
        nearest_demand = state.nearest_demand
        nearest_supply = state.nearest_supply
        
        # Calculate zone proximity scores
        demand_proximity = 0.0
        supply_proximity = 0.0
        demand_reliability = 0.0
        supply_reliability = 0.0
        
        if nearest_demand and self.supply_demand_engine:
            demand_dist = nearest_demand.distance_percent(state.current_price)
            demand_proximity = max(0, 1 - (demand_dist / 5.0))  # Max at 5% distance
            demand_reliability = self.supply_demand_engine.status_tracker.get_zone_reliability(nearest_demand)
        
        if nearest_supply and self.supply_demand_engine:
            supply_dist = nearest_supply.distance_percent(state.current_price)
            supply_proximity = max(0, 1 - (supply_dist / 5.0))
            supply_reliability = self.supply_demand_engine.status_tracker.get_zone_reliability(nearest_supply)
        
        return {
            "direction": direction,
            "confidence": bias_confidence,
            "equilibrium": state.equilibrium_state.name if state.equilibrium_state else None,
            "entry_signal": state.entry_signal.name if state.entry_signal else None,
            "signal_confidence": state.signal_confidence,
            "demand_zones": len(state.demand_zones),
            "supply_zones": len(state.supply_zones),
            "nearest_demand": {
                "upper": nearest_demand.boundary.upper,
                "lower": nearest_demand.boundary.lower,
                "strength": nearest_demand.strength.name,
                "status": nearest_demand.status.name,
                "proximity": demand_proximity,
                "reliability": demand_reliability,
            } if nearest_demand else None,
            "nearest_supply": {
                "upper": nearest_supply.boundary.upper,
                "lower": nearest_supply.boundary.lower,
                "strength": nearest_supply.strength.name,
                "status": nearest_supply.status.name,
                "proximity": supply_proximity,
                "reliability": supply_reliability,
            } if nearest_supply else None,
            "reasoning": bias_reasoning,
        }
    
    def _get_liquidity_concepts_signal(
        self,
        symbol: str,
        liquidity_concepts_state: Optional["LiquidityConceptsState"] = None
    ) -> Optional[Dict[str, Any]]:
        """Get Liquidity Concepts signal for a symbol.
        
        Analyzes:
        - Latent liquidity pools (buy-side/sell-side)
        - Strong/weak swing classification
        - Liquidity voids
        - Fractal market structure
        - Active inducements (stop hunts, sweeps)
        """
        state = liquidity_concepts_state
        if not state and self.liquidity_concepts_engine:
            state = self.liquidity_concepts_engine.get_state(symbol)
        
        if not state:
            return None
        
        # Get bias from liquidity concepts engine
        bias_direction = state.bias or "neutral"
        bias_confidence = state.bias_confidence or 0.0
        bias_reasoning = state.bias_reasoning or ""
        
        # Convert to numeric direction
        if bias_direction == "bullish" or bias_direction == "long":
            direction = 1.0
        elif bias_direction == "bearish" or bias_direction == "short":
            direction = -1.0
        else:
            direction = 0.0
        
        # Get nearest pools information
        nearest_buy_pool = state.nearest_buy_side_pool
        nearest_sell_pool = state.nearest_sell_side_pool
        
        # Calculate pool type for multiplier
        nearest_pool_type = None
        if direction > 0 and nearest_sell_pool:
            nearest_pool_type = nearest_sell_pool.pool_type.name
        elif direction < 0 and nearest_buy_pool:
            nearest_pool_type = nearest_buy_pool.pool_type.name
        
        # Get market structure type
        market_structure = None
        if state.range_structure:
            market_structure = state.range_structure.structure_type.name
        
        # Get active inducement if any
        active_inducement = None
        if state.active_inducement and state.active_inducement.reversal_detected:
            active_inducement = {
                "type": state.active_inducement.inducement_type.name,
                "confidence": state.active_inducement.confidence,
                "direction": state.active_inducement.signal_direction,
            }
            # Inducement signals have higher confidence
            if state.active_inducement.signal_direction == "long":
                direction = 1.0
                bias_confidence = max(bias_confidence, state.active_inducement.confidence)
            elif state.active_inducement.signal_direction == "short":
                direction = -1.0
                bias_confidence = max(bias_confidence, state.active_inducement.confidence)
        
        return {
            "direction": direction,
            "confidence": bias_confidence,
            "trend": state.trend_direction,
            "reasoning": bias_reasoning,
            # Pool information
            "buy_side_pools": len(state.buy_side_pools),
            "sell_side_pools": len(state.sell_side_pools),
            "nearest_buy_pool": {
                "price": nearest_buy_pool.price_level,
                "type": nearest_buy_pool.pool_type.name,
                "strength": nearest_buy_pool.get_pool_strength(),
            } if nearest_buy_pool else None,
            "nearest_sell_pool": {
                "price": nearest_sell_pool.price_level,
                "type": nearest_sell_pool.pool_type.name,
                "strength": nearest_sell_pool.get_pool_strength(),
            } if nearest_sell_pool else None,
            "nearest_pool_type": nearest_pool_type,
            # Swing classification
            "strong_highs": len(state.strong_highs),
            "strong_lows": len(state.strong_lows),
            "weak_highs": len(state.weak_highs),
            "weak_lows": len(state.weak_lows),
            # Void information
            "voids_detected": len(state.voids),
            "nearest_void_above": {
                "high": state.nearest_void_above.price_high,
                "low": state.nearest_void_above.price_low,
                "size": state.nearest_void_above.size,
            } if state.nearest_void_above else None,
            "nearest_void_below": {
                "high": state.nearest_void_below.price_high,
                "low": state.nearest_void_below.price_low,
                "size": state.nearest_void_below.size,
            } if state.nearest_void_below else None,
            # BOS events
            "bos_events": len(state.recent_bos),
            "last_bos_type": state.last_bos.bos_type.name if state.last_bos else None,
            # Market structure
            "market_structure": market_structure,
            # Inducement
            "active_inducement": active_inducement,
            "recent_inducements": len(state.recent_inducements),
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
        order_flow_signal: Optional[Dict[str, Any]],
        supply_demand_signal: Optional[Dict[str, Any]] = None,
        liquidity_concepts_signal: Optional[Dict[str, Any]] = None,
        base_signal: Optional[Dict[str, Any]] = None,
    ) -> Tuple[DirectionEnum, float, str]:
        """Combine all signals into final direction and confidence."""
        
        total_weight = 0.0
        weighted_direction = 0.0
        weighted_confidence = 0.0
        reasoning_parts = []
        
        # Process Wyckoff signal
        if wyckoff_signal and self.wyckoff_weight > 0:
            phase_mult = self.WYCKOFF_PHASE_MULT.get(wyckoff_signal.get("phase", "unknown"), 1.0)
            adj_conf = wyckoff_signal["confidence"] * phase_mult
            
            weighted_direction += wyckoff_signal["direction"] * adj_conf * self.wyckoff_weight
            weighted_confidence += adj_conf * self.wyckoff_weight
            total_weight += self.wyckoff_weight
            
            if wyckoff_signal["direction"] != 0:
                reasoning_parts.append(
                    f"Wyckoff: {wyckoff_signal['phase']} {wyckoff_signal['event']} "
                    f"({adj_conf:.0%})"
                )
        
        # Process ICT signal
        if ict_signal and self.ict_weight > 0:
            # Apply OTE zone boost
            zone_mult = 1.15 if ict_signal.get("in_ote") else 1.0
            adj_conf = ict_signal["confidence"] * zone_mult
            
            weighted_direction += ict_signal["direction"] * adj_conf * self.ict_weight
            weighted_confidence += adj_conf * self.ict_weight
            total_weight += self.ict_weight
            
            if ict_signal["direction"] != 0:
                ote_str = " (OTE)" if ict_signal.get("in_ote") else ""
                reasoning_parts.append(
                    f"ICT: {ict_signal.get('zone', 'N/A')}{ote_str} ({adj_conf:.0%})"
                )
        
        # Process Order Flow signal
        if order_flow_signal and self.order_flow_weight > 0:
            # Apply signal strength boost
            strength_mult = 1.0 + (order_flow_signal.get("signal_strength", 0) * 0.2)
            adj_conf = order_flow_signal["confidence"] * strength_mult
            
            weighted_direction += order_flow_signal["direction"] * adj_conf * self.order_flow_weight
            weighted_confidence += adj_conf * self.order_flow_weight
            total_weight += self.order_flow_weight
            
            if order_flow_signal["direction"] != 0:
                reasoning_parts.append(
                    f"OrderFlow: {order_flow_signal.get('auction_state', 'N/A')} ({adj_conf:.0%})"
                )
        
        # Process Supply/Demand signal
        if supply_demand_signal and self.supply_demand_weight > 0:
            # Apply zone strength/status multipliers
            strength_mult = 1.0
            status_mult = 1.0
            
            # Get relevant zone based on direction
            if supply_demand_signal["direction"] > 0 and supply_demand_signal.get("nearest_demand"):
                zone_info = supply_demand_signal["nearest_demand"]
                strength_mult = self.ZONE_STRENGTH_MULT.get(zone_info["strength"], 1.0)
                status_mult = self.ZONE_STATUS_MULT.get(zone_info["status"], 1.0)
            elif supply_demand_signal["direction"] < 0 and supply_demand_signal.get("nearest_supply"):
                zone_info = supply_demand_signal["nearest_supply"]
                strength_mult = self.ZONE_STRENGTH_MULT.get(zone_info["strength"], 1.0)
                status_mult = self.ZONE_STATUS_MULT.get(zone_info["status"], 1.0)
            
            adj_conf = supply_demand_signal["confidence"] * strength_mult * status_mult
            
            weighted_direction += supply_demand_signal["direction"] * adj_conf * self.supply_demand_weight
            weighted_confidence += adj_conf * self.supply_demand_weight
            total_weight += self.supply_demand_weight
            
            if supply_demand_signal["direction"] != 0:
                equilibrium = supply_demand_signal.get('equilibrium', 'N/A')
                entry = supply_demand_signal.get('entry_signal', '')
                reasoning_parts.append(
                    f"S/D: {equilibrium} {entry} ({adj_conf:.0%})"
                )
        
        # Process Liquidity Concepts signal
        if liquidity_concepts_signal and self.liquidity_concepts_weight > 0:
            # Apply inducement/pool type multipliers
            pool_type_mult = 1.0
            structure_mult = 1.0
            
            # Check for active inducement (highest priority)
            if liquidity_concepts_signal.get("active_inducement"):
                ind_type = liquidity_concepts_signal["active_inducement"].get("type", "")
                pool_type_mult = 1.20  # Inducement boost
            
            # Apply pool type multiplier if near a pool
            if liquidity_concepts_signal.get("nearest_pool_type"):
                pool_type_mult = self.LC_POOL_TYPE_MULT.get(
                    liquidity_concepts_signal["nearest_pool_type"], 1.0
                )
            
            # Apply structure multiplier
            if liquidity_concepts_signal.get("market_structure"):
                structure_mult = self.LC_STRUCTURE_MULT.get(
                    liquidity_concepts_signal["market_structure"], 1.0
                )
            
            adj_conf = liquidity_concepts_signal["confidence"] * pool_type_mult * structure_mult
            
            weighted_direction += liquidity_concepts_signal["direction"] * adj_conf * self.liquidity_concepts_weight
            weighted_confidence += adj_conf * self.liquidity_concepts_weight
            total_weight += self.liquidity_concepts_weight
            
            if liquidity_concepts_signal["direction"] != 0:
                trend = liquidity_concepts_signal.get("trend", "neutral")
                reason = liquidity_concepts_signal.get("reasoning", "")
                short_reason = reason[:30] + "..." if len(reason) > 30 else reason
                reasoning_parts.append(
                    f"LC: {trend} ({adj_conf:.0%})"
                )
        
        # Process base signal
        if base_signal and self.base_weight > 0:
            weighted_direction += base_signal["direction"] * base_signal["confidence"] * self.base_weight
            weighted_confidence += base_signal["confidence"] * self.base_weight
            total_weight += self.base_weight
        
        # Calculate final values
        if total_weight > 0:
            final_direction = weighted_direction / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            return DirectionEnum.NEUTRAL, 0.0, "No signals available"
        
        # Apply confluence bonus - now with 5 methodologies
        methods_agreeing = sum([
            1 for sig in [wyckoff_signal, ict_signal, order_flow_signal, supply_demand_signal, liquidity_concepts_signal]
            if sig and sig["direction"] != 0 and 
            (sig["direction"] > 0) == (final_direction > 0)
        ])
        
        if methods_agreeing >= 5:
            final_confidence *= 1.30  # 30% bonus for PENTA confluence (all 5)
            reasoning_parts.append("PENTA CONFLUENCE")
        elif methods_agreeing >= 4:
            final_confidence *= 1.25  # 25% bonus for QUAD confluence
            reasoning_parts.append("QUAD CONFLUENCE")
        elif methods_agreeing >= 3:
            final_confidence *= 1.15  # 15% bonus for triple confluence
            reasoning_parts.append("TRIPLE CONFLUENCE")
        elif methods_agreeing >= 2:
            final_confidence *= 1.08  # 8% bonus for double confluence
            reasoning_parts.append("Double confluence")
        
        # Cap confidence
        final_confidence = min(final_confidence, 0.95)
        
        # Convert to DirectionEnum
        if final_direction > 0.1:
            direction_enum = DirectionEnum.LONG
        elif final_direction < -0.1:
            direction_enum = DirectionEnum.SHORT
        else:
            direction_enum = DirectionEnum.NEUTRAL
        
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "Mixed signals"
        
        return direction_enum, final_confidence, reasoning
    
    def _analyze_timeframe(
        self,
        timeframe: str,
        snapshot: Any,
        wyckoff_state: Optional[Any],
        ict_state: Optional[Any],
        order_flow_state: Optional[OrderFlowState],
    ) -> Tuple[float, float, float, str]:
        """Analyze a single timeframe with all methodologies.
        
        Returns:
            Tuple of (direction, strength, confidence, reasoning)
        """
        base_imbalance = getattr(snapshot, 'bid_ask_imbalance', 0.0)
        base_depth = getattr(snapshot, 'total_depth', 0)
        
        # Start with base signal
        direction = np.sign(base_imbalance) if abs(base_imbalance) > 0.1 else 0.0
        strength = abs(base_imbalance)
        confidence = getattr(snapshot, 'confidence', 0.5)
        reasons = []
        
        # Apply Wyckoff context
        if wyckoff_state:
            phase_mult = self.WYCKOFF_PHASE_MULT.get(wyckoff_state.phase.value, 1.0)
            confidence *= phase_mult
            
            if wyckoff_state.trading_bias == "long" and direction >= 0:
                direction = max(direction, 0.3)
                reasons.append(f"Wyckoff {wyckoff_state.phase.value}")
            elif wyckoff_state.trading_bias == "short" and direction <= 0:
                direction = min(direction, -0.3)
                reasons.append(f"Wyckoff {wyckoff_state.phase.value}")
        
        # Apply ICT context
        if ict_state:
            if ict_state.daily_bias:
                if ict_state.daily_bias.bias == DailyBias.BULLISH and direction >= 0:
                    confidence *= 1.1
                    reasons.append("ICT bullish bias")
                elif ict_state.daily_bias.bias == DailyBias.BEARISH and direction <= 0:
                    confidence *= 1.1
                    reasons.append("ICT bearish bias")
            
            if ict_state.in_ote:
                confidence *= 1.15
                reasons.append("In OTE zone")
        
        # Apply Order Flow context
        if order_flow_state:
            # CVD confirmation
            cvd_trend = "bullish" if order_flow_state.current_cvd > order_flow_state.cvd_ma else "bearish"
            if cvd_trend == "bullish" and direction >= 0:
                confidence *= 1.12
                strength *= 1.1
                reasons.append("CVD bullish")
            elif cvd_trend == "bearish" and direction <= 0:
                confidence *= 1.12
                strength *= 1.1
                reasons.append("CVD bearish")
            
            # Footprint patterns
            if order_flow_state.patterns:
                for pattern in order_flow_state.patterns:
                    if pattern in self.ORDER_FLOW_ENTRY_PATTERNS:
                        pattern_dir, pattern_conf = self.ORDER_FLOW_ENTRY_PATTERNS[pattern]
                        if pattern_dir == "long" and direction >= 0:
                            confidence *= 1.1
                            reasons.append(pattern.name)
                        elif pattern_dir == "short" and direction <= 0:
                            confidence *= 1.1
                            reasons.append(pattern.name)
        
        # Cap confidence
        confidence = min(confidence, 0.95)
        
        reasoning = "; ".join(reasons) if reasons else f"Base imbalance: {base_imbalance:.2f}"
        
        return direction, strength, confidence, reasoning
    
    def _calculate_quad_confluence(
        self,
        wyckoff_analysis: Dict[str, Any],
        ict_analysis: Dict[str, Any],
        order_flow_analysis: Dict[str, Any],
        supply_demand_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate confluence from all four methodologies."""
        
        confluence = {
            "score": 0.0,
            "direction": "neutral",
            "reasoning": [],
            "best_entry_type": None,
            "methods_aligned": 0,
            "order_flow_confirms": False,
            "supply_demand_confirms": False,
        }
        
        # Count directional signals
        bullish_score = 0.0
        bearish_score = 0.0
        
        # Wyckoff contribution (22.5%)
        if wyckoff_analysis:
            bias = wyckoff_analysis.get("trading_bias")
            conf = wyckoff_analysis.get("confidence", 0)
            if bias == "long":
                bullish_score += conf * 0.225
                confluence["reasoning"].append(f"Wyckoff long ({conf:.0%})")
            elif bias == "short":
                bearish_score += conf * 0.225
                confluence["reasoning"].append(f"Wyckoff short ({conf:.0%})")
        
        # ICT contribution (22.5%)
        if ict_analysis:
            daily_bias = ict_analysis.get("daily_bias")
            conf = ict_analysis.get("daily_bias_confidence", 0)
            if daily_bias == "bullish":
                bullish_score += conf * 0.225
                confluence["reasoning"].append(f"ICT bullish bias ({conf:.0%})")
            elif daily_bias == "bearish":
                bearish_score += conf * 0.225
                confluence["reasoning"].append(f"ICT bearish bias ({conf:.0%})")
            
            # Zone bonus
            zone = ict_analysis.get("current_zone")
            if zone == "discount":
                bullish_score += 0.075
            elif zone == "premium":
                bearish_score += 0.075
        
        # Order Flow contribution (22.5%)
        if order_flow_analysis:
            auction = order_flow_analysis.get("auction_state")
            strength = order_flow_analysis.get("signal_strength", 0)
            
            if auction == "INITIATING_LONG":
                bullish_score += 0.20 + (strength * 0.075)
                confluence["reasoning"].append(f"Order Flow initiating long ({strength:.0%} strength)")
                confluence["order_flow_confirms"] = True
            elif auction == "INITIATING_SHORT":
                bearish_score += 0.20 + (strength * 0.075)
                confluence["reasoning"].append(f"Order Flow initiating short ({strength:.0%} strength)")
                confluence["order_flow_confirms"] = True
            
            # Check for key patterns
            patterns = order_flow_analysis.get("patterns", [])
            if "ABSORPTION_AT_LOW" in patterns:
                bullish_score += 0.10
                confluence["reasoning"].append("Absorption at lows")
            elif "ABSORPTION_AT_HIGH" in patterns:
                bearish_score += 0.10
                confluence["reasoning"].append("Absorption at highs")
            
            # CVD divergence
            signals = order_flow_analysis.get("signals", [])
            if "DIVERGENCE" in signals:
                confluence["reasoning"].append("CVD divergence detected")
        
        # Supply/Demand contribution (22.5%)
        if supply_demand_analysis:
            equilibrium = supply_demand_analysis.get("equilibrium")
            entry_signal = supply_demand_analysis.get("entry_signal")
            signal_conf = supply_demand_analysis.get("signal_confidence", 0)
            
            # Check equilibrium state
            if equilibrium == "SEEKING_HIGHER":
                bullish_score += 0.15
                confluence["reasoning"].append("S/D: Market seeking higher equilibrium")
                confluence["supply_demand_confirms"] = True
            elif equilibrium == "SEEKING_LOWER":
                bearish_score += 0.15
                confluence["reasoning"].append("S/D: Market seeking lower equilibrium")
                confluence["supply_demand_confirms"] = True
            
            # Check entry signals
            if entry_signal in ["DEMAND_ZONE_TOUCH", "DEMAND_ZONE_BOUNCE"]:
                bullish_score += signal_conf * 0.10
                confluence["reasoning"].append(f"S/D: {entry_signal} ({signal_conf:.0%})")
                confluence["supply_demand_confirms"] = True
            elif entry_signal in ["SUPPLY_ZONE_TOUCH", "SUPPLY_ZONE_BOUNCE"]:
                bearish_score += signal_conf * 0.10
                confluence["reasoning"].append(f"S/D: {entry_signal} ({signal_conf:.0%})")
                confluence["supply_demand_confirms"] = True
            
            # Fresh zone bonus
            nearest_demand = supply_demand_analysis.get("nearest_demand")
            nearest_supply = supply_demand_analysis.get("nearest_supply")
            
            if nearest_demand and nearest_demand.get("status") == "FRESH":
                bullish_score += 0.05
                confluence["reasoning"].append("Fresh demand zone nearby")
            if nearest_supply and nearest_supply.get("status") == "FRESH":
                bearish_score += 0.05
                confluence["reasoning"].append("Fresh supply zone nearby")
        
        # Determine direction and score
        if bullish_score > bearish_score:
            confluence["direction"] = "long"
            confluence["score"] = min(bullish_score, 0.95)
            confluence["methods_aligned"] = sum([
                1 if wyckoff_analysis.get("trading_bias") == "long" else 0,
                1 if ict_analysis.get("daily_bias") == "bullish" else 0,
                1 if order_flow_analysis.get("auction_state") in ["INITIATING_LONG", "RESPONSIVE_LONG"] else 0,
                1 if supply_demand_analysis.get("equilibrium") == "SEEKING_HIGHER" else 0,
            ])
        elif bearish_score > bullish_score:
            confluence["direction"] = "short"
            confluence["score"] = min(bearish_score, 0.95)
            confluence["methods_aligned"] = sum([
                1 if wyckoff_analysis.get("trading_bias") == "short" else 0,
                1 if ict_analysis.get("daily_bias") == "bearish" else 0,
                1 if order_flow_analysis.get("auction_state") in ["INITIATING_SHORT", "RESPONSIVE_SHORT"] else 0,
                1 if supply_demand_analysis.get("equilibrium") == "SEEKING_LOWER" else 0,
            ])
        
        # Determine best entry type (now with S/D zones as highest priority for fresh zones)
        if confluence["score"] >= 0.6:
            # Fresh S/D zones have highest priority
            if supply_demand_analysis:
                nearest_demand = supply_demand_analysis.get("nearest_demand")
                nearest_supply = supply_demand_analysis.get("nearest_supply")
                if (confluence["direction"] == "long" and nearest_demand and 
                    nearest_demand.get("status") == "FRESH" and 
                    nearest_demand.get("proximity", 0) > 0.5):
                    confluence["best_entry_type"] = "supply_demand_fresh_zone"
                elif (confluence["direction"] == "short" and nearest_supply and 
                      nearest_supply.get("status") == "FRESH" and 
                      nearest_supply.get("proximity", 0) > 0.5):
                    confluence["best_entry_type"] = "supply_demand_fresh_zone"
            
            # Then Order Flow initiative
            if not confluence["best_entry_type"]:
                if order_flow_analysis.get("auction_state") in ["INITIATING_LONG", "INITIATING_SHORT"]:
                    confluence["best_entry_type"] = "order_flow_initiative"
                elif ict_analysis.get("unfilled_fvgs", 0) > 0:
                    confluence["best_entry_type"] = "ict_fvg"
                elif wyckoff_analysis.get("current_event") in ["spring", "upthrust_after_distribution"]:
                    confluence["best_entry_type"] = "wyckoff_event"
                elif supply_demand_analysis.get("entry_signal") not in [None, "NO_SIGNAL"]:
                    confluence["best_entry_type"] = "supply_demand_zone"
                else:
                    confluence["best_entry_type"] = "quad_confluence_zone"
        
        return confluence
    
    # Keep old method for backwards compatibility
    def _calculate_triple_confluence(
        self,
        wyckoff_analysis: Dict[str, Any],
        ict_analysis: Dict[str, Any],
        order_flow_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate confluence from three methodologies (backwards compatible)."""
        return self._calculate_quad_confluence(
            wyckoff_analysis,
            ict_analysis,
            order_flow_analysis,
            {},  # Empty supply/demand analysis
        )


# Export for package
__all__ = ["LiquidityAgentV5"]
