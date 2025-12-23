"""Liquidity engine package.

Version History:
- v1: Basic liquidity metrics
- v2: Enhanced with depth analysis  
- v3: 0DTE support and gamma squeeze detection
- v4: Full Wyckoff Methodology Integration (VSA, Phases, Events)
- v5: UNIFIED Engine - Market Quality + PENTA Methodology

PENTA Methodology Sub-Engines:
- Wyckoff (v4): VSA, Phases, Events, Structures
- ICT: Inner Circle Trader (FVGs, Order Blocks, Liquidity Sweeps)
- OrderFlow: Footprint, CVD, Volume Profile
- SupplyDemand: Zones based on economic principles
- LiquidityConcepts: Pools, Voids, Strong/Weak H/L, Fractal, Inducement

Architecture:
    Data Adapters → LiquidityEngineV5 → LiquidityAgentV5 → Composer
    
The LiquidityEngineV5 contains all 5 PENTA sub-engines internally.
"""

from engines.liquidity.liquidity_engine_v1 import LiquidityEngineV1
from engines.liquidity.liquidity_engine_v2 import LiquidityEngineV2
from engines.liquidity.liquidity_engine_v3 import LiquidityEngineV3
from engines.liquidity.liquidity_engine_v4 import (
    LiquidityEngineV4,
    WyckoffPhase,
    WyckoffEvent,
    VSASignal,
    MarketStructure,
    WyckoffState,
    WyckoffSnapshot,
    VolumeSpreadAnalyzer,
    WyckoffStructureDetector,
    WyckoffEventDetector,
    WyckoffPhaseTracker,
)
from engines.liquidity.order_flow_engine import (
    OrderFlowEngine,
    create_order_flow_engine,
    # Analyzers
    FootprintAnalyzer,
    CVDAnalyzer,
    VolumeProfileAnalyzer,
    # Enums
    OrderFlowSignal,
    DeltaType,
    VolumeNodeType,
    AuctionState,
    FootprintPattern,
    ImbalanceType,
    MarketParticipant,
    OrderType,
    # Data Structures
    FootprintCell,
    FootprintBar,
    CVDPoint,
    VolumeNode,
    VolumeProfile,
    OrderFlowState,
    OrderFlowEntry,
)
from engines.liquidity.ict_engine import (
    ICTEngine,
    # Data Structures
    SwingPoint as ICTSwingPoint,  # Renamed to avoid conflict with S&D
    LiquidityLevel,
    PremiumDiscountZone,
    FairValueGap,
    VolumeImbalance,
    OrderBlock,
    DailyBiasResult,
    LiquiditySweep,
    ICTSnapshot,
    # Enums
    LiquidityType,
    SwingType,
    HighLowType,
    FVGType,
    FVGStatus,
    OrderBlockType,
    DailyBias,
    ZoneType as ICTZoneType,  # Renamed to avoid conflict with S&D
    # Components
    SwingPointDetector as ICTSwingPointDetector,  # Renamed to avoid conflict
    PremiumDiscountCalculator,
    FairValueGapDetector,
    OrderBlockDetector,
    DailyBiasCalculator,
    LiquiditySweepDetector,
)
from engines.liquidity.supply_demand_engine import (
    SupplyDemandEngine,
    create_supply_demand_engine,
    # Detectors
    SwingPointDetector as SDSwingPointDetector,
    SupplyDemandZoneDetector,
    ZoneBoundaryCalculator,
    ZoneStatusTracker,
    # Enums
    ZoneType as SDZoneType,
    ZoneStrength,
    ZoneStatus,
    MarketEquilibrium,
    ShiftType,
    EntrySignal as SDEntrySignal,
    # Data Structures
    SwingPoint as SDSwingPoint,
    ZoneBoundary,
    SupplyDemandZone,
    SupplyDemandState,
    ZoneEntry,
)
from engines.liquidity.liquidity_concepts_engine import (
    LiquidityConceptsEngine,
    create_liquidity_concepts_engine,
    # Analyzers
    ExtendedSwingAnalyzer,
    LiquidityPoolDetector,
    LiquidityVoidDetector,
    FractalStructureAnalyzer,
    LiquidityInducementDetector,
    # Enums
    LiquidityPoolType,
    LiquidityPoolSide,
    SwingStrength,
    MarketStructureType,
    BOSType,
    LiquidityInducementType,
    # Data Structures
    SwingPointExtended,
    LiquidityPool,
    LiquidityVoid,
    BreakOfStructure,
    FractalStructureAnalysis,
    LiquidityInducement,
    LiquidityConceptsState,
)
# V5: Unified Engine with PENTA Methodology
from engines.liquidity.liquidity_engine_v5 import (
    LiquidityEngineV5,
    LiquidityEngineV5Snapshot,
    PENTAState,
    MarketQualityGrade,
)

__all__ = [
    # Liquidity Engines
    "LiquidityEngineV1",
    "LiquidityEngineV2",
    "LiquidityEngineV3",
    "LiquidityEngineV4",
    "LiquidityEngineV5",  # Main unified engine with PENTA
    # V5 Components
    "LiquidityEngineV5Snapshot",
    "PENTAState",
    "MarketQualityGrade",
    # Wyckoff components
    "WyckoffPhase",
    "WyckoffEvent",
    "VSASignal",
    "MarketStructure",
    "WyckoffState",
    "WyckoffSnapshot",
    "VolumeSpreadAnalyzer",
    "WyckoffStructureDetector",
    "WyckoffEventDetector",
    "WyckoffPhaseTracker",
    # Order Flow Engine
    "OrderFlowEngine",
    "create_order_flow_engine",
    # Order Flow Analyzers
    "FootprintAnalyzer",
    "CVDAnalyzer",
    "VolumeProfileAnalyzer",
    # Order Flow Enums
    "OrderFlowSignal",
    "DeltaType",
    "VolumeNodeType",
    "AuctionState",
    "FootprintPattern",
    "ImbalanceType",
    "MarketParticipant",
    "OrderType",
    # Order Flow Data Structures
    "FootprintCell",
    "FootprintBar",
    "CVDPoint",
    "VolumeNode",
    "VolumeProfile",
    "OrderFlowState",
    "OrderFlowEntry",
    # ICT Engine
    "ICTEngine",
    # ICT Data Structures
    "ICTSwingPoint",
    "LiquidityLevel",
    "PremiumDiscountZone",
    "FairValueGap",
    "VolumeImbalance",
    "OrderBlock",
    "DailyBiasResult",
    "LiquiditySweep",
    "ICTSnapshot",
    # ICT Enums
    "LiquidityType",
    "SwingType",
    "HighLowType",
    "FVGType",
    "FVGStatus",
    "OrderBlockType",
    "DailyBias",
    "ICTZoneType",
    # ICT Components
    "ICTSwingPointDetector",
    "PremiumDiscountCalculator",
    "FairValueGapDetector",
    "OrderBlockDetector",
    "DailyBiasCalculator",
    "LiquiditySweepDetector",
    # Supply and Demand Engine
    "SupplyDemandEngine",
    "create_supply_demand_engine",
    # S&D Detectors
    "SDSwingPointDetector",
    "SupplyDemandZoneDetector",
    "ZoneBoundaryCalculator",
    "ZoneStatusTracker",
    # S&D Enums
    "SDZoneType",
    "ZoneStrength",
    "ZoneStatus",
    "MarketEquilibrium",
    "ShiftType",
    "SDEntrySignal",
    # S&D Data Structures
    "SDSwingPoint",
    "ZoneBoundary",
    "SupplyDemandZone",
    "SupplyDemandState",
    "ZoneEntry",
    # Liquidity Concepts Engine
    "LiquidityConceptsEngine",
    "create_liquidity_concepts_engine",
    # Liquidity Concepts Analyzers
    "ExtendedSwingAnalyzer",
    "LiquidityPoolDetector",
    "LiquidityVoidDetector",
    "FractalStructureAnalyzer",
    "LiquidityInducementDetector",
    # Liquidity Concepts Enums
    "LiquidityPoolType",
    "LiquidityPoolSide",
    "SwingStrength",
    "MarketStructureType",
    "BOSType",
    "LiquidityInducementType",
    # Liquidity Concepts Data Structures
    "SwingPointExtended",
    "LiquidityPool",
    "LiquidityVoid",
    "BreakOfStructure",
    "FractalStructureAnalysis",
    "LiquidityInducement",
    "LiquidityConceptsState",
]
