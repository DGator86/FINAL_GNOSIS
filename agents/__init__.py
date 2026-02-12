"""Agents package for Super Gnosis / DHPE v3.

Agent Architecture:
==================

This package contains agents that interpret engine outputs and generate
trading signals. Each agent type has multiple versions:

Canonical Versions (used in production):
- HedgeAgentV3: Energy-aware dealer positioning with LSTM integration
- HedgeAgentV4: Extended regime detection and multi-timeframe confluence
- LiquidityAgentV1: Basic liquidity scoring and tradability
- LiquidityAgentV3: Wyckoff-enhanced liquidity with VSA/phase/event analysis
- LiquidityAgentV4: Unified Wyckoff + ICT methodology
- LiquidityAgentV5: PENTA methodology (Wyckoff + ICT + Order Flow + S&D + Liquidity Concepts) [RECOMMENDED]
- SentimentAgentV1: Multi-source sentiment aggregation
- ComposerAgentV1: Weighted consensus building
- ComposerAgentV2: Multi-timeframe analysis with ConfidenceBuilder

Experimental Versions (for advanced use):
- LiquidityAgentV2: Multi-timeframe depth analysis (experimental)
- SentimentAgentV2/V3: NLP-enhanced sentiment

Methodology Integration:
-----------------------
LiquidityAgentV3 (Wyckoff only):
- Volume Spread Analysis (VSA)
- Wyckoff Phase Tracking (A-E)
- Seven Logical Events Detection
- Accumulation/Distribution Structure Recognition
- Spring/Upthrust Entry Signals

LiquidityAgentV4 (Wyckoff + ICT):
- All Wyckoff features from V3
- ICT Swing Points & Liquidity Levels
- Fair Value Gaps (FVG) - BISI/SIBI
- Order Blocks (High/Low Probability)
- Premium/Discount Zones & OTE
- Daily Bias Calculation
- Liquidity Sweep Detection
- Combined confluence scoring

LiquidityAgentV5 (Wyckoff + ICT + Order Flow + Supply/Demand) - RECOMMENDED:
- All features from V3 and V4
- Footprint Charts (bid/ask aggression, imbalance, absorption)
- Cumulative Volume Delta (CVD) analysis
- Volume Profile (POC, Value Area, HVN/LVN)
- Auction Market Theory integration
- Supply & Demand Zones (economic principles-based)
- Zone Strength & Status tracking (Fresh > Tested > Retested)
- Built-in risk management (Stop Loss, TP1-TP4 levels)
- QUAD confluence scoring (25% bonus for 4-method alignment)
- Order flow confirmed entries

Usage:
------
For standard pipeline usage:
    from agents import HedgeAgentV3, LiquidityAgentV1, SentimentAgentV1

For Wyckoff-only trading:
    from agents import LiquidityAgentV3
    agent = LiquidityAgentV3(config, liquidity_engine=wyckoff_engine)

For combined Wyckoff + ICT trading:
    from agents import LiquidityAgentV4
    agent = LiquidityAgentV4(config, wyckoff_engine=wyckoff_engine, ict_engine=ict_engine)

For QUAD methodology trading (RECOMMENDED):
    from agents import LiquidityAgentV5
    from engines.engine_factory import create_unified_analysis_engines
    
    engines = create_unified_analysis_engines()
    agent = LiquidityAgentV5(
        config,
        wyckoff_engine=engines['wyckoff_engine'],
        ict_engine=engines['ict_engine'],
        order_flow_engine=engines['order_flow_engine'],
        supply_demand_engine=engines['supply_demand_engine']
    )
    
    # Get high-confluence entry setups with built-in risk management
    setups = agent.get_entry_setups(symbol)
    
    # Get detailed analysis from all 4 methodologies
    analysis = agent.get_confluence_analysis(symbol)
    
    # Key features:
    # - setups include stop_loss, take_profit_1/2/3/4 from S&D zones
    # - QUAD CONFLUENCE bonus when all 4 methodologies align
    # - Fresh zones prioritized (highest probability trades)
"""

# Base classes
from agents.base import Agent

# Canonical agent versions (recommended for production)
from agents.hedge_agent_v3 import HedgeAgentV3
from agents.hedge_agent_v4 import HedgeAgentV4
from agents.liquidity_agent_v1 import LiquidityAgentV1
from agents.liquidity_agent_v3 import LiquidityAgentV3
from agents.liquidity_agent_v4 import LiquidityAgentV4
from agents.liquidity_agent_v5 import LiquidityAgentV5
from agents.sentiment_agent_v1 import SentimentAgentV1
from agents.composer.composer_agent_v1 import ComposerAgentV1
from agents.composer.composer_agent_v2 import ComposerAgentV2

__all__ = [
    # Base
    "Agent",
    # Canonical production agents
    "HedgeAgentV3",
    "HedgeAgentV4",
    "LiquidityAgentV1",
    "LiquidityAgentV3",  # Wyckoff-enhanced
    "LiquidityAgentV4",  # Wyckoff + ICT combined
    "LiquidityAgentV5",  # PENTA: Wyckoff + ICT + Order Flow + S&D + Liquidity Concepts (RECOMMENDED)
    "SentimentAgentV1",
    "ComposerAgentV1",
    "ComposerAgentV2",
]
