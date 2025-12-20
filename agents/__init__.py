"""Agents package for Super Gnosis / DHPE v3.

Agent Architecture:
==================

This package contains agents that interpret engine outputs and generate
trading signals. Each agent type has multiple versions:

Canonical Versions (used in production):
- HedgeAgentV3: Energy-aware dealer positioning with LSTM integration
- LiquidityAgentV1: Basic liquidity scoring and tradability
- SentimentAgentV1: Multi-source sentiment aggregation
- ComposerAgentV1: Weighted consensus building

Experimental Versions (for advanced use):
- HedgeAgentV4: Extended regime detection
- LiquidityAgentV2/V3: Multi-timeframe depth analysis
- SentimentAgentV2/V3: NLP-enhanced sentiment

Usage:
------
For standard pipeline usage, import from this module:
    from agents import HedgeAgentV3, LiquidityAgentV1, SentimentAgentV1

For experimental versions, import directly:
    from agents.liquidity_agent_v2 import LiquidityAgentV2
"""

# Base classes
from agents.base import Agent

# Canonical agent versions (recommended for production)
from agents.hedge_agent_v3 import HedgeAgentV3
from agents.liquidity_agent_v1 import LiquidityAgentV1
from agents.sentiment_agent_v1 import SentimentAgentV1
from agents.composer.composer_agent_v1 import ComposerAgentV1

__all__ = [
    # Base
    "Agent",
    # Canonical production agents
    "HedgeAgentV3",
    "LiquidityAgentV1", 
    "SentimentAgentV1",
    "ComposerAgentV1",
]
