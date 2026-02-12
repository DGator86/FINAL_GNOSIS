"""
Trade agent package.

Includes:
- Trade agents (v1, v2, v3, elite)
- GNOSIS Trade Agents (Full Gnosis + Alpha)
- ML-integrated trading engine
- Trading safety controls

GNOSIS Architecture:
    Composer Agent → Trade Agent Layer → Monitoring Agent Layer
    
Trade Agent Types:
1. FullGnosisTradeAgent(V2) - Full automated trading
2. AlphaTradeAgent(V2) - Signal-only for retail (Robinhood/Webull)
"""

from trade.trade_agent_v1 import TradeAgentV1
from trade.trade_agent_v2 import ProposedTrade, TradeAgent

__all__ = ["TradeAgentV1", "TradeAgent", "ProposedTrade"]

# GNOSIS Trade Agents V1
from trade.gnosis_trade_agent import (
    FullGnosisTradeAgent,
    AlphaTradeAgent,
    TradeAction as TradeActionV1,
    AlphaSignal as AlphaSignalV1,
    create_full_gnosis_agent,
    create_alpha_agent,
)

# GNOSIS Trade Agents V2 (Full Architecture)
from trade.gnosis_trade_agent_v2 import (
    FullGnosisTradeAgentV2,
    AlphaTradeAgentV2,
    TradeAction,
    AlphaSignalV2,
    TradeActionType,
    SignalStrength,
    SignalType,
    create_full_gnosis_agent_v2,
    create_alpha_agent_v2,
)

# ML Trading Engine
from trade.ml_trading_engine import (
    MLTradingState,
    MLTradingConfig,
    MLPosition,
    MLTradingStats,
    MLTradingEngine,
    create_ml_trading_engine,
)

# Trading Safety
from trade.trading_safety import (
    SafetyStatus,
    CircuitBreakerState,
    SafetyConfig,
    SafetyMetrics,
    TradeValidationResult,
    TradingSafetyManager,
    create_safety_manager,
)

__all__ = [
    # Legacy
    "TradeAgentV1",
    "TradeAgent",
    "ProposedTrade",
    # GNOSIS Trade Agents V1
    "FullGnosisTradeAgent",
    "AlphaTradeAgent",
    "TradeActionV1",
    "AlphaSignalV1",
    "create_full_gnosis_agent",
    "create_alpha_agent",
    # GNOSIS Trade Agents V2 (Recommended)
    "FullGnosisTradeAgentV2",
    "AlphaTradeAgentV2",
    "TradeAction",
    "AlphaSignalV2",
    "TradeActionType",
    "SignalStrength",
    "SignalType",
    "create_full_gnosis_agent_v2",
    "create_alpha_agent_v2",
    # ML Trading
    "MLTradingState",
    "MLTradingConfig",
    "MLPosition",
    "MLTradingStats",
    "MLTradingEngine",
    "create_ml_trading_engine",
    # Safety
    "SafetyStatus",
    "CircuitBreakerState",
    "SafetyConfig",
    "SafetyMetrics",
    "TradeValidationResult",
    "TradingSafetyManager",
    "create_safety_manager",
]
