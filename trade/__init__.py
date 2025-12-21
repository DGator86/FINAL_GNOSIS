"""
Trade agent package.

Includes:
- Trade agents (v1, v2, v3, elite)
- ML-integrated trading engine
- Trading safety controls
"""

from trade.trade_agent_v1 import TradeAgentV1
from trade.trade_agent_v2 import ProposedTrade, TradeAgent

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
