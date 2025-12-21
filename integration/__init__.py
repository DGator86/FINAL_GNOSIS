"""
Integration Module - Central Hub for Trading Components

This module provides the Trading Hub which connects all trading components:
- Paper Trading Engine
- WebSocket Broadcasting
- Anomaly Detection
- Options Flow Scanner
- Notification Bot
- Greeks Hedger
- Portfolio Analytics

Usage:
    from integration import get_trading_hub, start_trading_hub
    
    # Get hub instance
    hub = get_trading_hub()
    
    # Start hub with all components
    await start_trading_hub()

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from integration.trading_hub import (
    TradingHub,
    HubConfig,
    HubState,
    TradingAlert,
    AlertPriority,
    get_trading_hub,
    start_trading_hub,
    stop_trading_hub,
)

__all__ = [
    "TradingHub",
    "HubConfig",
    "HubState",
    "TradingAlert",
    "AlertPriority",
    "get_trading_hub",
    "start_trading_hub",
    "stop_trading_hub",
]
