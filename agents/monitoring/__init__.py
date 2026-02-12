"""
GNOSIS Monitoring Agent Layer

Monitors trade agents and provides feedback for continuous improvement.

Architecture:
    Trade Agent Layer → Monitoring Agent Layer → ML Feedback Loop

Monitor Types:
1. GnosisMonitor - Monitors Full Gnosis Trade Agent (positions, P&L, risk)
2. AlphaMonitor - Monitors Alpha Trade Agent (signal accuracy, win rate)

Both monitors support:
- PENTA methodology tracking
- Signal accuracy by confluence level
- Performance metrics per symbol
- Risk alerts and thresholds
"""

from agents.monitoring.gnosis_monitor import (
    # Enums
    EventType,
    AlertLevel,
    # Dataclasses
    GnosisMonitor,
    AlphaMonitor,
    MonitoringEvent,
    PerformanceMetrics,
    BaseMonitor,
    # Factory functions
    create_gnosis_monitor,
    create_alpha_monitor,
)

__all__ = [
    # Enums
    "EventType",
    "AlertLevel",
    # Monitor classes
    "GnosisMonitor",
    "AlphaMonitor",
    "BaseMonitor",
    # Data classes
    "MonitoringEvent",
    "PerformanceMetrics",
    # Factory functions
    "create_gnosis_monitor",
    "create_alpha_monitor",
]
