"""Lightweight base classes for agent implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class AgentSignal:
    """Structured signal emitted by an agent."""

    agent_id: str
    timestamp: datetime
    signal_type: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    position_size: Optional[float] = None


class BaseAgent:
    """Base class for agents that track configuration and state."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.state: Dict[str, Any] = {}

    def analyze(self, market_data: Dict[str, Any], features: Dict[str, Any]) -> AgentSignal:
        """Analyze inputs and produce an :class:`AgentSignal`."""

        raise NotImplementedError("analyze must be implemented by subclasses")

    def update_state(
        self, market_data: Dict[str, Any], execution_result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update internal state from market data and optional execution results."""

        raise NotImplementedError("update_state must be implemented by subclasses")
