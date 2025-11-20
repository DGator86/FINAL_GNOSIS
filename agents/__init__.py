"""Agents package for Super Gnosis / DHPE v3."""

from agents.base import Agent
from agents.regime_detection_agent import RegimeDetectionAgent

__all__ = ["Agent", "RegimeDetectionAgent"]
from agents.base import Agent, AgentSignal, AgentState, BaseAgent

__all__ = ["Agent", "AgentSignal", "AgentState", "BaseAgent"]
