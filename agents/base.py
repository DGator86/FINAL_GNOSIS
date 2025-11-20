"""
Base Agent Class for GNOSIS Intelligent Agents
Provides common functionality for all trading agents.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from schemas.core_schemas import AgentSuggestion, PipelineResult


@dataclass
class AgentSignal:
    """Trading signal from an agent"""

    agent_id: str
    timestamp: datetime
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-1
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSignal":
        """Create from dictionary"""
        data = {**data, "timestamp": datetime.fromisoformat(data["timestamp"])}
        return cls(**data)


@dataclass
class AgentState:
    """Agent internal state"""

    agent_id: str
    state_data: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "state_data": self.state_data,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseAgent(ABC):
    """Base class for all GNOSIS intelligent agents"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.logger = self._setup_logger()

        # Agent state
        self.state: Dict[str, Any] = {}
        self.signal_history: List[AgentSignal] = []
        self.performance_metrics: Dict[str, Any] = {}

        # Configuration
        self.min_confidence = config.get("min_confidence", 0.6)
        self.max_position_size = config.get("max_position_size", 0.1)
        self.risk_tolerance = config.get("risk_tolerance", 0.02)

        # Feature requirements
        self.required_features: List[str] = []

    def _setup_logger(self) -> logging.Logger:
        """Setup agent logger"""
        logger = logging.getLogger(f"gnosis.agent.{self.agent_id}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    @abstractmethod
    def analyze(
        self, market_data: Dict[str, Any], features: Dict[str, Any]
    ) -> AgentSignal:
        """
        Analyze market data and generate trading signal

        Args:
            market_data: Current market data (prices, volume, etc.)
            features: Computed features from feature builder

        Returns:
            AgentSignal with trading recommendation
        """

    @abstractmethod
    def update_state(
        self, market_data: Dict[str, Any], execution_result: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update agent internal state

        Args:
            market_data: Latest market data
            execution_result: Result of last signal execution (if any)
        """

    def validate_features(self, features: Dict[str, Any]) -> bool:
        """Validate required features are present"""
        for feature in self.required_features:
            if feature not in features:
                self.logger.warning(f"Missing required feature: {feature}")
                return False
        return True

    def calculate_position_size(
        self, confidence: float, current_price: float, account_balance: float
    ) -> float:
        """
        Calculate position size based on confidence and risk

        Args:
            confidence: Signal confidence (0-1)
            current_price: Current asset price
            account_balance: Available account balance

        Returns:
            Position size (fraction of balance)
        """
        # Base position size scaled by confidence
        base_size = self.max_position_size * confidence

        # Adjust for risk tolerance
        risk_adjusted_size = base_size * (1 - self.risk_tolerance)

        # Ensure minimum and maximum bounds
        position_size = float(np.clip(risk_adjusted_size, 0.01, self.max_position_size))

        return position_size

    def calculate_stop_loss(self, entry_price: float, signal_type: str, volatility: float) -> float:
        """
        Calculate stop loss price

        Args:
            entry_price: Entry price for position
            signal_type: 'buy' or 'sell'
            volatility: Current market volatility

        Returns:
            Stop loss price
        """
        # Stop loss at 2x volatility or risk tolerance, whichever is smaller
        stop_distance = min(2 * volatility, self.risk_tolerance) * entry_price

        if signal_type == "buy":
            return entry_price - stop_distance
        return entry_price + stop_distance

    def calculate_take_profit(
        self, entry_price: float, signal_type: str, confidence: float, volatility: float
    ) -> float:
        """
        Calculate take profit price

        Args:
            entry_price: Entry price for position
            signal_type: 'buy' or 'sell'
            confidence: Signal confidence
            volatility: Current market volatility

        Returns:
            Take profit price
        """
        # Take profit scaled by confidence and volatility
        profit_target = (2 + confidence) * volatility * entry_price

        if signal_type == "buy":
            return entry_price + profit_target
        return entry_price - profit_target

    def log_signal(self, signal: AgentSignal) -> None:
        """Log generated signal"""
        self.signal_history.append(signal)

        self.logger.info(
            "Generated %s signal with confidence %.2f",
            signal.signal_type,
            signal.confidence,
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get agent performance summary"""
        if not self.signal_history:
            return {"total_signals": 0}

        signals_by_type = {
            "buy": len([s for s in self.signal_history if s.signal_type == "buy"]),
            "sell": len([s for s in self.signal_history if s.signal_type == "sell"]),
            "hold": len([s for s in self.signal_history if s.signal_type == "hold"]),
        }

        confidences = [s.confidence for s in self.signal_history]

        return {
            "total_signals": len(self.signal_history),
            "signals_by_type": signals_by_type,
            "avg_confidence": float(np.mean(confidences)),
            "min_confidence": float(np.min(confidences)),
            "max_confidence": float(np.max(confidences)),
            "performance_metrics": self.performance_metrics,
        }

    def save_state(self, filepath: str) -> None:
        """Save agent state to file"""
        state_data = {
            "agent_id": self.agent_id,
            "config": self.config,
            "state": self.state,
            "signal_history": [s.to_dict() for s in self.signal_history],
            "performance_metrics": self.performance_metrics,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(state_data, f, indent=2)

        self.logger.info("Saved agent state to %s", filepath)

    def load_state(self, filepath: str) -> None:
        """Load agent state from file"""
        with open(filepath, "r") as f:
            state_data = json.load(f)

        self.state = state_data["state"]
        self.signal_history = [AgentSignal.from_dict(s) for s in state_data["signal_history"]]
        self.performance_metrics = state_data["performance_metrics"]

        self.logger.info("Loaded agent state from %s", filepath)


class Agent(Protocol):
    """Protocol for all agents."""

    def suggest(self, pipeline_result: PipelineResult, timestamp: datetime) -> AgentSuggestion:
        """Generate a suggestion based on pipeline results."""
        ...
