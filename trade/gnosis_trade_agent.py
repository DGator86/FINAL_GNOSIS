"""
GNOSIS Trade Agent - Full Automated Trading Agent

This is the Trade Agent Layer of the GNOSIS architecture.
It receives composed signals from the Composer Agent and executes trades.

Architecture:
    Composer Agent → Trade Agent → Monitoring Agent
    
Two Trade Agent Types:
1. FullGnosisTradeAgent - Full automated trading with position management
2. AlphaTradeAgent - Signal-only mode for retail traders (Robinhood/Webull)

Author: GNOSIS Trading System
Version: 1.0.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# Import schemas
try:
    from schemas.core_schemas import (
        DirectionEnum,
        TradeIdea,
        AgentSignal,
        PipelineResult,
    )
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    DirectionEnum = None

# Import composer
try:
    from agents.composer.composer_agent_v1 import ComposerAgentV1
    COMPOSER_AVAILABLE = True
except ImportError:
    COMPOSER_AVAILABLE = False
    ComposerAgentV1 = None


class TradeActionType(str, Enum):
    """Types of trade actions."""
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    ADJUST_STOP = "ADJUST_STOP"
    ADJUST_TARGET = "ADJUST_TARGET"
    SCALE_IN = "SCALE_IN"
    SCALE_OUT = "SCALE_OUT"
    HOLD = "HOLD"
    NO_ACTION = "NO_ACTION"


class SignalType(str, Enum):
    """Signal types for Alpha Agent."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TradeAction:
    """A trade action to be executed."""
    action_type: TradeActionType
    symbol: str
    direction: str  # LONG or SHORT
    quantity: int
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Order type
    order_type: str = "MARKET"  # MARKET, LIMIT, STOP
    limit_price: Optional[float] = None
    
    # Risk management
    risk_amount: Optional[float] = None
    position_size_pct: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "symbol": self.symbol,
            "direction": self.direction,
            "quantity": self.quantity,
            "price": self.price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "risk_amount": self.risk_amount,
            "position_size_pct": self.position_size_pct,
        }


@dataclass
class AlphaSignal:
    """A trading signal for retail traders (Alpha Agent output)."""
    symbol: str
    signal_type: SignalType
    direction: str  # BUY, SELL, HOLD
    confidence: float
    
    # Price levels
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Analysis
    reasoning: str = ""
    risk_factors: List[str] = field(default_factory=list)
    catalysts: List[str] = field(default_factory=list)
    
    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: Optional[datetime] = None
    holding_period_days: int = 3
    
    # Simple options (for retail)
    options_play: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "reasoning": self.reasoning,
            "risk_factors": self.risk_factors,
            "catalysts": self.catalysts,
            "timestamp": self.timestamp.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "holding_period_days": self.holding_period_days,
            "options_play": self.options_play,
        }


class BaseTradeAgent(ABC):
    """Base class for trade agents."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_confidence = config.get("min_confidence", 0.6)
        
    @abstractmethod
    def process_composer_output(
        self,
        composer_output: Dict[str, Any],
        symbol: str,
        current_price: float,
    ) -> Any:
        """Process output from Composer Agent."""
        pass


class FullGnosisTradeAgent(BaseTradeAgent):
    """
    Full GNOSIS Trade Agent - Automated Trading with Position Management.
    
    This agent:
    - Receives composed signals from Composer Agent
    - Manages positions (entry, exit, adjustments)
    - Sets stops and targets
    - Handles position sizing
    - Executes trades via broker API
    
    Architecture Position: Trade Agent Layer (receives from Composer)
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        config: Dict[str, Any],
        composer: Optional[Any] = None,
        broker_adapter: Optional[Any] = None,
    ):
        """
        Initialize Full Gnosis Trade Agent.
        
        Args:
            config: Agent configuration
            composer: Composer Agent (receives signals from)
            broker_adapter: Broker API adapter for execution
        """
        super().__init__(config)
        self.composer = composer
        self.broker_adapter = broker_adapter
        
        # Position management
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        
        # Risk parameters
        self.max_position_size = config.get("max_position_size", 0.1)  # 10% of portfolio
        self.max_total_exposure = config.get("max_total_exposure", 0.5)  # 50% max
        self.default_stop_pct = config.get("default_stop_pct", 0.02)  # 2% stop
        self.default_target_pct = config.get("default_target_pct", 0.04)  # 4% target
        
        # Execution parameters
        self.use_limit_orders = config.get("use_limit_orders", True)
        self.limit_offset_pct = config.get("limit_offset_pct", 0.001)  # 0.1%
        
        logger.info(f"FullGnosisTradeAgent v{self.VERSION} initialized")
    
    def process_composer_output(
        self,
        composer_output: Dict[str, Any],
        symbol: str,
        current_price: float,
    ) -> TradeAction:
        """
        Process Composer output and generate trade action.
        
        Args:
            composer_output: Output from Composer Agent
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            TradeAction to execute
        """
        direction = composer_output.get("direction", "NEUTRAL")
        confidence = composer_output.get("confidence", 0.0)
        reasoning = composer_output.get("reasoning", "")
        
        # Check if we have an existing position
        existing_position = self.positions.get(symbol)
        
        # Determine action based on signal and position
        if existing_position:
            return self._manage_existing_position(
                symbol, direction, confidence, current_price, existing_position, reasoning
            )
        else:
            return self._evaluate_new_entry(
                symbol, direction, confidence, current_price, reasoning
            )
    
    def _evaluate_new_entry(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        current_price: float,
        reasoning: str,
    ) -> TradeAction:
        """Evaluate whether to enter a new position."""
        # Check minimum confidence
        if confidence < self.min_confidence:
            return TradeAction(
                action_type=TradeActionType.NO_ACTION,
                symbol=symbol,
                direction="NONE",
                quantity=0,
                confidence=confidence,
                reasoning=f"Confidence {confidence:.2%} below threshold {self.min_confidence:.2%}",
            )
        
        # Determine direction
        if direction in ["LONG", "UP", "BUY", "BULLISH"]:
            action_type = TradeActionType.ENTER_LONG
            stop_loss = current_price * (1 - self.default_stop_pct)
            take_profit = current_price * (1 + self.default_target_pct)
            trade_direction = "LONG"
        elif direction in ["SHORT", "DOWN", "SELL", "BEARISH"]:
            action_type = TradeActionType.ENTER_SHORT
            stop_loss = current_price * (1 + self.default_stop_pct)
            take_profit = current_price * (1 - self.default_target_pct)
            trade_direction = "SHORT"
        else:
            return TradeAction(
                action_type=TradeActionType.NO_ACTION,
                symbol=symbol,
                direction="NONE",
                quantity=0,
                confidence=confidence,
                reasoning=f"Neutral direction: {direction}",
            )
        
        # Calculate position size
        quantity = self._calculate_position_size(symbol, current_price, stop_loss)
        
        return TradeAction(
            action_type=action_type,
            symbol=symbol,
            direction=trade_direction,
            quantity=quantity,
            price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reasoning=reasoning,
            order_type="LIMIT" if self.use_limit_orders else "MARKET",
            limit_price=current_price * (1 + self.limit_offset_pct) if trade_direction == "LONG" else current_price * (1 - self.limit_offset_pct),
        )
    
    def _manage_existing_position(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        current_price: float,
        position: Dict[str, Any],
        reasoning: str,
    ) -> TradeAction:
        """Manage an existing position."""
        position_direction = position.get("direction", "LONG")
        entry_price = position.get("entry_price", current_price)
        
        # Check for exit signals
        if position_direction == "LONG" and direction in ["SHORT", "DOWN", "SELL", "BEARISH"]:
            if confidence > self.min_confidence:
                return TradeAction(
                    action_type=TradeActionType.EXIT_LONG,
                    symbol=symbol,
                    direction="LONG",
                    quantity=position.get("quantity", 0),
                    price=current_price,
                    confidence=confidence,
                    reasoning=f"Exit LONG on bearish signal: {reasoning}",
                )
        
        if position_direction == "SHORT" and direction in ["LONG", "UP", "BUY", "BULLISH"]:
            if confidence > self.min_confidence:
                return TradeAction(
                    action_type=TradeActionType.EXIT_SHORT,
                    symbol=symbol,
                    direction="SHORT",
                    quantity=position.get("quantity", 0),
                    price=current_price,
                    confidence=confidence,
                    reasoning=f"Exit SHORT on bullish signal: {reasoning}",
                )
        
        # Check for stop/target adjustments
        # ... (trailing stop logic, etc.)
        
        return TradeAction(
            action_type=TradeActionType.HOLD,
            symbol=symbol,
            direction=position_direction,
            quantity=position.get("quantity", 0),
            price=current_price,
            confidence=confidence,
            reasoning="Holding position",
        )
    
    def _calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
    ) -> int:
        """Calculate position size based on risk parameters."""
        # Get portfolio value (from broker or config)
        portfolio_value = self.config.get("portfolio_value", 100000)
        risk_per_trade = self.config.get("risk_per_trade", 0.01)  # 1% risk
        
        # Calculate risk amount
        risk_amount = portfolio_value * risk_per_trade
        
        # Calculate per-share risk
        per_share_risk = abs(entry_price - stop_price)
        
        if per_share_risk <= 0:
            return 0
        
        # Calculate shares
        shares = int(risk_amount / per_share_risk)
        
        # Apply max position size limit
        max_shares = int((portfolio_value * self.max_position_size) / entry_price)
        
        return min(shares, max_shares)
    
    def execute_action(self, action: TradeAction) -> Dict[str, Any]:
        """Execute a trade action via broker."""
        if not self.broker_adapter:
            logger.warning("No broker adapter configured - simulating execution")
            return {"status": "simulated", "action": action.to_dict()}
        
        # Execute via broker
        try:
            if action.action_type in [TradeActionType.ENTER_LONG, TradeActionType.ENTER_SHORT]:
                result = self.broker_adapter.place_order(
                    symbol=action.symbol,
                    side="buy" if action.action_type == TradeActionType.ENTER_LONG else "sell",
                    quantity=action.quantity,
                    order_type=action.order_type.lower(),
                    limit_price=action.limit_price,
                )
                
                # Track position
                self.positions[action.symbol] = {
                    "direction": action.direction,
                    "quantity": action.quantity,
                    "entry_price": action.price,
                    "stop_loss": action.stop_loss,
                    "take_profit": action.take_profit,
                    "timestamp": action.timestamp,
                }
                
            elif action.action_type in [TradeActionType.EXIT_LONG, TradeActionType.EXIT_SHORT]:
                result = self.broker_adapter.place_order(
                    symbol=action.symbol,
                    side="sell" if action.action_type == TradeActionType.EXIT_LONG else "buy",
                    quantity=action.quantity,
                    order_type="market",
                )
                
                # Remove position
                self.positions.pop(action.symbol, None)
            
            else:
                result = {"status": "no_action", "action_type": action.action_type.value}
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute action: {e}")
            return {"status": "error", "error": str(e)}


class AlphaTradeAgent(BaseTradeAgent):
    """
    Alpha Trade Agent - Signal Generation for Retail Traders.
    
    This agent:
    - Receives composed signals from Composer Agent
    - Generates simple BUY/SELL/HOLD signals
    - Provides entry, stop, and target levels
    - Suggests simple options plays
    - Designed for Robinhood/Webull users
    
    Architecture Position: Trade Agent Layer (receives from Composer)
    
    Note: This is a SIGNAL-ONLY agent - it does NOT execute trades.
    Users execute trades manually on their platform.
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        config: Dict[str, Any],
        composer: Optional[Any] = None,
    ):
        """
        Initialize Alpha Trade Agent.
        
        Args:
            config: Agent configuration
            composer: Composer Agent (receives signals from)
        """
        super().__init__(config)
        self.composer = composer
        
        # Signal parameters
        self.strong_confidence_threshold = config.get("strong_confidence_threshold", 0.8)
        self.default_stop_pct = config.get("default_stop_pct", 0.03)  # 3% stop
        self.default_target_pct = config.get("default_target_pct", 0.05)  # 5% target
        
        # Holding period
        self.default_holding_days = config.get("default_holding_days", 3)
        self.signal_validity_hours = config.get("signal_validity_hours", 24)
        
        logger.info(f"AlphaTradeAgent v{self.VERSION} initialized")
    
    def process_composer_output(
        self,
        composer_output: Dict[str, Any],
        symbol: str,
        current_price: float,
    ) -> AlphaSignal:
        """
        Process Composer output and generate Alpha signal.
        
        Args:
            composer_output: Output from Composer Agent
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            AlphaSignal for retail trader
        """
        direction = composer_output.get("direction", "NEUTRAL")
        confidence = composer_output.get("confidence", 0.0)
        reasoning = composer_output.get("reasoning", "")
        
        # Determine signal type
        if confidence < self.min_confidence:
            signal_type = SignalType.HOLD
            signal_direction = "HOLD"
            stop_loss = None
            take_profit = None
        elif direction in ["LONG", "UP", "BUY", "BULLISH"]:
            if confidence >= self.strong_confidence_threshold:
                signal_type = SignalType.STRONG_BUY
            else:
                signal_type = SignalType.BUY
            signal_direction = "BUY"
            stop_loss = current_price * (1 - self.default_stop_pct)
            take_profit = current_price * (1 + self.default_target_pct)
        elif direction in ["SHORT", "DOWN", "SELL", "BEARISH"]:
            if confidence >= self.strong_confidence_threshold:
                signal_type = SignalType.STRONG_SELL
            else:
                signal_type = SignalType.SELL
            signal_direction = "SELL"
            stop_loss = current_price * (1 + self.default_stop_pct)
            take_profit = current_price * (1 - self.default_target_pct)
        else:
            signal_type = SignalType.HOLD
            signal_direction = "HOLD"
            stop_loss = None
            take_profit = None
        
        # Create signal
        now = datetime.now(timezone.utc)
        
        return AlphaSignal(
            symbol=symbol,
            signal_type=signal_type,
            direction=signal_direction,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            timestamp=now,
            valid_until=now + timedelta(hours=self.signal_validity_hours),
            holding_period_days=self.default_holding_days,
            options_play=self._suggest_options_play(symbol, signal_direction, current_price, confidence) if signal_direction != "HOLD" else None,
        )
    
    def _suggest_options_play(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        confidence: float,
    ) -> Optional[Dict[str, Any]]:
        """Suggest a simple options play for retail traders."""
        if direction == "HOLD":
            return None
        
        # Simple options suggestion
        if direction == "BUY":
            option_type = "CALL"
            strike_offset = 1.02  # 2% OTM
        else:
            option_type = "PUT"
            strike_offset = 0.98  # 2% OTM
        
        suggested_strike = round(current_price * strike_offset)
        
        return {
            "option_type": option_type,
            "suggested_strike": suggested_strike,
            "dte_range": "14-30 days",
            "strategy": "Long Call" if option_type == "CALL" else "Long Put",
            "risk_note": "Only risk what you can afford to lose. Options can expire worthless.",
        }
    
    def generate_signal(
        self,
        symbol: str,
        composer_output: Dict[str, Any],
        current_price: float,
    ) -> AlphaSignal:
        """Generate an Alpha signal for a symbol."""
        return self.process_composer_output(composer_output, symbol, current_price)


# Factory functions
def create_full_gnosis_agent(
    config: Optional[Dict[str, Any]] = None,
    composer: Optional[Any] = None,
    broker_adapter: Optional[Any] = None,
) -> FullGnosisTradeAgent:
    """Create a Full Gnosis Trade Agent."""
    return FullGnosisTradeAgent(
        config=config or {},
        composer=composer,
        broker_adapter=broker_adapter,
    )


def create_alpha_agent(
    config: Optional[Dict[str, Any]] = None,
    composer: Optional[Any] = None,
) -> AlphaTradeAgent:
    """Create an Alpha Trade Agent."""
    return AlphaTradeAgent(
        config=config or {},
        composer=composer,
    )


__all__ = [
    "TradeActionType",
    "SignalType",
    "TradeAction",
    "AlphaSignal",
    "BaseTradeAgent",
    "FullGnosisTradeAgent",
    "AlphaTradeAgent",
    "create_full_gnosis_agent",
    "create_alpha_agent",
]
