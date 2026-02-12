"""
GNOSIS Trade Agent V2 - Full GNOSIS Architecture Integration.

This is the Trade Agent Layer of the GNOSIS architecture.
It receives composed signals from the Composer Agent and generates trades.

Architecture:
    Composer Agent V4 → Trade Agent V2 → Monitoring Agent
    
Two Trade Agent Types:
1. FullGnosisTradeAgentV2 - Full automated trading with position management
2. AlphaTradeAgentV2 - Signal-only mode for retail traders (Robinhood/Webull)

Key Features:
- Proper integration with ComposerAgentV4
- PENTA methodology confluence bonuses
- Express Lane support (0DTE, Cheap Calls)
- Enhanced risk management

Author: GNOSIS Trading System
Version: 2.0.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# Import composer
try:
    from agents.composer.composer_agent_v4 import (
        ComposerAgentV4,
        ComposerOutput,
        ComposerMode,
    )
    COMPOSER_V4_AVAILABLE = True
except ImportError:
    COMPOSER_V4_AVAILABLE = False
    ComposerAgentV4 = None
    ComposerOutput = None
    ComposerMode = None

# Import monitoring
try:
    from agents.monitoring import GnosisMonitor, AlphaMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    GnosisMonitor = None
    AlphaMonitor = None


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


class SignalStrength(str, Enum):
    """Signal strength levels."""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


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
    
    # PENTA methodology
    penta_confluence: Optional[str] = None
    penta_bonus: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "symbol": self.symbol,
            "direction": self.direction,
            "quantity": self.quantity,
            "price": self.price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "risk_amount": self.risk_amount,
            "position_size_pct": self.position_size_pct,
            "penta_confluence": self.penta_confluence,
            "penta_bonus": self.penta_bonus,
        }


@dataclass
class AlphaSignalV2:
    """Enhanced trading signal for retail traders (Alpha Agent V2 output)."""
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
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
    agent_signals: Dict[str, str] = field(default_factory=dict)
    
    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: Optional[datetime] = None
    holding_period_days: int = 3
    
    # PENTA methodology
    penta_confluence: Optional[str] = None
    penta_confidence_bonus: float = 0.0
    
    # Simple options (for retail)
    options_play: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "strength": self.strength.value,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "reasoning": self.reasoning,
            "risk_factors": self.risk_factors,
            "catalysts": self.catalysts,
            "agent_signals": self.agent_signals,
            "timestamp": self.timestamp.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "holding_period_days": self.holding_period_days,
            "penta_confluence": self.penta_confluence,
            "penta_confidence_bonus": round(self.penta_confidence_bonus, 4),
            "options_play": self.options_play,
        }
    
    def to_robinhood_format(self) -> str:
        """Format signal for Robinhood-style display."""
        emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(self.direction, "⚪")
        strength_emoji = {"STRONG": "💪", "MODERATE": "👍", "WEAK": "👋"}.get(self.strength.value, "")
        
        lines = [
            f"{emoji} {self.symbol}: {self.direction} {strength_emoji}",
            f"Confidence: {self.confidence * 100:.0f}%",
        ]
        
        if self.penta_confluence:
            lines.append(f"🧠 {self.penta_confluence} confluence")
        
        if self.entry_price:
            lines.append(f"Entry: ${self.entry_price:.2f}")
        if self.stop_loss:
            lines.append(f"Stop: ${self.stop_loss:.2f}")
        if self.take_profit:
            lines.append(f"Target: ${self.take_profit:.2f}")
        
        lines.append(f"Hold: {self.holding_period_days} days")
        
        if self.risk_factors:
            lines.append(f"⚠️ {', '.join(self.risk_factors[:2])}")
        
        return "\n".join(lines)


class BaseTradeAgentV2(ABC):
    """Base class for GNOSIS trade agents V2."""
    
    VERSION = "2.0.0"
    
    def __init__(
        self,
        config: Dict[str, Any],
        composer: Optional[ComposerAgentV4] = None,
    ):
        self.config = config
        self.composer = composer
        self.min_confidence = config.get("min_confidence", 0.6)
        
    @abstractmethod
    def process_composer_output(
        self,
        composer_output: ComposerOutput,
        symbol: str,
        current_price: float,
    ) -> Any:
        """Process output from Composer Agent V4."""
        pass


class FullGnosisTradeAgentV2(BaseTradeAgentV2):
    """
    Full GNOSIS Trade Agent V2 - Automated Trading with Position Management.
    
    This agent:
    - Receives composed signals from ComposerAgentV4
    - Applies PENTA methodology confluence bonuses
    - Manages positions (entry, exit, adjustments)
    - Sets stops and targets based on market structure
    - Handles position sizing with risk management
    - Executes trades via broker API
    - Reports to GnosisMonitor
    
    Architecture Position: Trade Agent Layer (receives from Composer)
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        composer: Optional[ComposerAgentV4] = None,
        broker_adapter: Optional[Any] = None,
        monitor: Optional[GnosisMonitor] = None,
    ):
        """
        Initialize Full Gnosis Trade Agent V2.
        
        Args:
            config: Agent configuration
            composer: ComposerAgentV4 (receives signals from)
            broker_adapter: Broker API adapter for execution
            monitor: GnosisMonitor for tracking
        """
        super().__init__(config, composer)
        self.broker_adapter = broker_adapter
        self.monitor = monitor
        
        # Position management
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        
        # Risk parameters
        self.max_position_size = config.get("max_position_size", 0.1)  # 10% of portfolio
        self.max_total_exposure = config.get("max_total_exposure", 0.5)  # 50% max
        self.default_stop_pct = config.get("default_stop_pct", 0.02)  # 2% stop
        self.default_target_pct = config.get("default_target_pct", 0.04)  # 4% target
        
        # PENTA methodology adjustments
        self.penta_position_bonus = {
            "PENTA": 0.25,  # +25% position size for PENTA confluence
            "QUAD": 0.15,
            "TRIPLE": 0.10,
            "DOUBLE": 0.0,
        }
        
        # Execution parameters
        self.use_limit_orders = config.get("use_limit_orders", True)
        self.limit_offset_pct = config.get("limit_offset_pct", 0.001)
        
        logger.info(f"FullGnosisTradeAgentV2 v{self.VERSION} initialized")
    
    def process_composer_output(
        self,
        composer_output: ComposerOutput,
        symbol: str,
        current_price: float,
    ) -> TradeAction:
        """
        Process ComposerAgentV4 output and generate trade action.
        
        Args:
            composer_output: Output from ComposerAgentV4
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            TradeAction to execute
        """
        direction = composer_output.direction
        confidence = composer_output.confidence
        reasoning = composer_output.reasoning
        penta_confluence = composer_output.penta_confluence
        
        # Check if we have an existing position
        existing_position = self.positions.get(symbol)
        
        # Determine action based on signal and position
        if existing_position:
            return self._manage_existing_position(
                symbol, direction, confidence, current_price, 
                existing_position, reasoning, penta_confluence
            )
        else:
            return self._evaluate_new_entry(
                symbol, direction, confidence, current_price, 
                reasoning, penta_confluence
            )
    
    def _evaluate_new_entry(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        current_price: float,
        reasoning: str,
        penta_confluence: Optional[str] = None,
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
        
        # Calculate position size with PENTA bonus
        position_bonus = self.penta_position_bonus.get(penta_confluence, 0.0)
        quantity = self._calculate_position_size(
            symbol, current_price, stop_loss, bonus=position_bonus
        )
        
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
            limit_price=current_price * (1 + self.limit_offset_pct) 
                if trade_direction == "LONG" 
                else current_price * (1 - self.limit_offset_pct),
            penta_confluence=penta_confluence,
            penta_bonus=position_bonus,
        )
    
    def _manage_existing_position(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        current_price: float,
        position: Dict[str, Any],
        reasoning: str,
        penta_confluence: Optional[str] = None,
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
                    penta_confluence=penta_confluence,
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
                    penta_confluence=penta_confluence,
                )
        
        return TradeAction(
            action_type=TradeActionType.HOLD,
            symbol=symbol,
            direction=position_direction,
            quantity=position.get("quantity", 0),
            price=current_price,
            confidence=confidence,
            reasoning="Holding position",
            penta_confluence=penta_confluence,
        )
    
    def _calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        bonus: float = 0.0,
    ) -> int:
        """Calculate position size with optional PENTA bonus."""
        portfolio_value = self.config.get("portfolio_value", 100000)
        risk_per_trade = self.config.get("risk_per_trade", 0.01)  # 1% risk
        
        # Apply PENTA bonus
        adjusted_risk = risk_per_trade * (1 + bonus)
        risk_amount = portfolio_value * adjusted_risk
        
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
            result = {"status": "simulated", "action": action.to_dict()}
        else:
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
                        "penta_confluence": action.penta_confluence,
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
                
            except Exception as e:
                logger.error(f"Failed to execute action: {e}")
                result = {"status": "error", "error": str(e)}
        
        # Update monitor if available
        if self.monitor and action.action_type != TradeActionType.NO_ACTION:
            try:
                self.monitor.update(self.positions, {action.symbol: action.price or 0})
            except Exception as e:
                logger.error(f"Monitor update failed: {e}")
        
        return result


class AlphaTradeAgentV2(BaseTradeAgentV2):
    """
    Alpha Trade Agent V2 - Enhanced Signal Generation for Retail Traders.
    
    This agent:
    - Receives composed signals from ComposerAgentV4
    - Generates simple BUY/SELL/HOLD signals with strength
    - Provides entry, stop, and target levels
    - Applies PENTA methodology confluence bonuses
    - Suggests simple options plays
    - Designed for Robinhood/Webull users
    - Reports to AlphaMonitor
    
    Architecture Position: Trade Agent Layer (receives from Composer)
    
    Note: This is a SIGNAL-ONLY agent - it does NOT execute trades.
    Users execute trades manually on their platform.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        composer: Optional[ComposerAgentV4] = None,
        monitor: Optional[AlphaMonitor] = None,
    ):
        """
        Initialize Alpha Trade Agent V2.
        
        Args:
            config: Agent configuration
            composer: ComposerAgentV4 (receives signals from)
            monitor: AlphaMonitor for tracking
        """
        super().__init__(config, composer)
        self.monitor = monitor
        
        # Signal parameters
        self.strong_confidence_threshold = config.get("strong_confidence_threshold", 0.8)
        self.moderate_confidence_threshold = config.get("moderate_confidence_threshold", 0.65)
        self.default_stop_pct = config.get("default_stop_pct", 0.03)
        self.default_target_pct = config.get("default_target_pct", 0.05)
        
        # PENTA adjustments
        self.penta_target_bonus = {
            "PENTA": 0.50,  # +50% target for PENTA confluence
            "QUAD": 0.30,
            "TRIPLE": 0.15,
            "DOUBLE": 0.0,
        }
        
        # Holding period
        self.default_holding_days = config.get("default_holding_days", 3)
        self.signal_validity_hours = config.get("signal_validity_hours", 24)
        
        logger.info(f"AlphaTradeAgentV2 v{self.VERSION} initialized")
    
    def process_composer_output(
        self,
        composer_output: ComposerOutput,
        symbol: str,
        current_price: float,
    ) -> AlphaSignalV2:
        """
        Process ComposerAgentV4 output and generate Alpha signal.
        
        Args:
            composer_output: Output from ComposerAgentV4
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            AlphaSignalV2 for retail trader
        """
        direction = composer_output.direction
        confidence = composer_output.confidence
        reasoning = composer_output.reasoning
        penta_confluence = composer_output.penta_confluence
        penta_bonus = composer_output.penta_confidence_bonus
        
        # Determine signal type and strength
        if confidence < self.min_confidence:
            signal_type = SignalType.HOLD
            strength = SignalStrength.WEAK
            signal_direction = "HOLD"
            stop_loss = None
            take_profit = None
        elif direction in ["LONG", "UP", "BUY", "BULLISH"]:
            if confidence >= self.strong_confidence_threshold:
                signal_type = SignalType.STRONG_BUY
                strength = SignalStrength.STRONG
            elif confidence >= self.moderate_confidence_threshold:
                signal_type = SignalType.BUY
                strength = SignalStrength.MODERATE
            else:
                signal_type = SignalType.BUY
                strength = SignalStrength.WEAK
            
            signal_direction = "BUY"
            
            # Apply PENTA target bonus
            target_bonus = self.penta_target_bonus.get(penta_confluence, 0.0)
            target_pct = self.default_target_pct * (1 + target_bonus)
            
            stop_loss = current_price * (1 - self.default_stop_pct)
            take_profit = current_price * (1 + target_pct)
            
        elif direction in ["SHORT", "DOWN", "SELL", "BEARISH"]:
            if confidence >= self.strong_confidence_threshold:
                signal_type = SignalType.STRONG_SELL
                strength = SignalStrength.STRONG
            elif confidence >= self.moderate_confidence_threshold:
                signal_type = SignalType.SELL
                strength = SignalStrength.MODERATE
            else:
                signal_type = SignalType.SELL
                strength = SignalStrength.WEAK
            
            signal_direction = "SELL"
            
            # Apply PENTA target bonus
            target_bonus = self.penta_target_bonus.get(penta_confluence, 0.0)
            target_pct = self.default_target_pct * (1 + target_bonus)
            
            stop_loss = current_price * (1 + self.default_stop_pct)
            take_profit = current_price * (1 - target_pct)
        else:
            signal_type = SignalType.HOLD
            strength = SignalStrength.WEAK
            signal_direction = "HOLD"
            stop_loss = None
            take_profit = None
        
        # Create signal
        now = datetime.now(timezone.utc)
        
        signal = AlphaSignalV2(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            direction=signal_direction,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            risk_factors=composer_output.risk_factors,
            agent_signals=composer_output.agent_signals,
            timestamp=now,
            valid_until=now + timedelta(hours=self.signal_validity_hours),
            holding_period_days=self.default_holding_days,
            penta_confluence=penta_confluence,
            penta_confidence_bonus=penta_bonus,
            options_play=self._suggest_options_play(
                symbol, signal_direction, current_price, confidence, penta_confluence
            ) if signal_direction != "HOLD" else None,
        )
        
        # Update monitor if available
        if self.monitor:
            try:
                self.monitor.update(signal=signal.to_dict())
            except Exception as e:
                logger.error(f"Monitor update failed: {e}")
        
        return signal
    
    def _suggest_options_play(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        confidence: float,
        penta_confluence: Optional[str] = None,
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
        
        # Adjust DTE based on confluence
        if penta_confluence in ["PENTA", "QUAD"]:
            dte_range = "7-21 days"  # Shorter for high confluence
        else:
            dte_range = "14-30 days"
        
        return {
            "option_type": option_type,
            "suggested_strike": suggested_strike,
            "dte_range": dte_range,
            "strategy": "Long Call" if option_type == "CALL" else "Long Put",
            "penta_note": f"{penta_confluence} confluence detected" if penta_confluence else None,
            "risk_note": "Only risk what you can afford to lose. Options can expire worthless.",
        }
    
    def generate_signal(
        self,
        symbol: str,
        composer_output: ComposerOutput,
        current_price: float,
    ) -> AlphaSignalV2:
        """Generate an Alpha signal for a symbol."""
        return self.process_composer_output(composer_output, symbol, current_price)


# Factory functions
def create_full_gnosis_agent_v2(
    config: Optional[Dict[str, Any]] = None,
    composer: Optional[ComposerAgentV4] = None,
    broker_adapter: Optional[Any] = None,
    monitor: Optional[GnosisMonitor] = None,
) -> FullGnosisTradeAgentV2:
    """Create a Full Gnosis Trade Agent V2."""
    return FullGnosisTradeAgentV2(
        config=config or {},
        composer=composer,
        broker_adapter=broker_adapter,
        monitor=monitor,
    )


def create_alpha_agent_v2(
    config: Optional[Dict[str, Any]] = None,
    composer: Optional[ComposerAgentV4] = None,
    monitor: Optional[AlphaMonitor] = None,
) -> AlphaTradeAgentV2:
    """Create an Alpha Trade Agent V2."""
    return AlphaTradeAgentV2(
        config=config or {},
        composer=composer,
        monitor=monitor,
    )


__all__ = [
    "TradeActionType",
    "SignalStrength",
    "SignalType",
    "TradeAction",
    "AlphaSignalV2",
    "BaseTradeAgentV2",
    "FullGnosisTradeAgentV2",
    "AlphaTradeAgentV2",
    "create_full_gnosis_agent_v2",
    "create_alpha_agent_v2",
]
