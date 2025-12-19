"""Exit Rules Engine - Comprehensive exit condition management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from loguru import logger

from execution.position_manager import ManagedPosition, PositionStatus


class ExitReason(str, Enum):
    """Reasons for exiting a position."""
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TARGET_1 = "target_1"
    TARGET_2 = "target_2"
    TIME_STOP = "time_stop"
    SIGNAL_REVERSAL = "signal_reversal"
    MARKET_CLOSE = "market_close"
    EXPIRY_APPROACHING = "expiry_approaching"
    DRAWDOWN_LIMIT = "drawdown_limit"
    VOLATILITY_SPIKE = "volatility_spike"
    MANUAL = "manual"
    EXTERNAL = "external"


class ExitPriority(int, Enum):
    """Priority levels for exit execution."""
    CRITICAL = 1      # Stop loss, circuit breaker
    HIGH = 2          # Target hit, expiry
    MEDIUM = 3        # Time stop, signal reversal
    LOW = 4           # Optimization exits


@dataclass
class ExitRule:
    """Definition of an exit rule."""
    name: str
    reason: ExitReason
    priority: ExitPriority
    check_fn: Callable[[ManagedPosition, Dict[str, Any]], bool]
    enabled: bool = True
    description: str = ""


@dataclass
class ExitSignal:
    """Signal to exit a position."""
    symbol: str
    reason: ExitReason
    priority: ExitPriority
    exit_price: Optional[float] = None
    exit_quantity: Optional[float] = None  # None = full position
    message: str = ""
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExitRulesConfig:
    """Configuration for exit rules."""
    # Time-based exits
    max_hold_minutes: int = 60
    market_close_exit_minutes: int = 5  # Exit X mins before close

    # Profit targets
    target_1_pct: float = 0.5
    target_2_pct: float = 1.0
    partial_exit_at_target_1: bool = True
    partial_exit_pct: float = 0.5

    # Loss limits
    max_loss_pct: float = 2.0
    drawdown_from_high_pct: float = 1.5

    # Signal-based
    exit_on_signal_reversal: bool = True
    reversal_confidence_threshold: float = 0.6

    # Options-specific
    exit_hours_before_expiry: int = 2
    exit_at_theta_decay_pct: float = 0.5  # Exit when theta decay > X%

    # Volatility
    exit_on_vol_spike: bool = True
    vol_spike_threshold: float = 2.0  # X times normal vol


class ExitRulesEngine:
    """
    Evaluates all exit conditions and generates exit signals.

    Supports:
    - Stop loss and trailing stops
    - Profit targets (full and partial)
    - Time-based exits
    - Signal reversal exits
    - Market close exits
    - Options expiry exits
    - Volatility-based exits
    - Drawdown protection
    """

    def __init__(self, config: ExitRulesConfig = None):
        self.config = config or ExitRulesConfig()
        self.rules: List[ExitRule] = []

        # Market state
        self.market_close_time: Optional[datetime] = None
        self.current_signals: Dict[str, Any] = {}  # Latest signals per symbol
        self.volatility_map: Dict[str, float] = {}
        self.expiry_map: Dict[str, datetime] = {}

        # Initialize default rules
        self._init_default_rules()

        logger.info("ExitRulesEngine initialized with %d rules", len(self.rules))

    def _init_default_rules(self) -> None:
        """Initialize default exit rules."""

        # 1. Stop Loss
        self.rules.append(ExitRule(
            name="stop_loss",
            reason=ExitReason.STOP_LOSS,
            priority=ExitPriority.CRITICAL,
            check_fn=self._check_stop_loss,
            description="Exit when price hits stop loss",
        ))

        # 2. Trailing Stop
        self.rules.append(ExitRule(
            name="trailing_stop",
            reason=ExitReason.TRAILING_STOP,
            priority=ExitPriority.CRITICAL,
            check_fn=self._check_trailing_stop,
            description="Exit when price hits trailing stop",
        ))

        # 3. Target 2 (full exit)
        self.rules.append(ExitRule(
            name="target_2",
            reason=ExitReason.TARGET_2,
            priority=ExitPriority.HIGH,
            check_fn=self._check_target_2,
            description="Exit when price hits second profit target",
        ))

        # 4. Target 1 (partial exit)
        self.rules.append(ExitRule(
            name="target_1",
            reason=ExitReason.TARGET_1,
            priority=ExitPriority.HIGH,
            check_fn=self._check_target_1,
            description="Partial exit at first profit target",
        ))

        # 5. Time Stop
        self.rules.append(ExitRule(
            name="time_stop",
            reason=ExitReason.TIME_STOP,
            priority=ExitPriority.MEDIUM,
            check_fn=self._check_time_stop,
            description="Exit when max hold time exceeded",
        ))

        # 6. Signal Reversal
        self.rules.append(ExitRule(
            name="signal_reversal",
            reason=ExitReason.SIGNAL_REVERSAL,
            priority=ExitPriority.MEDIUM,
            check_fn=self._check_signal_reversal,
            enabled=self.config.exit_on_signal_reversal,
            description="Exit when signal reverses direction",
        ))

        # 7. Market Close
        self.rules.append(ExitRule(
            name="market_close",
            reason=ExitReason.MARKET_CLOSE,
            priority=ExitPriority.HIGH,
            check_fn=self._check_market_close,
            description="Exit before market close",
        ))

        # 8. Options Expiry
        self.rules.append(ExitRule(
            name="expiry_approaching",
            reason=ExitReason.EXPIRY_APPROACHING,
            priority=ExitPriority.HIGH,
            check_fn=self._check_expiry_approaching,
            description="Exit before options expiry",
        ))

        # 9. Drawdown from High
        self.rules.append(ExitRule(
            name="drawdown_limit",
            reason=ExitReason.DRAWDOWN_LIMIT,
            priority=ExitPriority.HIGH,
            check_fn=self._check_drawdown_limit,
            description="Exit when position gives back too much profit",
        ))

        # 10. Volatility Spike
        self.rules.append(ExitRule(
            name="volatility_spike",
            reason=ExitReason.VOLATILITY_SPIKE,
            priority=ExitPriority.MEDIUM,
            check_fn=self._check_volatility_spike,
            enabled=self.config.exit_on_vol_spike,
            description="Exit on extreme volatility spike",
        ))

    def evaluate(
        self,
        position: ManagedPosition,
        current_time: datetime,
        context: Dict[str, Any] = None,
    ) -> List[ExitSignal]:
        """
        Evaluate all exit rules for a position.

        Args:
            position: The managed position to evaluate
            current_time: Current timestamp
            context: Additional context (signals, volatility, etc.)

        Returns:
            List of exit signals, sorted by priority
        """
        context = context or {}
        context["current_time"] = current_time
        context["volatility"] = self.volatility_map.get(position.symbol, 0)
        context["expiry"] = self.expiry_map.get(position.symbol)
        context["signal"] = self.current_signals.get(position.symbol)
        context["market_close"] = self.market_close_time

        signals = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            try:
                if rule.check_fn(position, context):
                    signal = ExitSignal(
                        symbol=position.symbol,
                        reason=rule.reason,
                        priority=rule.priority,
                        exit_price=position.current_price,
                        message=rule.description,
                        triggered_at=current_time,
                        metadata={"rule": rule.name},
                    )

                    # Handle partial exits
                    if rule.reason == ExitReason.TARGET_1 and self.config.partial_exit_at_target_1:
                        signal.exit_quantity = position.quantity * self.config.partial_exit_pct

                    signals.append(signal)
                    logger.info(
                        f"{position.symbol}: Exit rule triggered - {rule.name}"
                    )
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")

        # Sort by priority (lower = more urgent)
        signals.sort(key=lambda s: s.priority.value)

        return signals

    def _check_stop_loss(self, position: ManagedPosition, ctx: Dict) -> bool:
        """Check if stop loss is hit."""
        if not position.initial_stop:
            return False

        if position.side == "long":
            return position.current_price <= position.initial_stop
        else:
            return position.current_price >= position.initial_stop

    def _check_trailing_stop(self, position: ManagedPosition, ctx: Dict) -> bool:
        """Check if trailing stop is hit."""
        if not position.current_stop:
            return False

        # Skip if this is the initial stop (not trailing yet)
        if position.current_stop == position.initial_stop:
            return False

        if position.side == "long":
            return position.current_price <= position.current_stop
        else:
            return position.current_price >= position.current_stop

    def _check_target_2(self, position: ManagedPosition, ctx: Dict) -> bool:
        """Check if second profit target is hit."""
        if not position.target_2:
            return False

        if position.side == "long":
            return position.current_price >= position.target_2
        else:
            return position.current_price <= position.target_2

    def _check_target_1(self, position: ManagedPosition, ctx: Dict) -> bool:
        """Check if first profit target is hit (for partial exit)."""
        if not position.target_1 or position.target_1_hit:
            return False

        if position.side == "long":
            return position.current_price >= position.target_1
        else:
            return position.current_price <= position.target_1

    def _check_time_stop(self, position: ManagedPosition, ctx: Dict) -> bool:
        """Check if max hold time exceeded."""
        current_time = ctx.get("current_time", datetime.utcnow())
        return position.is_time_expired(current_time)

    def _check_signal_reversal(self, position: ManagedPosition, ctx: Dict) -> bool:
        """Check if signal has reversed direction."""
        signal = ctx.get("signal")
        if not signal:
            return False

        signal_direction = signal.get("direction", "neutral")
        signal_confidence = signal.get("confidence", 0)

        if signal_confidence < self.config.reversal_confidence_threshold:
            return False

        # Long position, bearish signal = reversal
        if position.side == "long" and signal_direction == "short":
            return True

        # Short position, bullish signal = reversal
        if position.side == "short" and signal_direction == "long":
            return True

        return False

    def _check_market_close(self, position: ManagedPosition, ctx: Dict) -> bool:
        """Check if approaching market close."""
        market_close = ctx.get("market_close")
        if not market_close:
            return False

        current_time = ctx.get("current_time", datetime.utcnow())
        minutes_to_close = (market_close - current_time).total_seconds() / 60

        return minutes_to_close <= self.config.market_close_exit_minutes

    def _check_expiry_approaching(self, position: ManagedPosition, ctx: Dict) -> bool:
        """Check if options expiry is approaching."""
        expiry = ctx.get("expiry")
        if not expiry:
            return False

        current_time = ctx.get("current_time", datetime.utcnow())
        hours_to_expiry = (expiry - current_time).total_seconds() / 3600

        return hours_to_expiry <= self.config.exit_hours_before_expiry

    def _check_drawdown_limit(self, position: ManagedPosition, ctx: Dict) -> bool:
        """Check if position has given back too much profit."""
        if position.side == "long":
            peak_profit_pct = (position.highest_price - position.entry_price) / position.entry_price * 100
            current_profit_pct = (position.current_price - position.entry_price) / position.entry_price * 100
        else:
            peak_profit_pct = (position.entry_price - position.lowest_price) / position.entry_price * 100
            current_profit_pct = (position.entry_price - position.current_price) / position.entry_price * 100

        # Only check if we were profitable
        if peak_profit_pct <= 0:
            return False

        drawdown = peak_profit_pct - current_profit_pct
        return drawdown >= self.config.drawdown_from_high_pct

    def _check_volatility_spike(self, position: ManagedPosition, ctx: Dict) -> bool:
        """Check for extreme volatility spike."""
        current_vol = ctx.get("volatility", 0)
        if current_vol <= 0:
            return False

        # Compare to baseline (stored in position metadata or default)
        baseline_vol = position.metadata.get("entry_volatility", 0.2)
        if baseline_vol <= 0:
            return False

        vol_ratio = current_vol / baseline_vol
        return vol_ratio >= self.config.vol_spike_threshold

    def update_market_close(self, close_time: datetime) -> None:
        """Update market close time."""
        self.market_close_time = close_time

    def update_signal(self, symbol: str, signal: Dict[str, Any]) -> None:
        """Update latest signal for a symbol."""
        self.current_signals[symbol] = signal

    def update_volatility(self, symbol: str, volatility: float) -> None:
        """Update volatility for a symbol."""
        self.volatility_map[symbol] = volatility

    def update_expiry(self, symbol: str, expiry: datetime) -> None:
        """Update expiry time for a symbol."""
        self.expiry_map[symbol] = expiry

    def add_custom_rule(
        self,
        name: str,
        reason: ExitReason,
        priority: ExitPriority,
        check_fn: Callable[[ManagedPosition, Dict[str, Any]], bool],
        description: str = "",
    ) -> None:
        """Add a custom exit rule."""
        self.rules.append(ExitRule(
            name=name,
            reason=reason,
            priority=priority,
            check_fn=check_fn,
            description=description,
        ))
        logger.info(f"Added custom exit rule: {name}")

    def enable_rule(self, name: str) -> None:
        """Enable an exit rule by name."""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = True
                logger.info(f"Enabled exit rule: {name}")
                return

    def disable_rule(self, name: str) -> None:
        """Disable an exit rule by name."""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = False
                logger.info(f"Disabled exit rule: {name}")
                return

    def get_active_rules(self) -> List[str]:
        """Get list of active rule names."""
        return [r.name for r in self.rules if r.enabled]
