"""
GNOSIS Monitoring Agent Layer

This layer tracks trade agents and provides feedback for continuous improvement.

Architecture:
    Trade Agent → Monitoring Agent → Feedback Loop → ML Enhancement

Two Monitor Types:
1. GnosisMonitor - Monitors Full Gnosis Trade Agent (positions, P&L, risk)
2. AlphaMonitor - Monitors Alpha Trade Agent (signal accuracy, win rate)

Author: GNOSIS Trading System
Version: 1.0.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import statistics

from loguru import logger


class EventType(str, Enum):
    """Types of monitoring events."""
    # Position events
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    POSITION_ADJUSTED = "POSITION_ADJUSTED"
    STOP_HIT = "STOP_HIT"
    TARGET_HIT = "TARGET_HIT"
    
    # Risk events
    RISK_THRESHOLD_EXCEEDED = "RISK_THRESHOLD_EXCEEDED"
    DRAWDOWN_WARNING = "DRAWDOWN_WARNING"
    EXPOSURE_LIMIT = "EXPOSURE_LIMIT"
    
    # Signal events
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    SIGNAL_EXPIRED = "SIGNAL_EXPIRED"
    SIGNAL_CORRECT = "SIGNAL_CORRECT"
    SIGNAL_INCORRECT = "SIGNAL_INCORRECT"
    
    # Performance events
    WINNING_STREAK = "WINNING_STREAK"
    LOSING_STREAK = "LOSING_STREAK"
    NEW_HIGH = "NEW_HIGH"
    NEW_LOW = "NEW_LOW"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class MonitoringEvent:
    """A monitoring event."""
    event_type: EventType
    symbol: Optional[str]
    message: str
    alert_level: AlertLevel
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "symbol": self.symbol,
            "message": self.message,
            "alert_level": self.alert_level.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L metrics
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    
    # Signal metrics (for Alpha)
    signals_generated: int = 0
    signals_correct: int = 0
    signal_accuracy: float = 0.0
    
    # Time metrics
    average_holding_time: timedelta = timedelta(0)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "profit_factor": round(self.profit_factor, 2),
            "average_win": round(self.average_win, 2),
            "average_loss": round(self.average_loss, 2),
            "signals_generated": self.signals_generated,
            "signals_correct": self.signals_correct,
            "signal_accuracy": round(self.signal_accuracy, 4),
            "average_holding_time_hours": self.average_holding_time.total_seconds() / 3600,
            "last_updated": self.last_updated.isoformat(),
        }


class BaseMonitor(ABC):
    """Base class for monitoring agents."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.events: deque = deque(maxlen=config.get("max_events", 1000))
        self.metrics = PerformanceMetrics()
        
        # Alert thresholds
        self.drawdown_warning_pct = config.get("drawdown_warning_pct", 0.05)
        self.drawdown_critical_pct = config.get("drawdown_critical_pct", 0.10)
        self.losing_streak_warning = config.get("losing_streak_warning", 3)
        
        # Callbacks
        self.alert_callbacks: List[callable] = []
    
    def add_alert_callback(self, callback: callable) -> None:
        """Add a callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def _emit_event(self, event: MonitoringEvent) -> None:
        """Emit a monitoring event."""
        self.events.append(event)
        logger.log(
            event.alert_level.value,
            f"[{event.event_type.value}] {event.message}"
        )
        
        # Trigger callbacks for warnings and critical events
        if event.alert_level in [AlertLevel.WARNING, AlertLevel.CRITICAL]:
            for callback in self.alert_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def get_recent_events(self, limit: int = 50) -> List[MonitoringEvent]:
        """Get recent events."""
        return list(self.events)[-limit:]
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.metrics
    
    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update monitor with new data."""
        pass


class GnosisMonitor(BaseMonitor):
    """
    Full Gnosis Monitor - Tracks Full Gnosis Trade Agent.
    
    Monitors:
    - Open positions
    - P&L (realized and unrealized)
    - Risk exposure
    - Drawdown
    - Trade performance
    
    Architecture Position: Monitoring Agent Layer (receives from Trade Agent)
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Position tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        
        # P&L tracking
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.peak_equity: float = config.get("initial_equity", 100000)
        self.current_equity: float = self.peak_equity
        
        # Streak tracking
        self.current_streak: int = 0
        self.streak_type: Optional[str] = None  # "winning" or "losing"
        
        logger.info(f"GnosisMonitor v{self.VERSION} initialized")
    
    def update(
        self,
        positions: Dict[str, Dict[str, Any]],
        current_prices: Dict[str, float],
    ) -> None:
        """
        Update monitor with current position data.
        
        Args:
            positions: Current open positions
            current_prices: Current market prices
        """
        now = datetime.now(timezone.utc)
        
        # Track new positions
        for symbol, position in positions.items():
            if symbol not in self.positions:
                self._emit_event(MonitoringEvent(
                    event_type=EventType.POSITION_OPENED,
                    symbol=symbol,
                    message=f"Opened {position.get('direction', 'LONG')} position in {symbol}",
                    alert_level=AlertLevel.INFO,
                    data=position,
                ))
            self.positions[symbol] = position
        
        # Track closed positions
        for symbol in list(self.positions.keys()):
            if symbol not in positions:
                closed_position = self.positions.pop(symbol)
                self._record_closed_trade(closed_position, current_prices.get(symbol, 0))
        
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.get("entry_price", 0))
            entry_price = position.get("entry_price", current_price)
            quantity = position.get("quantity", 0)
            direction = position.get("direction", "LONG")
            
            if direction == "LONG":
                pnl = (current_price - entry_price) * quantity
            else:
                pnl = (entry_price - current_price) * quantity
            
            unrealized_pnl += pnl
        
        self.metrics.unrealized_pnl = unrealized_pnl
        self.current_equity = self.peak_equity + self.metrics.realized_pnl + unrealized_pnl
        
        # Update equity curve
        self.equity_curve.append((now, self.current_equity))
        
        # Check for new high
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
            self._emit_event(MonitoringEvent(
                event_type=EventType.NEW_HIGH,
                symbol=None,
                message=f"New equity high: ${self.current_equity:,.2f}",
                alert_level=AlertLevel.INFO,
            ))
        
        # Check drawdown
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        self.metrics.max_drawdown = max(self.metrics.max_drawdown, drawdown)
        
        if drawdown >= self.drawdown_critical_pct:
            self._emit_event(MonitoringEvent(
                event_type=EventType.DRAWDOWN_WARNING,
                symbol=None,
                message=f"CRITICAL: Drawdown at {drawdown:.2%}",
                alert_level=AlertLevel.CRITICAL,
                data={"drawdown": drawdown, "peak": self.peak_equity, "current": self.current_equity},
            ))
        elif drawdown >= self.drawdown_warning_pct:
            self._emit_event(MonitoringEvent(
                event_type=EventType.DRAWDOWN_WARNING,
                symbol=None,
                message=f"Warning: Drawdown at {drawdown:.2%}",
                alert_level=AlertLevel.WARNING,
                data={"drawdown": drawdown},
            ))
        
        self.metrics.last_updated = now
    
    def _record_closed_trade(
        self,
        position: Dict[str, Any],
        exit_price: float,
    ) -> None:
        """Record a closed trade."""
        entry_price = position.get("entry_price", exit_price)
        quantity = position.get("quantity", 0)
        direction = position.get("direction", "LONG")
        symbol = position.get("symbol", "UNKNOWN")
        
        # Calculate P&L
        if direction == "LONG":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        # Record trade
        trade = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "pnl": pnl,
            "timestamp": datetime.now(timezone.utc),
        }
        self.trade_history.append(trade)
        
        # Update metrics
        self.metrics.total_trades += 1
        self.metrics.realized_pnl += pnl
        self.metrics.total_pnl = self.metrics.realized_pnl + self.metrics.unrealized_pnl
        
        if pnl > 0:
            self.metrics.winning_trades += 1
            self._update_streak("winning")
        else:
            self.metrics.losing_trades += 1
            self._update_streak("losing")
        
        # Update win rate
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades
        
        # Calculate averages
        winning_pnls = [t["pnl"] for t in self.trade_history if t["pnl"] > 0]
        losing_pnls = [t["pnl"] for t in self.trade_history if t["pnl"] < 0]
        
        if winning_pnls:
            self.metrics.average_win = statistics.mean(winning_pnls)
        if losing_pnls:
            self.metrics.average_loss = abs(statistics.mean(losing_pnls))
        
        # Calculate profit factor
        total_wins = sum(winning_pnls) if winning_pnls else 0
        total_losses = abs(sum(losing_pnls)) if losing_pnls else 1
        self.metrics.profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Emit event
        event_type = EventType.TARGET_HIT if pnl > 0 else EventType.STOP_HIT
        self._emit_event(MonitoringEvent(
            event_type=event_type,
            symbol=symbol,
            message=f"Closed {direction} {symbol}: ${pnl:+,.2f}",
            alert_level=AlertLevel.INFO,
            data=trade,
        ))
    
    def _update_streak(self, streak_type: str) -> None:
        """Update winning/losing streak."""
        if self.streak_type == streak_type:
            self.current_streak += 1
        else:
            self.streak_type = streak_type
            self.current_streak = 1
        
        # Check for streak warnings
        if streak_type == "losing" and self.current_streak >= self.losing_streak_warning:
            self._emit_event(MonitoringEvent(
                event_type=EventType.LOSING_STREAK,
                symbol=None,
                message=f"Losing streak: {self.current_streak} consecutive losses",
                alert_level=AlertLevel.WARNING,
                data={"streak": self.current_streak},
            ))


class AlphaMonitor(BaseMonitor):
    """
    Alpha Monitor - Tracks Alpha Trade Agent Signal Accuracy.
    
    Monitors:
    - Signal accuracy
    - Win rate by signal type
    - Performance by symbol
    - User feedback integration
    
    Architecture Position: Monitoring Agent Layer (receives from Alpha Trade Agent)
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Signal tracking
        self.pending_signals: Dict[str, Dict[str, Any]] = {}
        self.signal_history: List[Dict[str, Any]] = []
        
        # Performance by symbol
        self.symbol_performance: Dict[str, Dict[str, Any]] = {}
        
        # Performance by signal type
        self.signal_type_performance: Dict[str, Dict[str, int]] = {}
        
        logger.info(f"AlphaMonitor v{self.VERSION} initialized")
    
    def update(
        self,
        signal: Optional[Dict[str, Any]] = None,
        outcome: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update monitor with signal or outcome data.
        
        Args:
            signal: New signal generated
            outcome: Outcome of a previous signal
        """
        now = datetime.now(timezone.utc)
        
        if signal:
            self._track_signal(signal)
        
        if outcome:
            self._record_outcome(outcome)
        
        self.metrics.last_updated = now
    
    def _track_signal(self, signal: Dict[str, Any]) -> None:
        """Track a new signal."""
        symbol = signal.get("symbol", "UNKNOWN")
        signal_type = signal.get("signal_type", "UNKNOWN")
        
        # Store pending signal
        self.pending_signals[symbol] = {
            **signal,
            "tracked_at": datetime.now(timezone.utc),
        }
        
        # Update counts
        self.metrics.signals_generated += 1
        
        if signal_type not in self.signal_type_performance:
            self.signal_type_performance[signal_type] = {"generated": 0, "correct": 0}
        self.signal_type_performance[signal_type]["generated"] += 1
        
        # Emit event
        self._emit_event(MonitoringEvent(
            event_type=EventType.SIGNAL_GENERATED,
            symbol=symbol,
            message=f"{signal_type} signal for {symbol} ({signal.get('confidence', 0):.0%} confidence)",
            alert_level=AlertLevel.INFO,
            data=signal,
        ))
    
    def _record_outcome(self, outcome: Dict[str, Any]) -> None:
        """Record the outcome of a signal."""
        symbol = outcome.get("symbol", "UNKNOWN")
        was_correct = outcome.get("correct", False)
        pnl = outcome.get("pnl", 0)
        
        # Get original signal
        original_signal = self.pending_signals.pop(symbol, {})
        signal_type = original_signal.get("signal_type", "UNKNOWN")
        
        # Record in history
        record = {
            **original_signal,
            "outcome": outcome,
            "was_correct": was_correct,
            "pnl": pnl,
            "resolved_at": datetime.now(timezone.utc),
        }
        self.signal_history.append(record)
        
        # Update metrics
        if was_correct:
            self.metrics.signals_correct += 1
            if signal_type in self.signal_type_performance:
                self.signal_type_performance[signal_type]["correct"] += 1
        
        # Calculate accuracy
        if self.metrics.signals_generated > 0:
            self.metrics.signal_accuracy = self.metrics.signals_correct / self.metrics.signals_generated
        
        # Update symbol performance
        if symbol not in self.symbol_performance:
            self.symbol_performance[symbol] = {"total": 0, "correct": 0, "pnl": 0}
        self.symbol_performance[symbol]["total"] += 1
        self.symbol_performance[symbol]["pnl"] += pnl
        if was_correct:
            self.symbol_performance[symbol]["correct"] += 1
        
        # Emit event
        event_type = EventType.SIGNAL_CORRECT if was_correct else EventType.SIGNAL_INCORRECT
        self._emit_event(MonitoringEvent(
            event_type=event_type,
            symbol=symbol,
            message=f"Signal {'correct' if was_correct else 'incorrect'} for {symbol}: ${pnl:+,.2f}",
            alert_level=AlertLevel.INFO,
            data=record,
        ))
    
    def get_symbol_accuracy(self, symbol: str) -> float:
        """Get accuracy for a specific symbol."""
        perf = self.symbol_performance.get(symbol, {})
        total = perf.get("total", 0)
        correct = perf.get("correct", 0)
        return correct / total if total > 0 else 0.0
    
    def get_signal_type_accuracy(self, signal_type: str) -> float:
        """Get accuracy for a signal type."""
        perf = self.signal_type_performance.get(signal_type, {})
        generated = perf.get("generated", 0)
        correct = perf.get("correct", 0)
        return correct / generated if generated > 0 else 0.0
    
    def expire_old_signals(self, max_age_hours: int = 24) -> int:
        """Expire old pending signals."""
        now = datetime.now(timezone.utc)
        expired = 0
        
        for symbol in list(self.pending_signals.keys()):
            signal = self.pending_signals[symbol]
            tracked_at = signal.get("tracked_at", now)
            age = now - tracked_at
            
            if age.total_seconds() > max_age_hours * 3600:
                self.pending_signals.pop(symbol)
                expired += 1
                self._emit_event(MonitoringEvent(
                    event_type=EventType.SIGNAL_EXPIRED,
                    symbol=symbol,
                    message=f"Signal expired for {symbol} (age: {age})",
                    alert_level=AlertLevel.INFO,
                ))
        
        return expired


# Factory functions
def create_gnosis_monitor(config: Optional[Dict[str, Any]] = None) -> GnosisMonitor:
    """Create a Gnosis Monitor."""
    return GnosisMonitor(config or {})


def create_alpha_monitor(config: Optional[Dict[str, Any]] = None) -> AlphaMonitor:
    """Create an Alpha Monitor."""
    return AlphaMonitor(config or {})


__all__ = [
    "EventType",
    "AlertLevel",
    "MonitoringEvent",
    "PerformanceMetrics",
    "BaseMonitor",
    "GnosisMonitor",
    "AlphaMonitor",
    "create_gnosis_monitor",
    "create_alpha_monitor",
]
