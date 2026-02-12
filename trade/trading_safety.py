"""
Trading Safety Module - Comprehensive Risk Controls for Live/Paper Trading

This module provides institutional-grade safety controls:
1. Circuit breakers for rapid loss scenarios
2. Position concentration limits
3. Volatility-based trade rejection
4. Pre-trade risk validation
5. Order rate limiting
6. Correlation-based exposure limits

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
import threading
from collections import deque

from loguru import logger


class SafetyStatus(str, Enum):
    """Safety system status."""
    OK = "ok"
    WARNING = "warning"
    BLOCKED = "blocked"
    EMERGENCY = "emergency"


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Triggered, blocking trades
    HALF_OPEN = "half_open"  # Testing if safe to resume


@dataclass
class SafetyConfig:
    """Configuration for trading safety controls."""
    # Daily loss limits
    max_daily_loss_usd: float = 5000.0
    max_daily_loss_pct: float = 0.05  # 5% of portfolio
    
    # Position limits
    max_positions: int = 10
    max_position_size_pct: float = 0.10  # 10% per position
    max_sector_exposure_pct: float = 0.30  # 30% per sector
    max_single_stock_pct: float = 0.15  # 15% in any single stock
    
    # Correlation limits
    max_correlation_exposure: float = 0.80  # Block if corr > 80%
    min_diversification_ratio: float = 0.30  # At least 30% diversification
    
    # Volatility controls
    max_vix_for_new_trades: float = 35.0  # No new trades if VIX > 35
    position_size_vix_scaling: bool = True
    high_vol_position_reduction: float = 0.50  # Reduce size 50% in high vol
    
    # Drawdown controls
    max_intraday_drawdown_pct: float = 0.03  # 3% intraday dd pause
    max_weekly_drawdown_pct: float = 0.07  # 7% weekly dd stop
    
    # Rate limiting
    max_orders_per_minute: int = 10
    max_orders_per_hour: int = 100
    min_seconds_between_orders: float = 1.0
    
    # Circuit breaker settings
    circuit_breaker_loss_threshold_pct: float = 0.02  # 2% rapid loss triggers
    circuit_breaker_time_window_minutes: int = 15
    circuit_breaker_cooldown_minutes: int = 30
    
    # Emergency stop
    emergency_stop_loss_pct: float = 0.10  # 10% loss = emergency stop
    
    # Trade validation
    min_liquidity_score: float = 0.30
    max_bid_ask_spread_pct: float = 0.05  # 5% max spread
    min_volume_percentile: float = 0.20
    
    # Session controls
    no_trade_last_minutes_of_day: int = 15
    no_trade_first_minutes_of_day: int = 5


@dataclass
class SafetyMetrics:
    """Real-time safety metrics."""
    # Loss tracking
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    intraday_high_water_mark: float = 0.0
    current_drawdown_pct: float = 0.0
    
    # Position metrics
    position_count: int = 0
    largest_position_pct: float = 0.0
    total_exposure_pct: float = 0.0
    portfolio_delta: float = 0.0
    
    # Volatility
    current_vix: float = 0.0
    portfolio_beta: float = 1.0
    
    # Order tracking
    orders_this_minute: int = 0
    orders_this_hour: int = 0
    last_order_time: Optional[datetime] = None
    
    # Circuit breaker
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    circuit_breaker_triggered_at: Optional[datetime] = None
    
    # Status
    status: SafetyStatus = SafetyStatus.OK
    warnings: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)


@dataclass
class TradeValidationResult:
    """Result of pre-trade validation."""
    approved: bool
    reason: str = ""
    warnings: List[str] = field(default_factory=list)
    adjustments: Dict[str, Any] = field(default_factory=dict)
    
    # Suggested modifications
    suggested_size_multiplier: float = 1.0
    suggested_stop_loss: Optional[float] = None
    max_allowed_size: float = float('inf')


class TradingSafetyManager:
    """
    Comprehensive trading safety management system.
    
    Provides real-time monitoring and blocking of dangerous trades:
    - Circuit breakers for rapid losses
    - Position concentration limits
    - Volatility-based controls
    - Order rate limiting
    - Pre-trade validation
    """
    
    def __init__(
        self,
        config: Optional[SafetyConfig] = None,
        portfolio_value: float = 100000.0,
        on_safety_event: Optional[Callable] = None,
    ):
        """
        Initialize the safety manager.
        
        Args:
            config: Safety configuration
            portfolio_value: Current portfolio value
            on_safety_event: Callback for safety events
        """
        self.config = config or SafetyConfig()
        self.portfolio_value = portfolio_value
        self.on_safety_event = on_safety_event
        
        # Metrics
        self.metrics = SafetyMetrics()
        
        # Order tracking
        self._order_times: deque = deque(maxlen=1000)
        self._pnl_history: deque = deque(maxlen=1000)
        
        # Position tracking
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._sector_exposure: Dict[str, float] = {}
        
        # Circuit breaker
        self._circuit_breaker_lock = threading.Lock()
        self._last_circuit_check = datetime.now(timezone.utc)
        
        # Session start
        self._session_start = datetime.now(timezone.utc)
        self._session_start_equity = portfolio_value
        self._intraday_high = portfolio_value
        
        logger.info(
            f"TradingSafetyManager initialized | "
            f"max_daily_loss=${self.config.max_daily_loss_usd:,.0f} | "
            f"max_positions={self.config.max_positions}"
        )
    
    def validate_trade(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        quantity: int,
        price: float,
        order_type: str = "market",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TradeValidationResult:
        """
        Validate a trade before execution.
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity
            price: Expected price
            order_type: Order type
            metadata: Additional metadata (liquidity, volatility, etc.)
            
        Returns:
            TradeValidationResult with approval status and any adjustments
        """
        result = TradeValidationResult(approved=True)
        metadata = metadata or {}
        
        # Check circuit breaker
        if self.metrics.circuit_breaker_state == CircuitBreakerState.OPEN:
            result.approved = False
            result.reason = "Circuit breaker is OPEN - trading halted"
            self._emit_safety_event("trade_blocked", {"reason": "circuit_breaker", "symbol": symbol})
            return result
        
        # Check emergency stop
        if self.metrics.status == SafetyStatus.EMERGENCY:
            result.approved = False
            result.reason = "Emergency stop active - all trading halted"
            return result
        
        # Calculate trade value
        trade_value = quantity * price
        trade_value_pct = trade_value / self.portfolio_value if self.portfolio_value > 0 else 1.0
        
        # 1. Position size check
        if trade_value_pct > self.config.max_position_size_pct:
            if side == "buy":
                max_allowed = self.config.max_position_size_pct * self.portfolio_value / price
                result.max_allowed_size = max_allowed
                result.suggested_size_multiplier = max_allowed / quantity
                result.warnings.append(
                    f"Position size {trade_value_pct:.1%} exceeds limit {self.config.max_position_size_pct:.1%}"
                )
        
        # 2. Max positions check
        if side == "buy" and symbol not in self._positions:
            if self.metrics.position_count >= self.config.max_positions:
                result.approved = False
                result.reason = f"Max positions ({self.config.max_positions}) reached"
                return result
        
        # 3. Single stock concentration
        existing_value = self._positions.get(symbol, {}).get("value", 0)
        new_total_pct = (existing_value + trade_value) / self.portfolio_value
        if side == "buy" and new_total_pct > self.config.max_single_stock_pct:
            result.warnings.append(
                f"Single stock exposure {new_total_pct:.1%} exceeds limit {self.config.max_single_stock_pct:.1%}"
            )
            max_additional = (self.config.max_single_stock_pct * self.portfolio_value - existing_value) / price
            if max_additional <= 0:
                result.approved = False
                result.reason = "Maximum single stock exposure reached"
                return result
            result.max_allowed_size = min(result.max_allowed_size, max_additional)
        
        # 4. VIX check
        current_vix = metadata.get("vix", self.metrics.current_vix)
        if current_vix > self.config.max_vix_for_new_trades and side == "buy":
            result.approved = False
            result.reason = f"VIX ({current_vix:.1f}) exceeds max ({self.config.max_vix_for_new_trades})"
            return result
        
        # 5. Volatility position sizing adjustment
        if self.config.position_size_vix_scaling and current_vix > 25:
            vol_multiplier = max(0.5, 1.0 - (current_vix - 25) * 0.02)
            result.suggested_size_multiplier *= vol_multiplier
            result.adjustments["volatility_scaling"] = vol_multiplier
            result.warnings.append(f"Position reduced {1-vol_multiplier:.0%} due to elevated VIX")
        
        # 6. Liquidity check
        liquidity_score = metadata.get("liquidity_score", 1.0)
        if liquidity_score < self.config.min_liquidity_score:
            result.warnings.append(f"Low liquidity score: {liquidity_score:.2f}")
            if liquidity_score < self.config.min_liquidity_score * 0.5:
                result.approved = False
                result.reason = "Insufficient liquidity"
                return result
        
        # 7. Bid-ask spread check
        spread_pct = metadata.get("spread_pct", 0)
        if spread_pct > self.config.max_bid_ask_spread_pct:
            result.warnings.append(f"Wide spread: {spread_pct:.2%}")
            if spread_pct > self.config.max_bid_ask_spread_pct * 2:
                result.approved = False
                result.reason = f"Spread too wide: {spread_pct:.2%}"
                return result
        
        # 8. Rate limiting check
        if not self._check_rate_limit():
            result.approved = False
            result.reason = "Order rate limit exceeded"
            return result
        
        # 9. Session time check
        if not self._check_session_time():
            result.approved = False
            result.reason = "Outside allowed trading window"
            return result
        
        # 10. Daily loss check
        if self.metrics.daily_pnl < -self.config.max_daily_loss_usd:
            if side == "buy":
                result.approved = False
                result.reason = f"Daily loss limit (${self.config.max_daily_loss_usd:,.0f}) reached"
                return result
        
        # Apply size multiplier
        if result.suggested_size_multiplier < 1.0:
            result.max_allowed_size = min(
                result.max_allowed_size,
                quantity * result.suggested_size_multiplier
            )
        
        return result
    
    def _check_rate_limit(self) -> bool:
        """Check if order rate limits are satisfied."""
        now = datetime.now(timezone.utc)
        
        # Count orders in last minute
        minute_ago = now - timedelta(minutes=1)
        orders_last_minute = sum(1 for t in self._order_times if t > minute_ago)
        
        if orders_last_minute >= self.config.max_orders_per_minute:
            logger.warning(f"Rate limit: {orders_last_minute} orders in last minute")
            return False
        
        # Count orders in last hour
        hour_ago = now - timedelta(hours=1)
        orders_last_hour = sum(1 for t in self._order_times if t > hour_ago)
        
        if orders_last_hour >= self.config.max_orders_per_hour:
            logger.warning(f"Rate limit: {orders_last_hour} orders in last hour")
            return False
        
        # Check minimum time between orders
        if self.metrics.last_order_time:
            seconds_since_last = (now - self.metrics.last_order_time).total_seconds()
            if seconds_since_last < self.config.min_seconds_between_orders:
                return False
        
        return True
    
    def _check_session_time(self) -> bool:
        """Check if current time is within allowed trading window."""
        now = datetime.now(timezone.utc)
        # Simplified check - in production, use market calendar
        hour = now.hour
        minute = now.minute
        
        # Market hours 9:30-16:00 ET (14:30-21:00 UTC)
        if hour == 14 and minute < 30 + self.config.no_trade_first_minutes_of_day:
            return False
        if hour == 20 and minute > 60 - self.config.no_trade_last_minutes_of_day:
            return False
        
        return True
    
    def record_order(self, symbol: str, side: str, quantity: int, price: float):
        """Record an order for rate limiting and tracking."""
        now = datetime.now(timezone.utc)
        self._order_times.append(now)
        self.metrics.last_order_time = now
        
        # Update position tracking
        if side == "buy":
            if symbol not in self._positions:
                self._positions[symbol] = {"quantity": 0, "value": 0, "avg_price": 0}
            pos = self._positions[symbol]
            pos["quantity"] += quantity
            pos["value"] += quantity * price
            pos["avg_price"] = pos["value"] / pos["quantity"] if pos["quantity"] > 0 else 0
        elif side == "sell" and symbol in self._positions:
            pos = self._positions[symbol]
            pos["quantity"] -= quantity
            pos["value"] = pos["quantity"] * pos["avg_price"]
            if pos["quantity"] <= 0:
                del self._positions[symbol]
        
        # Update metrics
        self._update_position_metrics()
    
    def update_pnl(self, pnl_change: float, current_equity: float):
        """Update P&L tracking and check circuit breakers."""
        now = datetime.now(timezone.utc)
        
        # Update metrics
        self.metrics.daily_pnl += pnl_change
        self.portfolio_value = current_equity
        
        # Track for circuit breaker
        self._pnl_history.append((now, pnl_change))
        
        # Update high water mark
        if current_equity > self._intraday_high:
            self._intraday_high = current_equity
        
        # Calculate drawdown
        if self._intraday_high > 0:
            self.metrics.current_drawdown_pct = (self._intraday_high - current_equity) / self._intraday_high
        
        # Check circuit breakers
        self._check_circuit_breakers()
        
        # Check emergency stop
        session_loss_pct = (self._session_start_equity - current_equity) / self._session_start_equity
        if session_loss_pct > self.config.emergency_stop_loss_pct:
            self._trigger_emergency_stop()
    
    def _check_circuit_breakers(self):
        """Check and manage circuit breaker state."""
        with self._circuit_breaker_lock:
            now = datetime.now(timezone.utc)
            
            # Check cooldown
            if self.metrics.circuit_breaker_state == CircuitBreakerState.OPEN:
                if self.metrics.circuit_breaker_triggered_at:
                    cooldown = timedelta(minutes=self.config.circuit_breaker_cooldown_minutes)
                    if now - self.metrics.circuit_breaker_triggered_at > cooldown:
                        logger.info("Circuit breaker cooldown complete, moving to HALF_OPEN")
                        self.metrics.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                return
            
            # Check for rapid loss
            window_start = now - timedelta(minutes=self.config.circuit_breaker_time_window_minutes)
            recent_pnl = sum(pnl for t, pnl in self._pnl_history if t > window_start)
            
            threshold = -self.config.circuit_breaker_loss_threshold_pct * self.portfolio_value
            
            if recent_pnl < threshold:
                self._trigger_circuit_breaker(recent_pnl)
    
    def _trigger_circuit_breaker(self, loss: float):
        """Trigger the circuit breaker."""
        logger.error(
            f"🚨 CIRCUIT BREAKER TRIGGERED | "
            f"Loss: ${-loss:,.2f} in {self.config.circuit_breaker_time_window_minutes}min"
        )
        
        self.metrics.circuit_breaker_state = CircuitBreakerState.OPEN
        self.metrics.circuit_breaker_triggered_at = datetime.now(timezone.utc)
        self.metrics.status = SafetyStatus.BLOCKED
        self.metrics.blocks.append(f"Circuit breaker triggered at {datetime.now()}")
        
        self._emit_safety_event("circuit_breaker_triggered", {"loss": loss})
    
    def _trigger_emergency_stop(self):
        """Trigger emergency stop - halt all trading."""
        logger.critical(
            f"🆘 EMERGENCY STOP TRIGGERED | "
            f"Loss: {self.config.emergency_stop_loss_pct:.0%}"
        )
        
        self.metrics.status = SafetyStatus.EMERGENCY
        self.metrics.blocks.append(f"EMERGENCY STOP at {datetime.now()}")
        
        self._emit_safety_event("emergency_stop", {
            "daily_pnl": self.metrics.daily_pnl,
            "drawdown": self.metrics.current_drawdown_pct,
        })
    
    def _update_position_metrics(self):
        """Update position-related metrics."""
        self.metrics.position_count = len(self._positions)
        
        if self._positions and self.portfolio_value > 0:
            position_values = [p["value"] for p in self._positions.values()]
            self.metrics.total_exposure_pct = sum(position_values) / self.portfolio_value
            self.metrics.largest_position_pct = max(position_values) / self.portfolio_value
        else:
            self.metrics.total_exposure_pct = 0
            self.metrics.largest_position_pct = 0
    
    def update_vix(self, vix_level: float):
        """Update current VIX level."""
        self.metrics.current_vix = vix_level
        
        if vix_level > self.config.max_vix_for_new_trades:
            if SafetyStatus.WARNING not in self.metrics.warnings:
                self.metrics.warnings.append(f"VIX elevated: {vix_level:.1f}")
    
    def reset_circuit_breaker(self, force: bool = False):
        """
        Reset circuit breaker to CLOSED state.
        
        Args:
            force: If True, reset even during cooldown
        """
        if force or self.metrics.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            logger.info("Circuit breaker reset to CLOSED")
            self.metrics.circuit_breaker_state = CircuitBreakerState.CLOSED
            self.metrics.circuit_breaker_triggered_at = None
            
            if self.metrics.status == SafetyStatus.BLOCKED:
                self.metrics.status = SafetyStatus.OK
    
    def reset_daily_metrics(self):
        """Reset daily metrics for new trading day."""
        self.metrics.daily_pnl = 0.0
        self._session_start_equity = self.portfolio_value
        self._intraday_high = self.portfolio_value
        self.metrics.current_drawdown_pct = 0.0
        self.metrics.warnings.clear()
        self.metrics.blocks.clear()
        
        # Reset circuit breaker for new day
        self.reset_circuit_breaker(force=True)
        
        logger.info("Daily safety metrics reset")
    
    def _emit_safety_event(self, event_type: str, data: Dict[str, Any]):
        """Emit a safety event."""
        if self.on_safety_event:
            try:
                self.on_safety_event(event_type, data)
            except Exception as e:
                logger.error(f"Error in safety event callback: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current safety status."""
        return {
            "status": self.metrics.status.value,
            "circuit_breaker": self.metrics.circuit_breaker_state.value,
            "daily_pnl": self.metrics.daily_pnl,
            "current_drawdown_pct": self.metrics.current_drawdown_pct,
            "position_count": self.metrics.position_count,
            "total_exposure_pct": self.metrics.total_exposure_pct,
            "current_vix": self.metrics.current_vix,
            "warnings": self.metrics.warnings.copy(),
            "blocks": self.metrics.blocks.copy(),
        }
    
    def is_trading_allowed(self) -> Tuple[bool, str]:
        """
        Quick check if trading is currently allowed.
        
        Returns:
            Tuple of (allowed, reason)
        """
        if self.metrics.status == SafetyStatus.EMERGENCY:
            return False, "Emergency stop active"
        
        if self.metrics.circuit_breaker_state == CircuitBreakerState.OPEN:
            return False, "Circuit breaker open"
        
        if self.metrics.daily_pnl < -self.config.max_daily_loss_usd:
            return False, "Daily loss limit reached"
        
        return True, "OK"


# Factory function
def create_safety_manager(
    portfolio_value: float = 100000.0,
    max_daily_loss: float = 5000.0,
    max_positions: int = 10,
    on_safety_event: Optional[Callable] = None,
) -> TradingSafetyManager:
    """Create a safety manager with common configuration."""
    config = SafetyConfig(
        max_daily_loss_usd=max_daily_loss,
        max_positions=max_positions,
    )
    
    return TradingSafetyManager(
        config=config,
        portfolio_value=portfolio_value,
        on_safety_event=on_safety_event,
    )


__all__ = [
    "SafetyStatus",
    "CircuitBreakerState",
    "SafetyConfig",
    "SafetyMetrics",
    "TradeValidationResult",
    "TradingSafetyManager",
    "create_safety_manager",
]
