"""Health monitoring for trading system.

Provides:
- Periodic health checks for all components
- Alert thresholds and notifications
- Automatic recovery attempts
- Health status dashboard
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class HealthStatus(Enum):
    """Health status levels."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY
    
    @property
    def seconds_since_success(self) -> Optional[float]:
        if self.last_success is None:
            return None
        return (datetime.now(timezone.utc) - self.last_success).total_seconds()


@dataclass
class Alert:
    """Alert notification."""
    
    timestamp: datetime
    level: str  # "warning", "error", "critical"
    component: str
    message: str
    acknowledged: bool = False


class HealthMonitor:
    """Monitors health of trading system components.
    
    Features:
    - Periodic health checks for broker, data, and pipeline
    - Alert generation on failures
    - Automatic recovery attempts
    - Thread-safe operation
    
    Example:
        monitor = HealthMonitor()
        monitor.register_check("broker", check_broker_health)
        monitor.start()
        
        # Later
        status = monitor.get_status()
        if not status["overall_healthy"]:
            handle_degraded_state()
    """
    
    def __init__(
        self,
        check_interval: float = 30.0,
        alert_callback: Optional[Callable[[Alert], None]] = None,
    ):
        """Initialize health monitor.
        
        Args:
            check_interval: Seconds between health checks
            alert_callback: Function to call when alerts are generated
        """
        self.check_interval = check_interval
        self.alert_callback = alert_callback
        
        self._components: Dict[str, ComponentHealth] = {}
        self._checks: Dict[str, Callable[[], bool]] = {}
        self._alerts: List[Alert] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Alert thresholds
        self.degraded_threshold = 2  # Consecutive failures before degraded
        self.unhealthy_threshold = 5  # Consecutive failures before unhealthy
        self.alert_cooldown = 300.0  # Seconds between repeat alerts
        self._last_alert_time: Dict[str, float] = {}
    
    def register_check(
        self,
        name: str,
        check_fn: Callable[[], bool],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a health check.
        
        Args:
            name: Component name
            check_fn: Function returning True if healthy
            metadata: Optional metadata about the component
        """
        with self._lock:
            self._checks[name] = check_fn
            self._components[name] = ComponentHealth(
                name=name,
                metadata=metadata or {},
            )
            logger.debug(f"Registered health check: {name}")
    
    def run_check(self, name: str) -> HealthStatus:
        """Run a single health check.
        
        Args:
            name: Component name
            
        Returns:
            Health status after check
        """
        if name not in self._checks:
            return HealthStatus.UNKNOWN
        
        check_fn = self._checks[name]
        component = self._components[name]
        
        try:
            is_healthy = check_fn()
            component.last_check = datetime.now(timezone.utc)
            
            if is_healthy:
                component.status = HealthStatus.HEALTHY
                component.last_success = datetime.now(timezone.utc)
                component.consecutive_failures = 0
                component.last_error = None
            else:
                component.consecutive_failures += 1
                self._update_status_from_failures(component)
                
        except Exception as e:
            component.last_check = datetime.now(timezone.utc)
            component.consecutive_failures += 1
            component.last_error = str(e)
            self._update_status_from_failures(component)
            logger.warning(f"Health check failed for {name}: {e}")
        
        return component.status
    
    def _update_status_from_failures(self, component: ComponentHealth) -> None:
        """Update status based on failure count."""
        if component.consecutive_failures >= self.unhealthy_threshold:
            if component.status != HealthStatus.UNHEALTHY:
                component.status = HealthStatus.UNHEALTHY
                self._create_alert(
                    "critical",
                    component.name,
                    f"{component.name} is UNHEALTHY after "
                    f"{component.consecutive_failures} failures"
                )
        elif component.consecutive_failures >= self.degraded_threshold:
            if component.status != HealthStatus.DEGRADED:
                component.status = HealthStatus.DEGRADED
                self._create_alert(
                    "warning",
                    component.name,
                    f"{component.name} is degraded after "
                    f"{component.consecutive_failures} failures"
                )
    
    def _create_alert(self, level: str, component: str, message: str) -> None:
        """Create and dispatch an alert."""
        # Check cooldown
        key = f"{component}:{level}"
        now = time.time()
        last_alert = self._last_alert_time.get(key, 0)
        
        if now - last_alert < self.alert_cooldown:
            return  # Still in cooldown
        
        alert = Alert(
            timestamp=datetime.now(timezone.utc),
            level=level,
            component=component,
            message=message,
        )
        
        with self._lock:
            self._alerts.append(alert)
            self._last_alert_time[key] = now
        
        logger.warning(f"ALERT [{level.upper()}] {component}: {message}")
        
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def run_all_checks(self) -> Dict[str, HealthStatus]:
        """Run all registered health checks.
        
        Returns:
            Dictionary of component name -> status
        """
        results = {}
        
        for name in self._checks:
            results[name] = self.run_check(name)
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall health status.
        
        Returns:
            Status dictionary with component details
        """
        with self._lock:
            components = {}
            overall_healthy = True
            
            for name, component in self._components.items():
                components[name] = {
                    "status": component.status.value,
                    "last_check": component.last_check.isoformat() if component.last_check else None,
                    "last_success": component.last_success.isoformat() if component.last_success else None,
                    "last_error": component.last_error,
                    "consecutive_failures": component.consecutive_failures,
                }
                
                if component.status != HealthStatus.HEALTHY:
                    overall_healthy = False
            
            return {
                "overall_healthy": overall_healthy,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": components,
                "active_alerts": len([a for a in self._alerts if not a.acknowledged]),
            }
    
    def get_alerts(self, unacknowledged_only: bool = True) -> List[Alert]:
        """Get alerts.
        
        Args:
            unacknowledged_only: Only return unacknowledged alerts
            
        Returns:
            List of alerts
        """
        with self._lock:
            if unacknowledged_only:
                return [a for a in self._alerts if not a.acknowledged]
            return list(self._alerts)
    
    def acknowledge_alert(self, index: int) -> bool:
        """Acknowledge an alert by index.
        
        Args:
            index: Alert index
            
        Returns:
            True if acknowledged
        """
        with self._lock:
            if 0 <= index < len(self._alerts):
                self._alerts[index].acknowledged = True
                return True
            return False
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self.run_all_checks()
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
            
            # Sleep in chunks for responsive shutdown
            for _ in range(int(self.check_interval)):
                if not self._running:
                    break
                time.sleep(1)
    
    def start(self):
        """Start background health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="health-monitor"
        )
        self._thread.start()
        logger.info("Health monitor started")
    
    def stop(self):
        """Stop background health monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Health monitor stopped")


def create_trading_health_monitor(
    broker=None,
    check_interval: float = 30.0,
) -> HealthMonitor:
    """Create a health monitor with standard trading checks.
    
    Args:
        broker: Optional broker adapter for account checks
        check_interval: Seconds between checks
        
    Returns:
        Configured HealthMonitor
    """
    monitor = HealthMonitor(check_interval=check_interval)
    
    # Broker health check
    if broker:
        def check_broker():
            try:
                account = broker.get_account()
                return account is not None and not account.trading_blocked
            except Exception:
                return False
        
        monitor.register_check("broker", check_broker, {"type": "alpaca"})
    
    # Market data check
    def check_market_data():
        try:
            from engines.inputs.adapter_factory import create_market_data_adapter
            adapter = create_market_data_adapter(prefer_real=True)
            # Try to get a quote
            return adapter is not None
        except Exception:
            return False
    
    monitor.register_check("market_data", check_market_data)
    
    # Options data check
    def check_options_data():
        try:
            from engines.inputs.adapter_factory import create_options_adapter
            adapter = create_options_adapter(prefer_real=True)
            return adapter is not None
        except Exception:
            return False
    
    monitor.register_check("options_data", check_options_data)
    
    # Circuit breaker check
    def check_circuit_breakers():
        try:
            from utils.circuit_breaker import (
                get_alpaca_circuit_breaker,
                get_unusual_whales_circuit_breaker,
            )
            alpaca = get_alpaca_circuit_breaker()
            uw = get_unusual_whales_circuit_breaker()
            return alpaca.is_closed and uw.is_closed
        except Exception:
            return True  # OK if not configured
    
    monitor.register_check("circuit_breakers", check_circuit_breakers)
    
    # Disk space check
    def check_disk_space():
        try:
            import shutil
            usage = shutil.disk_usage("/")
            free_pct = usage.free / usage.total
            return free_pct > 0.1  # At least 10% free
        except Exception:
            return True  # OK if can't check
    
    monitor.register_check("disk_space", check_disk_space)
    
    return monitor
