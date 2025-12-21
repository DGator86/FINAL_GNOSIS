"""
Prometheus Metrics for Trading System Monitoring

Exports comprehensive metrics for:
- Trading performance (P&L, returns, Sharpe)
- Portfolio Greeks (delta, gamma, theta, vega)
- Order execution (latency, fill rates, slippage)
- System health (API latency, error rates)
- Risk metrics (VaR, drawdown, exposure)

Features:
- Prometheus-compatible metrics endpoint
- Custom trading-specific metrics
- Automatic metric labeling
- Histogram buckets for latency tracking
- Gauge metrics for real-time values

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from collections import defaultdict
import threading

from loguru import logger


# =============================================================================
# Metric Types
# =============================================================================

class MetricType(str, Enum):
    """Types of Prometheus metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


@dataclass
class MetricValue:
    """Value of a metric with labels."""
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Metric Registry
# =============================================================================

class MetricsRegistry:
    """
    Registry for all metrics.
    
    Manages metric definitions, values, and Prometheus export.
    """
    
    # Default histogram buckets for different metric types
    LATENCY_BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    PRICE_BUCKETS = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    PNL_BUCKETS = [-10000, -5000, -1000, -500, -100, 0, 100, 500, 1000, 5000, 10000]
    
    def __init__(self, prefix: str = "gnosis"):
        """Initialize the registry.
        
        Args:
            prefix: Metric name prefix
        """
        self.prefix = prefix
        self._metrics: Dict[str, MetricDefinition] = {}
        self._values: Dict[str, List[MetricValue]] = defaultdict(list)
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._lock = threading.Lock()
        
        # Register default trading metrics
        self._register_default_metrics()
        
        logger.info(f"MetricsRegistry initialized with prefix: {prefix}")
    
    def _register_default_metrics(self) -> None:
        """Register default trading system metrics."""
        
        # ======================
        # Trading Performance
        # ======================
        self.register_metric(MetricDefinition(
            name="portfolio_value_dollars",
            type=MetricType.GAUGE,
            description="Current portfolio value in dollars",
        ))
        
        self.register_metric(MetricDefinition(
            name="daily_pnl_dollars",
            type=MetricType.GAUGE,
            description="Daily P&L in dollars",
        ))
        
        self.register_metric(MetricDefinition(
            name="daily_pnl_percent",
            type=MetricType.GAUGE,
            description="Daily P&L as percentage",
        ))
        
        self.register_metric(MetricDefinition(
            name="unrealized_pnl_dollars",
            type=MetricType.GAUGE,
            description="Unrealized P&L in dollars",
            labels=["symbol"],
        ))
        
        self.register_metric(MetricDefinition(
            name="realized_pnl_dollars",
            type=MetricType.GAUGE,
            description="Realized P&L in dollars",
        ))
        
        self.register_metric(MetricDefinition(
            name="sharpe_ratio",
            type=MetricType.GAUGE,
            description="Rolling Sharpe ratio",
        ))
        
        self.register_metric(MetricDefinition(
            name="win_rate",
            type=MetricType.GAUGE,
            description="Trade win rate (0-1)",
        ))
        
        # ======================
        # Portfolio Greeks
        # ======================
        self.register_metric(MetricDefinition(
            name="portfolio_delta",
            type=MetricType.GAUGE,
            description="Net portfolio delta",
        ))
        
        self.register_metric(MetricDefinition(
            name="portfolio_gamma",
            type=MetricType.GAUGE,
            description="Net portfolio gamma",
        ))
        
        self.register_metric(MetricDefinition(
            name="portfolio_theta",
            type=MetricType.GAUGE,
            description="Net portfolio theta (daily)",
        ))
        
        self.register_metric(MetricDefinition(
            name="portfolio_vega",
            type=MetricType.GAUGE,
            description="Net portfolio vega",
        ))
        
        self.register_metric(MetricDefinition(
            name="position_delta",
            type=MetricType.GAUGE,
            description="Position delta",
            labels=["symbol"],
        ))
        
        # ======================
        # Order Execution
        # ======================
        self.register_metric(MetricDefinition(
            name="orders_total",
            type=MetricType.COUNTER,
            description="Total orders submitted",
            labels=["side", "type", "status"],
        ))
        
        self.register_metric(MetricDefinition(
            name="order_latency_seconds",
            type=MetricType.HISTOGRAM,
            description="Order execution latency",
            labels=["side", "type"],
            buckets=self.LATENCY_BUCKETS,
        ))
        
        self.register_metric(MetricDefinition(
            name="order_fill_rate",
            type=MetricType.GAUGE,
            description="Order fill rate",
        ))
        
        self.register_metric(MetricDefinition(
            name="order_slippage_dollars",
            type=MetricType.HISTOGRAM,
            description="Order slippage in dollars",
            labels=["side"],
            buckets=self.PRICE_BUCKETS,
        ))
        
        # ======================
        # Risk Metrics
        # ======================
        self.register_metric(MetricDefinition(
            name="var_95_dollars",
            type=MetricType.GAUGE,
            description="Value at Risk (95%) in dollars",
        ))
        
        self.register_metric(MetricDefinition(
            name="max_drawdown_percent",
            type=MetricType.GAUGE,
            description="Maximum drawdown percentage",
        ))
        
        self.register_metric(MetricDefinition(
            name="current_drawdown_percent",
            type=MetricType.GAUGE,
            description="Current drawdown percentage",
        ))
        
        self.register_metric(MetricDefinition(
            name="gross_exposure_dollars",
            type=MetricType.GAUGE,
            description="Gross exposure in dollars",
        ))
        
        self.register_metric(MetricDefinition(
            name="net_exposure_dollars",
            type=MetricType.GAUGE,
            description="Net exposure in dollars",
        ))
        
        self.register_metric(MetricDefinition(
            name="leverage_ratio",
            type=MetricType.GAUGE,
            description="Portfolio leverage ratio",
        ))
        
        self.register_metric(MetricDefinition(
            name="risk_score",
            type=MetricType.GAUGE,
            description="Overall risk score (0-100)",
        ))
        
        # ======================
        # Position Metrics
        # ======================
        self.register_metric(MetricDefinition(
            name="positions_count",
            type=MetricType.GAUGE,
            description="Number of open positions",
        ))
        
        self.register_metric(MetricDefinition(
            name="position_size_dollars",
            type=MetricType.GAUGE,
            description="Position size in dollars",
            labels=["symbol"],
        ))
        
        self.register_metric(MetricDefinition(
            name="position_weight",
            type=MetricType.GAUGE,
            description="Position weight in portfolio",
            labels=["symbol"],
        ))
        
        # ======================
        # API/System Health
        # ======================
        self.register_metric(MetricDefinition(
            name="api_requests_total",
            type=MetricType.COUNTER,
            description="Total API requests",
            labels=["endpoint", "method", "status"],
        ))
        
        self.register_metric(MetricDefinition(
            name="api_latency_seconds",
            type=MetricType.HISTOGRAM,
            description="API request latency",
            labels=["endpoint"],
            buckets=self.LATENCY_BUCKETS,
        ))
        
        self.register_metric(MetricDefinition(
            name="api_errors_total",
            type=MetricType.COUNTER,
            description="Total API errors",
            labels=["endpoint", "error_type"],
        ))
        
        self.register_metric(MetricDefinition(
            name="market_data_latency_seconds",
            type=MetricType.HISTOGRAM,
            description="Market data feed latency",
            labels=["source"],
            buckets=self.LATENCY_BUCKETS,
        ))
        
        self.register_metric(MetricDefinition(
            name="websocket_connections",
            type=MetricType.GAUGE,
            description="Active WebSocket connections",
        ))
        
        # ======================
        # Trading Signals
        # ======================
        self.register_metric(MetricDefinition(
            name="signals_generated_total",
            type=MetricType.COUNTER,
            description="Trading signals generated",
            labels=["signal_type", "direction"],
        ))
        
        self.register_metric(MetricDefinition(
            name="signal_confidence",
            type=MetricType.GAUGE,
            description="Signal confidence score",
            labels=["signal_type"],
        ))
        
        # ======================
        # ML Model Metrics
        # ======================
        self.register_metric(MetricDefinition(
            name="model_prediction_accuracy",
            type=MetricType.GAUGE,
            description="ML model prediction accuracy",
            labels=["model_name"],
        ))
        
        self.register_metric(MetricDefinition(
            name="model_inference_latency_seconds",
            type=MetricType.HISTOGRAM,
            description="ML model inference latency",
            labels=["model_name"],
            buckets=self.LATENCY_BUCKETS,
        ))
    
    def register_metric(self, metric: MetricDefinition) -> None:
        """Register a new metric.
        
        Args:
            metric: Metric definition
        """
        full_name = f"{self.prefix}_{metric.name}"
        self._metrics[full_name] = metric
    
    def _get_label_key(self, labels: Dict[str, str]) -> str:
        """Generate a unique key from labels."""
        if not labels:
            return ""
        sorted_items = sorted(labels.items())
        return ",".join(f'{k}="{v}"' for k, v in sorted_items)
    
    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric.
        
        Args:
            name: Metric name (without prefix)
            value: Value to add
            labels: Optional labels
        """
        full_name = f"{self.prefix}_{name}"
        label_key = self._get_label_key(labels or {})
        
        with self._lock:
            self._counters[full_name][label_key] += value
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric value.
        
        Args:
            name: Metric name (without prefix)
            value: New value
            labels: Optional labels
        """
        full_name = f"{self.prefix}_{name}"
        label_key = self._get_label_key(labels or {})
        
        with self._lock:
            self._gauges[full_name][label_key] = value
    
    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a value for a histogram metric.
        
        Args:
            name: Metric name (without prefix)
            value: Observed value
            labels: Optional labels
        """
        full_name = f"{self.prefix}_{name}"
        label_key = self._get_label_key(labels or {})
        
        with self._lock:
            self._histograms[full_name][label_key].append(value)
    
    def get_prometheus_output(self) -> str:
        """Generate Prometheus-compatible metrics output.
        
        Returns:
            Prometheus text format metrics
        """
        lines = []
        
        with self._lock:
            # Output counters
            for name, label_values in self._counters.items():
                if name in self._metrics:
                    metric = self._metrics[name]
                    lines.append(f"# HELP {name} {metric.description}")
                    lines.append(f"# TYPE {name} counter")
                
                for label_key, value in label_values.items():
                    if label_key:
                        lines.append(f"{name}{{{label_key}}} {value}")
                    else:
                        lines.append(f"{name} {value}")
            
            # Output gauges
            for name, label_values in self._gauges.items():
                if name in self._metrics:
                    metric = self._metrics[name]
                    lines.append(f"# HELP {name} {metric.description}")
                    lines.append(f"# TYPE {name} gauge")
                
                for label_key, value in label_values.items():
                    if label_key:
                        lines.append(f"{name}{{{label_key}}} {value}")
                    else:
                        lines.append(f"{name} {value}")
            
            # Output histograms
            for name, label_values in self._histograms.items():
                if name in self._metrics:
                    metric = self._metrics[name]
                    lines.append(f"# HELP {name} {metric.description}")
                    lines.append(f"# TYPE {name} histogram")
                    
                    buckets = metric.buckets or self.LATENCY_BUCKETS
                    
                    for label_key, observations in label_values.items():
                        if not observations:
                            continue
                        
                        # Calculate bucket counts
                        for bucket in buckets:
                            count = sum(1 for v in observations if v <= bucket)
                            bucket_label = f'le="{bucket}"'
                            if label_key:
                                lines.append(f"{name}_bucket{{{label_key},{bucket_label}}} {count}")
                            else:
                                lines.append(f"{name}_bucket{{{bucket_label}}} {count}")
                        
                        # +Inf bucket
                        inf_label = 'le="+Inf"'
                        if label_key:
                            lines.append(f"{name}_bucket{{{label_key},{inf_label}}} {len(observations)}")
                        else:
                            lines.append(f"{name}_bucket{{{inf_label}}} {len(observations)}")
                        
                        # Sum and count
                        total = sum(observations)
                        if label_key:
                            lines.append(f"{name}_sum{{{label_key}}} {total}")
                            lines.append(f"{name}_count{{{label_key}}} {len(observations)}")
                        else:
                            lines.append(f"{name}_sum {total}")
                            lines.append(f"{name}_count {len(observations)}")
        
        return "\n".join(lines)
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary.
        
        Returns:
            Dictionary of all metric values
        """
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: {k: len(v) for k, v in label_values.items()}
                    for name, label_values in self._histograms.items()
                },
            }
    
    def reset(self) -> None:
        """Reset all metric values."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# =============================================================================
# Global Metrics Instance
# =============================================================================

metrics = MetricsRegistry()


# =============================================================================
# Decorators for Automatic Metrics
# =============================================================================

F = TypeVar('F', bound=Callable[..., Any])


def track_latency(metric_name: str, labels: Optional[Dict[str, str]] = None) -> Callable[[F], F]:
    """Decorator to track function execution latency.
    
    Args:
        metric_name: Name of the latency histogram
        labels: Optional labels
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start_time
                metrics.observe_histogram(metric_name, elapsed, labels)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.time() - start_time
                metrics.observe_histogram(metric_name, elapsed, labels)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore
    
    return decorator


def count_calls(metric_name: str, labels: Optional[Dict[str, str]] = None) -> Callable[[F], F]:
    """Decorator to count function calls.
    
    Args:
        metric_name: Name of the counter
        labels: Optional labels
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics.inc_counter(metric_name, 1, labels)
            return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics.inc_counter(metric_name, 1, labels)
            return await func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Metrics Collector (Gathers from System)
# =============================================================================

class MetricsCollector:
    """
    Collects metrics from various system components.
    
    Periodically gathers and updates metrics from:
    - Portfolio manager
    - Order executor
    - Risk manager
    - API health
    """
    
    def __init__(self, registry: MetricsRegistry):
        """Initialize collector.
        
        Args:
            registry: Metrics registry to update
        """
        self.registry = registry
        self._running = False
    
    async def start(self, interval: float = 5.0) -> None:
        """Start periodic metrics collection.
        
        Args:
            interval: Collection interval in seconds
        """
        self._running = True
        
        while self._running:
            try:
                await self._collect_all()
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
            
            await asyncio.sleep(interval)
    
    def stop(self) -> None:
        """Stop metrics collection."""
        self._running = False
    
    async def _collect_all(self) -> None:
        """Collect all metrics."""
        # These would integrate with actual system components
        # For now, they serve as placeholders showing the structure
        
        await self._collect_portfolio_metrics()
        await self._collect_risk_metrics()
        await self._collect_system_metrics()
    
    async def _collect_portfolio_metrics(self) -> None:
        """Collect portfolio-related metrics."""
        # Would integrate with actual portfolio manager
        pass
    
    async def _collect_risk_metrics(self) -> None:
        """Collect risk-related metrics."""
        # Would integrate with actual risk manager
        pass
    
    async def _collect_system_metrics(self) -> None:
        """Collect system health metrics."""
        # Would check API health, connections, etc.
        pass


# Need asyncio for the collector
import asyncio

collector = MetricsCollector(metrics)


# =============================================================================
# Convenience Functions
# =============================================================================

def update_portfolio_metrics(
    total_value: float,
    daily_pnl: float,
    unrealized_pnl: float,
    realized_pnl: float,
) -> None:
    """Update portfolio performance metrics.
    
    Args:
        total_value: Total portfolio value
        daily_pnl: Daily P&L in dollars
        unrealized_pnl: Unrealized P&L
        realized_pnl: Realized P&L
    """
    metrics.set_gauge("portfolio_value_dollars", total_value)
    metrics.set_gauge("daily_pnl_dollars", daily_pnl)
    metrics.set_gauge("daily_pnl_percent", (daily_pnl / total_value * 100) if total_value > 0 else 0)
    metrics.set_gauge("unrealized_pnl_dollars", unrealized_pnl)
    metrics.set_gauge("realized_pnl_dollars", realized_pnl)


def update_greeks_metrics(
    delta: float,
    gamma: float,
    theta: float,
    vega: float,
) -> None:
    """Update portfolio Greeks metrics.
    
    Args:
        delta: Portfolio delta
        gamma: Portfolio gamma
        theta: Portfolio theta
        vega: Portfolio vega
    """
    metrics.set_gauge("portfolio_delta", delta)
    metrics.set_gauge("portfolio_gamma", gamma)
    metrics.set_gauge("portfolio_theta", theta)
    metrics.set_gauge("portfolio_vega", vega)


def update_risk_metrics(
    var_95: float,
    max_drawdown: float,
    current_drawdown: float,
    risk_score: float,
) -> None:
    """Update risk metrics.
    
    Args:
        var_95: Value at Risk (95%)
        max_drawdown: Maximum drawdown percentage
        current_drawdown: Current drawdown percentage
        risk_score: Overall risk score
    """
    metrics.set_gauge("var_95_dollars", var_95)
    metrics.set_gauge("max_drawdown_percent", max_drawdown)
    metrics.set_gauge("current_drawdown_percent", current_drawdown)
    metrics.set_gauge("risk_score", risk_score)


def record_order(
    side: str,
    order_type: str,
    status: str,
    latency: Optional[float] = None,
    slippage: Optional[float] = None,
) -> None:
    """Record order execution metrics.
    
    Args:
        side: Order side (buy/sell)
        order_type: Order type (market/limit)
        status: Order status (filled/rejected/etc)
        latency: Execution latency in seconds
        slippage: Slippage in dollars
    """
    labels = {"side": side, "type": order_type, "status": status}
    metrics.inc_counter("orders_total", 1, labels)
    
    if latency is not None:
        metrics.observe_histogram("order_latency_seconds", latency, {"side": side, "type": order_type})
    
    if slippage is not None:
        metrics.observe_histogram("order_slippage_dollars", slippage, {"side": side})


def record_api_request(
    endpoint: str,
    method: str,
    status: int,
    latency: float,
) -> None:
    """Record API request metrics.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        status: Response status code
        latency: Request latency in seconds
    """
    labels = {"endpoint": endpoint, "method": method, "status": str(status)}
    metrics.inc_counter("api_requests_total", 1, labels)
    metrics.observe_histogram("api_latency_seconds", latency, {"endpoint": endpoint})
    
    if status >= 400:
        metrics.inc_counter("api_errors_total", 1, {"endpoint": endpoint, "error_type": str(status)})
