"""
Anomaly Detection System

Detects market anomalies and generates alerts:
- Price anomalies
- Volume spikes
- Volatility regime changes
- Correlation breakdowns
- Technical indicator divergences

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from collections import deque

from loguru import logger


class AnomalyType(str, Enum):
    """Types of market anomalies."""
    PRICE_SPIKE = "price_spike"
    PRICE_GAP = "price_gap"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    VOLATILITY_REGIME_CHANGE = "volatility_regime_change"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    TECHNICAL_DIVERGENCE = "technical_divergence"
    SPREAD_ANOMALY = "spread_anomaly"
    LIQUIDITY_ANOMALY = "liquidity_anomaly"
    FLASH_CRASH = "flash_crash"


class AnomalySeverity(str, Enum):
    """Anomaly severity levels."""
    INFO = "info"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"


@dataclass
class AnomalyAlert:
    """Alert for detected anomaly."""
    alert_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    
    # Context
    symbol: str
    detected_at: datetime
    
    # Values
    current_value: float
    expected_value: float
    deviation: float  # Standard deviations or percentage
    
    # Analysis
    description: str
    recommendation: str
    
    # Historical context
    historical_frequency: Optional[str] = None  # e.g., "1 in 100 days"
    
    # Related data
    related_symbols: List[str] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "symbol": self.symbol,
            "detected_at": self.detected_at.isoformat(),
            "current_value": self.current_value,
            "expected_value": self.expected_value,
            "deviation": self.deviation,
            "description": self.description,
            "recommendation": self.recommendation,
            "historical_frequency": self.historical_frequency,
            "related_symbols": self.related_symbols,
            "contributing_factors": self.contributing_factors,
        }


@dataclass
class MarketDataPoint:
    """Single market data point."""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float
    ask: float
    
    # Optional
    high: Optional[float] = None
    low: Optional[float] = None
    vwap: Optional[float] = None


@dataclass
class AnomalyDetectorConfig:
    """Configuration for anomaly detector."""
    # Price anomaly thresholds
    price_spike_zscore: float = 3.0  # Z-score for price spike
    price_gap_threshold: float = 0.02  # 2% gap
    
    # Volume thresholds
    volume_spike_multiplier: float = 5.0  # 5x average
    
    # Volatility thresholds
    volatility_spike_zscore: float = 2.5
    regime_change_window: int = 20  # Days
    
    # Correlation thresholds
    correlation_breakdown_threshold: float = 0.3  # Change in correlation
    
    # Technical thresholds
    divergence_threshold: float = 0.05  # 5% divergence
    
    # Spread/liquidity
    spread_anomaly_multiplier: float = 3.0
    
    # Flash crash detection
    flash_crash_threshold: float = 0.05  # 5% in short time
    flash_crash_window_seconds: int = 60
    
    # Historical data
    lookback_periods: int = 100
    min_data_points: int = 20


class StatisticalTracker:
    """Tracks statistics for a series."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize tracker."""
        self.data = deque(maxlen=max_size)
        self._mean = 0.0
        self._std = 0.0
        self._min = float('inf')
        self._max = float('-inf')
    
    def update(self, value: float) -> None:
        """Add new value and update statistics."""
        self.data.append(value)
        
        if len(self.data) >= 2:
            self._mean = sum(self.data) / len(self.data)
            self._std = statistics.stdev(self.data)
            self._min = min(self.data)
            self._max = max(self.data)
    
    @property
    def mean(self) -> float:
        return self._mean
    
    @property
    def std(self) -> float:
        return self._std if self._std > 0 else 1e-6
    
    def zscore(self, value: float) -> float:
        """Calculate z-score for value."""
        return (value - self._mean) / self.std
    
    def percentile(self, value: float) -> float:
        """Calculate percentile rank."""
        if not self.data:
            return 50.0
        below = sum(1 for v in self.data if v < value)
        return below / len(self.data) * 100


class AnomalyDetector:
    """
    Market anomaly detection system.
    
    Features:
    - Real-time anomaly detection
    - Multiple anomaly types
    - Statistical analysis
    - Alert generation
    - Historical context
    """
    
    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        """Initialize anomaly detector."""
        self.config = config or AnomalyDetectorConfig()
        
        # Trackers per symbol
        self._price_trackers: Dict[str, StatisticalTracker] = {}
        self._volume_trackers: Dict[str, StatisticalTracker] = {}
        self._volatility_trackers: Dict[str, StatisticalTracker] = {}
        self._return_trackers: Dict[str, StatisticalTracker] = {}
        self._spread_trackers: Dict[str, StatisticalTracker] = {}
        
        # Recent data for flash crash detection
        self._recent_prices: Dict[str, deque] = {}
        
        # Alert history
        self._alerts: List[AnomalyAlert] = []
        self._alert_counter = 0
        
        # Callbacks
        self._alert_callbacks: List[Callable] = []
        
        # Correlation tracking
        self._correlation_pairs: Dict[str, StatisticalTracker] = {}
        
        logger.info("AnomalyDetector initialized")
    
    def register_alert_callback(self, callback: Callable) -> None:
        """Register callback for alerts."""
        self._alert_callbacks.append(callback)
    
    def _get_tracker(
        self,
        trackers: Dict[str, StatisticalTracker],
        symbol: str,
    ) -> StatisticalTracker:
        """Get or create tracker for symbol."""
        if symbol not in trackers:
            trackers[symbol] = StatisticalTracker(self.config.lookback_periods)
        return trackers[symbol]
    
    def process_data(self, data: MarketDataPoint) -> List[AnomalyAlert]:
        """
        Process new market data and detect anomalies.
        
        Returns list of detected anomalies.
        """
        alerts = []
        
        # Get/create trackers
        price_tracker = self._get_tracker(self._price_trackers, data.symbol)
        volume_tracker = self._get_tracker(self._volume_trackers, data.symbol)
        spread_tracker = self._get_tracker(self._spread_trackers, data.symbol)
        return_tracker = self._get_tracker(self._return_trackers, data.symbol)
        
        # Calculate derived values
        spread = data.ask - data.bid
        spread_pct = spread / data.price if data.price > 0 else 0
        
        # Calculate return if we have previous data
        if price_tracker.data:
            prev_price = price_tracker.data[-1]
            ret = (data.price - prev_price) / prev_price if prev_price > 0 else 0
            return_tracker.update(ret)
        
        # Check for various anomalies
        
        # 1. Price spike
        if len(price_tracker.data) >= self.config.min_data_points:
            zscore = price_tracker.zscore(data.price)
            if abs(zscore) > self.config.price_spike_zscore:
                alert = self._create_price_spike_alert(data, zscore)
                alerts.append(alert)
        
        # 2. Volume spike
        if len(volume_tracker.data) >= self.config.min_data_points:
            vol_ratio = data.volume / volume_tracker.mean if volume_tracker.mean > 0 else 1
            if vol_ratio > self.config.volume_spike_multiplier:
                alert = self._create_volume_spike_alert(data, vol_ratio)
                alerts.append(alert)
        
        # 3. Spread anomaly
        if len(spread_tracker.data) >= self.config.min_data_points:
            spread_ratio = spread_pct / spread_tracker.mean if spread_tracker.mean > 0 else 1
            if spread_ratio > self.config.spread_anomaly_multiplier:
                alert = self._create_spread_anomaly_alert(data, spread_ratio, spread_pct)
                alerts.append(alert)
        
        # 4. Flash crash detection
        flash_alert = self._detect_flash_crash(data)
        if flash_alert:
            alerts.append(flash_alert)
        
        # 5. Price gap detection
        if price_tracker.data:
            prev_price = price_tracker.data[-1]
            gap_pct = abs(data.price - prev_price) / prev_price if prev_price > 0 else 0
            if gap_pct > self.config.price_gap_threshold:
                alert = self._create_gap_alert(data, gap_pct, prev_price)
                alerts.append(alert)
        
        # Update trackers
        price_tracker.update(data.price)
        volume_tracker.update(data.volume)
        spread_tracker.update(spread_pct)
        
        # Store and notify alerts
        for alert in alerts:
            self._alerts.append(alert)
            self._notify_alert(alert)
        
        return alerts
    
    def _create_price_spike_alert(
        self,
        data: MarketDataPoint,
        zscore: float,
    ) -> AnomalyAlert:
        """Create price spike alert."""
        self._alert_counter += 1
        
        tracker = self._price_trackers[data.symbol]
        direction = "up" if zscore > 0 else "down"
        
        severity = AnomalySeverity.ALERT if abs(zscore) > 4 else AnomalySeverity.WARNING
        
        return AnomalyAlert(
            alert_id=f"PRICE_{self._alert_counter}",
            anomaly_type=AnomalyType.PRICE_SPIKE,
            severity=severity,
            symbol=data.symbol,
            detected_at=data.timestamp,
            current_value=data.price,
            expected_value=tracker.mean,
            deviation=zscore,
            description=f"Price spike {direction}: {data.price:.2f} ({zscore:.1f} std devs from mean {tracker.mean:.2f})",
            recommendation=f"Review position in {data.symbol}. Consider reducing exposure if long/short {direction}.",
            historical_frequency=f"~{self._estimate_frequency(abs(zscore))}",
        )
    
    def _create_volume_spike_alert(
        self,
        data: MarketDataPoint,
        ratio: float,
    ) -> AnomalyAlert:
        """Create volume spike alert."""
        self._alert_counter += 1
        
        tracker = self._volume_trackers[data.symbol]
        
        severity = AnomalySeverity.ALERT if ratio > 10 else AnomalySeverity.WARNING
        
        return AnomalyAlert(
            alert_id=f"VOL_{self._alert_counter}",
            anomaly_type=AnomalyType.VOLUME_SPIKE,
            severity=severity,
            symbol=data.symbol,
            detected_at=data.timestamp,
            current_value=float(data.volume),
            expected_value=tracker.mean,
            deviation=ratio,
            description=f"Volume spike: {data.volume:,} ({ratio:.1f}x average of {tracker.mean:,.0f})",
            recommendation="Monitor for potential breakout or breakdown. Check for news or events.",
            contributing_factors=["Possible institutional activity", "News catalyst", "Technical breakout"],
        )
    
    def _create_spread_anomaly_alert(
        self,
        data: MarketDataPoint,
        ratio: float,
        spread_pct: float,
    ) -> AnomalyAlert:
        """Create spread anomaly alert."""
        self._alert_counter += 1
        
        return AnomalyAlert(
            alert_id=f"SPREAD_{self._alert_counter}",
            anomaly_type=AnomalyType.SPREAD_ANOMALY,
            severity=AnomalySeverity.WARNING,
            symbol=data.symbol,
            detected_at=data.timestamp,
            current_value=spread_pct * 100,
            expected_value=self._spread_trackers[data.symbol].mean * 100,
            deviation=ratio,
            description=f"Bid-ask spread anomaly: {spread_pct*100:.3f}% ({ratio:.1f}x normal)",
            recommendation="Liquidity may be thin. Use limit orders and reduce position size.",
        )
    
    def _create_gap_alert(
        self,
        data: MarketDataPoint,
        gap_pct: float,
        prev_price: float,
    ) -> AnomalyAlert:
        """Create price gap alert."""
        self._alert_counter += 1
        
        direction = "up" if data.price > prev_price else "down"
        severity = AnomalySeverity.ALERT if gap_pct > 0.05 else AnomalySeverity.WARNING
        
        return AnomalyAlert(
            alert_id=f"GAP_{self._alert_counter}",
            anomaly_type=AnomalyType.PRICE_GAP,
            severity=severity,
            symbol=data.symbol,
            detected_at=data.timestamp,
            current_value=data.price,
            expected_value=prev_price,
            deviation=gap_pct * 100,
            description=f"Price gap {direction}: {gap_pct*100:.1f}% ({prev_price:.2f} → {data.price:.2f})",
            recommendation=f"Gap {'may fill' if gap_pct < 0.03 else 'indicates strong momentum'}. Review catalyst.",
        )
    
    def _detect_flash_crash(self, data: MarketDataPoint) -> Optional[AnomalyAlert]:
        """Detect flash crash conditions."""
        if data.symbol not in self._recent_prices:
            self._recent_prices[data.symbol] = deque(maxlen=100)
        
        recent = self._recent_prices[data.symbol]
        recent.append((data.timestamp, data.price))
        
        # Look for rapid decline
        if len(recent) < 2:
            return None
        
        window_start = data.timestamp - timedelta(seconds=self.config.flash_crash_window_seconds)
        
        prices_in_window = [(t, p) for t, p in recent if t >= window_start]
        if len(prices_in_window) < 2:
            return None
        
        start_price = prices_in_window[0][1]
        current_price = prices_in_window[-1][1]
        
        change_pct = (current_price - start_price) / start_price if start_price > 0 else 0
        
        if abs(change_pct) > self.config.flash_crash_threshold:
            self._alert_counter += 1
            
            direction = "crash" if change_pct < 0 else "spike"
            
            return AnomalyAlert(
                alert_id=f"FLASH_{self._alert_counter}",
                anomaly_type=AnomalyType.FLASH_CRASH,
                severity=AnomalySeverity.CRITICAL,
                symbol=data.symbol,
                detected_at=data.timestamp,
                current_value=current_price,
                expected_value=start_price,
                deviation=change_pct * 100,
                description=f"Flash {direction}: {change_pct*100:.1f}% in {self.config.flash_crash_window_seconds}s",
                recommendation="IMMEDIATE REVIEW REQUIRED. Check for erroneous prints or market dislocation.",
            )
        
        return None
    
    def detect_volatility_regime_change(
        self,
        symbol: str,
        current_vol: float,
    ) -> Optional[AnomalyAlert]:
        """Detect volatility regime changes."""
        tracker = self._get_tracker(self._volatility_trackers, symbol)
        
        if len(tracker.data) < self.config.regime_change_window:
            tracker.update(current_vol)
            return None
        
        # Compare recent vol to historical
        recent_avg = statistics.mean(list(tracker.data)[-10:])
        historical_avg = statistics.mean(list(tracker.data)[:-10])
        
        if historical_avg > 0:
            change_ratio = recent_avg / historical_avg
            
            if change_ratio > 1.5 or change_ratio < 0.67:
                self._alert_counter += 1
                
                direction = "expansion" if change_ratio > 1 else "compression"
                
                alert = AnomalyAlert(
                    alert_id=f"REGIME_{self._alert_counter}",
                    anomaly_type=AnomalyType.VOLATILITY_REGIME_CHANGE,
                    severity=AnomalySeverity.ALERT,
                    symbol=symbol,
                    detected_at=datetime.now(),
                    current_value=recent_avg,
                    expected_value=historical_avg,
                    deviation=change_ratio,
                    description=f"Volatility regime {direction}: {recent_avg:.1%} vs historical {historical_avg:.1%}",
                    recommendation=f"Adjust position sizes and option strategies for {'higher' if direction == 'expansion' else 'lower'} volatility.",
                )
                
                self._alerts.append(alert)
                self._notify_alert(alert)
                
                tracker.update(current_vol)
                return alert
        
        tracker.update(current_vol)
        return None
    
    def detect_correlation_breakdown(
        self,
        symbol1: str,
        symbol2: str,
        returns1: List[float],
        returns2: List[float],
    ) -> Optional[AnomalyAlert]:
        """Detect correlation breakdown between symbols."""
        if len(returns1) < self.config.min_data_points or len(returns2) < self.config.min_data_points:
            return None
        
        # Calculate current correlation
        n = min(len(returns1), len(returns2))
        r1 = returns1[-n:]
        r2 = returns2[-n:]
        
        mean1 = sum(r1) / n
        mean2 = sum(r2) / n
        
        cov = sum((r1[i] - mean1) * (r2[i] - mean2) for i in range(n)) / n
        std1 = math.sqrt(sum((x - mean1) ** 2 for x in r1) / n)
        std2 = math.sqrt(sum((x - mean2) ** 2 for x in r2) / n)
        
        current_corr = cov / (std1 * std2) if std1 > 0 and std2 > 0 else 0
        
        # Track correlation
        pair_key = f"{symbol1}_{symbol2}"
        tracker = self._get_tracker(self._correlation_pairs, pair_key)
        
        if len(tracker.data) >= self.config.min_data_points:
            historical_corr = tracker.mean
            change = abs(current_corr - historical_corr)
            
            if change > self.config.correlation_breakdown_threshold:
                self._alert_counter += 1
                
                alert = AnomalyAlert(
                    alert_id=f"CORR_{self._alert_counter}",
                    anomaly_type=AnomalyType.CORRELATION_BREAKDOWN,
                    severity=AnomalySeverity.ALERT,
                    symbol=f"{symbol1}/{symbol2}",
                    detected_at=datetime.now(),
                    current_value=current_corr,
                    expected_value=historical_corr,
                    deviation=change,
                    description=f"Correlation breakdown: {symbol1}/{symbol2} correlation changed from {historical_corr:.2f} to {current_corr:.2f}",
                    recommendation="Review pair trades and hedge ratios. Market regime may be changing.",
                    related_symbols=[symbol1, symbol2],
                )
                
                self._alerts.append(alert)
                self._notify_alert(alert)
                
                tracker.update(current_corr)
                return alert
        
        tracker.update(current_corr)
        return None
    
    def detect_technical_divergence(
        self,
        symbol: str,
        price: float,
        indicator_value: float,
        indicator_name: str,
        prices: List[float],
        indicator_values: List[float],
    ) -> Optional[AnomalyAlert]:
        """Detect price/indicator divergence."""
        if len(prices) < 10 or len(indicator_values) < 10:
            return None
        
        # Check for divergence: price making new high/low but indicator not confirming
        recent_price_high = max(prices[-10:])
        recent_price_low = min(prices[-10:])
        recent_ind_high = max(indicator_values[-10:])
        recent_ind_low = min(indicator_values[-10:])
        
        # Bullish divergence: price lower low, indicator higher low
        if price <= recent_price_low * 1.01 and indicator_value > recent_ind_low * 1.05:
            self._alert_counter += 1
            
            return AnomalyAlert(
                alert_id=f"DIV_{self._alert_counter}",
                anomaly_type=AnomalyType.TECHNICAL_DIVERGENCE,
                severity=AnomalySeverity.INFO,
                symbol=symbol,
                detected_at=datetime.now(),
                current_value=indicator_value,
                expected_value=recent_ind_low,
                deviation=0,
                description=f"Bullish divergence: {symbol} price at low but {indicator_name} showing higher low",
                recommendation="Potential reversal signal. Watch for confirmation before acting.",
            )
        
        # Bearish divergence: price higher high, indicator lower high
        if price >= recent_price_high * 0.99 and indicator_value < recent_ind_high * 0.95:
            self._alert_counter += 1
            
            return AnomalyAlert(
                alert_id=f"DIV_{self._alert_counter}",
                anomaly_type=AnomalyType.TECHNICAL_DIVERGENCE,
                severity=AnomalySeverity.INFO,
                symbol=symbol,
                detected_at=datetime.now(),
                current_value=indicator_value,
                expected_value=recent_ind_high,
                deviation=0,
                description=f"Bearish divergence: {symbol} price at high but {indicator_name} showing lower high",
                recommendation="Potential reversal signal. Consider reducing long exposure.",
            )
        
        return None
    
    def _estimate_frequency(self, zscore: float) -> str:
        """Estimate historical frequency of z-score."""
        # Approximate based on normal distribution
        if abs(zscore) >= 4:
            return "1 in 15,000 observations"
        elif abs(zscore) >= 3:
            return "1 in 370 observations"
        elif abs(zscore) >= 2.5:
            return "1 in 160 observations"
        elif abs(zscore) >= 2:
            return "1 in 44 observations"
        else:
            return "1 in 20 observations"
    
    def _notify_alert(self, alert: AnomalyAlert) -> None:
        """Notify registered callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def get_alerts(
        self,
        symbol: Optional[str] = None,
        anomaly_type: Optional[AnomalyType] = None,
        severity: Optional[AnomalySeverity] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AnomalyAlert]:
        """Get filtered alerts."""
        alerts = self._alerts.copy()
        
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol]
        
        if anomaly_type:
            alerts = [a for a in alerts if a.anomaly_type == anomaly_type]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if since:
            alerts = [a for a in alerts if a.detected_at >= since]
        
        return sorted(alerts, key=lambda a: a.detected_at, reverse=True)[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "total_alerts": len(self._alerts),
            "symbols_tracked": len(self._price_trackers),
            "alerts_by_type": {
                t.value: sum(1 for a in self._alerts if a.anomaly_type == t)
                for t in AnomalyType
            },
            "alerts_by_severity": {
                s.value: sum(1 for a in self._alerts if a.severity == s)
                for s in AnomalySeverity
            },
        }
    
    def clear_old_alerts(self, hours: int = 24) -> int:
        """Clear alerts older than specified hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        original_count = len(self._alerts)
        self._alerts = [a for a in self._alerts if a.detected_at >= cutoff]
        return original_count - len(self._alerts)


# Singleton instance
anomaly_detector = AnomalyDetector()


# Convenience functions
def process_market_data(data: MarketDataPoint) -> List[AnomalyAlert]:
    """Process data through global detector."""
    return anomaly_detector.process_data(data)


def get_anomaly_alerts(**kwargs) -> List[AnomalyAlert]:
    """Get alerts from global detector."""
    return anomaly_detector.get_alerts(**kwargs)


def register_anomaly_callback(callback: Callable) -> None:
    """Register callback for anomaly alerts."""
    anomaly_detector.register_alert_callback(callback)
