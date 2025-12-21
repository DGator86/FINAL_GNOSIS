"""
Options Flow Scanner - Institutional Flow Detection

Detects unusual options activity that may indicate institutional positioning:
- Large block trades
- Unusual volume spikes
- Sweep orders across exchanges
- Dark pool activity
- Smart money flow indicators

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import statistics

from loguru import logger


class FlowType(str, Enum):
    """Types of options flow."""
    BLOCK = "block"              # Large single trade
    SWEEP = "sweep"              # Multiple exchanges hit rapidly
    SPLIT = "split"              # Large order split into pieces
    UNUSUAL_VOLUME = "unusual_volume"  # Volume spike
    DARK_POOL = "dark_pool"      # Off-exchange activity
    OPENING = "opening"          # New position opening
    CLOSING = "closing"          # Position closing


class FlowSentiment(str, Enum):
    """Flow sentiment classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class TradeAggressor(str, Enum):
    """Trade aggressor side."""
    BUYER = "buyer"      # Bought at ask
    SELLER = "seller"    # Sold at bid
    UNKNOWN = "unknown"  # Mid-price


@dataclass
class OptionTrade:
    """Single options trade record."""
    symbol: str
    underlying: str
    strike: float
    expiration: datetime
    option_type: str  # 'call' or 'put'
    
    price: float
    size: int  # Number of contracts
    premium: float  # Total premium (price * size * 100)
    
    bid: float
    ask: float
    underlying_price: float
    
    timestamp: datetime
    exchange: str
    trade_id: str
    
    # Computed fields
    aggressor: TradeAggressor = TradeAggressor.UNKNOWN
    is_sweep: bool = False
    is_block: bool = False
    
    def __post_init__(self):
        """Compute aggressor and trade type."""
        # Determine aggressor
        mid = (self.bid + self.ask) / 2
        spread = self.ask - self.bid
        
        if spread > 0:
            if self.price >= self.ask - 0.01:
                self.aggressor = TradeAggressor.BUYER
            elif self.price <= self.bid + 0.01:
                self.aggressor = TradeAggressor.SELLER
        
        # Block trade threshold (100+ contracts or $100k+ premium)
        self.is_block = self.size >= 100 or self.premium >= 100000
    
    @property
    def moneyness(self) -> str:
        """Determine if option is ITM/ATM/OTM."""
        if self.option_type.lower() == 'call':
            if self.underlying_price > self.strike * 1.02:
                return "ITM"
            elif self.underlying_price < self.strike * 0.98:
                return "OTM"
        else:
            if self.underlying_price < self.strike * 0.98:
                return "ITM"
            elif self.underlying_price > self.strike * 1.02:
                return "OTM"
        return "ATM"
    
    @property
    def days_to_expiry(self) -> int:
        """Days until expiration."""
        return max(0, (self.expiration - datetime.now()).days)


@dataclass
class FlowAlert:
    """Options flow alert."""
    alert_id: str
    flow_type: FlowType
    sentiment: FlowSentiment
    
    symbol: str
    underlying: str
    
    # Trade details
    total_premium: float
    total_contracts: int
    avg_price: float
    
    strike: float
    expiration: datetime
    option_type: str
    
    # Context
    underlying_price: float
    volume_ratio: float  # Current vs average volume
    oi_ratio: float      # Volume vs open interest
    
    # Timing
    timestamp: datetime
    trades: List[OptionTrade] = field(default_factory=list)
    
    # Scoring
    score: float = 0.0  # Alert importance score (0-100)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "flow_type": self.flow_type.value,
            "sentiment": self.sentiment.value,
            "symbol": self.symbol,
            "underlying": self.underlying,
            "total_premium": self.total_premium,
            "total_contracts": self.total_contracts,
            "avg_price": self.avg_price,
            "strike": self.strike,
            "expiration": self.expiration.isoformat(),
            "option_type": self.option_type,
            "underlying_price": self.underlying_price,
            "volume_ratio": self.volume_ratio,
            "oi_ratio": self.oi_ratio,
            "timestamp": self.timestamp.isoformat(),
            "score": self.score,
            "trade_count": len(self.trades),
        }


@dataclass
class FlowScannerConfig:
    """Options flow scanner configuration."""
    # Block trade thresholds
    block_size_threshold: int = 100  # Contracts
    block_premium_threshold: float = 100000  # Dollars
    
    # Sweep detection
    sweep_time_window: int = 60  # Seconds
    sweep_min_exchanges: int = 2
    sweep_min_contracts: int = 50
    
    # Volume spike detection
    volume_lookback_days: int = 20
    volume_spike_multiplier: float = 3.0
    
    # OI analysis
    oi_volume_ratio_threshold: float = 0.5  # Volume > 50% of OI
    
    # Alert filtering
    min_premium_alert: float = 25000  # Min premium for alerts
    min_score_alert: float = 50.0  # Min score for alerts
    
    # Sector focus (optional)
    focus_symbols: List[str] = field(default_factory=list)
    exclude_symbols: List[str] = field(default_factory=list)


class OptionsFlowScanner:
    """
    Scans options market for unusual flow patterns.
    
    Detects:
    - Block trades (large single orders)
    - Sweep orders (hitting multiple exchanges)
    - Unusual volume spikes
    - Opening vs closing activity
    - Smart money indicators
    """
    
    def __init__(self, config: Optional[FlowScannerConfig] = None):
        """Initialize flow scanner."""
        self.config = config or FlowScannerConfig()
        
        # Trade tracking
        self._trades: Dict[str, List[OptionTrade]] = defaultdict(list)
        self._alerts: List[FlowAlert] = []
        
        # Historical data for comparison
        self._volume_history: Dict[str, List[int]] = defaultdict(list)
        self._oi_data: Dict[str, int] = {}
        
        # Alert ID counter
        self._alert_counter = 0
        
        # Callbacks
        self._alert_callbacks: List[callable] = []
        
        logger.info("OptionsFlowScanner initialized")
    
    def register_alert_callback(self, callback: callable) -> None:
        """Register callback for new alerts."""
        self._alert_callbacks.append(callback)
    
    async def process_trade(self, trade: OptionTrade) -> Optional[FlowAlert]:
        """
        Process incoming options trade.
        
        Returns alert if trade triggers one.
        """
        # Store trade
        key = f"{trade.underlying}_{trade.strike}_{trade.expiration.date()}_{trade.option_type}"
        self._trades[key].append(trade)
        
        # Check for various alert conditions
        alert = None
        
        # 1. Block trade detection
        if trade.is_block:
            alert = self._create_block_alert(trade)
        
        # 2. Sweep detection
        if not alert:
            alert = self._detect_sweep(trade, key)
        
        # 3. Volume spike detection
        if not alert:
            alert = self._detect_volume_spike(trade, key)
        
        # Process and notify
        if alert and alert.score >= self.config.min_score_alert:
            self._alerts.append(alert)
            await self._notify_alert(alert)
            return alert
        
        return None
    
    def _create_block_alert(self, trade: OptionTrade) -> FlowAlert:
        """Create alert for block trade."""
        self._alert_counter += 1
        
        # Determine sentiment
        if trade.option_type.lower() == 'call':
            sentiment = FlowSentiment.BULLISH if trade.aggressor == TradeAggressor.BUYER else FlowSentiment.BEARISH
        else:
            sentiment = FlowSentiment.BEARISH if trade.aggressor == TradeAggressor.BUYER else FlowSentiment.BULLISH
        
        # Calculate score
        score = self._calculate_alert_score(
            premium=trade.premium,
            contracts=trade.size,
            flow_type=FlowType.BLOCK,
            aggressor=trade.aggressor,
            moneyness=trade.moneyness,
            days_to_expiry=trade.days_to_expiry,
        )
        
        return FlowAlert(
            alert_id=f"BLOCK_{self._alert_counter}",
            flow_type=FlowType.BLOCK,
            sentiment=sentiment,
            symbol=trade.symbol,
            underlying=trade.underlying,
            total_premium=trade.premium,
            total_contracts=trade.size,
            avg_price=trade.price,
            strike=trade.strike,
            expiration=trade.expiration,
            option_type=trade.option_type,
            underlying_price=trade.underlying_price,
            volume_ratio=1.0,
            oi_ratio=self._get_oi_ratio(trade),
            timestamp=trade.timestamp,
            trades=[trade],
            score=score,
        )
    
    def _detect_sweep(self, trade: OptionTrade, key: str) -> Optional[FlowAlert]:
        """Detect sweep orders across exchanges."""
        recent_trades = self._trades[key]
        
        # Look for trades in time window
        window_start = trade.timestamp - timedelta(seconds=self.config.sweep_time_window)
        window_trades = [t for t in recent_trades if t.timestamp >= window_start]
        
        if len(window_trades) < 2:
            return None
        
        # Check exchange diversity
        exchanges = set(t.exchange for t in window_trades)
        if len(exchanges) < self.config.sweep_min_exchanges:
            return None
        
        # Check total size
        total_contracts = sum(t.size for t in window_trades)
        if total_contracts < self.config.sweep_min_contracts:
            return None
        
        # Check aggressor consistency (sweeps typically same direction)
        aggressors = [t.aggressor for t in window_trades if t.aggressor != TradeAggressor.UNKNOWN]
        if not aggressors:
            return None
        
        buyer_count = sum(1 for a in aggressors if a == TradeAggressor.BUYER)
        dominant_aggressor = TradeAggressor.BUYER if buyer_count > len(aggressors) / 2 else TradeAggressor.SELLER
        
        # Create sweep alert
        self._alert_counter += 1
        total_premium = sum(t.premium for t in window_trades)
        avg_price = sum(t.price * t.size for t in window_trades) / total_contracts
        
        # Sentiment from sweep direction
        if trade.option_type.lower() == 'call':
            sentiment = FlowSentiment.BULLISH if dominant_aggressor == TradeAggressor.BUYER else FlowSentiment.BEARISH
        else:
            sentiment = FlowSentiment.BEARISH if dominant_aggressor == TradeAggressor.BUYER else FlowSentiment.BULLISH
        
        score = self._calculate_alert_score(
            premium=total_premium,
            contracts=total_contracts,
            flow_type=FlowType.SWEEP,
            aggressor=dominant_aggressor,
            moneyness=trade.moneyness,
            days_to_expiry=trade.days_to_expiry,
        )
        
        return FlowAlert(
            alert_id=f"SWEEP_{self._alert_counter}",
            flow_type=FlowType.SWEEP,
            sentiment=sentiment,
            symbol=trade.symbol,
            underlying=trade.underlying,
            total_premium=total_premium,
            total_contracts=total_contracts,
            avg_price=avg_price,
            strike=trade.strike,
            expiration=trade.expiration,
            option_type=trade.option_type,
            underlying_price=trade.underlying_price,
            volume_ratio=1.0,
            oi_ratio=self._get_oi_ratio(trade),
            timestamp=trade.timestamp,
            trades=window_trades,
            score=score,
        )
    
    def _detect_volume_spike(self, trade: OptionTrade, key: str) -> Optional[FlowAlert]:
        """Detect unusual volume spikes."""
        # Get historical volume
        history = self._volume_history.get(key, [])
        if len(history) < 5:  # Need history
            return None
        
        avg_volume = statistics.mean(history)
        if avg_volume == 0:
            return None
        
        # Current session volume
        today_trades = [t for t in self._trades[key] 
                       if t.timestamp.date() == trade.timestamp.date()]
        current_volume = sum(t.size for t in today_trades)
        
        # Check spike
        volume_ratio = current_volume / avg_volume
        if volume_ratio < self.config.volume_spike_multiplier:
            return None
        
        # Create volume spike alert
        self._alert_counter += 1
        total_premium = sum(t.premium for t in today_trades)
        
        if total_premium < self.config.min_premium_alert:
            return None
        
        # Analyze direction
        buyer_premium = sum(t.premium for t in today_trades if t.aggressor == TradeAggressor.BUYER)
        seller_premium = sum(t.premium for t in today_trades if t.aggressor == TradeAggressor.SELLER)
        
        if buyer_premium > seller_premium * 1.5:
            dominant = TradeAggressor.BUYER
        elif seller_premium > buyer_premium * 1.5:
            dominant = TradeAggressor.SELLER
        else:
            dominant = TradeAggressor.UNKNOWN
        
        # Sentiment
        if dominant == TradeAggressor.UNKNOWN:
            sentiment = FlowSentiment.MIXED
        elif trade.option_type.lower() == 'call':
            sentiment = FlowSentiment.BULLISH if dominant == TradeAggressor.BUYER else FlowSentiment.BEARISH
        else:
            sentiment = FlowSentiment.BEARISH if dominant == TradeAggressor.BUYER else FlowSentiment.BULLISH
        
        score = self._calculate_alert_score(
            premium=total_premium,
            contracts=current_volume,
            flow_type=FlowType.UNUSUAL_VOLUME,
            aggressor=dominant,
            moneyness=trade.moneyness,
            days_to_expiry=trade.days_to_expiry,
            volume_ratio=volume_ratio,
        )
        
        return FlowAlert(
            alert_id=f"VOL_{self._alert_counter}",
            flow_type=FlowType.UNUSUAL_VOLUME,
            sentiment=sentiment,
            symbol=trade.symbol,
            underlying=trade.underlying,
            total_premium=total_premium,
            total_contracts=current_volume,
            avg_price=total_premium / current_volume / 100 if current_volume > 0 else 0,
            strike=trade.strike,
            expiration=trade.expiration,
            option_type=trade.option_type,
            underlying_price=trade.underlying_price,
            volume_ratio=volume_ratio,
            oi_ratio=self._get_oi_ratio(trade),
            timestamp=trade.timestamp,
            trades=today_trades,
            score=score,
        )
    
    def _calculate_alert_score(
        self,
        premium: float,
        contracts: int,
        flow_type: FlowType,
        aggressor: TradeAggressor,
        moneyness: str,
        days_to_expiry: int,
        volume_ratio: float = 1.0,
    ) -> float:
        """
        Calculate alert importance score (0-100).
        
        Factors:
        - Premium size (larger = more significant)
        - Contract count
        - Flow type (sweeps more significant than blocks)
        - Aggressor clarity (clear direction = higher score)
        - Moneyness (OTM = higher risk/reward)
        - DTE (shorter = more urgent)
        - Volume ratio
        """
        score = 0.0
        
        # Premium contribution (0-30 points)
        if premium >= 1000000:
            score += 30
        elif premium >= 500000:
            score += 25
        elif premium >= 250000:
            score += 20
        elif premium >= 100000:
            score += 15
        elif premium >= 50000:
            score += 10
        elif premium >= 25000:
            score += 5
        
        # Contract contribution (0-15 points)
        if contracts >= 1000:
            score += 15
        elif contracts >= 500:
            score += 12
        elif contracts >= 200:
            score += 9
        elif contracts >= 100:
            score += 6
        elif contracts >= 50:
            score += 3
        
        # Flow type (0-15 points)
        flow_scores = {
            FlowType.SWEEP: 15,
            FlowType.BLOCK: 12,
            FlowType.UNUSUAL_VOLUME: 10,
            FlowType.DARK_POOL: 13,
            FlowType.OPENING: 8,
            FlowType.CLOSING: 5,
            FlowType.SPLIT: 7,
        }
        score += flow_scores.get(flow_type, 5)
        
        # Aggressor clarity (0-10 points)
        if aggressor in (TradeAggressor.BUYER, TradeAggressor.SELLER):
            score += 10
        else:
            score += 3
        
        # Moneyness (0-10 points)
        moneyness_scores = {"OTM": 10, "ATM": 7, "ITM": 4}
        score += moneyness_scores.get(moneyness, 5)
        
        # DTE urgency (0-10 points)
        if days_to_expiry <= 7:
            score += 10
        elif days_to_expiry <= 14:
            score += 8
        elif days_to_expiry <= 30:
            score += 6
        elif days_to_expiry <= 60:
            score += 4
        else:
            score += 2
        
        # Volume ratio bonus (0-10 points)
        if volume_ratio >= 10:
            score += 10
        elif volume_ratio >= 5:
            score += 7
        elif volume_ratio >= 3:
            score += 4
        
        return min(100.0, score)
    
    def _get_oi_ratio(self, trade: OptionTrade) -> float:
        """Get volume/OI ratio for trade."""
        key = f"{trade.symbol}"
        oi = self._oi_data.get(key, 0)
        if oi == 0:
            return 0.0
        return trade.size / oi
    
    async def _notify_alert(self, alert: FlowAlert) -> None:
        """Notify registered callbacks of new alert."""
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def update_volume_history(self, symbol: str, strike: float, 
                              expiration: datetime, option_type: str,
                              daily_volume: int) -> None:
        """Update historical volume data."""
        key = f"{symbol}_{strike}_{expiration.date()}_{option_type}"
        history = self._volume_history[key]
        history.append(daily_volume)
        
        # Keep only recent history
        if len(history) > self.config.volume_lookback_days:
            self._volume_history[key] = history[-self.config.volume_lookback_days:]
    
    def update_open_interest(self, symbol: str, open_interest: int) -> None:
        """Update open interest data."""
        self._oi_data[symbol] = open_interest
    
    def get_alerts(
        self,
        min_score: float = 0.0,
        flow_types: Optional[List[FlowType]] = None,
        sentiment: Optional[FlowSentiment] = None,
        underlying: Optional[str] = None,
        limit: int = 100,
    ) -> List[FlowAlert]:
        """
        Get filtered alerts.
        
        Args:
            min_score: Minimum alert score
            flow_types: Filter by flow types
            sentiment: Filter by sentiment
            underlying: Filter by underlying symbol
            limit: Maximum alerts to return
        """
        alerts = self._alerts.copy()
        
        # Apply filters
        if min_score > 0:
            alerts = [a for a in alerts if a.score >= min_score]
        
        if flow_types:
            alerts = [a for a in alerts if a.flow_type in flow_types]
        
        if sentiment:
            alerts = [a for a in alerts if a.sentiment == sentiment]
        
        if underlying:
            alerts = [a for a in alerts if a.underlying.upper() == underlying.upper()]
        
        # Sort by score and timestamp
        alerts.sort(key=lambda a: (a.score, a.timestamp), reverse=True)
        
        return alerts[:limit]
    
    def get_flow_summary(self, underlying: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of recent flow activity.
        
        Returns aggregated statistics.
        """
        alerts = self.get_alerts(underlying=underlying)
        
        if not alerts:
            return {
                "total_alerts": 0,
                "total_premium": 0,
                "sentiment_breakdown": {},
                "top_symbols": [],
            }
        
        # Aggregate
        total_premium = sum(a.total_premium for a in alerts)
        
        sentiment_counts = defaultdict(int)
        sentiment_premium = defaultdict(float)
        symbol_premium = defaultdict(float)
        
        for alert in alerts:
            sentiment_counts[alert.sentiment.value] += 1
            sentiment_premium[alert.sentiment.value] += alert.total_premium
            symbol_premium[alert.underlying] += alert.total_premium
        
        # Top symbols
        top_symbols = sorted(
            symbol_premium.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "total_alerts": len(alerts),
            "total_premium": total_premium,
            "sentiment_breakdown": {
                s: {"count": sentiment_counts[s], "premium": sentiment_premium[s]}
                for s in sentiment_counts
            },
            "top_symbols": [
                {"symbol": s, "premium": p} for s, p in top_symbols
            ],
            "avg_score": sum(a.score for a in alerts) / len(alerts),
        }
    
    def clear_old_data(self, hours: int = 24) -> int:
        """Clear data older than specified hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Clear old trades
        cleared = 0
        for key in list(self._trades.keys()):
            original_len = len(self._trades[key])
            self._trades[key] = [t for t in self._trades[key] if t.timestamp >= cutoff]
            cleared += original_len - len(self._trades[key])
            
            if not self._trades[key]:
                del self._trades[key]
        
        # Clear old alerts
        original_alerts = len(self._alerts)
        self._alerts = [a for a in self._alerts if a.timestamp >= cutoff]
        cleared += original_alerts - len(self._alerts)
        
        logger.info(f"Cleared {cleared} old flow records")
        return cleared


# Singleton instance
flow_scanner = OptionsFlowScanner()


# Convenience functions
async def process_flow_trade(trade: OptionTrade) -> Optional[FlowAlert]:
    """Process trade through global scanner."""
    return await flow_scanner.process_trade(trade)


def get_flow_alerts(**kwargs) -> List[FlowAlert]:
    """Get alerts from global scanner."""
    return flow_scanner.get_alerts(**kwargs)


def get_flow_summary(underlying: Optional[str] = None) -> Dict[str, Any]:
    """Get flow summary from global scanner."""
    return flow_scanner.get_flow_summary(underlying)
