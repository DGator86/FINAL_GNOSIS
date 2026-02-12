"""
Earnings Calendar Integration

Tracks corporate earnings events and integrates with trading strategy:
- Earnings date tracking
- Pre/post earnings volatility analysis
- Historical earnings surprise data
- IV crush prediction
- Position management around earnings

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import json

from loguru import logger


class EarningsTime(str, Enum):
    """When earnings are released."""
    BMO = "bmo"      # Before Market Open
    AMC = "amc"      # After Market Close
    DMH = "dmh"      # During Market Hours
    UNKNOWN = "unknown"


class EarningsSurprise(str, Enum):
    """Earnings result vs expectations."""
    BEAT = "beat"
    MISS = "miss"
    INLINE = "inline"
    UNKNOWN = "unknown"


@dataclass
class EarningsEvent:
    """Single earnings event."""
    symbol: str
    company_name: str
    earnings_date: date
    earnings_time: EarningsTime
    
    # Estimates
    eps_estimate: Optional[float] = None
    revenue_estimate: Optional[float] = None  # Millions
    
    # Actuals (filled after release)
    eps_actual: Optional[float] = None
    revenue_actual: Optional[float] = None
    
    # Historical context
    avg_move_percent: Optional[float] = None  # Average post-earnings move
    implied_move_percent: Optional[float] = None  # Options-implied move
    
    # IV data
    iv_before: Optional[float] = None
    iv_after: Optional[float] = None
    
    # Metadata
    fiscal_quarter: Optional[str] = None
    fiscal_year: Optional[int] = None
    confirmed: bool = False
    
    @property
    def eps_surprise(self) -> Optional[EarningsSurprise]:
        """Calculate EPS surprise."""
        if self.eps_actual is None or self.eps_estimate is None:
            return None
        
        diff = self.eps_actual - self.eps_estimate
        threshold = abs(self.eps_estimate) * 0.02 if self.eps_estimate != 0 else 0.01
        
        if diff > threshold:
            return EarningsSurprise.BEAT
        elif diff < -threshold:
            return EarningsSurprise.MISS
        return EarningsSurprise.INLINE
    
    @property
    def eps_surprise_percent(self) -> Optional[float]:
        """Calculate EPS surprise percentage."""
        if self.eps_actual is None or self.eps_estimate is None:
            return None
        if self.eps_estimate == 0:
            return None
        return ((self.eps_actual - self.eps_estimate) / abs(self.eps_estimate)) * 100
    
    @property
    def revenue_surprise_percent(self) -> Optional[float]:
        """Calculate revenue surprise percentage."""
        if self.revenue_actual is None or self.revenue_estimate is None:
            return None
        if self.revenue_estimate == 0:
            return None
        return ((self.revenue_actual - self.revenue_estimate) / self.revenue_estimate) * 100
    
    @property
    def iv_crush_percent(self) -> Optional[float]:
        """Calculate IV crush percentage."""
        if self.iv_before is None or self.iv_after is None:
            return None
        if self.iv_before == 0:
            return None
        return ((self.iv_after - self.iv_before) / self.iv_before) * 100
    
    @property
    def days_until(self) -> int:
        """Days until earnings."""
        return (self.earnings_date - date.today()).days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "company_name": self.company_name,
            "earnings_date": self.earnings_date.isoformat(),
            "earnings_time": self.earnings_time.value,
            "eps_estimate": self.eps_estimate,
            "eps_actual": self.eps_actual,
            "eps_surprise": self.eps_surprise.value if self.eps_surprise else None,
            "eps_surprise_percent": self.eps_surprise_percent,
            "revenue_estimate": self.revenue_estimate,
            "revenue_actual": self.revenue_actual,
            "revenue_surprise_percent": self.revenue_surprise_percent,
            "avg_move_percent": self.avg_move_percent,
            "implied_move_percent": self.implied_move_percent,
            "iv_crush_percent": self.iv_crush_percent,
            "days_until": self.days_until,
            "fiscal_quarter": self.fiscal_quarter,
            "confirmed": self.confirmed,
        }


@dataclass
class EarningsHistory:
    """Historical earnings data for a symbol."""
    symbol: str
    events: List[EarningsEvent] = field(default_factory=list)
    
    @property
    def avg_eps_surprise(self) -> Optional[float]:
        """Average EPS surprise percentage."""
        surprises = [e.eps_surprise_percent for e in self.events 
                    if e.eps_surprise_percent is not None]
        return sum(surprises) / len(surprises) if surprises else None
    
    @property
    def beat_rate(self) -> float:
        """Percentage of earnings beats."""
        results = [e.eps_surprise for e in self.events 
                  if e.eps_surprise is not None]
        if not results:
            return 0.0
        beats = sum(1 for r in results if r == EarningsSurprise.BEAT)
        return beats / len(results) * 100
    
    @property
    def avg_post_earnings_move(self) -> Optional[float]:
        """Average absolute post-earnings move."""
        moves = [e.avg_move_percent for e in self.events 
                if e.avg_move_percent is not None]
        return sum(abs(m) for m in moves) / len(moves) if moves else None
    
    @property
    def avg_iv_crush(self) -> Optional[float]:
        """Average IV crush."""
        crushes = [e.iv_crush_percent for e in self.events 
                  if e.iv_crush_percent is not None]
        return sum(crushes) / len(crushes) if crushes else None


@dataclass
class EarningsCalendarConfig:
    """Earnings calendar configuration."""
    # Lookback/lookahead
    days_lookback: int = 90
    days_lookahead: int = 45
    
    # Alert thresholds
    high_iv_threshold: float = 80.0  # High IV percentile
    large_move_threshold: float = 10.0  # Large expected move %
    
    # Position management
    close_positions_before_earnings: bool = False
    days_before_close: int = 1
    
    # Data sources (for future API integration)
    data_sources: List[str] = field(default_factory=lambda: ["yahoo", "nasdaq"])


class EarningsCalendar:
    """
    Manages earnings calendar data and analysis.
    
    Features:
    - Track upcoming earnings
    - Historical earnings analysis
    - IV crush prediction
    - Position alerts around earnings
    """
    
    def __init__(self, config: Optional[EarningsCalendarConfig] = None):
        """Initialize earnings calendar."""
        self.config = config or EarningsCalendarConfig()
        
        # Data storage
        self._upcoming: Dict[str, EarningsEvent] = {}
        self._history: Dict[str, EarningsHistory] = {}
        
        # Watchlist
        self._watchlist: set = set()
        
        # Callbacks
        self._alert_callbacks: List[callable] = []
        
        logger.info("EarningsCalendar initialized")
    
    def add_earnings_event(self, event: EarningsEvent) -> None:
        """Add or update earnings event."""
        key = f"{event.symbol}_{event.earnings_date}"
        self._upcoming[key] = event
        
        # Add to history if past
        if event.earnings_date < date.today():
            self._add_to_history(event)
        
        logger.debug(f"Added earnings event: {event.symbol} on {event.earnings_date}")
    
    def _add_to_history(self, event: EarningsEvent) -> None:
        """Add event to historical data."""
        if event.symbol not in self._history:
            self._history[event.symbol] = EarningsHistory(symbol=event.symbol)
        
        # Avoid duplicates
        existing_dates = {e.earnings_date for e in self._history[event.symbol].events}
        if event.earnings_date not in existing_dates:
            self._history[event.symbol].events.append(event)
            self._history[event.symbol].events.sort(key=lambda e: e.earnings_date, reverse=True)
    
    def get_upcoming_earnings(
        self,
        days_ahead: Optional[int] = None,
        symbols: Optional[List[str]] = None,
    ) -> List[EarningsEvent]:
        """
        Get upcoming earnings events.
        
        Args:
            days_ahead: Number of days to look ahead
            symbols: Filter to specific symbols
        """
        days = days_ahead or self.config.days_lookahead
        cutoff = date.today() + timedelta(days=days)
        
        events = []
        for event in self._upcoming.values():
            if event.earnings_date < date.today():
                continue
            if event.earnings_date > cutoff:
                continue
            if symbols and event.symbol not in symbols:
                continue
            events.append(event)
        
        return sorted(events, key=lambda e: e.earnings_date)
    
    def get_earnings_this_week(self) -> List[EarningsEvent]:
        """Get earnings for current week."""
        today = date.today()
        # Monday of this week
        monday = today - timedelta(days=today.weekday())
        friday = monday + timedelta(days=4)
        
        events = []
        for event in self._upcoming.values():
            if monday <= event.earnings_date <= friday:
                events.append(event)
        
        return sorted(events, key=lambda e: (e.earnings_date, e.symbol))
    
    def get_earnings_by_date(self, target_date: date) -> List[EarningsEvent]:
        """Get all earnings for a specific date."""
        return [e for e in self._upcoming.values() if e.earnings_date == target_date]
    
    def get_earnings_event(self, symbol: str) -> Optional[EarningsEvent]:
        """Get next upcoming earnings for symbol."""
        events = [e for e in self._upcoming.values() 
                 if e.symbol.upper() == symbol.upper() and e.earnings_date >= date.today()]
        
        if not events:
            return None
        
        return min(events, key=lambda e: e.earnings_date)
    
    def get_history(self, symbol: str) -> Optional[EarningsHistory]:
        """Get historical earnings data."""
        return self._history.get(symbol.upper())
    
    def add_to_watchlist(self, symbol: str) -> None:
        """Add symbol to earnings watchlist."""
        self._watchlist.add(symbol.upper())
    
    def remove_from_watchlist(self, symbol: str) -> None:
        """Remove symbol from watchlist."""
        self._watchlist.discard(symbol.upper())
    
    def get_watchlist_earnings(self) -> List[EarningsEvent]:
        """Get upcoming earnings for watchlist symbols."""
        return self.get_upcoming_earnings(symbols=list(self._watchlist))
    
    def calculate_expected_move(
        self,
        symbol: str,
        atm_iv: float,
        days_to_earnings: int,
        stock_price: float,
    ) -> Dict[str, float]:
        """
        Calculate expected move around earnings.
        
        Uses ATM IV to estimate expected move range.
        """
        # Standard expected move formula
        # Expected Move = Stock Price × IV × sqrt(DTE/365)
        if days_to_earnings <= 0:
            days_to_earnings = 1
        
        expected_move_pct = atm_iv * (days_to_earnings / 365) ** 0.5
        expected_move_dollars = stock_price * expected_move_pct
        
        # Get historical context
        history = self.get_history(symbol)
        historical_avg = history.avg_post_earnings_move if history else None
        
        return {
            "expected_move_percent": expected_move_pct * 100,
            "expected_move_dollars": expected_move_dollars,
            "upper_bound": stock_price + expected_move_dollars,
            "lower_bound": stock_price - expected_move_dollars,
            "historical_avg_move": historical_avg,
            "iv_used": atm_iv,
        }
    
    def predict_iv_crush(
        self,
        symbol: str,
        current_iv: float,
    ) -> Dict[str, Any]:
        """
        Predict post-earnings IV crush.
        
        Uses historical data to estimate IV behavior.
        """
        history = self.get_history(symbol)
        
        if not history or not history.avg_iv_crush:
            # Use market average
            estimated_crush = -35.0  # Typical 30-40% IV crush
        else:
            estimated_crush = history.avg_iv_crush
        
        post_iv = current_iv * (1 + estimated_crush / 100)
        
        return {
            "current_iv": current_iv,
            "estimated_post_iv": max(0.1, post_iv),  # IV can't go below 10%
            "estimated_crush_percent": estimated_crush,
            "historical_avg_crush": history.avg_iv_crush if history else None,
            "confidence": "high" if history and len(history.events) >= 4 else "low",
        }
    
    def get_earnings_trade_suggestions(
        self,
        symbol: str,
        current_price: float,
        current_iv: float,
        risk_tolerance: str = "moderate",
    ) -> List[Dict[str, Any]]:
        """
        Generate trade suggestions for earnings.
        
        Strategies based on IV and historical patterns.
        """
        event = self.get_earnings_event(symbol)
        if not event:
            return []
        
        history = self.get_history(symbol)
        expected_move = self.calculate_expected_move(
            symbol, current_iv, event.days_until, current_price
        )
        iv_prediction = self.predict_iv_crush(symbol, current_iv)
        
        suggestions = []
        
        # High IV - sell premium strategies
        if current_iv > 0.5:  # IV > 50%
            # Iron Condor
            suggestions.append({
                "strategy": "iron_condor",
                "description": "Sell iron condor to capture IV crush",
                "rationale": f"High IV ({current_iv*100:.1f}%) likely to crush post-earnings",
                "strikes": {
                    "short_put": current_price - expected_move["expected_move_dollars"],
                    "long_put": current_price - expected_move["expected_move_dollars"] * 1.5,
                    "short_call": current_price + expected_move["expected_move_dollars"],
                    "long_call": current_price + expected_move["expected_move_dollars"] * 1.5,
                },
                "risk_level": "moderate",
                "max_profit": "premium collected",
                "max_loss": "width of spread - premium",
            })
            
            # Straddle sell (high risk)
            if risk_tolerance == "aggressive":
                suggestions.append({
                    "strategy": "short_straddle",
                    "description": "Sell ATM straddle for maximum premium",
                    "rationale": "Capture full IV crush, unlimited risk",
                    "strikes": {"atm": current_price},
                    "risk_level": "high",
                    "max_profit": "premium collected",
                    "max_loss": "unlimited",
                })
        
        # Directional plays based on history
        if history and history.beat_rate > 70:
            suggestions.append({
                "strategy": "bull_call_spread",
                "description": "Bullish spread for likely beat",
                "rationale": f"Historical beat rate: {history.beat_rate:.0f}%",
                "strikes": {
                    "long_call": current_price,
                    "short_call": current_price + expected_move["expected_move_dollars"],
                },
                "risk_level": "moderate",
            })
        elif history and history.beat_rate < 30:
            suggestions.append({
                "strategy": "bear_put_spread",
                "description": "Bearish spread for likely miss",
                "rationale": f"Historical beat rate: {history.beat_rate:.0f}%",
                "strikes": {
                    "long_put": current_price,
                    "short_put": current_price - expected_move["expected_move_dollars"],
                },
                "risk_level": "moderate",
            })
        
        # Volatility play
        suggestions.append({
            "strategy": "calendar_spread",
            "description": "Calendar spread to capture term structure",
            "rationale": "Sell front-month high IV, buy back-month",
            "strikes": {"atm": current_price},
            "expirations": {
                "front": "weekly containing earnings",
                "back": "monthly after earnings",
            },
            "risk_level": "moderate",
        })
        
        return suggestions
    
    def check_position_alerts(
        self,
        positions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Check positions for earnings-related alerts.
        
        Returns list of alerts for positions with upcoming earnings.
        """
        alerts = []
        
        for position in positions:
            symbol = position.get("symbol", "").split("_")[0]  # Handle option symbols
            event = self.get_earnings_event(symbol)
            
            if not event:
                continue
            
            days = event.days_until
            
            # Position before earnings
            if 0 <= days <= 5:
                alert_type = "imminent" if days <= 1 else "upcoming"
                
                alerts.append({
                    "alert_type": f"earnings_{alert_type}",
                    "symbol": symbol,
                    "position": position,
                    "earnings_date": event.earnings_date.isoformat(),
                    "earnings_time": event.earnings_time.value,
                    "days_until": days,
                    "message": f"{symbol} reports earnings in {days} day(s)",
                    "suggested_action": self._suggest_action(position, event, days),
                })
        
        return alerts
    
    def _suggest_action(
        self,
        position: Dict[str, Any],
        event: EarningsEvent,
        days: int,
    ) -> str:
        """Suggest action for position near earnings."""
        position_type = position.get("type", "stock")
        
        if position_type == "stock":
            return "Consider hedging with protective put or collar"
        
        # Options position
        if days <= 1:
            if self.config.close_positions_before_earnings:
                return "Close position before earnings announcement"
            return "Monitor closely - high IV crush risk"
        
        return "Review position sizing and risk exposure"
    
    def get_calendar_summary(self) -> Dict[str, Any]:
        """Get summary of earnings calendar."""
        upcoming = self.get_upcoming_earnings(days_ahead=7)
        this_week = self.get_earnings_this_week()
        watchlist = self.get_watchlist_earnings()
        
        # Group by date
        by_date = defaultdict(list)
        for event in upcoming:
            by_date[event.earnings_date.isoformat()].append(event.symbol)
        
        return {
            "total_upcoming_7days": len(upcoming),
            "this_week_count": len(this_week),
            "watchlist_count": len(watchlist),
            "by_date": dict(by_date),
            "watchlist_symbols": list(self._watchlist),
            "next_earnings": upcoming[0].to_dict() if upcoming else None,
        }
    
    async def fetch_earnings_data(self, symbol: str) -> Optional[EarningsEvent]:
        """
        Fetch earnings data from external source.
        
        Note: This is a placeholder - implement with actual API.
        """
        # Placeholder - would integrate with Yahoo Finance, Alpha Vantage, etc.
        logger.info(f"Would fetch earnings data for {symbol}")
        return None
    
    def export_calendar(self, format: str = "json") -> str:
        """Export calendar data."""
        data = {
            "upcoming": [e.to_dict() for e in self.get_upcoming_earnings()],
            "watchlist": list(self._watchlist),
            "exported_at": datetime.now().isoformat(),
        }
        
        if format == "json":
            return json.dumps(data, indent=2)
        
        # CSV format
        lines = ["symbol,date,time,eps_estimate,implied_move"]
        for event in self.get_upcoming_earnings():
            lines.append(
                f"{event.symbol},{event.earnings_date},{event.earnings_time.value},"
                f"{event.eps_estimate},{event.implied_move_percent}"
            )
        return "\n".join(lines)


# Singleton instance
earnings_calendar = EarningsCalendar()


# Convenience functions
def get_upcoming_earnings(**kwargs) -> List[EarningsEvent]:
    """Get upcoming earnings."""
    return earnings_calendar.get_upcoming_earnings(**kwargs)


def get_earnings_event(symbol: str) -> Optional[EarningsEvent]:
    """Get earnings event for symbol."""
    return earnings_calendar.get_earnings_event(symbol)


def add_to_watchlist(symbol: str) -> None:
    """Add to earnings watchlist."""
    earnings_calendar.add_to_watchlist(symbol)
