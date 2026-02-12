"""
Event Risk Manager - Institutional-Grade Event Calendar and Risk Adjustment

Manages event risk including earnings, dividends, economic releases, and corporate actions.

EVENT TYPES:
├── Earnings: Quarterly reports with IV crush/expansion
├── Dividends: Ex-dates affect options pricing
├── Economic: FOMC, CPI, NFP, GDP releases
├── Corporate: Splits, M&A, spinoffs
└── Market: VIX expiration, OpEx, rebalancing

RISK ADJUSTMENTS:
- Pre-earnings: Reduce position size, widen stops
- Binary events: Avoid or use defined-risk strategies
- High IV events: Prefer credit strategies
- Ex-dividend: Adjust call assignment risk

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class EventType(str, Enum):
    """Types of market events."""
    EARNINGS = "earnings"
    DIVIDEND = "dividend"
    FOMC = "fomc"
    CPI = "cpi"
    NFP = "nfp"
    GDP = "gdp"
    PPI = "ppi"
    RETAIL_SALES = "retail_sales"
    UNEMPLOYMENT = "unemployment"
    OPTIONS_EXPIRATION = "opex"
    VIX_EXPIRATION = "vix_exp"
    TRIPLE_WITCHING = "triple_witching"
    STOCK_SPLIT = "split"
    MERGER = "merger"
    SPINOFF = "spinoff"
    FDA_DECISION = "fda"
    PRODUCT_LAUNCH = "product_launch"
    CONFERENCE = "conference"
    OTHER = "other"


class EventImpact(str, Enum):
    """Expected impact level of event."""
    HIGH = "high"      # Major price move expected (>5%)
    MEDIUM = "medium"  # Moderate move expected (2-5%)
    LOW = "low"        # Minor impact (<2%)
    UNKNOWN = "unknown"


class RiskAction(str, Enum):
    """Risk management actions for events."""
    AVOID = "avoid"           # Don't trade
    REDUCE_SIZE = "reduce"    # Reduce position size
    WIDEN_STOPS = "widen"     # Widen stop losses
    DEFINED_RISK = "defined"  # Use defined-risk strategies only
    CLOSE_BEFORE = "close"    # Close positions before event
    NORMAL = "normal"         # No special action


@dataclass
class MarketEvent:
    """A market event with risk implications."""
    event_type: EventType
    date: date
    symbol: Optional[str] = None  # None for market-wide events
    time: Optional[str] = None    # "pre-market", "after-hours", "09:30", etc.
    description: str = ""
    impact: EventImpact = EventImpact.MEDIUM
    
    # Historical data
    historical_move_avg: float = 0.0  # Average historical move %
    historical_move_max: float = 0.0  # Max historical move %
    historical_iv_crush: float = 0.0  # Average IV crush %
    
    # Dividend specific
    dividend_amount: float = 0.0
    ex_dividend_date: Optional[date] = None
    
    # Earnings specific
    expected_eps: float = 0.0
    consensus_revenue: float = 0.0
    
    # Risk action
    recommended_action: RiskAction = RiskAction.NORMAL
    position_size_multiplier: float = 1.0
    stop_loss_multiplier: float = 1.0


@dataclass
class EventRiskAssessment:
    """Risk assessment for upcoming events."""
    symbol: str
    assessment_date: date = field(default_factory=date.today)
    
    # Events found
    events: List[MarketEvent] = field(default_factory=list)
    days_to_next_event: int = 999
    next_event: Optional[MarketEvent] = None
    
    # Risk multipliers
    position_size_multiplier: float = 1.0
    stop_loss_multiplier: float = 1.0
    
    # Recommendations
    risk_level: str = "normal"  # "normal", "elevated", "high", "extreme"
    recommended_action: RiskAction = RiskAction.NORMAL
    strategy_adjustments: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Event-specific adjustments
    avoid_calls_before_ex_div: bool = False
    prefer_credit_strategies: bool = False
    use_defined_risk_only: bool = False


class EventRiskManager:
    """
    Manages event-based risk for trading decisions.
    
    Features:
    - Earnings calendar integration
    - Economic event calendar
    - Dividend tracking
    - Risk adjustment recommendations
    - Strategy filtering based on events
    """
    
    # Known high-impact economic events (monthly schedule)
    ECONOMIC_CALENDAR_TEMPLATE = {
        "FOMC": [
            {"day": "wednesday", "week": 3, "months": [1, 3, 5, 6, 7, 9, 11, 12]},
        ],
        "CPI": [
            {"day": "tuesday", "week": 2, "months": list(range(1, 13))},
        ],
        "NFP": [
            {"day": "friday", "week": 1, "months": list(range(1, 13))},
        ],
        "PPI": [
            {"day": "thursday", "week": 2, "months": list(range(1, 13))},
        ],
    }
    
    # Options expiration schedule
    MONTHLY_OPEX_WEEK = 3  # Third Friday
    
    # Historical earnings move data (simplified)
    EARNINGS_MOVE_HISTORY: Dict[str, Dict[str, float]] = {
        "NVDA": {"avg_move": 8.5, "max_move": 24.0, "iv_crush": 15.0},
        "TSLA": {"avg_move": 7.0, "max_move": 21.0, "iv_crush": 12.0},
        "AAPL": {"avg_move": 3.5, "max_move": 8.0, "iv_crush": 8.0},
        "MSFT": {"avg_move": 3.0, "max_move": 6.5, "iv_crush": 6.0},
        "GOOGL": {"avg_move": 4.0, "max_move": 9.0, "iv_crush": 7.0},
        "META": {"avg_move": 8.0, "max_move": 26.0, "iv_crush": 14.0},
        "AMZN": {"avg_move": 5.0, "max_move": 14.0, "iv_crush": 10.0},
        "AMD": {"avg_move": 6.5, "max_move": 15.0, "iv_crush": 11.0},
        "COIN": {"avg_move": 12.0, "max_move": 30.0, "iv_crush": 18.0},
        "GME": {"avg_move": 15.0, "max_move": 50.0, "iv_crush": 20.0},
        "SPY": {"avg_move": 1.0, "max_move": 3.0, "iv_crush": 3.0},
        "QQQ": {"avg_move": 1.5, "max_move": 4.0, "iv_crush": 4.0},
        # Default for unknown symbols
        "_default": {"avg_move": 5.0, "max_move": 12.0, "iv_crush": 10.0},
    }
    
    def __init__(
        self,
        lookforward_days: int = 14,
        earnings_api_key: Optional[str] = None,
    ):
        """Initialize the Event Risk Manager.
        
        Args:
            lookforward_days: Days to look ahead for events
            earnings_api_key: API key for earnings data (optional)
        """
        self.lookforward_days = lookforward_days
        self.earnings_api_key = earnings_api_key or os.getenv("EARNINGS_API_KEY")
        
        # Cache for earnings dates
        self._earnings_cache: Dict[str, List[MarketEvent]] = {}
        self._dividend_cache: Dict[str, List[MarketEvent]] = {}
        self._economic_events: List[MarketEvent] = []
        
        # Build economic calendar for current quarter
        self._build_economic_calendar()
        
        logger.info(
            f"EventRiskManager initialized | lookforward={lookforward_days} days"
        )
    
    def _build_economic_calendar(self) -> None:
        """Build economic event calendar for next 3 months."""
        today = date.today()
        end_date = today + timedelta(days=90)
        
        events = []
        
        # Add FOMC meetings (approximate - 8 per year)
        fomc_2024_2025 = [
            # 2024
            date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1),
            date(2024, 6, 12), date(2024, 7, 31), date(2024, 9, 18),
            date(2024, 11, 7), date(2024, 12, 18),
            # 2025
            date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
            date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
            date(2025, 11, 5), date(2025, 12, 17),
        ]
        
        for fomc_date in fomc_2024_2025:
            if today <= fomc_date <= end_date:
                events.append(MarketEvent(
                    event_type=EventType.FOMC,
                    date=fomc_date,
                    time="14:00",
                    description="FOMC Interest Rate Decision",
                    impact=EventImpact.HIGH,
                    historical_move_avg=1.5,
                    recommended_action=RiskAction.REDUCE_SIZE,
                    position_size_multiplier=0.5,
                    stop_loss_multiplier=1.5,
                ))
        
        # Add monthly OpEx (third Friday)
        current = today.replace(day=1)
        while current <= end_date:
            # Find third Friday
            first_day = current.replace(day=1)
            # Days until first Friday
            days_to_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_to_friday)
            third_friday = first_friday + timedelta(weeks=2)
            
            if today <= third_friday <= end_date:
                # Check if it's triple witching (March, June, September, December)
                is_triple = current.month in [3, 6, 9, 12]
                
                events.append(MarketEvent(
                    event_type=EventType.TRIPLE_WITCHING if is_triple else EventType.OPTIONS_EXPIRATION,
                    date=third_friday,
                    time="16:00",
                    description="Triple Witching" if is_triple else "Monthly Options Expiration",
                    impact=EventImpact.HIGH if is_triple else EventImpact.MEDIUM,
                    historical_move_avg=1.0 if is_triple else 0.5,
                    recommended_action=RiskAction.WIDEN_STOPS,
                    position_size_multiplier=0.75 if is_triple else 0.9,
                    stop_loss_multiplier=1.25,
                ))
            
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        # Add CPI releases (typically second Tuesday)
        current = today.replace(day=1)
        while current <= end_date:
            # Find second Tuesday
            first_day = current.replace(day=1)
            days_to_tuesday = (1 - first_day.weekday()) % 7
            first_tuesday = first_day + timedelta(days=days_to_tuesday)
            second_tuesday = first_tuesday + timedelta(weeks=1)
            
            if today <= second_tuesday <= end_date:
                events.append(MarketEvent(
                    event_type=EventType.CPI,
                    date=second_tuesday,
                    time="08:30",
                    description="Consumer Price Index Release",
                    impact=EventImpact.HIGH,
                    historical_move_avg=1.2,
                    recommended_action=RiskAction.REDUCE_SIZE,
                    position_size_multiplier=0.6,
                    stop_loss_multiplier=1.3,
                ))
            
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        # Add NFP (first Friday)
        current = today.replace(day=1)
        while current <= end_date:
            # Find first Friday
            first_day = current.replace(day=1)
            days_to_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_to_friday)
            
            if today <= first_friday <= end_date:
                events.append(MarketEvent(
                    event_type=EventType.NFP,
                    date=first_friday,
                    time="08:30",
                    description="Non-Farm Payrolls Report",
                    impact=EventImpact.HIGH,
                    historical_move_avg=1.0,
                    recommended_action=RiskAction.REDUCE_SIZE,
                    position_size_multiplier=0.7,
                    stop_loss_multiplier=1.25,
                ))
            
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        self._economic_events = sorted(events, key=lambda e: e.date)
        logger.debug(f"Built economic calendar with {len(events)} events")
    
    def get_earnings_dates(self, symbol: str) -> List[MarketEvent]:
        """Get upcoming earnings dates for a symbol.
        
        Uses cache or fetches from API if available.
        """
        if symbol in self._earnings_cache:
            return self._earnings_cache[symbol]
        
        # Try to fetch from API
        events = self._fetch_earnings_from_api(symbol)
        
        if not events:
            # Use estimated quarterly schedule as fallback
            events = self._estimate_earnings_dates(symbol)
        
        self._earnings_cache[symbol] = events
        return events
    
    def _fetch_earnings_from_api(self, symbol: str) -> List[MarketEvent]:
        """Fetch earnings dates from external API."""
        if not HAS_REQUESTS or not self.earnings_api_key:
            return []
        
        # This is a placeholder - would integrate with actual earnings API
        # Such as Alpha Vantage, Finnhub, or Polygon.io
        return []
    
    def _estimate_earnings_dates(self, symbol: str) -> List[MarketEvent]:
        """Estimate earnings dates based on typical quarterly schedule."""
        today = date.today()
        events = []
        
        # Get historical move data
        hist = self.EARNINGS_MOVE_HISTORY.get(
            symbol, 
            self.EARNINGS_MOVE_HISTORY["_default"]
        )
        
        # Typical earnings months by quarter
        # Q1: Apr/May, Q2: Jul/Aug, Q3: Oct/Nov, Q4: Jan/Feb
        current_year = today.year
        potential_earnings = [
            date(current_year, 1, 25),   # Q4 earnings
            date(current_year, 4, 25),   # Q1 earnings
            date(current_year, 7, 25),   # Q2 earnings
            date(current_year, 10, 25),  # Q3 earnings
            date(current_year + 1, 1, 25),  # Next Q4
        ]
        
        for earn_date in potential_earnings:
            if today <= earn_date <= today + timedelta(days=self.lookforward_days):
                # Determine impact based on historical moves
                impact = EventImpact.HIGH if hist["avg_move"] > 5 else EventImpact.MEDIUM
                
                events.append(MarketEvent(
                    event_type=EventType.EARNINGS,
                    date=earn_date,
                    symbol=symbol,
                    time="after-hours",  # Most earnings are AH
                    description=f"{symbol} Quarterly Earnings",
                    impact=impact,
                    historical_move_avg=hist["avg_move"],
                    historical_move_max=hist["max_move"],
                    historical_iv_crush=hist["iv_crush"],
                    recommended_action=RiskAction.DEFINED_RISK if hist["avg_move"] > 7 else RiskAction.REDUCE_SIZE,
                    position_size_multiplier=0.3 if hist["avg_move"] > 7 else 0.5,
                    stop_loss_multiplier=2.0 if hist["avg_move"] > 7 else 1.5,
                ))
        
        return events
    
    def assess_event_risk(
        self,
        symbol: str,
        dte: int = 30,
        current_date: Optional[date] = None,
    ) -> EventRiskAssessment:
        """Assess event risk for a symbol within a given time horizon.
        
        Args:
            symbol: Stock symbol
            dte: Days to expiration for options
            current_date: Date to assess from (defaults to today)
            
        Returns:
            EventRiskAssessment with risk multipliers and recommendations
        """
        today = current_date or date.today()
        end_date = today + timedelta(days=max(dte, self.lookforward_days))
        
        assessment = EventRiskAssessment(
            symbol=symbol,
            assessment_date=today,
        )
        
        # Gather all relevant events
        all_events: List[MarketEvent] = []
        
        # 1. Symbol-specific earnings
        earnings_events = self.get_earnings_dates(symbol)
        for event in earnings_events:
            if today <= event.date <= end_date:
                all_events.append(event)
        
        # 2. Economic events (affect all symbols, especially indices)
        for event in self._economic_events:
            if today <= event.date <= end_date:
                all_events.append(event)
        
        # 3. Check for dividends (simplified - would use real API)
        dividend_events = self._get_dividend_events(symbol, today, end_date)
        all_events.extend(dividend_events)
        
        # Sort by date
        all_events.sort(key=lambda e: e.date)
        assessment.events = all_events
        
        if not all_events:
            assessment.risk_level = "normal"
            return assessment
        
        # Find next event
        assessment.next_event = all_events[0]
        assessment.days_to_next_event = (all_events[0].date - today).days
        
        # Calculate aggregate risk multipliers
        min_size_mult = 1.0
        max_stop_mult = 1.0
        
        high_impact_events = []
        warnings = []
        strategy_adjustments = []
        
        for event in all_events:
            # Track multipliers
            min_size_mult = min(min_size_mult, event.position_size_multiplier)
            max_stop_mult = max(max_stop_mult, event.stop_loss_multiplier)
            
            if event.impact == EventImpact.HIGH:
                high_impact_events.append(event)
            
            # Event-specific logic
            if event.event_type == EventType.EARNINGS:
                days_to_event = (event.date - today).days
                
                if days_to_event <= 3:
                    warnings.append(
                        f"⚠️ EARNINGS in {days_to_event} days - "
                        f"Expected move: {event.historical_move_avg:.1f}% "
                        f"(max: {event.historical_move_max:.1f}%)"
                    )
                    strategy_adjustments.append(
                        "Use DEFINED-RISK strategies only (spreads, iron condors)"
                    )
                    assessment.use_defined_risk_only = True
                    
                    if event.historical_move_avg > 7:
                        strategy_adjustments.append(
                            f"Consider IV crush play - avg {event.historical_iv_crush:.1f}% IV drop post-earnings"
                        )
                        assessment.prefer_credit_strategies = True
                
                elif days_to_event <= 7:
                    warnings.append(
                        f"Earnings approaching in {days_to_event} days"
                    )
                    strategy_adjustments.append(
                        "Reduce position size and use wider stops"
                    )
            
            elif event.event_type == EventType.DIVIDEND:
                days_to_ex = (event.date - today).days
                if days_to_ex <= 5 and event.dividend_amount > 0:
                    warnings.append(
                        f"Ex-dividend in {days_to_ex} days "
                        f"(${event.dividend_amount:.2f})"
                    )
                    assessment.avoid_calls_before_ex_div = True
                    strategy_adjustments.append(
                        "Avoid ITM calls due to early assignment risk"
                    )
            
            elif event.event_type == EventType.FOMC:
                days_to_event = (event.date - today).days
                if days_to_event <= 2:
                    warnings.append(
                        f"FOMC decision in {days_to_event} days"
                    )
                    strategy_adjustments.append(
                        "Reduce directional exposure - market may gap"
                    )
            
            elif event.event_type in [EventType.OPTIONS_EXPIRATION, EventType.TRIPLE_WITCHING]:
                days_to_event = (event.date - today).days
                if days_to_event <= 3:
                    warnings.append(
                        f"{'Triple Witching' if event.event_type == EventType.TRIPLE_WITCHING else 'OpEx'} "
                        f"in {days_to_event} days - expect increased volatility"
                    )
        
        assessment.position_size_multiplier = min_size_mult
        assessment.stop_loss_multiplier = max_stop_mult
        assessment.warnings = warnings
        assessment.strategy_adjustments = list(set(strategy_adjustments))
        
        # Determine overall risk level
        if len(high_impact_events) >= 2 or (
            high_impact_events and assessment.days_to_next_event <= 2
        ):
            assessment.risk_level = "extreme"
            assessment.recommended_action = RiskAction.AVOID
        elif high_impact_events and assessment.days_to_next_event <= 5:
            assessment.risk_level = "high"
            assessment.recommended_action = RiskAction.DEFINED_RISK
        elif assessment.days_to_next_event <= 7:
            assessment.risk_level = "elevated"
            assessment.recommended_action = RiskAction.REDUCE_SIZE
        else:
            assessment.risk_level = "normal"
            assessment.recommended_action = RiskAction.NORMAL
        
        logger.debug(
            f"Event risk for {symbol}: {assessment.risk_level} | "
            f"next_event={assessment.next_event.event_type.value if assessment.next_event else 'none'} "
            f"in {assessment.days_to_next_event}d | size_mult={assessment.position_size_multiplier:.2f}"
        )
        
        return assessment
    
    def _get_dividend_events(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> List[MarketEvent]:
        """Get dividend events for a symbol."""
        # This is a simplified implementation
        # In production, would fetch from API
        
        # Known high-dividend stocks (simplified quarterly schedule)
        dividend_stocks = {
            "AAPL": 0.25,
            "MSFT": 0.75,
            "JPM": 1.25,
            "JNJ": 1.24,
            "XOM": 0.99,
            "SPY": 1.80,
        }
        
        if symbol not in dividend_stocks:
            return []
        
        events = []
        amount = dividend_stocks[symbol]
        
        # Estimate quarterly ex-dates
        year = start.year
        for month in [2, 5, 8, 11]:  # Typical ex-date months
            ex_date = date(year, month, 10)
            if start <= ex_date <= end:
                events.append(MarketEvent(
                    event_type=EventType.DIVIDEND,
                    date=ex_date,
                    symbol=symbol,
                    description=f"{symbol} Ex-Dividend",
                    impact=EventImpact.LOW,
                    dividend_amount=amount,
                    ex_dividend_date=ex_date,
                    recommended_action=RiskAction.NORMAL,
                    position_size_multiplier=1.0,
                    stop_loss_multiplier=1.0,
                ))
        
        return events
    
    def filter_strategies_for_events(
        self,
        symbol: str,
        strategies: List[str],
        dte: int,
    ) -> Tuple[List[str], List[str]]:
        """Filter strategies based on upcoming events.
        
        Args:
            symbol: Stock symbol
            strategies: List of strategy names to consider
            dte: Days to expiration
            
        Returns:
            Tuple of (allowed_strategies, filtered_out_strategies)
        """
        assessment = self.assess_event_risk(symbol, dte)
        
        allowed = []
        filtered = []
        
        # Define risk categories
        defined_risk_strategies = {
            "bull_put_spread", "bear_call_spread", "iron_condor",
            "iron_butterfly", "credit_spread", "debit_spread",
            "long_call", "long_put", "straddle", "strangle",
        }
        
        undefined_risk_strategies = {
            "naked_put", "naked_call", "short_straddle",
            "short_strangle", "ratio_spread",
        }
        
        call_strategies = {
            "long_call", "bull_call_spread", "naked_call",
            "covered_call", "call_calendar",
        }
        
        for strategy in strategies:
            strategy_lower = strategy.lower()
            
            # Check if we should avoid based on risk level
            if assessment.risk_level == "extreme":
                filtered.append(strategy)
                continue
            
            # Check defined risk requirement
            if assessment.use_defined_risk_only:
                if strategy_lower in undefined_risk_strategies:
                    filtered.append(strategy)
                    continue
            
            # Check call restriction before ex-div
            if assessment.avoid_calls_before_ex_div:
                if strategy_lower in call_strategies:
                    # Only filter if ITM calls likely
                    filtered.append(strategy)
                    continue
            
            allowed.append(strategy)
        
        if filtered:
            logger.info(
                f"Event filter for {symbol}: "
                f"allowed={len(allowed)}, filtered={len(filtered)} | "
                f"reason: {assessment.risk_level} risk"
            )
        
        return allowed, filtered
    
    def get_upcoming_high_impact_events(
        self,
        days: int = 7,
    ) -> List[MarketEvent]:
        """Get all high-impact events in the next N days.
        
        Args:
            days: Number of days to look ahead
            
        Returns:
            List of high-impact events
        """
        today = date.today()
        end_date = today + timedelta(days=days)
        
        events = []
        
        # Economic events
        for event in self._economic_events:
            if today <= event.date <= end_date and event.impact == EventImpact.HIGH:
                events.append(event)
        
        return sorted(events, key=lambda e: e.date)
    
    def get_earnings_calendar(
        self,
        symbols: List[str],
        days: int = 14,
    ) -> Dict[str, List[MarketEvent]]:
        """Get earnings calendar for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            days: Days to look ahead
            
        Returns:
            Dict mapping symbol to list of earnings events
        """
        today = date.today()
        end_date = today + timedelta(days=days)
        
        calendar = {}
        
        for symbol in symbols:
            earnings = self.get_earnings_dates(symbol)
            upcoming = [e for e in earnings if today <= e.date <= end_date]
            if upcoming:
                calendar[symbol] = upcoming
        
        return calendar
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of upcoming events."""
        today = date.today()
        week_ahead = today + timedelta(days=7)
        
        upcoming_economic = [
            {
                "type": e.event_type.value,
                "date": e.date.isoformat(),
                "description": e.description,
                "impact": e.impact.value,
            }
            for e in self._economic_events
            if today <= e.date <= week_ahead
        ]
        
        return {
            "date": today.isoformat(),
            "upcoming_week_economic_events": len(upcoming_economic),
            "economic_events": upcoming_economic,
            "cached_earnings_symbols": len(self._earnings_cache),
        }


# Factory function
def create_event_risk_manager(
    lookforward_days: int = 14,
) -> EventRiskManager:
    """Create an EventRiskManager instance.
    
    Args:
        lookforward_days: Days to look ahead for events
        
    Returns:
        Configured EventRiskManager
    """
    return EventRiskManager(lookforward_days=lookforward_days)
