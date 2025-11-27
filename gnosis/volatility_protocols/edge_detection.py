"""
Edge Detection Module
=====================

Mathematical edge calculations for volatility trading:
- Vol Edge Score (IV vs RV)
- IV Rank (252-day percentile)
- Skew Analysis (Put/Call skew)
- Term Structure Premium (Contango/Backwardation)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from datetime import datetime


@dataclass
class VolEdgeScore:
    """Complete volatility edge analysis"""
    vol_edge: float  # Percentage difference IV vs RV
    iv_rank: float  # 0-100 percentile
    iv_current: float
    rv_20day: float
    iv_252_low: float
    iv_252_high: float
    timestamp: datetime

    def meets_threshold(self, strategy_type: str) -> bool:
        """Check if vol edge meets minimum threshold for strategy"""
        thresholds = {
            'short_vol': 15.0,  # Minimum +15% for short vol
            'long_vol': -10.0,  # Maximum +10% for long vol (can be negative)
            'neutral_vol': 5.0,  # Minimum +5% for neutral
            'directional': None,  # No threshold for directional
        }

        threshold = thresholds.get(strategy_type)
        if threshold is None:
            return True

        if strategy_type == 'long_vol':
            return self.vol_edge <= 10.0  # Want low or negative
        else:
            return self.vol_edge >= threshold

    def in_optimal_range(self, strategy_type: str) -> bool:
        """Check if vol edge is in optimal range"""
        ranges = {
            'short_vol': (20.0, 40.0),
            'long_vol': (-15.0, 5.0),
            'neutral_vol': (10.0, 25.0),
        }

        range_vals = ranges.get(strategy_type)
        if range_vals is None:
            return True

        return range_vals[0] <= self.vol_edge <= range_vals[1]


@dataclass
class SkewAnalysis:
    """Put/Call skew measurements"""
    put_skew: float  # (25Δ Put IV - ATM IV) / ATM IV * 100
    call_skew: float  # (25Δ Call IV - ATM IV) / ATM IV * 100
    atm_iv: float
    put_25delta_iv: float
    call_25delta_iv: float

    def jade_lizard_favorable(self) -> bool:
        """Check if skew favors Jade Lizard entry"""
        return self.put_skew > 8.0

    def put_backspread_favorable(self) -> bool:
        """Check if skew favors Put Backspread"""
        return self.put_skew < 3.0

    def call_backspread_favorable(self) -> bool:
        """Check if skew favors Call Backspread"""
        return self.call_skew < 0.0  # Negative call skew


@dataclass
class TermStructure:
    """Term structure analysis"""
    term_premium: float  # Percentage difference back/front
    front_month_iv: float
    back_month_iv: float
    front_month_dte: int
    back_month_dte: int
    is_contango: bool
    is_backwardation: bool

    def calendar_favorable(self) -> bool:
        """Check if term structure favors calendar spreads"""
        return (
            self.is_contango and
            self.term_premium > 5.0 and
            self.front_month_dte < 21 and
            self.back_month_dte > 35
        )

    def in_optimal_range(self) -> bool:
        """Check if term premium in optimal range for calendars"""
        return 8.0 <= self.term_premium <= 15.0


def calculate_vol_edge(
    iv_current: float,
    rv_20day: float,
    iv_252_low: float,
    iv_252_high: float,
) -> VolEdgeScore:
    """
    Calculate Vol Edge Score

    Formula: Vol Edge = (IV_current - RV_20day) / RV_20day * 100

    Args:
        iv_current: Current implied volatility (%)
        rv_20day: 20-day realized volatility (%)
        iv_252_low: 252-day IV low (%)
        iv_252_high: 252-day IV high (%)

    Returns:
        VolEdgeScore with all metrics
    """
    # Vol Edge calculation
    if rv_20day == 0:
        rv_20day = 0.01  # Avoid division by zero

    vol_edge = ((iv_current - rv_20day) / rv_20day) * 100

    # IV Rank calculation
    if iv_252_high == iv_252_low:
        iv_rank = 50.0  # Default to middle if no range
    else:
        iv_rank = ((iv_current - iv_252_low) / (iv_252_high - iv_252_low)) * 100

    # Clamp IV Rank to 0-100
    iv_rank = max(0.0, min(100.0, iv_rank))

    return VolEdgeScore(
        vol_edge=vol_edge,
        iv_rank=iv_rank,
        iv_current=iv_current,
        rv_20day=rv_20day,
        iv_252_low=iv_252_low,
        iv_252_high=iv_252_high,
        timestamp=datetime.now(),
    )


def calculate_iv_rank(
    iv_current: float,
    iv_history: np.ndarray,
    period: int = 252,
) -> float:
    """
    Calculate IV Rank (percentile position in historical range)

    Formula: IV Rank = (IV_current - IV_min) / (IV_max - IV_min) * 100

    Args:
        iv_current: Current IV value
        iv_history: Array of historical IV values
        period: Lookback period (default 252 trading days)

    Returns:
        IV Rank as percentage (0-100)
    """
    if len(iv_history) < 2:
        return 50.0  # Default to middle

    # Use last 'period' values
    recent_history = iv_history[-period:] if len(iv_history) > period else iv_history

    iv_min = np.min(recent_history)
    iv_max = np.max(recent_history)

    if iv_max == iv_min:
        return 50.0

    iv_rank = ((iv_current - iv_min) / (iv_max - iv_min)) * 100

    return max(0.0, min(100.0, iv_rank))


def calculate_skew(
    atm_iv: float,
    put_25delta_iv: float,
    call_25delta_iv: float,
) -> SkewAnalysis:
    """
    Calculate Put and Call Skew

    Put Skew = (25Δ Put IV - ATM IV) / ATM IV * 100
    Call Skew = (25Δ Call IV - ATM IV) / ATM IV * 100

    Args:
        atm_iv: At-the-money implied volatility
        put_25delta_iv: 25-delta put implied volatility
        call_25delta_iv: 25-delta call implied volatility

    Returns:
        SkewAnalysis object
    """
    if atm_iv == 0:
        atm_iv = 0.01  # Avoid division by zero

    put_skew = ((put_25delta_iv - atm_iv) / atm_iv) * 100
    call_skew = ((call_25delta_iv - atm_iv) / atm_iv) * 100

    return SkewAnalysis(
        put_skew=put_skew,
        call_skew=call_skew,
        atm_iv=atm_iv,
        put_25delta_iv=put_25delta_iv,
        call_25delta_iv=call_25delta_iv,
    )


def calculate_term_premium(
    front_month_iv: float,
    back_month_iv: float,
    front_month_dte: int,
    back_month_dte: int,
) -> TermStructure:
    """
    Calculate Term Structure Premium (Contango/Backwardation)

    Term Premium = (Back Month IV - Front Month IV) / Front Month IV * 100

    Args:
        front_month_iv: Front month implied volatility
        back_month_iv: Back month implied volatility
        front_month_dte: Days to expiration (front)
        back_month_dte: Days to expiration (back)

    Returns:
        TermStructure object
    """
    if front_month_iv == 0:
        front_month_iv = 0.01

    term_premium = ((back_month_iv - front_month_iv) / front_month_iv) * 100

    is_contango = term_premium > 0
    is_backwardation = term_premium < -2.0  # Small negative tolerance

    return TermStructure(
        term_premium=term_premium,
        front_month_iv=front_month_iv,
        back_month_iv=back_month_iv,
        front_month_dte=front_month_dte,
        back_month_dte=back_month_dte,
        is_contango=is_contango,
        is_backwardation=is_backwardation,
    )


def calculate_spread_quality(
    bid: float,
    ask: float,
    mid: Optional[float] = None,
) -> float:
    """
    Calculate bid-ask spread quality

    Spread Quality = (Ask - Bid) / Mid * 100

    Args:
        bid: Bid price
        ask: Ask price
        mid: Mid price (optional, will calculate if not provided)

    Returns:
        Spread quality percentage
    """
    if mid is None:
        mid = (bid + ask) / 2

    if mid == 0:
        return 100.0  # Poor quality

    spread_quality = ((ask - bid) / mid) * 100

    return spread_quality


def meets_liquidity_standards(
    spread_quality: float,
    open_interest: int,
    daily_volume: int,
    asset_type: str = 'etf',
) -> Tuple[bool, str]:
    """
    Check if option meets minimum liquidity standards

    Args:
        spread_quality: Spread quality percentage
        open_interest: Open interest
        daily_volume: Daily volume
        asset_type: 'etf', 'large_cap', or 'small_cap'

    Returns:
        (meets_standards, reason)
    """
    # Spread quality thresholds
    spread_thresholds = {
        'etf': 3.0,
        'large_cap': 5.0,
        'small_cap': 8.0,
    }

    max_spread = spread_thresholds.get(asset_type, 5.0)

    if spread_quality > max_spread:
        return False, f"Spread quality {spread_quality:.2f}% exceeds {max_spread}%"

    # Open interest minimums
    oi_minimums = {
        'etf': 100,
        'large_cap': 50,
        'small_cap': 25,
    }

    min_oi = oi_minimums.get(asset_type, 50)

    if open_interest < min_oi:
        return False, f"Open interest {open_interest} below minimum {min_oi}"

    # Volume check (should be >20% of OI)
    min_volume = open_interest * 0.20

    if daily_volume < min_volume:
        return False, f"Volume {daily_volume} below 20% of OI ({min_volume:.0f})"

    return True, "Liquidity standards met"


# Entry threshold matrix
ENTRY_THRESHOLDS = {
    'short_vol': {
        'min_vol_edge': 15.0,
        'optimal_range': (20.0, 40.0),
        'min_iv_rank': 70.0,
        'description': 'Short Straddles, Strangles',
    },
    'long_vol': {
        'min_vol_edge': -10.0,  # Can be negative
        'optimal_range': (-15.0, 5.0),
        'max_iv_rank': 30.0,
        'description': 'Long Straddles, Backspreads',
    },
    'neutral_vol': {
        'min_vol_edge': 5.0,
        'optimal_range': (10.0, 25.0),
        'min_iv_rank': 40.0,
        'max_iv_rank': 80.0,
        'description': 'Iron Condors, Calendars',
    },
    'directional': {
        'min_vol_edge': None,  # No requirement
        'description': 'Verticals, Diagonals',
    },
}
