#!/usr/bin/env python3
"""
Synthetic Options Data Generator for GNOSIS V2 Backtesting

Generates realistic historical options data including:
1. Options chains with proper strikes and expirations
2. Greeks (Delta, Gamma, Theta, Vega, Rho)
3. Implied Volatility surfaces (smile/skew)
4. Open Interest patterns (liquidity pools)
5. Volume patterns (unusual activity detection)
6. Dealer positioning estimates (gamma exposure)
7. Max Pain calculations
8. Put/Call ratios

Based on real market dynamics:
- IV typically increases before earnings
- Put skew steepens in down markets
- OI concentrates at round strikes
- Gamma exposure affects price pinning
- Volume spikes indicate smart money

Author: GNOSIS Trading System
Version: 1.0.0
"""

import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import interp1d

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class OptionContract:
    """Single option contract with full data."""
    symbol: str
    underlying: str
    strike: float
    expiration: datetime
    option_type: str  # "call" or "put"
    
    # Pricing
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    mid: float = 0.0
    
    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    
    # Volatility
    implied_volatility: float = 0.0
    iv_percentile: float = 0.0
    
    # Activity
    volume: int = 0
    open_interest: int = 0
    volume_oi_ratio: float = 0.0
    
    # Calculated
    intrinsic_value: float = 0.0
    extrinsic_value: float = 0.0
    days_to_expiry: int = 0
    moneyness: float = 0.0  # strike/spot
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "underlying": self.underlying,
            "strike": self.strike,
            "expiration": self.expiration.isoformat(),
            "option_type": self.option_type,
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "implied_volatility": self.implied_volatility,
            "volume": self.volume,
            "open_interest": self.open_interest,
        }


@dataclass
class OptionsChain:
    """Complete options chain for a symbol on a date."""
    underlying: str
    date: datetime
    spot_price: float
    
    # Chain data
    calls: List[OptionContract] = field(default_factory=list)
    puts: List[OptionContract] = field(default_factory=list)
    expirations: List[datetime] = field(default_factory=list)
    strikes: List[float] = field(default_factory=list)
    
    # Aggregated metrics
    total_call_oi: int = 0
    total_put_oi: int = 0
    total_call_volume: int = 0
    total_put_volume: int = 0
    put_call_ratio: float = 0.0
    put_call_oi_ratio: float = 0.0
    
    # Key levels
    max_pain: float = 0.0
    gamma_flip: float = 0.0
    call_wall: float = 0.0  # Highest call OI strike
    put_wall: float = 0.0   # Highest put OI strike
    
    # Dealer exposure
    net_gamma_exposure: float = 0.0
    dealer_positioning: str = "neutral"  # "long_gamma", "short_gamma", "neutral"


@dataclass
class OptionsFlow:
    """Options flow/unusual activity record."""
    timestamp: datetime
    underlying: str
    strike: float
    expiration: datetime
    option_type: str
    
    # Trade details
    price: float = 0.0
    size: int = 0
    premium: float = 0.0
    
    # Classification
    side: str = "unknown"  # "buy", "sell"
    sentiment: str = "neutral"  # "bullish", "bearish", "neutral"
    trade_type: str = "unknown"  # "sweep", "block", "split"
    is_unusual: bool = False
    unusual_score: float = 0.0


# ============================================================================
# BLACK-SCHOLES PRICING
# ============================================================================

class BlackScholes:
    """Black-Scholes option pricing and Greeks."""
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call option price."""
        if T <= 0:
            return max(0, S - K)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option price."""
        if T <= 0:
            return max(0, K - S)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate delta."""
        if T <= 0:
            if option_type == "call":
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        if option_type == "call":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate gamma (same for calls and puts)."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate theta (per day)."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if option_type == "call":
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        return (term1 + term2) / 365  # Per day
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate vega (per 1% IV change)."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100
    
    @staticmethod
    def implied_volatility(
        price: float, S: float, K: float, T: float, r: float, option_type: str,
        max_iterations: int = 100, tolerance: float = 1e-6
    ) -> float:
        """Calculate implied volatility using Newton-Raphson."""
        if T <= 0:
            return 0.0
        
        sigma = 0.3  # Initial guess
        
        for _ in range(max_iterations):
            if option_type == "call":
                model_price = BlackScholes.call_price(S, K, T, r, sigma)
            else:
                model_price = BlackScholes.put_price(S, K, T, r, sigma)
            
            diff = model_price - price
            
            if abs(diff) < tolerance:
                return sigma
            
            vega = BlackScholes.vega(S, K, T, r, sigma) * 100  # Adjust for percentage
            
            if vega < 1e-10:
                break
            
            sigma = sigma - diff / vega
            sigma = max(0.01, min(5.0, sigma))  # Bound sigma
        
        return sigma


# ============================================================================
# IV SURFACE GENERATOR
# ============================================================================

class IVSurfaceGenerator:
    """
    Generate realistic implied volatility surfaces.
    
    Models:
    - Volatility smile (higher IV for OTM options)
    - Put skew (higher IV for OTM puts)
    - Term structure (IV varies by expiration)
    - Event premium (higher IV before earnings)
    """
    
    def __init__(
        self,
        base_iv: float = 0.25,
        skew_factor: float = 0.15,
        smile_factor: float = 0.10,
        term_slope: float = 0.02,
    ):
        self.base_iv = base_iv
        self.skew_factor = skew_factor
        self.smile_factor = smile_factor
        self.term_slope = term_slope
    
    def get_iv(
        self,
        spot: float,
        strike: float,
        days_to_expiry: int,
        option_type: str,
        market_regime: str = "neutral",
        days_to_earnings: Optional[int] = None,
    ) -> float:
        """
        Calculate implied volatility for a specific option.
        
        Args:
            spot: Current stock price
            strike: Strike price
            days_to_expiry: Days until expiration
            option_type: "call" or "put"
            market_regime: "bullish", "bearish", "volatile", "neutral"
            days_to_earnings: Days until next earnings (if known)
        """
        moneyness = np.log(strike / spot)
        
        # Base IV with regime adjustment
        regime_adj = {
            "bullish": -0.03,
            "bearish": 0.05,
            "volatile": 0.10,
            "neutral": 0.0,
        }
        iv = self.base_iv + regime_adj.get(market_regime, 0)
        
        # Smile effect (higher IV for far OTM)
        smile = self.smile_factor * (moneyness ** 2)
        iv += smile
        
        # Skew effect (higher IV for OTM puts)
        if option_type == "put" and moneyness < 0:
            skew = self.skew_factor * abs(moneyness)
            iv += skew
        elif option_type == "call" and moneyness > 0:
            # Slight call skew for high strikes
            skew = self.skew_factor * 0.3 * moneyness
            iv += skew
        
        # Term structure (longer dated = slightly higher IV typically)
        if days_to_expiry > 0:
            term_adj = self.term_slope * np.log(days_to_expiry / 30)
            iv += term_adj * 0.5
        
        # Earnings premium
        if days_to_earnings is not None and days_to_earnings <= days_to_expiry:
            # IV increases as earnings approach
            if days_to_earnings <= 7:
                earnings_premium = 0.15 * (1 - days_to_earnings / 7)
            elif days_to_earnings <= 14:
                earnings_premium = 0.08 * (1 - (days_to_earnings - 7) / 7)
            else:
                earnings_premium = 0.02
            iv += earnings_premium
        
        # Add some noise
        noise = np.random.normal(0, 0.01)
        iv += noise
        
        # Bound IV to realistic range
        return max(0.05, min(2.0, iv))


# ============================================================================
# OPEN INTEREST GENERATOR
# ============================================================================

class OpenInterestGenerator:
    """
    Generate realistic open interest patterns.
    
    Models:
    - Concentration at round strikes (100, 150, 200, etc.)
    - Higher OI near ATM
    - Put walls below spot, call walls above
    - Accumulation patterns over time
    """
    
    def __init__(self, base_oi: int = 5000):
        self.base_oi = base_oi
    
    def get_open_interest(
        self,
        spot: float,
        strike: float,
        days_to_expiry: int,
        option_type: str,
        is_weekly: bool = False,
    ) -> int:
        """Generate realistic open interest for an option."""
        
        # Base OI depends on expiration
        if is_weekly:
            base = self.base_oi * 0.5
        elif days_to_expiry <= 7:
            base = self.base_oi * 2  # High OI in near-term
        elif days_to_expiry <= 30:
            base = self.base_oi * 1.5
        elif days_to_expiry <= 60:
            base = self.base_oi
        else:
            base = self.base_oi * 0.7
        
        # Distance from ATM effect
        moneyness = abs(strike - spot) / spot
        atm_factor = np.exp(-5 * moneyness)  # Higher OI near ATM
        
        # Round strike premium
        round_premium = 1.0
        if strike % 50 == 0:
            round_premium = 3.0
        elif strike % 25 == 0:
            round_premium = 2.0
        elif strike % 10 == 0:
            round_premium = 1.5
        elif strike % 5 == 0:
            round_premium = 1.2
        
        # Put/Call bias based on position relative to spot
        if option_type == "put":
            if strike < spot * 0.95:  # OTM puts
                type_factor = 1.3  # Higher OI for protective puts
            else:
                type_factor = 1.0
        else:  # call
            if strike > spot * 1.05:  # OTM calls
                type_factor = 1.1
            else:
                type_factor = 1.0
        
        # Calculate OI
        oi = base * atm_factor * round_premium * type_factor
        
        # Add randomness
        oi *= np.random.lognormal(0, 0.3)
        
        return max(10, int(oi))


# ============================================================================
# VOLUME GENERATOR
# ============================================================================

class VolumeGenerator:
    """
    Generate realistic options volume patterns.
    
    Models:
    - Volume correlated with OI
    - Unusual volume spikes
    - Time-of-day patterns
    - Event-driven volume
    """
    
    def __init__(self, base_volume_ratio: float = 0.1):
        self.base_volume_ratio = base_volume_ratio
    
    def get_volume(
        self,
        open_interest: int,
        spot: float,
        strike: float,
        days_to_expiry: int,
        option_type: str,
        is_unusual_day: bool = False,
        unusual_direction: str = None,  # "bullish" or "bearish"
    ) -> Tuple[int, bool, float]:
        """
        Generate volume and detect unusual activity.
        
        Returns:
            (volume, is_unusual, unusual_score)
        """
        # Base volume as ratio of OI
        base_vol = open_interest * self.base_volume_ratio
        
        # Near-term options trade more actively
        if days_to_expiry <= 7:
            dte_factor = 2.5
        elif days_to_expiry <= 14:
            dte_factor = 1.8
        elif days_to_expiry <= 30:
            dte_factor = 1.3
        else:
            dte_factor = 1.0
        
        # ATM options trade more
        moneyness = abs(strike - spot) / spot
        atm_factor = np.exp(-3 * moneyness) + 0.3
        
        volume = base_vol * dte_factor * atm_factor
        
        # Add unusual activity
        is_unusual = False
        unusual_score = 0.0
        
        if is_unusual_day:
            # Check if this strike would be targeted
            should_be_unusual = False
            
            if unusual_direction == "bullish" and option_type == "call":
                if strike > spot and strike < spot * 1.10:  # OTM calls
                    should_be_unusual = True
            elif unusual_direction == "bearish" and option_type == "put":
                if strike < spot and strike > spot * 0.90:  # OTM puts
                    should_be_unusual = True
            
            if should_be_unusual and np.random.random() > 0.5:
                volume *= np.random.uniform(3, 10)  # 3-10x normal
                is_unusual = True
                unusual_score = min(1.0, volume / (open_interest * 0.5))
        
        # Add noise
        volume *= np.random.lognormal(0, 0.4)
        
        return max(0, int(volume)), is_unusual, unusual_score


# ============================================================================
# MAIN OPTIONS DATA GENERATOR
# ============================================================================

class SyntheticOptionsGenerator:
    """
    Main class for generating comprehensive synthetic options data.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.05,
        base_iv: float = 0.25,
        seed: int = 42,
    ):
        self.risk_free_rate = risk_free_rate
        self.seed = seed
        np.random.seed(seed)
        
        # Initialize sub-generators
        self.iv_generator = IVSurfaceGenerator(base_iv=base_iv)
        self.oi_generator = OpenInterestGenerator()
        self.volume_generator = VolumeGenerator()
    
    def generate_strikes(
        self,
        spot: float,
        num_strikes: int = 20,
        strike_width_pct: float = 0.025,
    ) -> List[float]:
        """Generate realistic strike prices centered around spot."""
        
        # Determine strike increment based on price level
        if spot < 50:
            increment = 1
        elif spot < 100:
            increment = 2.5
        elif spot < 200:
            increment = 5
        elif spot < 500:
            increment = 10
        else:
            increment = 25
        
        # Round spot to nearest increment
        atm_strike = round(spot / increment) * increment
        
        # Generate strikes around ATM
        strikes = []
        for i in range(-num_strikes // 2, num_strikes // 2 + 1):
            strike = atm_strike + i * increment
            if strike > 0:
                strikes.append(strike)
        
        return sorted(strikes)
    
    def generate_expirations(
        self,
        current_date: datetime,
        num_weekly: int = 4,
        num_monthly: int = 3,
        num_quarterly: int = 2,
    ) -> List[datetime]:
        """Generate realistic expiration dates."""
        expirations = []
        
        # Find next Friday for weeklies
        days_to_friday = (4 - current_date.weekday()) % 7
        if days_to_friday == 0:
            days_to_friday = 7
        
        next_friday = current_date + timedelta(days=days_to_friday)
        
        # Add weeklies
        for i in range(num_weekly):
            exp = next_friday + timedelta(weeks=i)
            expirations.append(exp)
        
        # Add monthlies (third Friday of each month)
        month = current_date.month
        year = current_date.year
        
        for i in range(num_monthly):
            # Move to next month
            month += 1
            if month > 12:
                month = 1
                year += 1
            
            # Find third Friday
            first_day = datetime(year, month, 1, tzinfo=current_date.tzinfo)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(weeks=2)
            
            if third_friday > current_date:
                expirations.append(third_friday)
        
        # Add quarterly (LEAPS-style)
        for i in range(num_quarterly):
            quarter_month = ((current_date.month - 1) // 3 + 1 + i) * 3
            quarter_year = current_date.year
            if quarter_month > 12:
                quarter_month -= 12
                quarter_year += 1
            
            # Third Friday of quarter-end month
            first_day = datetime(quarter_year, quarter_month, 1, tzinfo=current_date.tzinfo)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(weeks=2)
            
            if third_friday > current_date and third_friday not in expirations:
                expirations.append(third_friday)
        
        return sorted(set(expirations))
    
    def generate_option_contract(
        self,
        underlying: str,
        spot: float,
        strike: float,
        expiration: datetime,
        current_date: datetime,
        option_type: str,
        market_regime: str = "neutral",
        days_to_earnings: Optional[int] = None,
        is_unusual_day: bool = False,
        unusual_direction: Optional[str] = None,
    ) -> OptionContract:
        """Generate a single option contract with full data."""
        
        days_to_expiry = max(0, (expiration - current_date).days)
        T = days_to_expiry / 365.0
        
        # Generate IV
        iv = self.iv_generator.get_iv(
            spot, strike, days_to_expiry, option_type,
            market_regime, days_to_earnings
        )
        
        # Calculate theoretical price
        if option_type == "call":
            theo_price = BlackScholes.call_price(spot, strike, T, self.risk_free_rate, iv)
        else:
            theo_price = BlackScholes.put_price(spot, strike, T, self.risk_free_rate, iv)
        
        # Generate bid/ask spread based on liquidity
        moneyness = abs(strike - spot) / spot
        base_spread_pct = 0.02 + 0.05 * moneyness + 0.01 * (days_to_expiry / 30)
        spread = theo_price * base_spread_pct
        spread = max(0.01, min(spread, theo_price * 0.2))
        
        bid = max(0.01, theo_price - spread / 2)
        ask = theo_price + spread / 2
        mid = (bid + ask) / 2
        
        # Calculate Greeks
        delta = BlackScholes.delta(spot, strike, T, self.risk_free_rate, iv, option_type)
        gamma = BlackScholes.gamma(spot, strike, T, self.risk_free_rate, iv)
        theta = BlackScholes.theta(spot, strike, T, self.risk_free_rate, iv, option_type)
        vega = BlackScholes.vega(spot, strike, T, self.risk_free_rate, iv)
        
        # Rho (simplified)
        if option_type == "call":
            rho = strike * T * np.exp(-self.risk_free_rate * T) * norm.cdf(
                BlackScholes.d2(spot, strike, T, self.risk_free_rate, iv)
            ) / 100
        else:
            rho = -strike * T * np.exp(-self.risk_free_rate * T) * norm.cdf(
                -BlackScholes.d2(spot, strike, T, self.risk_free_rate, iv)
            ) / 100
        
        # Generate OI
        is_weekly = days_to_expiry <= 7 and expiration.weekday() == 4
        oi = self.oi_generator.get_open_interest(
            spot, strike, days_to_expiry, option_type, is_weekly
        )
        
        # Generate volume
        volume, is_unusual, unusual_score = self.volume_generator.get_volume(
            oi, spot, strike, days_to_expiry, option_type,
            is_unusual_day, unusual_direction
        )
        
        # Calculate intrinsic/extrinsic
        if option_type == "call":
            intrinsic = max(0, spot - strike)
        else:
            intrinsic = max(0, strike - spot)
        extrinsic = max(0, mid - intrinsic)
        
        # Create symbol (OCC format)
        exp_str = expiration.strftime("%y%m%d")
        opt_type = "C" if option_type == "call" else "P"
        strike_str = f"{int(strike * 1000):08d}"
        symbol = f"{underlying}{exp_str}{opt_type}{strike_str}"
        
        return OptionContract(
            symbol=symbol,
            underlying=underlying,
            strike=strike,
            expiration=expiration,
            option_type=option_type,
            bid=round(bid, 2),
            ask=round(ask, 2),
            last=round(mid + np.random.uniform(-spread/4, spread/4), 2),
            mid=round(mid, 2),
            delta=round(delta, 4),
            gamma=round(gamma, 6),
            theta=round(theta, 4),
            vega=round(vega, 4),
            rho=round(rho, 4),
            implied_volatility=round(iv, 4),
            iv_percentile=round(np.random.uniform(0.2, 0.8), 2),
            volume=volume,
            open_interest=oi,
            volume_oi_ratio=round(volume / max(1, oi), 4),
            intrinsic_value=round(intrinsic, 2),
            extrinsic_value=round(extrinsic, 2),
            days_to_expiry=days_to_expiry,
            moneyness=round(strike / spot, 4),
        )
    
    def generate_options_chain(
        self,
        underlying: str,
        spot: float,
        current_date: datetime,
        market_regime: str = "neutral",
        days_to_earnings: Optional[int] = None,
        is_unusual_day: bool = False,
        unusual_direction: Optional[str] = None,
    ) -> OptionsChain:
        """Generate complete options chain for a symbol."""
        
        # Generate strikes and expirations
        strikes = self.generate_strikes(spot)
        expirations = self.generate_expirations(current_date)
        
        calls = []
        puts = []
        
        # Generate all contracts
        for exp in expirations:
            for strike in strikes:
                # Generate call
                call = self.generate_option_contract(
                    underlying, spot, strike, exp, current_date, "call",
                    market_regime, days_to_earnings, is_unusual_day, unusual_direction
                )
                calls.append(call)
                
                # Generate put
                put = self.generate_option_contract(
                    underlying, spot, strike, exp, current_date, "put",
                    market_regime, days_to_earnings, is_unusual_day, unusual_direction
                )
                puts.append(put)
        
        # Calculate aggregated metrics
        total_call_oi = sum(c.open_interest for c in calls)
        total_put_oi = sum(p.open_interest for p in puts)
        total_call_volume = sum(c.volume for c in calls)
        total_put_volume = sum(p.volume for p in puts)
        
        put_call_ratio = total_put_volume / max(1, total_call_volume)
        put_call_oi_ratio = total_put_oi / max(1, total_call_oi)
        
        # Find max pain (strike with max OI sum)
        strike_oi = {}
        for c in calls:
            strike_oi[c.strike] = strike_oi.get(c.strike, 0) + c.open_interest
        for p in puts:
            strike_oi[p.strike] = strike_oi.get(p.strike, 0) + p.open_interest
        
        max_pain = max(strike_oi.keys(), key=lambda k: strike_oi[k]) if strike_oi else spot
        
        # Find call/put walls (highest OI strikes)
        call_oi_by_strike = {}
        put_oi_by_strike = {}
        for c in calls:
            if c.strike > spot:  # OTM calls
                call_oi_by_strike[c.strike] = call_oi_by_strike.get(c.strike, 0) + c.open_interest
        for p in puts:
            if p.strike < spot:  # OTM puts
                put_oi_by_strike[p.strike] = put_oi_by_strike.get(p.strike, 0) + p.open_interest
        
        call_wall = max(call_oi_by_strike.keys(), key=lambda k: call_oi_by_strike[k]) if call_oi_by_strike else spot * 1.05
        put_wall = max(put_oi_by_strike.keys(), key=lambda k: put_oi_by_strike[k]) if put_oi_by_strike else spot * 0.95
        
        # Calculate net gamma exposure (simplified dealer model)
        # Dealers are typically short calls and long puts from retail
        net_gamma = 0
        for c in calls:
            # Dealers short calls = negative gamma when spot rises
            net_gamma -= c.gamma * c.open_interest * 100 * spot  # Dollar gamma
        for p in puts:
            # Dealers long puts = negative gamma when spot falls
            net_gamma += p.gamma * p.open_interest * 100 * spot
        
        # Gamma flip is where net gamma crosses zero
        gamma_flip = max_pain  # Simplified: near max pain
        
        # Determine dealer positioning
        if net_gamma > 1e6:
            dealer_positioning = "long_gamma"
        elif net_gamma < -1e6:
            dealer_positioning = "short_gamma"
        else:
            dealer_positioning = "neutral"
        
        return OptionsChain(
            underlying=underlying,
            date=current_date,
            spot_price=spot,
            calls=calls,
            puts=puts,
            expirations=expirations,
            strikes=strikes,
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            total_call_volume=total_call_volume,
            total_put_volume=total_put_volume,
            put_call_ratio=round(put_call_ratio, 3),
            put_call_oi_ratio=round(put_call_oi_ratio, 3),
            max_pain=max_pain,
            gamma_flip=gamma_flip,
            call_wall=call_wall,
            put_wall=put_wall,
            net_gamma_exposure=net_gamma,
            dealer_positioning=dealer_positioning,
        )
    
    def generate_unusual_flow(
        self,
        chain: OptionsChain,
        num_unusual: int = 5,
        bias: str = "neutral",  # "bullish", "bearish", "neutral"
    ) -> List[OptionsFlow]:
        """Generate unusual options flow records."""
        
        flows = []
        
        for _ in range(num_unusual):
            # Select option type based on bias
            if bias == "bullish":
                option_type = "call" if np.random.random() > 0.3 else "put"
                sentiment = "bullish" if option_type == "call" else "bearish"
            elif bias == "bearish":
                option_type = "put" if np.random.random() > 0.3 else "call"
                sentiment = "bearish" if option_type == "put" else "bullish"
            else:
                option_type = "call" if np.random.random() > 0.5 else "put"
                sentiment = "bullish" if option_type == "call" else "bearish"
            
            # Select from chain
            contracts = chain.calls if option_type == "call" else chain.puts
            
            # Filter to OTM with decent OI
            otm_contracts = [
                c for c in contracts
                if (c.strike > chain.spot_price if option_type == "call" else c.strike < chain.spot_price)
                and c.open_interest > 100
                and c.days_to_expiry > 0
                and c.days_to_expiry <= 45
            ]
            
            if not otm_contracts:
                continue
            
            # Weight by OI (higher OI = more likely target)
            weights = np.array([c.open_interest for c in otm_contracts])
            weights = weights / weights.sum()
            
            contract = np.random.choice(otm_contracts, p=weights)
            
            # Generate flow details
            size = int(np.random.lognormal(5, 1))  # 50-500+ contracts typical
            price = contract.mid
            premium = size * price * 100
            
            # Determine trade type
            if size > 500:
                trade_type = "block"
            elif premium > 100000:
                trade_type = "sweep"
            else:
                trade_type = "split"
            
            # Side (buy vs sell)
            if bias == "bullish" and option_type == "call":
                side = "buy"
            elif bias == "bearish" and option_type == "put":
                side = "buy"
            else:
                side = np.random.choice(["buy", "sell"], p=[0.6, 0.4])
            
            # Unusual score
            vol_oi_ratio = size / max(1, contract.open_interest)
            unusual_score = min(1.0, vol_oi_ratio * 2 + premium / 500000)
            
            flow = OptionsFlow(
                timestamp=chain.date,
                underlying=chain.underlying,
                strike=contract.strike,
                expiration=contract.expiration,
                option_type=option_type,
                price=price,
                size=size,
                premium=premium,
                side=side,
                sentiment=sentiment,
                trade_type=trade_type,
                is_unusual=True,
                unusual_score=round(unusual_score, 3),
            )
            flows.append(flow)
        
        return flows
    
    def generate_historical_data(
        self,
        underlying: str,
        price_data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        earnings_dates: Optional[List[datetime]] = None,
    ) -> Dict[datetime, OptionsChain]:
        """
        Generate historical options data for backtesting.
        
        Args:
            underlying: Stock symbol
            price_data: DataFrame with OHLCV data
            start_date: Start date
            end_date: End date
            earnings_dates: List of earnings dates for IV premium
        
        Returns:
            Dictionary mapping dates to OptionsChain objects
        """
        
        historical_chains = {}
        
        # Determine market regime from price action
        def get_regime(prices: np.ndarray) -> str:
            if len(prices) < 10:
                return "neutral"
            
            returns = np.diff(prices) / prices[:-1]
            trend = (prices[-1] - prices[0]) / prices[0]
            vol = np.std(returns) * np.sqrt(252)
            
            if vol > 0.35:
                return "volatile"
            elif trend > 0.05:
                return "bullish"
            elif trend < -0.05:
                return "bearish"
            else:
                return "neutral"
        
        # Generate unusual activity days (roughly 20% of days)
        unusual_days = set()
        trading_days = [d for d in price_data.index if start_date <= d <= end_date]
        num_unusual = int(len(trading_days) * 0.2)
        unusual_indices = np.random.choice(len(trading_days), num_unusual, replace=False)
        for idx in unusual_indices:
            unusual_days.add(trading_days[idx])
        
        # Generate chains for each day
        for date in trading_days:
            if date.weekday() >= 5:  # Skip weekends
                continue
            
            # Get spot price
            try:
                spot = float(price_data.loc[date, 'close'])
            except:
                continue
            
            # Determine regime from recent prices
            lookback = price_data.loc[:date].tail(20)
            if len(lookback) >= 10:
                regime = get_regime(lookback['close'].values)
            else:
                regime = "neutral"
            
            # Check for earnings
            days_to_earnings = None
            if earnings_dates:
                future_earnings = [e for e in earnings_dates if e > date]
                if future_earnings:
                    next_earnings = min(future_earnings)
                    days_to_earnings = (next_earnings - date).days
            
            # Check if unusual day
            is_unusual = date in unusual_days
            unusual_direction = None
            if is_unusual:
                # Bias based on next day's return (forward looking for simulation)
                try:
                    next_idx = price_data.index.get_loc(date) + 1
                    if next_idx < len(price_data):
                        next_price = price_data.iloc[next_idx]['close']
                        if next_price > spot * 1.01:
                            unusual_direction = "bullish"
                        elif next_price < spot * 0.99:
                            unusual_direction = "bearish"
                except:
                    pass
            
            # Generate chain
            chain = self.generate_options_chain(
                underlying=underlying,
                spot=spot,
                current_date=date,
                market_regime=regime,
                days_to_earnings=days_to_earnings,
                is_unusual_day=is_unusual,
                unusual_direction=unusual_direction,
            )
            
            # Add unusual flow if applicable
            if is_unusual:
                chain.unusual_flows = self.generate_unusual_flow(
                    chain,
                    num_unusual=np.random.randint(3, 8),
                    bias=unusual_direction or "neutral",
                )
            
            historical_chains[date] = chain
        
        return historical_chains


# ============================================================================
# DEMO / TEST
# ============================================================================

def demo_options_generator():
    """Demonstrate the synthetic options generator."""
    
    print("\n" + "="*80)
    print("  SYNTHETIC OPTIONS DATA GENERATOR - DEMO")
    print("="*80)
    
    generator = SyntheticOptionsGenerator(seed=42)
    
    # Generate a single chain
    current_date = datetime(2024, 1, 15, tzinfo=timezone.utc)
    spot = 450.0
    
    print(f"\nGenerating options chain for SPY @ ${spot}")
    print(f"Date: {current_date.date()}")
    
    chain = generator.generate_options_chain(
        underlying="SPY",
        spot=spot,
        current_date=current_date,
        market_regime="neutral",
        days_to_earnings=None,
        is_unusual_day=True,
        unusual_direction="bullish",
    )
    
    print(f"\n📊 Chain Summary:")
    print(f"  Expirations: {len(chain.expirations)}")
    print(f"  Strikes: {len(chain.strikes)}")
    print(f"  Total Calls: {len(chain.calls)}")
    print(f"  Total Puts: {len(chain.puts)}")
    
    print(f"\n📈 Open Interest:")
    print(f"  Total Call OI: {chain.total_call_oi:,}")
    print(f"  Total Put OI: {chain.total_put_oi:,}")
    print(f"  Put/Call OI Ratio: {chain.put_call_oi_ratio:.2f}")
    
    print(f"\n📉 Volume:")
    print(f"  Total Call Volume: {chain.total_call_volume:,}")
    print(f"  Total Put Volume: {chain.total_put_volume:,}")
    print(f"  Put/Call Ratio: {chain.put_call_ratio:.2f}")
    
    print(f"\n🎯 Key Levels:")
    print(f"  Max Pain: ${chain.max_pain}")
    print(f"  Call Wall: ${chain.call_wall}")
    print(f"  Put Wall: ${chain.put_wall}")
    print(f"  Gamma Flip: ${chain.gamma_flip}")
    print(f"  Dealer Positioning: {chain.dealer_positioning}")
    
    # Show sample ATM options
    print(f"\n📋 Sample ATM Options (nearest expiry):")
    nearest_exp = chain.expirations[0]
    atm_strike = min(chain.strikes, key=lambda x: abs(x - spot))
    
    atm_call = next((c for c in chain.calls if c.strike == atm_strike and c.expiration == nearest_exp), None)
    atm_put = next((p for p in chain.puts if p.strike == atm_strike and p.expiration == nearest_exp), None)
    
    if atm_call:
        print(f"\n  ATM CALL (Strike ${atm_strike}, Exp {nearest_exp.date()}):")
        print(f"    Bid/Ask: ${atm_call.bid:.2f} / ${atm_call.ask:.2f}")
        print(f"    IV: {atm_call.implied_volatility:.1%}")
        print(f"    Delta: {atm_call.delta:.3f}")
        print(f"    Gamma: {atm_call.gamma:.5f}")
        print(f"    Theta: ${atm_call.theta:.3f}/day")
        print(f"    Vega: ${atm_call.vega:.3f}")
        print(f"    OI: {atm_call.open_interest:,}")
        print(f"    Volume: {atm_call.volume:,}")
    
    if atm_put:
        print(f"\n  ATM PUT (Strike ${atm_strike}, Exp {nearest_exp.date()}):")
        print(f"    Bid/Ask: ${atm_put.bid:.2f} / ${atm_put.ask:.2f}")
        print(f"    IV: {atm_put.implied_volatility:.1%}")
        print(f"    Delta: {atm_put.delta:.3f}")
        print(f"    Gamma: {atm_put.gamma:.5f}")
        print(f"    Theta: ${atm_put.theta:.3f}/day")
        print(f"    Vega: ${atm_put.vega:.3f}")
        print(f"    OI: {atm_put.open_interest:,}")
        print(f"    Volume: {atm_put.volume:,}")
    
    # Generate unusual flow
    print(f"\n🔥 Unusual Options Activity:")
    flows = generator.generate_unusual_flow(chain, num_unusual=5, bias="bullish")
    
    for i, flow in enumerate(flows, 1):
        print(f"  {i}. {flow.option_type.upper()} ${flow.strike} exp {flow.expiration.date()}")
        print(f"     Size: {flow.size} | Premium: ${flow.premium:,.0f} | {flow.side.upper()}")
        print(f"     Type: {flow.trade_type} | Sentiment: {flow.sentiment} | Score: {flow.unusual_score:.2f}")
    
    # Show IV surface
    print(f"\n📊 IV Surface (nearest expiry):")
    print(f"  {'Strike':>8} {'Call IV':>10} {'Put IV':>10} {'Skew':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    
    for strike in sorted(chain.strikes)[::2]:  # Every other strike
        call = next((c for c in chain.calls if c.strike == strike and c.expiration == nearest_exp), None)
        put = next((p for p in chain.puts if p.strike == strike and p.expiration == nearest_exp), None)
        
        if call and put:
            skew = put.implied_volatility - call.implied_volatility
            marker = "<<< ATM" if abs(strike - spot) < 5 else ""
            print(f"  ${strike:>7.0f} {call.implied_volatility:>9.1%} {put.implied_volatility:>9.1%} {skew:>+9.1%} {marker}")
    
    print("\n" + "="*80)
    print("  DEMO COMPLETE")
    print("="*80 + "\n")
    
    return chain


if __name__ == "__main__":
    demo_options_generator()
