#!/usr/bin/env python3
"""
GNOSIS Options Backtest Engine - Theoretical Options on Real Stock Data

This engine uses real stock data to generate theoretical options trades using
the full GNOSIS framework:

1. PREDICTIVE ENGINE:
   - Price forecast cones (multi-horizon predictions)
   - Support/Resistance levels from price action
   - Confidence bands around predictions

2. SENTIMENT ENGINE:
   - RSI (momentum oscillator)
   - MACD (trend following + momentum)
   - Stochastic oscillator
   - Multi-timeframe sentiment agreement

3. LIQUIDITY ENGINE:
   - Accumulation/Distribution line
   - Bollinger Bands (volatility + mean reversion)
   - Volume Profile (VAH/VAL/POC)
   - Order flow imbalance

4. PRICE PARTICLE PHYSICS:
   - Mass = Market Cap / Float (determines inertia)
   - Volume = Energy (required to move price)
   - Momentum = Mass × Velocity
   - Resistance as friction zones
   - Support as gravity wells

5. OPTIONS STRATEGY SELECTION:
   - Based on volatility regime: Debit spreads, Credit spreads, Straddles
   - Based on directional confidence: Calls, Puts, Verticals
   - Based on time horizon: Weekly, Monthly expiries
   - Dynamic Greeks management

Author: GNOSIS Trading System
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum
import json
import math

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class VolatilityRegime(Enum):
    """Market volatility regime classification."""
    VERY_LOW = "very_low"      # IV < 15%
    LOW = "low"                # 15% <= IV < 25%
    NORMAL = "normal"          # 25% <= IV < 35%
    HIGH = "high"              # 35% <= IV < 50%
    VERY_HIGH = "very_high"    # IV >= 50%


class OptionsStrategy(Enum):
    """Available options strategies."""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    BULL_PUT_SPREAD = "bull_put_spread"      # Credit spread
    BEAR_CALL_SPREAD = "bear_call_spread"    # Credit spread
    LONG_STRADDLE = "long_straddle"
    LONG_STRANGLE = "long_strangle"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    CALENDAR_SPREAD = "calendar_spread"


class SignalDirection(Enum):
    """Trading signal direction."""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PriceParticle:
    """
    Price modeled as a particle with physics properties.
    
    The price particle has:
    - Mass: Resistance to movement (based on market cap, float, volume)
    - Velocity: Rate of price change
    - Momentum: Mass × Velocity
    - Kinetic Energy: 0.5 × Mass × Velocity²
    - Potential Energy: Distance from key levels (S/R, MAs)
    """
    price: float
    mass: float              # Normalized 0-1 (higher = harder to move)
    velocity: float          # Price change rate (normalized)
    momentum: float          # mass × velocity
    kinetic_energy: float    # 0.5 × mass × velocity²
    potential_energy: float  # Distance from equilibrium (S/R)
    
    # Friction zones
    resistance_friction: float  # How close to resistance (0-1)
    support_gravity: float      # How close to support (0-1)
    
    # Volume energy
    volume_energy: float     # Normalized volume (energy to move price)
    energy_required: float   # Volume needed to break current level
    
    @property
    def total_energy(self) -> float:
        return self.kinetic_energy + self.potential_energy
    
    @property
    def can_break_resistance(self) -> bool:
        """Check if volume energy exceeds required energy to break resistance."""
        return self.volume_energy > self.energy_required * (1 + self.resistance_friction)
    
    @property
    def near_support(self) -> bool:
        """Check if price is in support gravity well."""
        return self.support_gravity > 0.7


@dataclass
class SentimentSignals:
    """Sentiment engine signals (RSI, MACD, etc.)."""
    
    # RSI
    rsi: float = 50.0
    rsi_signal: str = "neutral"  # overbought, oversold, neutral
    rsi_divergence: Optional[str] = None  # bullish_div, bearish_div
    
    # MACD
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_crossover: Optional[str] = None  # bullish_cross, bearish_cross
    
    # Stochastic
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    stoch_signal: str = "neutral"
    
    # Combined sentiment
    combined_score: float = 0.0  # -1 to 1
    confidence: float = 0.0
    direction: SignalDirection = SignalDirection.NEUTRAL


@dataclass
class LiquiditySignals:
    """Liquidity engine signals (Accumulation, Bollinger, etc.)."""
    
    # Accumulation/Distribution
    adl: float = 0.0
    adl_trend: str = "neutral"  # accumulating, distributing
    
    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    bb_percent: float = 0.5  # Price position in BB (0=lower, 1=upper)
    bb_squeeze: bool = False
    
    # Volume Profile
    poc: float = 0.0          # Point of Control
    vah: float = 0.0          # Value Area High
    val: float = 0.0          # Value Area Low
    
    # Order Flow
    buy_pressure: float = 0.5
    sell_pressure: float = 0.5
    flow_imbalance: float = 0.0  # Positive = buying, Negative = selling
    
    # Combined
    combined_score: float = 0.0
    confidence: float = 0.0


@dataclass
class PredictiveCone:
    """Predictive price cone with confidence bands."""
    
    horizon_days: int
    center_price: float
    upper_1sigma: float
    lower_1sigma: float
    upper_2sigma: float
    lower_2sigma: float
    direction_bias: float  # -1 to 1
    confidence: float


@dataclass
class SupportResistance:
    """Support and resistance levels."""
    
    strong_support: List[float] = field(default_factory=list)
    weak_support: List[float] = field(default_factory=list)
    strong_resistance: List[float] = field(default_factory=list)
    weak_resistance: List[float] = field(default_factory=list)
    
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    
    # Key psychological levels
    round_numbers: List[float] = field(default_factory=list)


@dataclass
class GnosisSignal:
    """Combined GNOSIS signal for options trading."""
    
    timestamp: datetime
    symbol: str
    current_price: float
    
    # Components
    particle: PriceParticle
    sentiment: SentimentSignals
    liquidity: LiquiditySignals
    predictive_cone: PredictiveCone
    support_resistance: SupportResistance
    
    # Volatility
    historical_vol: float  # Historical volatility
    implied_vol: float     # Estimated IV
    vol_regime: VolatilityRegime
    
    # Combined Direction
    direction: SignalDirection
    confidence: float
    
    # Recommended Strategy
    recommended_strategy: OptionsStrategy
    strategy_reason: str
    
    # Greeks targets
    target_delta: float
    target_dte: int
    strike_offset_pct: float  # Distance from ATM


@dataclass
class TheoreticalOption:
    """Theoretical option contract."""
    
    symbol: str
    underlying_price: float
    strike: float
    expiry: datetime
    option_type: str  # call, put
    
    # Greeks (theoretical)
    delta: float
    gamma: float
    theta: float
    vega: float
    
    # Pricing
    theoretical_price: float
    bid: float
    ask: float
    iv: float
    
    dte: int


@dataclass
class OptionsPosition:
    """Options position for backtesting."""
    
    position_id: str
    symbol: str
    strategy: OptionsStrategy
    
    entry_date: datetime
    entry_underlying_price: float
    
    # Legs
    legs: List[TheoreticalOption] = field(default_factory=list)
    
    # Position info
    quantity: int = 1
    entry_cost: float = 0.0  # Debit or credit
    max_loss: float = 0.0
    max_profit: float = 0.0
    breakeven_prices: List[float] = field(default_factory=list)
    
    # Exit
    exit_date: Optional[datetime] = None
    exit_price: float = 0.0
    exit_underlying_price: float = 0.0
    exit_reason: str = ""
    
    # P&L
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    
    # Signal context
    entry_confidence: float = 0.0
    entry_direction: SignalDirection = SignalDirection.NEUTRAL
    vol_regime_at_entry: VolatilityRegime = VolatilityRegime.NORMAL


# ============================================================================
# PHYSICS ENGINE - Price Particle Dynamics
# ============================================================================

class PricePhysicsEngine:
    """
    Models price as a particle with mass, momentum, and energy.
    
    Key concepts:
    - Mass: Inversely proportional to volatility and liquidity
    - Velocity: Rate of price change
    - Momentum: Persistence of price movement
    - Energy: Volume required to move price
    - Friction: Resistance from S/R levels
    - Gravity: Attraction to support levels
    """
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
    
    def calculate_particle(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        support_levels: List[float],
        resistance_levels: List[float],
    ) -> PriceParticle:
        """Calculate price particle properties."""
        
        current_price = prices.iloc[-1]
        
        # Calculate mass (based on volatility - higher vol = lower mass)
        returns = prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        
        # Normalized mass (0-1, higher = more inertia)
        # Low volatility = high mass (hard to move)
        mass = 1.0 - min(volatility / 0.5, 1.0)  # 50% vol = minimum mass
        
        # Calculate velocity (rate of change)
        short_ma = prices.tail(5).mean()
        long_ma = prices.tail(20).mean()
        velocity = (short_ma - long_ma) / long_ma if long_ma > 0 else 0
        velocity = np.clip(velocity, -0.1, 0.1) / 0.1  # Normalize to -1 to 1
        
        # Momentum = mass × velocity
        momentum = mass * velocity
        
        # Kinetic energy = 0.5 × mass × velocity²
        kinetic_energy = 0.5 * mass * velocity ** 2
        
        # Calculate potential energy (distance from key levels)
        nearest_support = min(support_levels, default=current_price * 0.9)
        nearest_resistance = max(resistance_levels, default=current_price * 1.1)
        
        # Distance to equilibrium (mid-point of S/R)
        equilibrium = (nearest_support + nearest_resistance) / 2
        distance = abs(current_price - equilibrium) / equilibrium
        potential_energy = distance * mass
        
        # Friction from resistance
        if nearest_resistance > current_price:
            resistance_friction = 1.0 - (nearest_resistance - current_price) / current_price
        else:
            resistance_friction = 1.0
        resistance_friction = max(0, min(1, resistance_friction))
        
        # Gravity from support
        if current_price > nearest_support:
            support_gravity = 1.0 - (current_price - nearest_support) / current_price
        else:
            support_gravity = 1.0
        support_gravity = max(0, min(1, support_gravity))
        
        # Volume energy
        avg_volume = volumes.mean()
        current_volume = volumes.iloc[-1]
        volume_energy = current_volume / avg_volume if avg_volume > 0 else 1.0
        volume_energy = min(volume_energy, 5.0) / 5.0  # Normalize to 0-1
        
        # Energy required to break level (based on historical tests)
        # More tests of a level = more energy required
        atr = (highs - lows).tail(14).mean()
        energy_required = atr / current_price * 10  # Normalized
        
        return PriceParticle(
            price=current_price,
            mass=mass,
            velocity=velocity,
            momentum=momentum,
            kinetic_energy=kinetic_energy,
            potential_energy=potential_energy,
            resistance_friction=resistance_friction,
            support_gravity=support_gravity,
            volume_energy=volume_energy,
            energy_required=energy_required,
        )


# ============================================================================
# SENTIMENT ENGINE - Technical Indicators
# ============================================================================

class SentimentEngine:
    """
    Generates sentiment signals from technical indicators.
    
    Indicators:
    - RSI (14-period)
    - MACD (12, 26, 9)
    - Stochastic (14, 3, 3)
    """
    
    def __init__(self):
        pass
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> Tuple[float, str, Optional[str]]:
        """Calculate RSI and interpret signal."""
        
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gain = gains.rolling(period).mean()
        avg_loss = losses.rolling(period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        
        # Interpret
        if current_rsi >= 70:
            signal = "overbought"
        elif current_rsi <= 30:
            signal = "oversold"
        else:
            signal = "neutral"
        
        # Check for divergence
        divergence = None
        if len(prices) >= 20:
            price_trend = prices.iloc[-1] > prices.iloc[-10]
            rsi_trend = rsi.iloc[-1] > rsi.iloc[-10] if not pd.isna(rsi.iloc[-10]) else True
            
            if price_trend and not rsi_trend and current_rsi > 50:
                divergence = "bearish_div"
            elif not price_trend and rsi_trend and current_rsi < 50:
                divergence = "bullish_div"
        
        return current_rsi, signal, divergence
    
    def calculate_macd(
        self, 
        prices: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[float, float, float, Optional[str]]:
        """Calculate MACD and interpret signal."""
        
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_hist = histogram.iloc[-1]
        
        # Check for crossover
        crossover = None
        if len(macd_line) >= 2:
            prev_macd = macd_line.iloc[-2]
            prev_signal = signal_line.iloc[-2]
            
            if prev_macd <= prev_signal and current_macd > current_signal:
                crossover = "bullish_cross"
            elif prev_macd >= prev_signal and current_macd < current_signal:
                crossover = "bearish_cross"
        
        return current_macd, current_signal, current_hist, crossover
    
    def calculate_stochastic(
        self, 
        highs: pd.Series, 
        lows: pd.Series, 
        closes: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[float, float, str]:
        """Calculate Stochastic oscillator."""
        
        lowest_low = lows.rolling(k_period).min()
        highest_high = highs.rolling(k_period).max()
        
        k_line = 100 * (closes - lowest_low) / (highest_high - lowest_low + 1e-10)
        d_line = k_line.rolling(d_period).mean()
        
        current_k = k_line.iloc[-1] if not pd.isna(k_line.iloc[-1]) else 50.0
        current_d = d_line.iloc[-1] if not pd.isna(d_line.iloc[-1]) else 50.0
        
        # Interpret
        if current_k >= 80:
            signal = "overbought"
        elif current_k <= 20:
            signal = "oversold"
        else:
            signal = "neutral"
        
        return current_k, current_d, signal
    
    def generate_signals(
        self,
        prices: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
    ) -> SentimentSignals:
        """Generate combined sentiment signals."""
        
        signals = SentimentSignals()
        
        # RSI
        signals.rsi, signals.rsi_signal, signals.rsi_divergence = self.calculate_rsi(prices)
        
        # MACD
        signals.macd, signals.macd_signal, signals.macd_histogram, signals.macd_crossover = \
            self.calculate_macd(prices)
        
        # Stochastic
        signals.stoch_k, signals.stoch_d, signals.stoch_signal = \
            self.calculate_stochastic(highs, lows, prices)
        
        # Calculate combined score (-1 to 1)
        score = 0.0
        weights = 0.0
        
        # RSI contribution
        rsi_score = (signals.rsi - 50) / 50  # -1 to 1
        if signals.rsi_signal == "overbought":
            rsi_score = -0.5  # Likely reversal
        elif signals.rsi_signal == "oversold":
            rsi_score = 0.5  # Likely reversal
        score += rsi_score * 0.3
        weights += 0.3
        
        # MACD contribution
        macd_score = np.sign(signals.macd_histogram) * min(abs(signals.macd_histogram) / 2, 1)
        if signals.macd_crossover == "bullish_cross":
            macd_score = 0.8
        elif signals.macd_crossover == "bearish_cross":
            macd_score = -0.8
        score += macd_score * 0.4
        weights += 0.4
        
        # Stochastic contribution
        stoch_score = (signals.stoch_k - 50) / 50
        if signals.stoch_signal == "overbought":
            stoch_score = -0.3
        elif signals.stoch_signal == "oversold":
            stoch_score = 0.3
        score += stoch_score * 0.3
        weights += 0.3
        
        signals.combined_score = score / weights if weights > 0 else 0.0
        
        # Confidence based on agreement
        agreement = 0
        if signals.rsi > 50 and signals.macd_histogram > 0:
            agreement += 1
        elif signals.rsi < 50 and signals.macd_histogram < 0:
            agreement += 1
        
        if signals.stoch_k > 50 and signals.macd_histogram > 0:
            agreement += 1
        elif signals.stoch_k < 50 and signals.macd_histogram < 0:
            agreement += 1
        
        signals.confidence = agreement / 2.0
        
        # Direction
        if signals.combined_score > 0.5:
            signals.direction = SignalDirection.STRONG_BULLISH
        elif signals.combined_score > 0.2:
            signals.direction = SignalDirection.BULLISH
        elif signals.combined_score < -0.5:
            signals.direction = SignalDirection.STRONG_BEARISH
        elif signals.combined_score < -0.2:
            signals.direction = SignalDirection.BEARISH
        else:
            signals.direction = SignalDirection.NEUTRAL
        
        return signals


# ============================================================================
# LIQUIDITY ENGINE - Volume & Accumulation Analysis
# ============================================================================

class LiquidityEngine:
    """
    Generates liquidity signals from volume-based indicators.
    
    Indicators:
    - Accumulation/Distribution Line (ADL)
    - Bollinger Bands
    - Volume Profile (simplified)
    - Order Flow Imbalance
    """
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0):
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def calculate_adl(
        self,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        volumes: pd.Series,
    ) -> Tuple[float, str]:
        """Calculate Accumulation/Distribution Line."""
        
        # Money Flow Multiplier
        mf_mult = ((closes - lows) - (highs - closes)) / (highs - lows + 1e-10)
        
        # Money Flow Volume
        mf_volume = mf_mult * volumes
        
        # Accumulation/Distribution Line
        adl = mf_volume.cumsum()
        
        current_adl = adl.iloc[-1]
        
        # Determine trend
        if len(adl) >= 10:
            adl_ma = adl.rolling(10).mean().iloc[-1]
            if current_adl > adl_ma:
                trend = "accumulating"
            else:
                trend = "distributing"
        else:
            trend = "neutral"
        
        return current_adl, trend
    
    def calculate_bollinger(
        self,
        prices: pd.Series,
    ) -> Tuple[float, float, float, float, float, bool]:
        """Calculate Bollinger Bands."""
        
        middle = prices.rolling(self.bb_period).mean()
        std = prices.rolling(self.bb_period).std()
        
        upper = middle + (std * self.bb_std)
        lower = middle - (std * self.bb_std)
        
        current_middle = middle.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        
        # Width
        width = (current_upper - current_lower) / current_middle if current_middle > 0 else 0
        
        # Percent B
        current_price = prices.iloc[-1]
        bb_pct = (current_price - current_lower) / (current_upper - current_lower + 1e-10)
        
        # Squeeze detection (narrow bands)
        avg_width = ((upper - lower) / middle).tail(50).mean()
        squeeze = width < avg_width * 0.75
        
        return current_upper, current_middle, current_lower, width, bb_pct, squeeze
    
    def calculate_volume_profile(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        bins: int = 20,
    ) -> Tuple[float, float, float]:
        """Calculate simplified Volume Profile (POC, VAH, VAL)."""
        
        min_price = prices.min()
        max_price = prices.max()
        
        # Create price bins
        bin_edges = np.linspace(min_price, max_price, bins + 1)
        
        # Accumulate volume in each bin
        volume_by_bin = np.zeros(bins)
        for i in range(len(prices)):
            price = prices.iloc[i]
            volume = volumes.iloc[i]
            bin_idx = min(int((price - min_price) / (max_price - min_price) * bins), bins - 1)
            volume_by_bin[bin_idx] += volume
        
        # POC = bin with max volume
        poc_idx = np.argmax(volume_by_bin)
        poc = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
        
        # Value Area (70% of volume)
        total_vol = volume_by_bin.sum()
        target_vol = total_vol * 0.7
        
        accumulated = volume_by_bin[poc_idx]
        low_idx = poc_idx
        high_idx = poc_idx
        
        while accumulated < target_vol:
            # Expand to include next highest volume bin
            low_vol = volume_by_bin[low_idx - 1] if low_idx > 0 else 0
            high_vol = volume_by_bin[high_idx + 1] if high_idx < bins - 1 else 0
            
            if low_vol >= high_vol and low_idx > 0:
                low_idx -= 1
                accumulated += low_vol
            elif high_idx < bins - 1:
                high_idx += 1
                accumulated += high_vol
            else:
                break
        
        val = (bin_edges[low_idx] + bin_edges[low_idx + 1]) / 2
        vah = (bin_edges[high_idx] + bin_edges[high_idx + 1]) / 2
        
        return poc, vah, val
    
    def calculate_order_flow(
        self,
        opens: pd.Series,
        closes: pd.Series,
        volumes: pd.Series,
    ) -> Tuple[float, float, float]:
        """Calculate order flow imbalance."""
        
        # Classify volume as buying or selling
        buying_volume = volumes.where(closes > opens, 0)
        selling_volume = volumes.where(closes < opens, 0)
        
        recent_buying = buying_volume.tail(10).sum()
        recent_selling = selling_volume.tail(10).sum()
        total_recent = recent_buying + recent_selling
        
        buy_pressure = recent_buying / total_recent if total_recent > 0 else 0.5
        sell_pressure = recent_selling / total_recent if total_recent > 0 else 0.5
        
        imbalance = buy_pressure - sell_pressure  # -1 to 1
        
        return buy_pressure, sell_pressure, imbalance
    
    def generate_signals(
        self,
        opens: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        volumes: pd.Series,
    ) -> LiquiditySignals:
        """Generate combined liquidity signals."""
        
        signals = LiquiditySignals()
        
        # ADL
        signals.adl, signals.adl_trend = self.calculate_adl(highs, lows, closes, volumes)
        
        # Bollinger Bands
        signals.bb_upper, signals.bb_middle, signals.bb_lower, \
        signals.bb_width, signals.bb_percent, signals.bb_squeeze = \
            self.calculate_bollinger(closes)
        
        # Volume Profile
        signals.poc, signals.vah, signals.val = \
            self.calculate_volume_profile(closes, volumes)
        
        # Order Flow
        signals.buy_pressure, signals.sell_pressure, signals.flow_imbalance = \
            self.calculate_order_flow(opens, closes, volumes)
        
        # Combined score
        score = 0.0
        
        # ADL trend
        if signals.adl_trend == "accumulating":
            score += 0.3
        elif signals.adl_trend == "distributing":
            score -= 0.3
        
        # BB position
        if signals.bb_percent > 0.8:
            score -= 0.2  # Near upper band = potential reversal
        elif signals.bb_percent < 0.2:
            score += 0.2  # Near lower band = potential bounce
        
        # Flow imbalance
        score += signals.flow_imbalance * 0.5
        
        signals.combined_score = np.clip(score, -1, 1)
        
        # Confidence
        signals.confidence = abs(signals.flow_imbalance) * 0.5 + \
                            (1 if signals.bb_squeeze else 0) * 0.3 + \
                            0.2
        
        return signals


# ============================================================================
# PREDICTIVE ENGINE - Price Forecasting
# ============================================================================

class PredictiveEngine:
    """
    Generates price forecast cones using statistical methods.
    
    Methods:
    - Monte Carlo simulation
    - Mean-reverting models
    - Volatility-adjusted projections
    """
    
    def __init__(self, simulations: int = 1000):
        self.simulations = simulations
    
    def calculate_forecast_cone(
        self,
        prices: pd.Series,
        horizon_days: int = 5,
    ) -> PredictiveCone:
        """Generate price forecast cone."""
        
        current_price = prices.iloc[-1]
        
        # Calculate returns and volatility
        returns = prices.pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        
        # Monte Carlo simulation
        dt = 1.0  # Daily
        sqrt_dt = np.sqrt(dt)
        
        final_prices = []
        for _ in range(self.simulations):
            price = current_price
            for _ in range(horizon_days):
                # Geometric Brownian Motion
                random_shock = np.random.normal()
                price = price * np.exp((mu - 0.5 * sigma**2) * dt + sigma * sqrt_dt * random_shock)
            final_prices.append(price)
        
        final_prices = np.array(final_prices)
        
        # Calculate confidence bands
        center = np.median(final_prices)
        upper_1sigma = np.percentile(final_prices, 84.1)
        lower_1sigma = np.percentile(final_prices, 15.9)
        upper_2sigma = np.percentile(final_prices, 97.7)
        lower_2sigma = np.percentile(final_prices, 2.3)
        
        # Direction bias
        up_prob = (final_prices > current_price).mean()
        direction_bias = up_prob * 2 - 1  # -1 to 1
        
        # Confidence based on cone width
        width_pct = (upper_1sigma - lower_1sigma) / center
        confidence = max(0, 1 - width_pct)  # Narrower = more confident
        
        return PredictiveCone(
            horizon_days=horizon_days,
            center_price=center,
            upper_1sigma=upper_1sigma,
            lower_1sigma=lower_1sigma,
            upper_2sigma=upper_2sigma,
            lower_2sigma=lower_2sigma,
            direction_bias=direction_bias,
            confidence=confidence,
        )
    
    def calculate_support_resistance(
        self,
        prices: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        lookback: int = 50,
    ) -> SupportResistance:
        """Calculate support and resistance levels."""
        
        sr = SupportResistance()
        current_price = prices.iloc[-1]
        
        # Find pivot points
        for i in range(2, min(lookback, len(prices) - 2)):
            # Pivot high (resistance)
            if highs.iloc[-i] > highs.iloc[-i-1] and highs.iloc[-i] > highs.iloc[-i+1] and \
               highs.iloc[-i] > highs.iloc[-i-2] and highs.iloc[-i] > highs.iloc[-i+2]:
                level = highs.iloc[-i]
                if level > current_price:
                    sr.strong_resistance.append(level)
                else:
                    sr.strong_support.append(level)
            
            # Pivot low (support)
            if lows.iloc[-i] < lows.iloc[-i-1] and lows.iloc[-i] < lows.iloc[-i+1] and \
               lows.iloc[-i] < lows.iloc[-i-2] and lows.iloc[-i] < lows.iloc[-i+2]:
                level = lows.iloc[-i]
                if level < current_price:
                    sr.strong_support.append(level)
                else:
                    sr.weak_resistance.append(level)
        
        # Round numbers
        base = 10 ** (len(str(int(current_price))) - 1)
        for mult in range(-3, 4):
            round_level = round(current_price / base) * base + mult * base / 2
            sr.round_numbers.append(round_level)
        
        # Nearest levels
        if sr.strong_support:
            sr.nearest_support = max(sr.strong_support)
        else:
            sr.nearest_support = current_price * 0.95
        
        if sr.strong_resistance:
            sr.nearest_resistance = min(sr.strong_resistance)
        else:
            sr.nearest_resistance = current_price * 1.05
        
        return sr


# ============================================================================
# OPTIONS PRICING ENGINE
# ============================================================================

class OptionsEngine:
    """
    Black-Scholes options pricing and Greeks calculation.
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
    
    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
    
    def black_scholes(
        self,
        spot: float,
        strike: float,
        dte: int,
        vol: float,
        option_type: str,
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate Black-Scholes price and Greeks.
        
        Returns: (price, delta, gamma, theta, vega)
        """
        
        T = dte / 365.0
        if T <= 0:
            T = 1 / 365.0
        
        r = self.risk_free_rate
        
        d1 = (math.log(spot / strike) + (r + 0.5 * vol ** 2) * T) / (vol * math.sqrt(T))
        d2 = d1 - vol * math.sqrt(T)
        
        if option_type.lower() == "call":
            price = spot * self._norm_cdf(d1) - strike * math.exp(-r * T) * self._norm_cdf(d2)
            delta = self._norm_cdf(d1)
        else:  # put
            price = strike * math.exp(-r * T) * self._norm_cdf(-d2) - spot * self._norm_cdf(-d1)
            delta = self._norm_cdf(d1) - 1
        
        # Gamma
        gamma = self._norm_pdf(d1) / (spot * vol * math.sqrt(T))
        
        # Theta (per day)
        theta = -(spot * self._norm_pdf(d1) * vol) / (2 * math.sqrt(T)) - \
                r * strike * math.exp(-r * T) * self._norm_cdf(d2 if option_type.lower() == "call" else -d2)
        theta = theta / 365
        
        # Vega (per 1% vol change)
        vega = spot * math.sqrt(T) * self._norm_pdf(d1) / 100
        
        return price, delta, gamma, theta, vega
    
    def create_option(
        self,
        symbol: str,
        spot: float,
        strike: float,
        expiry: datetime,
        option_type: str,
        iv: float,
        current_date: datetime,
    ) -> TheoreticalOption:
        """Create a theoretical option contract."""
        
        dte = max(1, (expiry - current_date).days)
        
        price, delta, gamma, theta, vega = self.black_scholes(
            spot, strike, dte, iv, option_type
        )
        
        # Add bid-ask spread (0.5-2% based on liquidity)
        spread = price * 0.01
        bid = max(0.01, price - spread / 2)
        ask = price + spread / 2
        
        return TheoreticalOption(
            symbol=symbol,
            underlying_price=spot,
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            theoretical_price=price,
            bid=bid,
            ask=ask,
            iv=iv,
            dte=dte,
        )


# ============================================================================
# STRATEGY SELECTOR
# ============================================================================

class StrategySelector:
    """
    Selects optimal options strategy based on market conditions.
    """
    
    def select_strategy(
        self,
        signal: GnosisSignal,
    ) -> Tuple[OptionsStrategy, str, float, int, float]:
        """
        Select optimal options strategy.
        
        Returns: (strategy, reason, target_delta, target_dte, strike_offset)
        """
        
        direction = signal.direction
        vol_regime = signal.vol_regime
        confidence = signal.confidence
        
        # High confidence directional plays
        if confidence > 0.7:
            if direction == SignalDirection.STRONG_BULLISH:
                if vol_regime in [VolatilityRegime.HIGH, VolatilityRegime.VERY_HIGH]:
                    return (
                        OptionsStrategy.BULL_CALL_SPREAD,
                        "High IV + Strong bullish = Debit spread to limit vega risk",
                        0.6, 30, 0.02
                    )
                else:
                    return (
                        OptionsStrategy.LONG_CALL,
                        "Low IV + Strong bullish = Long call for unlimited upside",
                        0.5, 30, 0.0
                    )
            
            elif direction == SignalDirection.STRONG_BEARISH:
                if vol_regime in [VolatilityRegime.HIGH, VolatilityRegime.VERY_HIGH]:
                    return (
                        OptionsStrategy.BEAR_PUT_SPREAD,
                        "High IV + Strong bearish = Debit spread to limit vega risk",
                        -0.6, 30, -0.02
                    )
                else:
                    return (
                        OptionsStrategy.LONG_PUT,
                        "Low IV + Strong bearish = Long put for protection",
                        -0.5, 30, 0.0
                    )
        
        # Moderate confidence plays
        if confidence > 0.4:
            if direction in [SignalDirection.BULLISH, SignalDirection.STRONG_BULLISH]:
                if vol_regime in [VolatilityRegime.HIGH, VolatilityRegime.VERY_HIGH]:
                    return (
                        OptionsStrategy.BULL_PUT_SPREAD,
                        "High IV + Bullish = Sell put spread for premium",
                        0.3, 45, -0.05
                    )
                else:
                    return (
                        OptionsStrategy.BULL_CALL_SPREAD,
                        "Normal IV + Bullish = Call spread for defined risk",
                        0.5, 30, 0.02
                    )
            
            elif direction in [SignalDirection.BEARISH, SignalDirection.STRONG_BEARISH]:
                if vol_regime in [VolatilityRegime.HIGH, VolatilityRegime.VERY_HIGH]:
                    return (
                        OptionsStrategy.BEAR_CALL_SPREAD,
                        "High IV + Bearish = Sell call spread for premium",
                        -0.3, 45, 0.05
                    )
                else:
                    return (
                        OptionsStrategy.BEAR_PUT_SPREAD,
                        "Normal IV + Bearish = Put spread for defined risk",
                        -0.5, 30, -0.02
                    )
        
        # Low confidence / Neutral = Sell premium or play volatility
        if vol_regime in [VolatilityRegime.HIGH, VolatilityRegime.VERY_HIGH]:
            return (
                OptionsStrategy.IRON_CONDOR,
                "High IV + Neutral = Iron condor to collect premium",
                0.0, 45, 0.0
            )
        elif signal.liquidity.bb_squeeze:
            return (
                OptionsStrategy.LONG_STRADDLE,
                "BB Squeeze detected = Long straddle for breakout",
                0.0, 30, 0.0
            )
        else:
            return (
                OptionsStrategy.IRON_BUTTERFLY,
                "Low IV + Neutral = Iron butterfly for premium",
                0.0, 30, 0.0
            )


# ============================================================================
# GNOSIS OPTIONS BACKTEST ENGINE
# ============================================================================

class GnosisOptionsBacktestEngine:
    """
    Main backtest engine for GNOSIS options strategies.
    """
    
    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100_000.0,
        max_position_size: float = 0.10,
        max_positions: int = 5,
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        
        # Engines
        self.physics_engine = PricePhysicsEngine()
        self.sentiment_engine = SentimentEngine()
        self.liquidity_engine = LiquidityEngine()
        self.predictive_engine = PredictiveEngine()
        self.options_engine = OptionsEngine()
        self.strategy_selector = StrategySelector()
        
        # State
        self.capital = initial_capital
        self.positions: List[OptionsPosition] = []
        self.closed_positions: List[OptionsPosition] = []
        self.equity_curve: List[Dict] = []
        self.trade_counter = 0
        
        logger.info(
            f"GnosisOptionsBacktestEngine initialized | "
            f"symbols={symbols} | "
            f"period={start_date} to {end_date}"
        )
    
    def _fetch_data(self, symbol: str) -> pd.DataFrame:
        """Fetch historical data for a symbol."""
        
        # Try Massive.com first
        try:
            from config.credentials import get_massive_api_keys
            from massive import RESTClient
            
            primary, secondary = get_massive_api_keys()
            api_key = primary or secondary
            
            if api_key:
                client = RESTClient(api_key=api_key)
                
                aggs = list(client.get_aggs(
                    ticker=symbol,
                    multiplier=1,
                    timespan="day",
                    from_=self.start_date,
                    to=self.end_date,
                    adjusted=True,
                    limit=50000,
                ))
                
                if aggs:
                    data = []
                    for agg in aggs:
                        data.append({
                            'timestamp': datetime.fromtimestamp(agg.timestamp / 1000, tz=timezone.utc),
                            'open': float(agg.open),
                            'high': float(agg.high),
                            'low': float(agg.low),
                            'close': float(agg.close),
                            'volume': float(agg.volume),
                        })
                    
                    df = pd.DataFrame(data)
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    logger.info(f"Fetched {len(df)} daily bars for {symbol} from Massive.com")
                    return df
                    
        except Exception as e:
            logger.warning(f"Massive.com fetch failed for {symbol}: {e}")
        
        # Fallback to Alpaca
        try:
            from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
            
            adapter = AlpacaMarketDataAdapter()
            start = datetime.strptime(self.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end = datetime.strptime(self.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            
            bars = adapter.get_bars(symbol=symbol, start=start, end=end, timeframe="1Day")
            
            if bars:
                df = pd.DataFrame([
                    {
                        'timestamp': bar.timestamp,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume,
                    }
                    for bar in bars
                ])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                logger.info(f"Fetched {len(df)} daily bars for {symbol} from Alpaca")
                return df
                
        except Exception as e:
            logger.error(f"Alpaca fetch failed for {symbol}: {e}")
        
        raise ValueError(f"Could not fetch data for {symbol}")
    
    def _calculate_iv(self, prices: pd.Series) -> Tuple[float, VolatilityRegime]:
        """Calculate implied volatility estimate from historical volatility."""
        
        returns = prices.pct_change().dropna()
        hv = returns.std() * np.sqrt(252)
        
        # Add typical IV premium
        iv = hv * 1.2
        
        # Classify regime
        if iv < 0.15:
            regime = VolatilityRegime.VERY_LOW
        elif iv < 0.25:
            regime = VolatilityRegime.LOW
        elif iv < 0.35:
            regime = VolatilityRegime.NORMAL
        elif iv < 0.50:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.VERY_HIGH
        
        return iv, regime
    
    def _generate_gnosis_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        idx: int,
    ) -> GnosisSignal:
        """Generate comprehensive GNOSIS signal."""
        
        # Get data window
        window = min(50, idx)
        data = df.iloc[max(0, idx - window):idx + 1]
        
        timestamp = data['timestamp'].iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Calculate IV and regime
        hv, vol_regime = self._calculate_iv(data['close'])
        iv = hv * 1.2  # Simple IV estimate
        
        # Calculate support/resistance first
        sr = self.predictive_engine.calculate_support_resistance(
            data['close'], data['high'], data['low']
        )
        
        # Price particle physics
        particle = self.physics_engine.calculate_particle(
            data['close'], data['volume'],
            data['high'], data['low'],
            sr.strong_support, sr.strong_resistance
        )
        
        # Sentiment signals
        sentiment = self.sentiment_engine.generate_signals(
            data['close'], data['high'], data['low']
        )
        
        # Liquidity signals
        liquidity = self.liquidity_engine.generate_signals(
            data['open'], data['high'], data['low'],
            data['close'], data['volume']
        )
        
        # Predictive cone
        cone = self.predictive_engine.calculate_forecast_cone(
            data['close'], horizon_days=5
        )
        
        # Combine for overall direction
        combined_score = (
            sentiment.combined_score * 0.35 +
            liquidity.combined_score * 0.25 +
            cone.direction_bias * 0.25 +
            particle.momentum * 0.15
        )
        
        # Adjust for physics
        if particle.can_break_resistance and combined_score > 0:
            combined_score += 0.1
        if particle.near_support and combined_score < 0:
            combined_score = combined_score * 0.5  # Support likely to hold
        
        # Determine direction
        if combined_score > 0.4:
            direction = SignalDirection.STRONG_BULLISH
        elif combined_score > 0.15:
            direction = SignalDirection.BULLISH
        elif combined_score < -0.4:
            direction = SignalDirection.STRONG_BEARISH
        elif combined_score < -0.15:
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL
        
        # Confidence
        confidence = (
            abs(combined_score) * 0.3 +
            sentiment.confidence * 0.25 +
            liquidity.confidence * 0.25 +
            cone.confidence * 0.2
        )
        
        # Select strategy
        signal = GnosisSignal(
            timestamp=timestamp,
            symbol=symbol,
            current_price=current_price,
            particle=particle,
            sentiment=sentiment,
            liquidity=liquidity,
            predictive_cone=cone,
            support_resistance=sr,
            historical_vol=hv,
            implied_vol=iv,
            vol_regime=vol_regime,
            direction=direction,
            confidence=confidence,
            recommended_strategy=OptionsStrategy.LONG_CALL,  # Placeholder
            strategy_reason="",
            target_delta=0.5,
            target_dte=30,
            strike_offset_pct=0.0,
        )
        
        # Get strategy recommendation
        strategy, reason, delta, dte, offset = self.strategy_selector.select_strategy(signal)
        signal.recommended_strategy = strategy
        signal.strategy_reason = reason
        signal.target_delta = delta
        signal.target_dte = dte
        signal.strike_offset_pct = offset
        
        return signal
    
    def _create_position(
        self,
        signal: GnosisSignal,
    ) -> OptionsPosition:
        """Create options position based on signal."""
        
        self.trade_counter += 1
        
        # Calculate expiry
        expiry = signal.timestamp + timedelta(days=signal.target_dte)
        
        # Calculate strike
        strike = signal.current_price * (1 + signal.strike_offset_pct)
        strike = round(strike, 0)  # Round to whole number
        
        position = OptionsPosition(
            position_id=f"OPT{self.trade_counter:05d}",
            symbol=signal.symbol,
            strategy=signal.recommended_strategy,
            entry_date=signal.timestamp,
            entry_underlying_price=signal.current_price,
            entry_confidence=signal.confidence,
            entry_direction=signal.direction,
            vol_regime_at_entry=signal.vol_regime,
        )
        
        # Create option legs based on strategy
        if signal.recommended_strategy == OptionsStrategy.LONG_CALL:
            call = self.options_engine.create_option(
                signal.symbol, signal.current_price, strike, expiry,
                "call", signal.implied_vol, signal.timestamp
            )
            position.legs = [call]
            position.entry_cost = call.ask  # Pay ask
            position.max_loss = call.ask
            position.max_profit = float('inf')
            position.breakeven_prices = [strike + call.ask]
        
        elif signal.recommended_strategy == OptionsStrategy.LONG_PUT:
            put = self.options_engine.create_option(
                signal.symbol, signal.current_price, strike, expiry,
                "put", signal.implied_vol, signal.timestamp
            )
            position.legs = [put]
            position.entry_cost = put.ask
            position.max_loss = put.ask
            position.max_profit = strike - put.ask
            position.breakeven_prices = [strike - put.ask]
        
        elif signal.recommended_strategy == OptionsStrategy.BULL_CALL_SPREAD:
            # Buy lower strike call, sell higher strike call
            lower_strike = strike
            upper_strike = strike * 1.05
            
            long_call = self.options_engine.create_option(
                signal.symbol, signal.current_price, lower_strike, expiry,
                "call", signal.implied_vol, signal.timestamp
            )
            short_call = self.options_engine.create_option(
                signal.symbol, signal.current_price, upper_strike, expiry,
                "call", signal.implied_vol, signal.timestamp
            )
            
            position.legs = [long_call, short_call]
            position.entry_cost = long_call.ask - short_call.bid
            position.max_loss = position.entry_cost
            position.max_profit = (upper_strike - lower_strike) - position.entry_cost
            position.breakeven_prices = [lower_strike + position.entry_cost]
        
        elif signal.recommended_strategy == OptionsStrategy.BEAR_PUT_SPREAD:
            # Buy higher strike put, sell lower strike put
            higher_strike = strike
            lower_strike = strike * 0.95
            
            long_put = self.options_engine.create_option(
                signal.symbol, signal.current_price, higher_strike, expiry,
                "put", signal.implied_vol, signal.timestamp
            )
            short_put = self.options_engine.create_option(
                signal.symbol, signal.current_price, lower_strike, expiry,
                "put", signal.implied_vol, signal.timestamp
            )
            
            position.legs = [long_put, short_put]
            position.entry_cost = long_put.ask - short_put.bid
            position.max_loss = position.entry_cost
            position.max_profit = (higher_strike - lower_strike) - position.entry_cost
            position.breakeven_prices = [higher_strike - position.entry_cost]
        
        elif signal.recommended_strategy == OptionsStrategy.IRON_CONDOR:
            # Sell OTM put spread + sell OTM call spread
            put_short = strike * 0.95
            put_long = strike * 0.90
            call_short = strike * 1.05
            call_long = strike * 1.10
            
            long_put = self.options_engine.create_option(
                signal.symbol, signal.current_price, put_long, expiry,
                "put", signal.implied_vol, signal.timestamp
            )
            short_put = self.options_engine.create_option(
                signal.symbol, signal.current_price, put_short, expiry,
                "put", signal.implied_vol, signal.timestamp
            )
            short_call = self.options_engine.create_option(
                signal.symbol, signal.current_price, call_short, expiry,
                "call", signal.implied_vol, signal.timestamp
            )
            long_call = self.options_engine.create_option(
                signal.symbol, signal.current_price, call_long, expiry,
                "call", signal.implied_vol, signal.timestamp
            )
            
            position.legs = [long_put, short_put, short_call, long_call]
            credit = (short_put.bid + short_call.bid) - (long_put.ask + long_call.ask)
            position.entry_cost = -credit  # Negative = credit received
            position.max_profit = credit
            position.max_loss = (put_short - put_long) - credit
            position.breakeven_prices = [put_short - credit, call_short + credit]
        
        else:
            # Default to long call
            call = self.options_engine.create_option(
                signal.symbol, signal.current_price, strike, expiry,
                "call", signal.implied_vol, signal.timestamp
            )
            position.legs = [call]
            position.entry_cost = call.ask
            position.max_loss = call.ask
        
        # Position size (based on max loss)
        max_risk = self.capital * self.max_position_size
        if position.max_loss > 0:
            position.quantity = max(1, int(max_risk / (position.max_loss * 100)))
        else:
            position.quantity = max(1, int(max_risk / 500))  # Credit spread default
        
        return position
    
    def _evaluate_position(
        self,
        position: OptionsPosition,
        current_price: float,
        current_date: datetime,
        iv: float,
    ) -> Tuple[float, bool, str]:
        """Evaluate position P&L and check for exit."""
        
        # Calculate time remaining
        if position.legs:
            dte = max(1, (position.legs[0].expiry - current_date).days)
        else:
            dte = 1
        
        # Reprice options
        total_value = 0.0
        for leg in position.legs:
            price, _, _, _, _ = self.options_engine.black_scholes(
                current_price, leg.strike, dte, iv, leg.option_type
            )
            
            # Long or short?
            if leg.delta > 0:  # Long call or short put
                total_value += price
            else:  # Long put or short call
                total_value += price
        
        # Calculate P&L
        if position.entry_cost > 0:  # Debit trade
            pnl = (total_value - position.entry_cost) * 100 * position.quantity
            pnl_pct = (total_value / position.entry_cost - 1) if position.entry_cost > 0 else 0
        else:  # Credit trade
            credit = abs(position.entry_cost)
            pnl = (credit - total_value) * 100 * position.quantity
            pnl_pct = (credit - total_value) / credit if credit > 0 else 0
        
        # Exit conditions
        should_exit = False
        exit_reason = ""
        
        # Expiration
        if dte <= 1:
            should_exit = True
            exit_reason = "expiration"
        
        # Take profit (50% of max profit)
        if pnl > 0 and position.max_profit > 0:
            if pnl >= position.max_profit * 0.5 * 100 * position.quantity:
                should_exit = True
                exit_reason = "take_profit"
        
        # Stop loss (100% of premium for debit, 200% for credit)
        if pnl < 0:
            if position.entry_cost > 0:  # Debit
                if pnl <= -position.entry_cost * 100 * position.quantity * 0.5:
                    should_exit = True
                    exit_reason = "stop_loss"
            else:  # Credit
                if pnl <= -abs(position.entry_cost) * 100 * position.quantity * 2:
                    should_exit = True
                    exit_reason = "stop_loss"
        
        return pnl, should_exit, exit_reason
    
    def run_backtest(self) -> Dict[str, Any]:
        """Run the full backtest."""
        
        print("\n" + "="*70)
        print("  GNOSIS OPTIONS BACKTEST ENGINE")
        print("  Theoretical Options on Real Stock Data")
        print("="*70)
        print(f"\nSymbols: {', '.join(self.symbols)}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print()
        
        # Fetch data for all symbols
        all_data: Dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            try:
                df = self._fetch_data(symbol)
                if len(df) > 50:
                    all_data[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        
        if not all_data:
            raise ValueError("No data available for any symbol")
        
        # Get all unique dates
        all_dates = set()
        for df in all_data.values():
            all_dates.update(df['timestamp'].tolist())
        all_dates = sorted(all_dates)
        
        print(f"Data loaded: {len(all_data)} symbols, {len(all_dates)} trading days")
        print()
        
        # Warmup period
        warmup = 50
        
        # Main backtest loop
        for i in range(warmup, len(all_dates)):
            current_date = all_dates[i]
            
            # Evaluate existing positions
            for position in list(self.positions):
                if position.symbol in all_data:
                    df = all_data[position.symbol]
                    mask = df['timestamp'] <= current_date
                    if mask.sum() > 0:
                        current_price = df[mask]['close'].iloc[-1]
                        iv, _ = self._calculate_iv(df[mask]['close'].tail(30))
                        
                        pnl, should_exit, reason = self._evaluate_position(
                            position, current_price, current_date, iv
                        )
                        
                        if should_exit:
                            position.exit_date = current_date
                            position.exit_underlying_price = current_price
                            position.exit_reason = reason
                            position.realized_pnl = pnl
                            position.realized_pnl_pct = pnl / (abs(position.entry_cost) * 100 * position.quantity) \
                                if position.entry_cost != 0 else 0
                            
                            self.capital += pnl + (abs(position.entry_cost) * 100 * position.quantity)
                            self.closed_positions.append(position)
                            self.positions.remove(position)
                            
                            logger.debug(
                                f"CLOSE {position.strategy.value} {position.symbol} | "
                                f"P&L: ${pnl:,.2f} | {reason}"
                            )
            
            # Generate signals and open new positions
            if len(self.positions) < self.max_positions:
                for symbol, df in all_data.items():
                    # Skip if already have position
                    if any(p.symbol == symbol for p in self.positions):
                        continue
                    
                    mask = df['timestamp'] <= current_date
                    idx = mask.sum() - 1
                    
                    if idx < warmup:
                        continue
                    
                    # Generate signal
                    signal = self._generate_gnosis_signal(symbol, df, idx)
                    
                    # Entry criteria
                    if signal.confidence >= 0.4 and signal.direction != SignalDirection.NEUTRAL:
                        position = self._create_position(signal)
                        
                        # Check capital
                        cost = abs(position.entry_cost) * 100 * position.quantity
                        if cost <= self.capital * self.max_position_size:
                            self.capital -= cost
                            self.positions.append(position)
                            
                            logger.debug(
                                f"OPEN {position.strategy.value} {symbol} | "
                                f"Direction: {signal.direction.value} | "
                                f"Confidence: {signal.confidence:.2f}"
                            )
                            
                            if len(self.positions) >= self.max_positions:
                                break
            
            # Record equity
            position_value = 0
            for pos in self.positions:
                if pos.symbol in all_data:
                    df = all_data[pos.symbol]
                    mask = df['timestamp'] <= current_date
                    if mask.sum() > 0:
                        current_price = df[mask]['close'].iloc[-1]
                        iv, _ = self._calculate_iv(df[mask]['close'].tail(30))
                        pnl, _, _ = self._evaluate_position(pos, current_price, current_date, iv)
                        position_value += abs(pos.entry_cost) * 100 * pos.quantity + pnl
            
            self.equity_curve.append({
                'date': current_date,
                'equity': self.capital + position_value,
                'positions': len(self.positions),
            })
        
        # Close remaining positions
        final_date = all_dates[-1]
        for position in list(self.positions):
            if position.symbol in all_data:
                df = all_data[position.symbol]
                current_price = df['close'].iloc[-1]
                iv, _ = self._calculate_iv(df['close'].tail(30))
                
                pnl, _, _ = self._evaluate_position(position, current_price, final_date, iv)
                
                position.exit_date = final_date
                position.exit_underlying_price = current_price
                position.exit_reason = "end_of_test"
                position.realized_pnl = pnl
                
                self.capital += pnl + (abs(position.entry_cost) * 100 * position.quantity)
                self.closed_positions.append(position)
        
        self.positions = []
        
        # Calculate results
        results = self._calculate_results()
        self._print_results(results)
        self._save_results(results)
        
        return results
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate comprehensive results."""
        
        results = {
            'config': {
                'symbols': self.symbols,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_capital': self.initial_capital,
            },
            'capital': {
                'initial': self.initial_capital,
                'final': self.capital,
                'return': self.capital - self.initial_capital,
                'return_pct': (self.capital / self.initial_capital - 1) * 100,
            }
        }
        
        if self.closed_positions:
            winners = [p for p in self.closed_positions if p.realized_pnl > 0]
            losers = [p for p in self.closed_positions if p.realized_pnl <= 0]
            
            results['trades'] = {
                'total': len(self.closed_positions),
                'winners': len(winners),
                'losers': len(losers),
                'win_rate': len(winners) / len(self.closed_positions) * 100,
                'avg_win': np.mean([p.realized_pnl for p in winners]) if winners else 0,
                'avg_loss': np.mean([p.realized_pnl for p in losers]) if losers else 0,
                'largest_win': max([p.realized_pnl for p in winners]) if winners else 0,
                'largest_loss': min([p.realized_pnl for p in losers]) if losers else 0,
            }
            
            # Profit factor
            gross_profit = sum(p.realized_pnl for p in winners)
            gross_loss = abs(sum(p.realized_pnl for p in losers))
            results['trades']['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Strategy breakdown
            strategy_stats = {}
            for pos in self.closed_positions:
                strat = pos.strategy.value
                if strat not in strategy_stats:
                    strategy_stats[strat] = {'trades': 0, 'pnl': 0, 'winners': 0}
                strategy_stats[strat]['trades'] += 1
                strategy_stats[strat]['pnl'] += pos.realized_pnl
                if pos.realized_pnl > 0:
                    strategy_stats[strat]['winners'] += 1
            
            for strat, stats in strategy_stats.items():
                stats['win_rate'] = stats['winners'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
            
            results['by_strategy'] = strategy_stats
            
            # Direction breakdown
            direction_stats = {}
            for pos in self.closed_positions:
                dir_val = pos.entry_direction.value
                if dir_val not in direction_stats:
                    direction_stats[dir_val] = {'trades': 0, 'pnl': 0, 'winners': 0}
                direction_stats[dir_val]['trades'] += 1
                direction_stats[dir_val]['pnl'] += pos.realized_pnl
                if pos.realized_pnl > 0:
                    direction_stats[dir_val]['winners'] += 1
            
            results['by_direction'] = direction_stats
            
            # Confidence analysis
            high_conf = [p for p in self.closed_positions if p.entry_confidence >= 0.6]
            low_conf = [p for p in self.closed_positions if p.entry_confidence < 0.6]
            
            results['confidence_analysis'] = {
                'high_confidence': {
                    'trades': len(high_conf),
                    'win_rate': len([p for p in high_conf if p.realized_pnl > 0]) / len(high_conf) * 100 if high_conf else 0,
                    'total_pnl': sum(p.realized_pnl for p in high_conf),
                },
                'low_confidence': {
                    'trades': len(low_conf),
                    'win_rate': len([p for p in low_conf if p.realized_pnl > 0]) / len(low_conf) * 100 if low_conf else 0,
                    'total_pnl': sum(p.realized_pnl for p in low_conf),
                }
            }
        
        # Risk metrics
        if self.equity_curve:
            equities = [e['equity'] for e in self.equity_curve]
            peak = equities[0]
            max_dd = 0
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak
                if dd > max_dd:
                    max_dd = dd
            
            results['risk'] = {
                'max_drawdown_pct': max_dd * 100,
            }
            
            # Sharpe ratio
            if len(equities) > 1:
                returns = pd.Series(equities).pct_change().dropna()
                if returns.std() > 0:
                    results['risk']['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
        
        return results
    
    def _print_results(self, results: Dict[str, Any]):
        """Print formatted results."""
        
        print("\n" + "="*70)
        print("  GNOSIS OPTIONS BACKTEST RESULTS")
        print("="*70)
        
        cap = results['capital']
        print(f"\n{'CAPITAL PERFORMANCE':=^50}")
        print(f"  Initial Capital:  ${cap['initial']:>12,.2f}")
        print(f"  Final Capital:    ${cap['final']:>12,.2f}")
        print(f"  Total Return:     ${cap['return']:>12,.2f} ({cap['return_pct']:.2f}%)")
        
        if 'trades' in results:
            t = results['trades']
            print(f"\n{'TRADE STATISTICS':=^50}")
            print(f"  Total Trades:     {t['total']:>12}")
            print(f"  Winners:          {t['winners']:>12} ({t['win_rate']:.1f}%)")
            print(f"  Losers:           {t['losers']:>12}")
            print(f"  Profit Factor:    {t['profit_factor']:>12.2f}")
            print(f"  Avg Win:          ${t['avg_win']:>11,.2f}")
            print(f"  Avg Loss:         ${t['avg_loss']:>11,.2f}")
            print(f"  Largest Win:      ${t['largest_win']:>11,.2f}")
            print(f"  Largest Loss:     ${t['largest_loss']:>11,.2f}")
        
        if 'by_strategy' in results:
            print(f"\n{'STRATEGY BREAKDOWN':=^50}")
            for strat, stats in sorted(results['by_strategy'].items(), key=lambda x: -x[1]['pnl']):
                print(f"  {strat:20s}: {stats['trades']:3d} trades, "
                      f"{stats['win_rate']:.1f}% win, ${stats['pnl']:>10,.2f}")
        
        if 'by_direction' in results:
            print(f"\n{'DIRECTION BREAKDOWN':=^50}")
            for direction, stats in results['by_direction'].items():
                wr = stats['winners'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
                print(f"  {direction:20s}: {stats['trades']:3d} trades, "
                      f"{wr:.1f}% win, ${stats['pnl']:>10,.2f}")
        
        if 'confidence_analysis' in results:
            ca = results['confidence_analysis']
            print(f"\n{'CONFIDENCE ANALYSIS':=^50}")
            print(f"  High Confidence (>=60%): {ca['high_confidence']['trades']} trades, "
                  f"{ca['high_confidence']['win_rate']:.1f}% win, "
                  f"${ca['high_confidence']['total_pnl']:,.2f}")
            print(f"  Low Confidence (<60%):   {ca['low_confidence']['trades']} trades, "
                  f"{ca['low_confidence']['win_rate']:.1f}% win, "
                  f"${ca['low_confidence']['total_pnl']:,.2f}")
        
        if 'risk' in results:
            r = results['risk']
            print(f"\n{'RISK METRICS':=^50}")
            print(f"  Max Drawdown:     {r['max_drawdown_pct']:>12.2f}%")
            if 'sharpe_ratio' in r:
                print(f"  Sharpe Ratio:     {r['sharpe_ratio']:>12.2f}")
        
        print("\n" + "="*70)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        
        output_dir = Path("runs/gnosis_options_backtests")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gnosis_options_{timestamp}.json"
        
        # Convert positions to serializable format
        trades_data = []
        for p in self.closed_positions:
            trades_data.append({
                'position_id': p.position_id,
                'symbol': p.symbol,
                'strategy': p.strategy.value,
                'entry_date': p.entry_date.isoformat() if p.entry_date else None,
                'exit_date': p.exit_date.isoformat() if p.exit_date else None,
                'entry_price': p.entry_underlying_price,
                'exit_price': p.exit_underlying_price,
                'entry_cost': p.entry_cost,
                'quantity': p.quantity,
                'realized_pnl': p.realized_pnl,
                'exit_reason': p.exit_reason,
                'entry_confidence': p.entry_confidence,
                'entry_direction': p.entry_direction.value,
                'vol_regime': p.vol_regime_at_entry.value,
            })
        
        output = {
            'summary': results,
            'trades': trades_data,
        }
        
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run GNOSIS options backtest."""
    
    engine = GnosisOptionsBacktestEngine(
        symbols=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL"],
        start_date="2020-01-01",
        end_date="2024-12-01",
        initial_capital=100_000.0,
        max_position_size=0.10,
        max_positions=5,
    )
    
    results = engine.run_backtest()
    return results


if __name__ == "__main__":
    main()
