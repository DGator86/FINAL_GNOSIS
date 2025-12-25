#!/usr/bin/env python3
"""
GNOSIS Options Backtest Engine
==============================

A sophisticated options backtesting system that uses:
1. Real stock data from Massive.com/Alpaca
2. Theoretical options pricing (Black-Scholes)
3. Full GNOSIS framework integration:
   - Price-as-Particle Physics (mass, energy, momentum)
   - Sentiment Engine (RSI, MACD, momentum indicators)
   - Liquidity Engine (Accumulation/Distribution, Bollinger Bands, volume)
   - Predictive Cones (volatility-based future price ranges)
   - Support/Resistance detection
   - Multi-timeframe confirmation

The system can profit in ANY market condition through:
- Directional trades (calls/puts) when trend is clear
- Spreads when range-bound
- Straddles/Strangles for volatility plays
- Iron Condors for low-vol environments

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


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class MarketRegime(Enum):
    """Market regime classification."""
    STRONG_BULL = "strong_bull"      # Trending up strongly
    BULL = "bull"                     # Uptrend
    NEUTRAL = "neutral"               # Range-bound
    BEAR = "bear"                     # Downtrend
    STRONG_BEAR = "strong_bear"       # Trending down strongly
    HIGH_VOL = "high_volatility"      # Volatile, direction unclear
    LOW_VOL = "low_volatility"        # Compressed, breakout pending


class OptionsStrategy(Enum):
    """Options strategy types."""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    LONG_STRADDLE = "long_straddle"
    LONG_STRANGLE = "long_strangle"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    CALENDAR_SPREAD = "calendar_spread"


class SignalStrength(Enum):
    """Signal strength classification."""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1
    NEUTRAL = 0


# =============================================================================
# PRICE-AS-PARTICLE PHYSICS MODEL
# =============================================================================

@dataclass
class PriceParticle:
    """
    Models price as a particle with physical properties.
    
    Core concept: Price has "mass" (inertia) that determines how much
    "energy" (volume) is required to move it. Larger caps = more mass = 
    harder to move.
    """
    
    # Current state
    price: float = 0.0
    velocity: float = 0.0          # Rate of price change (momentum)
    acceleration: float = 0.0      # Change in velocity
    
    # Physical properties
    mass: float = 1.0              # Market cap relative mass (higher = harder to move)
    energy: float = 0.0            # Volume-weighted energy
    kinetic_energy: float = 0.0    # 0.5 * mass * velocity^2
    potential_energy: float = 0.0  # Distance from equilibrium (moving averages)
    
    # Derived metrics
    momentum: float = 0.0          # mass * velocity
    force: float = 0.0             # Volume pressure causing movement
    friction: float = 0.0          # Resistance to movement (spread, low liquidity)
    
    # Quantum uncertainty (volatility-based)
    position_uncertainty: float = 0.0  # Price uncertainty range
    momentum_uncertainty: float = 0.0  # Velocity uncertainty
    
    def calculate_momentum(self) -> float:
        """p = m * v"""
        self.momentum = self.mass * self.velocity
        return self.momentum
    
    def calculate_kinetic_energy(self) -> float:
        """KE = 0.5 * m * v^2"""
        self.kinetic_energy = 0.5 * self.mass * (self.velocity ** 2)
        return self.kinetic_energy
    
    def calculate_force_required(self, target_velocity: float, time_period: float = 1.0) -> float:
        """F = m * a, where a = (v_target - v_current) / t"""
        if time_period <= 0:
            return float('inf')
        required_acceleration = (target_velocity - self.velocity) / time_period
        return self.mass * required_acceleration
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'price': self.price,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'mass': self.mass,
            'energy': self.energy,
            'kinetic_energy': self.kinetic_energy,
            'potential_energy': self.potential_energy,
            'momentum': self.momentum,
            'force': self.force,
            'position_uncertainty': self.position_uncertainty,
        }


class PricePhysicsEngine:
    """
    Computes price physics from market data.
    
    Key insights:
    - Large caps (AAPL, MSFT) have high mass - need huge volume to move
    - Small caps have low mass - can move violently on small volume
    - Volume = energy input into the system
    - Price momentum = mass * velocity
    - Breakouts require energy > potential barrier
    """
    
    # Market cap tiers for mass calculation (in billions)
    MEGA_CAP = 1000   # AAPL, MSFT, GOOGL
    LARGE_CAP = 200
    MID_CAP = 10
    SMALL_CAP = 2
    
    def __init__(self):
        self.particles: Dict[str, PriceParticle] = {}
    
    def compute_particle_state(
        self,
        symbol: str,
        prices: pd.Series,
        volumes: pd.Series,
        market_cap: Optional[float] = None,
        float_shares: Optional[float] = None,
    ) -> PriceParticle:
        """
        Compute the physical state of a price particle.
        
        Args:
            symbol: Stock symbol
            prices: Historical prices (close)
            volumes: Historical volumes
            market_cap: Market capitalization in billions
            float_shares: Floating shares in millions
        """
        particle = PriceParticle()
        
        if len(prices) < 2:
            return particle
        
        # Current price
        particle.price = float(prices.iloc[-1])
        
        # === MASS CALCULATION ===
        # Mass is proportional to market cap and inversely to float turnover
        if market_cap:
            if market_cap >= self.MEGA_CAP:
                particle.mass = 10.0  # Very hard to move
            elif market_cap >= self.LARGE_CAP:
                particle.mass = 5.0
            elif market_cap >= self.MID_CAP:
                particle.mass = 2.0
            elif market_cap >= self.SMALL_CAP:
                particle.mass = 1.0
            else:
                particle.mass = 0.5  # Easy to move
        else:
            # Estimate mass from average volume and price
            avg_volume = volumes.mean()
            avg_price = prices.mean()
            implied_value = avg_volume * avg_price
            # Normalize to 1-10 scale
            particle.mass = min(10.0, max(0.5, np.log10(implied_value) / 2))
        
        # === VELOCITY (Price Rate of Change) ===
        returns = prices.pct_change().dropna()
        if len(returns) >= 1:
            particle.velocity = float(returns.iloc[-1]) * 100  # As percentage
        
        # Smoothed velocity (momentum)
        if len(returns) >= 5:
            particle.velocity = float(returns.tail(5).mean()) * 100
        
        # === ACCELERATION (Change in velocity) ===
        if len(returns) >= 10:
            recent_velocity = returns.tail(5).mean()
            prior_velocity = returns.tail(10).head(5).mean()
            particle.acceleration = float(recent_velocity - prior_velocity) * 100
        
        # === ENERGY (Volume-weighted) ===
        avg_volume = volumes.mean()
        current_volume = volumes.iloc[-1]
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            # Energy is volume above average
            particle.energy = max(0, (volume_ratio - 1.0)) * 100
        
        # === FORCE (Volume pressure) ===
        # Force = Energy input * Direction
        if len(returns) >= 1:
            direction = 1 if returns.iloc[-1] > 0 else -1
            particle.force = particle.energy * direction
        
        # === KINETIC ENERGY ===
        particle.calculate_kinetic_energy()
        
        # === POTENTIAL ENERGY (Distance from equilibrium) ===
        # Equilibrium is the 20-period moving average
        if len(prices) >= 20:
            equilibrium = prices.tail(20).mean()
            particle.potential_energy = abs(particle.price - equilibrium) / equilibrium * 100
        
        # === MOMENTUM ===
        particle.calculate_momentum()
        
        # === UNCERTAINTY (Volatility-based) ===
        if len(returns) >= 20:
            volatility = returns.tail(20).std()
            particle.position_uncertainty = volatility * particle.price * 2  # 2 std range
            particle.momentum_uncertainty = volatility * 100
        
        # === FRICTION (Inverse of liquidity) ===
        # High volume = low friction, low volume = high friction
        if avg_volume > 0:
            particle.friction = 1.0 / (volumes.iloc[-1] / avg_volume + 0.1)
        
        self.particles[symbol] = particle
        return particle
    
    def predict_movement(
        self,
        particle: PriceParticle,
        incoming_energy: float,
        direction: int = 1,
    ) -> Tuple[float, float]:
        """
        Predict price movement given incoming energy (volume).
        
        Returns:
            (expected_move_pct, confidence)
        """
        if particle.mass <= 0:
            return 0.0, 0.0
        
        # F = m * a => a = F / m
        net_force = (incoming_energy * direction) - particle.friction
        acceleration = net_force / particle.mass
        
        # New velocity = old velocity + acceleration
        new_velocity = particle.velocity + acceleration
        
        # Expected move is velocity (as percentage)
        expected_move = new_velocity
        
        # Confidence based on energy vs mass ratio
        energy_mass_ratio = incoming_energy / particle.mass
        confidence = min(1.0, energy_mass_ratio / 10)
        
        return expected_move, confidence


# =============================================================================
# SENTIMENT ENGINE
# =============================================================================

@dataclass
class SentimentState:
    """Combined sentiment from multiple indicators."""
    
    # RSI
    rsi: float = 50.0
    rsi_signal: str = "neutral"  # overbought, oversold, neutral
    rsi_divergence: str = "none"  # bullish_div, bearish_div, none
    
    # MACD
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_cross: str = "none"  # bullish_cross, bearish_cross, none
    macd_trend: str = "neutral"  # bullish, bearish, neutral
    
    # Momentum
    momentum_5: float = 0.0
    momentum_10: float = 0.0
    momentum_20: float = 0.0
    momentum_signal: str = "neutral"
    
    # Stochastic
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    stoch_signal: str = "neutral"
    
    # Williams %R
    williams_r: float = -50.0
    williams_signal: str = "neutral"
    
    # Combined
    overall_sentiment: float = 0.0  # -1 to 1
    sentiment_strength: SignalStrength = SignalStrength.NEUTRAL
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rsi': self.rsi,
            'rsi_signal': self.rsi_signal,
            'macd': self.macd,
            'macd_histogram': self.macd_histogram,
            'macd_cross': self.macd_cross,
            'momentum_signal': self.momentum_signal,
            'overall_sentiment': self.overall_sentiment,
            'confidence': self.confidence,
        }


class SentimentEngine:
    """
    Technical sentiment analysis using momentum indicators.
    
    Signals generated from:
    - RSI (overbought/oversold, divergences)
    - MACD (crosses, histogram direction)
    - Momentum (rate of change)
    - Stochastic oscillator
    - Williams %R
    """
    
    def __init__(self):
        pass
    
    def compute_sentiment(
        self,
        prices: pd.Series,
        highs: Optional[pd.Series] = None,
        lows: Optional[pd.Series] = None,
    ) -> SentimentState:
        """Compute sentiment from price data."""
        
        state = SentimentState()
        
        if len(prices) < 26:
            return state
        
        close = prices
        high = highs if highs is not None else prices
        low = lows if lows is not None else prices
        
        # === RSI ===
        state.rsi = self._compute_rsi(close, 14)
        if state.rsi >= 70:
            state.rsi_signal = "overbought"
        elif state.rsi <= 30:
            state.rsi_signal = "oversold"
        else:
            state.rsi_signal = "neutral"
        
        # Check for RSI divergence
        state.rsi_divergence = self._check_rsi_divergence(close, state.rsi)
        
        # === MACD ===
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        state.macd = float(macd_line.iloc[-1])
        state.macd_signal = float(signal_line.iloc[-1])
        state.macd_histogram = state.macd - state.macd_signal
        
        # Check for MACD cross
        if len(macd_line) >= 2:
            prev_macd = float(macd_line.iloc[-2])
            prev_signal = float(signal_line.iloc[-2])
            
            if prev_macd <= prev_signal and state.macd > state.macd_signal:
                state.macd_cross = "bullish_cross"
            elif prev_macd >= prev_signal and state.macd < state.macd_signal:
                state.macd_cross = "bearish_cross"
        
        state.macd_trend = "bullish" if state.macd > state.macd_signal else "bearish"
        
        # === MOMENTUM ===
        state.momentum_5 = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0
        state.momentum_10 = float((close.iloc[-1] / close.iloc[-11] - 1) * 100) if len(close) >= 11 else 0
        state.momentum_20 = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else 0
        
        if state.momentum_5 > 0 and state.momentum_10 > 0:
            state.momentum_signal = "bullish"
        elif state.momentum_5 < 0 and state.momentum_10 < 0:
            state.momentum_signal = "bearish"
        else:
            state.momentum_signal = "neutral"
        
        # === STOCHASTIC ===
        if len(close) >= 14:
            lowest_low = low.tail(14).min()
            highest_high = high.tail(14).max()
            if highest_high != lowest_low:
                state.stoch_k = float((close.iloc[-1] - lowest_low) / (highest_high - lowest_low) * 100)
            state.stoch_d = float(pd.Series([state.stoch_k]).rolling(3).mean().iloc[-1]) if state.stoch_k else 50
            
            if state.stoch_k >= 80:
                state.stoch_signal = "overbought"
            elif state.stoch_k <= 20:
                state.stoch_signal = "oversold"
        
        # === WILLIAMS %R ===
        if len(close) >= 14:
            highest_high = high.tail(14).max()
            lowest_low = low.tail(14).min()
            if highest_high != lowest_low:
                state.williams_r = float((highest_high - close.iloc[-1]) / (highest_high - lowest_low) * -100)
            
            if state.williams_r >= -20:
                state.williams_signal = "overbought"
            elif state.williams_r <= -80:
                state.williams_signal = "oversold"
        
        # === COMBINED SENTIMENT ===
        signals = []
        
        # RSI contribution
        if state.rsi_signal == "oversold":
            signals.append(0.7)  # Bullish
        elif state.rsi_signal == "overbought":
            signals.append(-0.7)  # Bearish
        else:
            signals.append((50 - state.rsi) / 50 * -0.3)  # Slight lean
        
        # MACD contribution
        if state.macd_cross == "bullish_cross":
            signals.append(0.8)
        elif state.macd_cross == "bearish_cross":
            signals.append(-0.8)
        else:
            signals.append(0.3 if state.macd_trend == "bullish" else -0.3)
        
        # Momentum contribution
        if state.momentum_signal == "bullish":
            signals.append(0.5)
        elif state.momentum_signal == "bearish":
            signals.append(-0.5)
        else:
            signals.append(0)
        
        # Stochastic contribution
        if state.stoch_signal == "oversold":
            signals.append(0.5)
        elif state.stoch_signal == "overbought":
            signals.append(-0.5)
        else:
            signals.append(0)
        
        # Divergence boost
        if state.rsi_divergence == "bullish_div":
            signals.append(0.6)
        elif state.rsi_divergence == "bearish_div":
            signals.append(-0.6)
        
        state.overall_sentiment = np.clip(np.mean(signals), -1, 1)
        
        # Confidence based on signal agreement
        signal_std = np.std(signals)
        state.confidence = 1.0 - min(1.0, signal_std)
        
        # Strength classification
        abs_sentiment = abs(state.overall_sentiment)
        if abs_sentiment >= 0.7:
            state.sentiment_strength = SignalStrength.VERY_STRONG
        elif abs_sentiment >= 0.5:
            state.sentiment_strength = SignalStrength.STRONG
        elif abs_sentiment >= 0.3:
            state.sentiment_strength = SignalStrength.MODERATE
        elif abs_sentiment >= 0.1:
            state.sentiment_strength = SignalStrength.WEAK
        else:
            state.sentiment_strength = SignalStrength.NEUTRAL
        
        return state
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Compute RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _check_rsi_divergence(self, prices: pd.Series, current_rsi: float) -> str:
        """Check for RSI divergence."""
        if len(prices) < 20:
            return "none"
        
        # Simple divergence check: price making new highs/lows but RSI not
        recent_high = prices.tail(10).max()
        prior_high = prices.tail(20).head(10).max()
        
        recent_low = prices.tail(10).min()
        prior_low = prices.tail(20).head(10).min()
        
        current_price = prices.iloc[-1]
        
        # Bearish divergence: price higher high, RSI lower
        if current_price >= recent_high and current_rsi < 60:
            return "bearish_div"
        
        # Bullish divergence: price lower low, RSI higher
        if current_price <= recent_low and current_rsi > 40:
            return "bullish_div"
        
        return "none"


# =============================================================================
# LIQUIDITY ENGINE
# =============================================================================

@dataclass
class LiquidityState:
    """Liquidity and accumulation analysis state."""
    
    # Accumulation/Distribution
    ad_line: float = 0.0
    ad_trend: str = "neutral"  # accumulation, distribution, neutral
    money_flow: float = 0.0    # Money Flow Index
    
    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0      # Normalized width (volatility)
    bb_position: float = 0.5   # 0 = at lower, 1 = at upper
    bb_squeeze: bool = False   # Tight bands = pending breakout
    
    # Volume Analysis
    volume_sma: float = 0.0
    volume_ratio: float = 1.0
    obv: float = 0.0           # On-Balance Volume
    obv_trend: str = "neutral"
    vwap: float = 0.0
    
    # Keltner Channels (for squeeze detection)
    kc_upper: float = 0.0
    kc_lower: float = 0.0
    
    # Combined
    liquidity_score: float = 0.0  # -1 to 1 (distribution to accumulation)
    breakout_probability: float = 0.0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ad_trend': self.ad_trend,
            'money_flow': self.money_flow,
            'bb_width': self.bb_width,
            'bb_position': self.bb_position,
            'bb_squeeze': self.bb_squeeze,
            'volume_ratio': self.volume_ratio,
            'obv_trend': self.obv_trend,
            'liquidity_score': self.liquidity_score,
            'breakout_probability': self.breakout_probability,
        }


class LiquidityEngine:
    """
    Liquidity and accumulation analysis.
    
    Analyzes:
    - Accumulation/Distribution (smart money flow)
    - Bollinger Bands (volatility and mean reversion)
    - On-Balance Volume (volume trend)
    - Money Flow Index (volume-weighted RSI)
    - VWAP (institutional reference)
    - Squeeze detection (Bollinger inside Keltner)
    """
    
    def __init__(self):
        pass
    
    def compute_liquidity(
        self,
        prices: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        volumes: pd.Series,
    ) -> LiquidityState:
        """Compute liquidity metrics."""
        
        state = LiquidityState()
        
        if len(prices) < 20:
            return state
        
        close = prices
        high = highs
        low = lows
        volume = volumes
        
        # === ACCUMULATION/DISTRIBUTION LINE ===
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        ad = (clv * volume).cumsum()
        state.ad_line = float(ad.iloc[-1])
        
        # A/D trend
        if len(ad) >= 10:
            ad_sma = ad.rolling(10).mean()
            if ad.iloc[-1] > ad_sma.iloc[-1]:
                state.ad_trend = "accumulation"
            else:
                state.ad_trend = "distribution"
        
        # === MONEY FLOW INDEX ===
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        delta = typical_price.diff()
        positive_flow = money_flow.where(delta > 0, 0).rolling(14).sum()
        negative_flow = money_flow.where(delta < 0, 0).rolling(14).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
        state.money_flow = float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50
        
        # === BOLLINGER BANDS ===
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        
        state.bb_middle = float(sma_20.iloc[-1])
        state.bb_upper = float(sma_20.iloc[-1] + 2 * std_20.iloc[-1])
        state.bb_lower = float(sma_20.iloc[-1] - 2 * std_20.iloc[-1])
        
        # Band width (normalized)
        state.bb_width = (state.bb_upper - state.bb_lower) / state.bb_middle
        
        # Position within bands
        if state.bb_upper != state.bb_lower:
            state.bb_position = (close.iloc[-1] - state.bb_lower) / (state.bb_upper - state.bb_lower)
        
        # === KELTNER CHANNELS (for squeeze) ===
        atr = self._compute_atr(high, low, close, 20)
        ema_20 = close.ewm(span=20, adjust=False).mean()
        state.kc_upper = float(ema_20.iloc[-1] + 1.5 * atr)
        state.kc_lower = float(ema_20.iloc[-1] - 1.5 * atr)
        
        # Squeeze: Bollinger inside Keltner
        state.bb_squeeze = (state.bb_lower > state.kc_lower and state.bb_upper < state.kc_upper)
        
        # === ON-BALANCE VOLUME ===
        obv = (np.sign(close.diff()) * volume).cumsum()
        state.obv = float(obv.iloc[-1])
        
        if len(obv) >= 10:
            obv_sma = obv.rolling(10).mean()
            if obv.iloc[-1] > obv_sma.iloc[-1]:
                state.obv_trend = "bullish"
            else:
                state.obv_trend = "bearish"
        
        # === VOLUME ANALYSIS ===
        state.volume_sma = float(volume.rolling(20).mean().iloc[-1])
        if state.volume_sma > 0:
            state.volume_ratio = float(volume.iloc[-1] / state.volume_sma)
        
        # === VWAP ===
        cumulative_tp_vol = (typical_price * volume).cumsum()
        cumulative_vol = volume.cumsum()
        vwap = cumulative_tp_vol / cumulative_vol
        state.vwap = float(vwap.iloc[-1])
        
        # === COMBINED LIQUIDITY SCORE ===
        scores = []
        
        # A/D contribution
        if state.ad_trend == "accumulation":
            scores.append(0.5)
        elif state.ad_trend == "distribution":
            scores.append(-0.5)
        else:
            scores.append(0)
        
        # Money flow contribution
        if state.money_flow >= 80:
            scores.append(0.6)
        elif state.money_flow <= 20:
            scores.append(-0.6)
        else:
            scores.append((state.money_flow - 50) / 50 * 0.3)
        
        # OBV contribution
        if state.obv_trend == "bullish":
            scores.append(0.4)
        elif state.obv_trend == "bearish":
            scores.append(-0.4)
        else:
            scores.append(0)
        
        # Volume confirmation
        if state.volume_ratio >= 1.5:
            # High volume confirms direction
            if close.iloc[-1] > close.iloc[-2]:
                scores.append(0.3)
            else:
                scores.append(-0.3)
        
        state.liquidity_score = np.clip(np.mean(scores), -1, 1)
        
        # Breakout probability (squeeze + volume building)
        if state.bb_squeeze:
            state.breakout_probability = min(1.0, 0.5 + state.volume_ratio * 0.2)
        else:
            state.breakout_probability = min(0.3, state.volume_ratio * 0.1)
        
        # Confidence
        state.confidence = 1.0 - np.std(scores)
        
        return state
    
    def _compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
        """Compute Average True Range."""
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        
        return float(tr.rolling(period).mean().iloc[-1])


# =============================================================================
# PREDICTIVE CONES & SUPPORT/RESISTANCE
# =============================================================================

@dataclass
class PredictiveCone:
    """Future price probability cone."""
    
    current_price: float = 0.0
    
    # 1 standard deviation bounds (68% probability)
    upper_1std: List[float] = field(default_factory=list)
    lower_1std: List[float] = field(default_factory=list)
    
    # 2 standard deviation bounds (95% probability)
    upper_2std: List[float] = field(default_factory=list)
    lower_2std: List[float] = field(default_factory=list)
    
    # Expected path
    expected_path: List[float] = field(default_factory=list)
    
    # Time horizons (days)
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 21])
    
    # Volatility used
    annualized_vol: float = 0.0


@dataclass
class SupportResistance:
    """Support and resistance levels."""
    
    # Key levels
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    
    # Nearest levels
    nearest_support: float = 0.0
    nearest_resistance: float = 0.0
    
    # Distance to levels (as percentage)
    distance_to_support_pct: float = 0.0
    distance_to_resistance_pct: float = 0.0
    
    # Strength of levels
    support_strength: float = 0.0
    resistance_strength: float = 0.0
    
    # Pivot points
    pivot: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    s1: float = 0.0
    s2: float = 0.0


class PredictionEngine:
    """
    Predictive cone and support/resistance analysis.
    
    Features:
    - Volatility-based future price cones
    - Support/Resistance detection from price action
    - Pivot point calculation
    - Trend projection
    """
    
    def __init__(self):
        pass
    
    def compute_predictive_cone(
        self,
        prices: pd.Series,
        horizons: List[int] = [1, 5, 10, 21],
        drift: float = 0.0,
    ) -> PredictiveCone:
        """
        Compute future price probability cone using geometric Brownian motion.
        """
        cone = PredictiveCone()
        
        if len(prices) < 20:
            return cone
        
        cone.current_price = float(prices.iloc[-1])
        cone.horizons = horizons
        
        # Calculate historical volatility
        returns = prices.pct_change().dropna()
        daily_vol = returns.std()
        cone.annualized_vol = daily_vol * np.sqrt(252)
        
        # Calculate drift if not provided (use historical mean return)
        if drift == 0:
            drift = returns.mean()
        
        # Generate cone for each horizon
        for days in horizons:
            # Standard deviation at this horizon
            sigma_t = daily_vol * np.sqrt(days)
            
            # Expected price (with drift)
            expected = cone.current_price * np.exp(drift * days)
            cone.expected_path.append(expected)
            
            # 1 std bounds
            cone.upper_1std.append(cone.current_price * np.exp(sigma_t))
            cone.lower_1std.append(cone.current_price * np.exp(-sigma_t))
            
            # 2 std bounds
            cone.upper_2std.append(cone.current_price * np.exp(2 * sigma_t))
            cone.lower_2std.append(cone.current_price * np.exp(-2 * sigma_t))
        
        return cone
    
    def compute_support_resistance(
        self,
        prices: pd.Series,
        highs: pd.Series,
        lows: pd.Series,
        lookback: int = 50,
    ) -> SupportResistance:
        """Compute support and resistance levels."""
        
        sr = SupportResistance()
        
        if len(prices) < lookback:
            lookback = len(prices)
        
        if lookback < 5:
            return sr
        
        close = prices.tail(lookback)
        high = highs.tail(lookback)
        low = lows.tail(lookback)
        current = float(prices.iloc[-1])
        
        # === PIVOT POINTS ===
        prev_high = float(high.iloc[-2])
        prev_low = float(low.iloc[-2])
        prev_close = float(close.iloc[-2])
        
        sr.pivot = (prev_high + prev_low + prev_close) / 3
        sr.r1 = 2 * sr.pivot - prev_low
        sr.r2 = sr.pivot + (prev_high - prev_low)
        sr.s1 = 2 * sr.pivot - prev_high
        sr.s2 = sr.pivot - (prev_high - prev_low)
        
        # === FIND SWING HIGHS/LOWS ===
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(high) - 2):
            # Swing high: higher than 2 bars on each side
            if (high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i-2] and
                high.iloc[i] > high.iloc[i+1] and high.iloc[i] > high.iloc[i+2]):
                swing_highs.append(float(high.iloc[i]))
            
            # Swing low: lower than 2 bars on each side
            if (low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i-2] and
                low.iloc[i] < low.iloc[i+1] and low.iloc[i] < low.iloc[i+2]):
                swing_lows.append(float(low.iloc[i]))
        
        # Cluster nearby levels
        sr.resistance_levels = self._cluster_levels(swing_highs, threshold=0.02)
        sr.support_levels = self._cluster_levels(swing_lows, threshold=0.02)
        
        # Add pivot levels
        sr.resistance_levels.extend([sr.r1, sr.r2])
        sr.support_levels.extend([sr.s1, sr.s2])
        
        # Sort and deduplicate
        sr.resistance_levels = sorted(set([r for r in sr.resistance_levels if r > current]))
        sr.support_levels = sorted(set([s for s in sr.support_levels if s < current]), reverse=True)
        
        # Nearest levels
        if sr.resistance_levels:
            sr.nearest_resistance = sr.resistance_levels[0]
            sr.distance_to_resistance_pct = (sr.nearest_resistance - current) / current * 100
        
        if sr.support_levels:
            sr.nearest_support = sr.support_levels[0]
            sr.distance_to_support_pct = (current - sr.nearest_support) / current * 100
        
        return sr
    
    def _cluster_levels(self, levels: List[float], threshold: float = 0.02) -> List[float]:
        """Cluster nearby price levels."""
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if level <= current_cluster[-1] * (1 + threshold):
                current_cluster.append(level)
            else:
                # Average the cluster
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clustered.append(np.mean(current_cluster))
        return clustered


# =============================================================================
# OPTIONS STRATEGY SELECTOR
# =============================================================================

@dataclass 
class OptionsPosition:
    """Represents a theoretical options position."""
    
    strategy: OptionsStrategy = OptionsStrategy.LONG_CALL
    
    # Legs
    legs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Entry
    entry_date: datetime = None
    entry_price: float = 0.0      # Total premium paid/received
    underlying_price: float = 0.0
    
    # Greeks at entry
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    # Position details
    contracts: int = 1
    max_profit: float = 0.0       # Maximum theoretical profit
    max_loss: float = 0.0         # Maximum theoretical loss
    breakeven: List[float] = field(default_factory=list)
    
    # Exit
    exit_date: datetime = None
    exit_price: float = 0.0
    exit_underlying: float = 0.0
    
    # P&L
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    pnl_pct: float = 0.0
    
    # Context
    signal_strength: SignalStrength = SignalStrength.NEUTRAL
    confidence: float = 0.0
    regime: MarketRegime = MarketRegime.NEUTRAL


class OptionsStrategySelector:
    """
    Selects optimal options strategy based on market conditions.
    
    Strategy Selection Matrix:
    
    | Regime         | Sentiment  | Vol   | Strategy              |
    |----------------|------------|-------|------------------------|
    | Strong Bull    | Bullish    | Any   | Long Call / Bull Spread|
    | Strong Bear    | Bearish    | Any   | Long Put / Bear Spread |
    | Neutral        | Mixed      | High  | Iron Condor            |
    | Neutral        | Mixed      | Low   | Long Straddle (breakout)|
    | High Vol       | Unclear    | High  | Iron Condor / Strangle |
    | Low Vol        | Any        | Low   | Long Straddle/Strangle |
    """
    
    # Risk-free rate for Black-Scholes
    RISK_FREE_RATE = 0.05
    
    def __init__(self):
        pass
    
    def select_strategy(
        self,
        regime: MarketRegime,
        sentiment: SentimentState,
        liquidity: LiquidityState,
        particle: PriceParticle,
        cone: PredictiveCone,
        sr: SupportResistance,
        iv_percentile: float = 50.0,
    ) -> Tuple[OptionsStrategy, Dict[str, Any]]:
        """
        Select optimal options strategy based on all inputs.
        
        Returns:
            (strategy, parameters)
        """
        
        # Determine direction bias
        direction_score = (
            sentiment.overall_sentiment * 0.4 +
            liquidity.liquidity_score * 0.3 +
            (particle.momentum / 10) * 0.3  # Normalize momentum
        )
        direction_score = np.clip(direction_score, -1, 1)
        
        # Determine volatility expectation
        vol_expanding = liquidity.bb_squeeze or liquidity.breakout_probability > 0.5
        iv_high = iv_percentile > 60
        iv_low = iv_percentile < 40
        
        # Confidence in direction
        confidence = (sentiment.confidence + liquidity.confidence) / 2
        
        params = {
            'direction_score': direction_score,
            'confidence': confidence,
            'iv_percentile': iv_percentile,
        }
        
        # === STRATEGY SELECTION LOGIC ===
        
        # Strong directional with high confidence
        if abs(direction_score) >= 0.6 and confidence >= 0.6:
            if direction_score > 0:
                # Bullish
                if iv_high:
                    # High IV - use spread to reduce cost
                    return OptionsStrategy.BULL_CALL_SPREAD, params
                else:
                    # Low/Normal IV - long call
                    return OptionsStrategy.LONG_CALL, params
            else:
                # Bearish
                if iv_high:
                    return OptionsStrategy.BEAR_PUT_SPREAD, params
                else:
                    return OptionsStrategy.LONG_PUT, params
        
        # Moderate directional
        elif abs(direction_score) >= 0.3:
            if direction_score > 0:
                return OptionsStrategy.BULL_CALL_SPREAD, params
            else:
                return OptionsStrategy.BEAR_PUT_SPREAD, params
        
        # Low directional conviction
        else:
            # Volatility play
            if vol_expanding or liquidity.bb_squeeze:
                # Expect volatility expansion
                if iv_low:
                    # Cheap vol - buy straddle
                    return OptionsStrategy.LONG_STRADDLE, params
                else:
                    # Moderate/high vol - strangle
                    return OptionsStrategy.LONG_STRANGLE, params
            else:
                # Range-bound, sell premium
                if iv_high:
                    return OptionsStrategy.IRON_CONDOR, params
                else:
                    # Low vol, expect breakout
                    return OptionsStrategy.LONG_STRADDLE, params
        
        return OptionsStrategy.LONG_CALL, params
    
    def price_option_bs(
        self,
        S: float,          # Underlying price
        K: float,          # Strike price
        T: float,          # Time to expiration (years)
        r: float,          # Risk-free rate
        sigma: float,      # Volatility
        option_type: str,  # 'call' or 'put'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Black-Scholes option pricing with Greeks.
        
        Returns:
            (option_price, greeks_dict)
        """
        if T <= 0:
            # Expired
            if option_type == 'call':
                return max(0, S - K), {'delta': 1.0 if S > K else 0.0, 'gamma': 0, 'theta': 0, 'vega': 0}
            else:
                return max(0, K - S), {'delta': -1.0 if K > S else 0.0, 'gamma': 0, 'theta': 0, 'vega': 0}
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Standard normal CDF and PDF
        from scipy.stats import norm
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
        
        # Greeks
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2))
        theta = theta / 365  # Daily theta
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% IV change
        
        greeks = {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
        }
        
        return price, greeks
    
    def construct_position(
        self,
        strategy: OptionsStrategy,
        underlying_price: float,
        volatility: float,
        dte: int = 30,
        params: Dict[str, Any] = None,
    ) -> OptionsPosition:
        """
        Construct a theoretical options position.
        
        Args:
            strategy: Strategy type
            underlying_price: Current stock price
            volatility: Annualized volatility
            dte: Days to expiration
            params: Additional parameters
        """
        position = OptionsPosition(strategy=strategy)
        position.underlying_price = underlying_price
        position.entry_date = datetime.now()
        
        T = dte / 365
        r = self.RISK_FREE_RATE
        S = underlying_price
        
        # ATM strike
        atm_strike = round(S / 5) * 5  # Round to nearest $5
        
        if strategy == OptionsStrategy.LONG_CALL:
            price, greeks = self.price_option_bs(S, atm_strike, T, r, volatility, 'call')
            position.legs = [{'type': 'call', 'strike': atm_strike, 'action': 'buy', 'price': price}]
            position.entry_price = price
            position.delta = greeks['delta']
            position.gamma = greeks['gamma']
            position.theta = greeks['theta']
            position.vega = greeks['vega']
            position.max_loss = price
            position.max_profit = float('inf')
            position.breakeven = [atm_strike + price]
            
        elif strategy == OptionsStrategy.LONG_PUT:
            price, greeks = self.price_option_bs(S, atm_strike, T, r, volatility, 'put')
            position.legs = [{'type': 'put', 'strike': atm_strike, 'action': 'buy', 'price': price}]
            position.entry_price = price
            position.delta = greeks['delta']
            position.gamma = greeks['gamma']
            position.theta = greeks['theta']
            position.vega = greeks['vega']
            position.max_loss = price
            position.max_profit = atm_strike - price
            position.breakeven = [atm_strike - price]
            
        elif strategy == OptionsStrategy.BULL_CALL_SPREAD:
            # Buy ATM call, sell OTM call
            long_strike = atm_strike
            short_strike = atm_strike + 10
            
            long_price, long_greeks = self.price_option_bs(S, long_strike, T, r, volatility, 'call')
            short_price, short_greeks = self.price_option_bs(S, short_strike, T, r, volatility, 'call')
            
            position.legs = [
                {'type': 'call', 'strike': long_strike, 'action': 'buy', 'price': long_price},
                {'type': 'call', 'strike': short_strike, 'action': 'sell', 'price': short_price},
            ]
            position.entry_price = long_price - short_price
            position.delta = long_greeks['delta'] - short_greeks['delta']
            position.max_loss = position.entry_price
            position.max_profit = (short_strike - long_strike) - position.entry_price
            position.breakeven = [long_strike + position.entry_price]
            
        elif strategy == OptionsStrategy.BEAR_PUT_SPREAD:
            # Buy ATM put, sell OTM put
            long_strike = atm_strike
            short_strike = atm_strike - 10
            
            long_price, long_greeks = self.price_option_bs(S, long_strike, T, r, volatility, 'put')
            short_price, short_greeks = self.price_option_bs(S, short_strike, T, r, volatility, 'put')
            
            position.legs = [
                {'type': 'put', 'strike': long_strike, 'action': 'buy', 'price': long_price},
                {'type': 'put', 'strike': short_strike, 'action': 'sell', 'price': short_price},
            ]
            position.entry_price = long_price - short_price
            position.delta = long_greeks['delta'] - short_greeks['delta']
            position.max_loss = position.entry_price
            position.max_profit = (long_strike - short_strike) - position.entry_price
            position.breakeven = [long_strike - position.entry_price]
            
        elif strategy == OptionsStrategy.LONG_STRADDLE:
            # Buy ATM call and ATM put
            call_price, call_greeks = self.price_option_bs(S, atm_strike, T, r, volatility, 'call')
            put_price, put_greeks = self.price_option_bs(S, atm_strike, T, r, volatility, 'put')
            
            position.legs = [
                {'type': 'call', 'strike': atm_strike, 'action': 'buy', 'price': call_price},
                {'type': 'put', 'strike': atm_strike, 'action': 'buy', 'price': put_price},
            ]
            position.entry_price = call_price + put_price
            position.delta = call_greeks['delta'] + put_greeks['delta']
            position.gamma = call_greeks['gamma'] + put_greeks['gamma']
            position.vega = call_greeks['vega'] + put_greeks['vega']
            position.max_loss = position.entry_price
            position.max_profit = float('inf')
            position.breakeven = [atm_strike - position.entry_price, atm_strike + position.entry_price]
            
        elif strategy == OptionsStrategy.LONG_STRANGLE:
            # Buy OTM call and OTM put
            call_strike = atm_strike + 10
            put_strike = atm_strike - 10
            
            call_price, call_greeks = self.price_option_bs(S, call_strike, T, r, volatility, 'call')
            put_price, put_greeks = self.price_option_bs(S, put_strike, T, r, volatility, 'put')
            
            position.legs = [
                {'type': 'call', 'strike': call_strike, 'action': 'buy', 'price': call_price},
                {'type': 'put', 'strike': put_strike, 'action': 'buy', 'price': put_price},
            ]
            position.entry_price = call_price + put_price
            position.delta = call_greeks['delta'] + put_greeks['delta']
            position.max_loss = position.entry_price
            position.max_profit = float('inf')
            position.breakeven = [put_strike - position.entry_price, call_strike + position.entry_price]
            
        elif strategy == OptionsStrategy.IRON_CONDOR:
            # Sell OTM put spread + Sell OTM call spread
            put_long_strike = atm_strike - 20
            put_short_strike = atm_strike - 10
            call_short_strike = atm_strike + 10
            call_long_strike = atm_strike + 20
            
            put_long_price, _ = self.price_option_bs(S, put_long_strike, T, r, volatility, 'put')
            put_short_price, _ = self.price_option_bs(S, put_short_strike, T, r, volatility, 'put')
            call_short_price, _ = self.price_option_bs(S, call_short_strike, T, r, volatility, 'call')
            call_long_price, _ = self.price_option_bs(S, call_long_strike, T, r, volatility, 'call')
            
            credit = (put_short_price - put_long_price) + (call_short_price - call_long_price)
            
            position.legs = [
                {'type': 'put', 'strike': put_long_strike, 'action': 'buy', 'price': put_long_price},
                {'type': 'put', 'strike': put_short_strike, 'action': 'sell', 'price': put_short_price},
                {'type': 'call', 'strike': call_short_strike, 'action': 'sell', 'price': call_short_price},
                {'type': 'call', 'strike': call_long_strike, 'action': 'buy', 'price': call_long_price},
            ]
            position.entry_price = -credit  # Negative = credit received
            position.max_profit = credit
            position.max_loss = 10 - credit  # Wing width minus credit
            position.breakeven = [put_short_strike - credit, call_short_strike + credit]
        
        return position
    
    def calculate_exit_value(
        self,
        position: OptionsPosition,
        underlying_price: float,
        remaining_dte: int,
        current_vol: float,
    ) -> float:
        """Calculate current value of position."""
        
        T = remaining_dte / 365
        r = self.RISK_FREE_RATE
        S = underlying_price
        
        total_value = 0.0
        
        for leg in position.legs:
            price, _ = self.price_option_bs(
                S, leg['strike'], T, r, current_vol, leg['type']
            )
            
            if leg['action'] == 'buy':
                total_value += price
            else:
                total_value -= price
        
        return total_value


# =============================================================================
# GNOSIS OPTIONS BACKTEST ENGINE
# =============================================================================

@dataclass
class GnosisOptionsBacktestConfig:
    """Configuration for GNOSIS options backtesting."""
    
    # Symbols
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL"])
    
    # Date range
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-01"
    
    # Capital
    initial_capital: float = 100_000.0
    max_position_pct: float = 0.05        # Max 5% per trade
    max_positions: int = 5
    
    # Options parameters
    default_dte: int = 30                  # Days to expiration
    min_dte_at_entry: int = 21
    close_at_dte: int = 7                  # Close before expiration
    
    # Signal thresholds
    min_confidence: float = 0.5
    min_signal_strength: int = 3           # SignalStrength.MODERATE
    
    # Risk management
    stop_loss_pct: float = 0.50            # 50% of premium
    take_profit_pct: float = 1.00          # 100% profit
    
    # Output
    output_dir: str = "runs/gnosis_options_backtests"


class GnosisOptionsBacktester:
    """
    GNOSIS Options Backtesting Engine.
    
    Integrates:
    - Price Physics (mass, energy, momentum)
    - Sentiment (RSI, MACD, momentum)
    - Liquidity (A/D, Bollinger, OBV)
    - Predictive Cones
    - Support/Resistance
    
    Generates theoretical options trades that can profit in any market condition.
    """
    
    def __init__(self, config: GnosisOptionsBacktestConfig):
        self.config = config
        
        # Engines
        self.physics_engine = PricePhysicsEngine()
        self.sentiment_engine = SentimentEngine()
        self.liquidity_engine = LiquidityEngine()
        self.prediction_engine = PredictionEngine()
        self.strategy_selector = OptionsStrategySelector()
        
        # State
        self.capital = config.initial_capital
        self.positions: Dict[str, OptionsPosition] = {}
        self.trades: List[OptionsPosition] = []
        self.equity_curve: List[Dict] = []
        
        logger.info(f"GnosisOptionsBacktester initialized | symbols={config.symbols}")
    
    def run_backtest(self) -> Dict[str, Any]:
        """Run the full GNOSIS options backtest."""
        
        print("\n" + "="*70)
        print("  GNOSIS OPTIONS BACKTEST ENGINE")
        print("  Price Physics + Sentiment + Liquidity + Predictive Cones")
        print("="*70)
        print(f"\nSymbols: {', '.join(self.config.symbols)}")
        print(f"Period: {self.config.start_date} to {self.config.end_date}")
        print(f"Initial Capital: ${self.config.initial_capital:,.2f}")
        print()
        
        # Fetch data for all symbols
        all_data = self._fetch_all_data()
        
        if not all_data:
            raise ValueError("No data fetched")
        
        # Get common date range
        all_dates = set()
        for symbol, df in all_data.items():
            all_dates.update(df['timestamp'].tolist())
        all_dates = sorted(all_dates)
        
        print(f"Data: {len(all_dates)} trading days")
        print(f"Period: {all_dates[0]} to {all_dates[-1]}")
        print()
        
        # Reset state
        self.capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        warmup = 50
        
        # Process each day
        for i in range(warmup, len(all_dates)):
            date = all_dates[i]
            
            for symbol, df in all_data.items():
                # Get data up to current date
                mask = df['timestamp'] <= date
                if mask.sum() < warmup:
                    continue
                
                current_df = df[mask].tail(warmup + 1)
                
                # Get current bar
                current_bar = current_df.iloc[-1]
                history = current_df.iloc[:-1]
                
                # Compute all signals
                signals = self._compute_signals(symbol, current_bar, history)
                
                # Check exits
                if symbol in self.positions:
                    self._check_exit(symbol, current_bar, date, signals)
                
                # Check entries
                if symbol not in self.positions:
                    self._check_entry(symbol, current_bar, date, signals)
            
            # Record equity
            self._record_equity(date, all_data, all_dates[i])
        
        # Close remaining positions
        for symbol in list(self.positions.keys()):
            if symbol in all_data:
                final_bar = all_data[symbol].iloc[-1]
                self._close_position(symbol, final_bar, all_dates[-1], "end_of_backtest")
        
        # Calculate results
        results = self._calculate_results()
        self._print_results(results)
        self._save_results(results)
        
        return results
    
    def _fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all symbols."""
        
        all_data = {}
        
        for symbol in self.config.symbols:
            try:
                df = self._fetch_symbol_data(symbol)
                if df is not None and len(df) > 0:
                    all_data[symbol] = df
                    logger.info(f"Fetched {len(df)} bars for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        
        return all_data
    
    def _fetch_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch data for a single symbol."""
        
        # Try Massive.com (Polygon) first
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
                    from_=self.config.start_date,
                    to=self.config.end_date,
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
                    return df
                    
        except Exception as e:
            logger.warning(f"Massive.com fetch failed for {symbol}: {e}")
        
        # Fallback to Alpaca
        try:
            from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
            
            adapter = AlpacaMarketDataAdapter()
            
            start = datetime.strptime(self.config.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end = datetime.strptime(self.config.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            
            bars = adapter.get_bars(
                symbol=symbol,
                start=start,
                end=end,
                timeframe="1Day",
            )
            
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
                return df
                
        except Exception as e:
            logger.error(f"Alpaca fetch failed for {symbol}: {e}")
        
        return None
    
    def _compute_signals(
        self,
        symbol: str,
        current_bar: pd.Series,
        history: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute all GNOSIS signals for a symbol."""
        
        prices = pd.concat([history['close'], pd.Series([current_bar['close']])])
        highs = pd.concat([history['high'], pd.Series([current_bar['high']])])
        lows = pd.concat([history['low'], pd.Series([current_bar['low']])])
        volumes = pd.concat([history['volume'], pd.Series([current_bar['volume']])])
        
        # Price Physics
        particle = self.physics_engine.compute_particle_state(
            symbol, prices, volumes
        )
        
        # Sentiment
        sentiment = self.sentiment_engine.compute_sentiment(prices, highs, lows)
        
        # Liquidity
        liquidity = self.liquidity_engine.compute_liquidity(prices, highs, lows, volumes)
        
        # Predictive Cone
        cone = self.prediction_engine.compute_predictive_cone(prices)
        
        # Support/Resistance
        sr = self.prediction_engine.compute_support_resistance(prices, highs, lows)
        
        # Market Regime
        regime = self._classify_regime(sentiment, liquidity, particle)
        
        # IV percentile (estimate from historical vol)
        returns = prices.pct_change().dropna()
        current_vol = returns.tail(20).std() * np.sqrt(252)
        historical_vol = returns.rolling(252).std() * np.sqrt(252)
        iv_percentile = (historical_vol < current_vol).mean() * 100 if len(historical_vol) >= 252 else 50
        
        # Select strategy
        strategy, params = self.strategy_selector.select_strategy(
            regime, sentiment, liquidity, particle, cone, sr, iv_percentile
        )
        
        return {
            'particle': particle,
            'sentiment': sentiment,
            'liquidity': liquidity,
            'cone': cone,
            'sr': sr,
            'regime': regime,
            'strategy': strategy,
            'params': params,
            'volatility': current_vol if current_vol > 0 else 0.20,
            'iv_percentile': iv_percentile,
        }
    
    def _classify_regime(
        self,
        sentiment: SentimentState,
        liquidity: LiquidityState,
        particle: PriceParticle,
    ) -> MarketRegime:
        """Classify current market regime."""
        
        combined_score = (
            sentiment.overall_sentiment * 0.4 +
            liquidity.liquidity_score * 0.3 +
            np.clip(particle.momentum / 5, -1, 1) * 0.3
        )
        
        # Check for volatility regimes
        if liquidity.bb_squeeze:
            return MarketRegime.LOW_VOL
        
        if liquidity.bb_width > 0.10:  # Wide bands
            return MarketRegime.HIGH_VOL
        
        # Directional regimes
        if combined_score >= 0.6:
            return MarketRegime.STRONG_BULL
        elif combined_score >= 0.3:
            return MarketRegime.BULL
        elif combined_score <= -0.6:
            return MarketRegime.STRONG_BEAR
        elif combined_score <= -0.3:
            return MarketRegime.BEAR
        else:
            return MarketRegime.NEUTRAL
    
    def _check_entry(
        self,
        symbol: str,
        current_bar: pd.Series,
        date: datetime,
        signals: Dict[str, Any],
    ):
        """Check if we should enter a new position."""
        
        # Check position limits
        if len(self.positions) >= self.config.max_positions:
            return
        
        # Check confidence
        confidence = signals['params']['confidence']
        if confidence < self.config.min_confidence:
            return
        
        # Check signal strength
        if signals['sentiment'].sentiment_strength.value < self.config.min_signal_strength:
            return
        
        # Construct position
        strategy = signals['strategy']
        position = self.strategy_selector.construct_position(
            strategy=strategy,
            underlying_price=current_bar['close'],
            volatility=signals['volatility'],
            dte=self.config.default_dte,
            params=signals['params'],
        )
        
        position.entry_date = date
        position.confidence = confidence
        position.regime = signals['regime']
        position.signal_strength = signals['sentiment'].sentiment_strength
        
        # Calculate position size
        max_risk = self.capital * self.config.max_position_pct
        
        if position.max_loss > 0:
            contracts = int(max_risk / (position.max_loss * 100))
            contracts = max(1, min(contracts, 10))  # 1-10 contracts
        else:
            contracts = 1
        
        position.contracts = contracts
        position.entry_price *= contracts  # Total premium
        position.max_loss *= contracts
        position.max_profit = position.max_profit * contracts if position.max_profit != float('inf') else float('inf')
        
        # Deduct premium from capital
        self.capital -= abs(position.entry_price) * 100  # Per contract multiplier
        
        self.positions[symbol] = position
        
        logger.debug(
            f"OPEN {strategy.value} on {symbol} @ ${current_bar['close']:.2f} | "
            f"Premium: ${position.entry_price:.2f} | "
            f"Conf: {confidence:.2f} | Regime: {signals['regime'].value}"
        )
    
    def _check_exit(
        self,
        symbol: str,
        current_bar: pd.Series,
        date: datetime,
        signals: Dict[str, Any],
    ):
        """Check if we should exit a position."""
        
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Calculate current value
        dte_remaining = max(1, self.config.default_dte - (date - position.entry_date).days)
        current_value = self.strategy_selector.calculate_exit_value(
            position,
            current_bar['close'],
            dte_remaining,
            signals['volatility'],
        )
        
        pnl_pct = (current_value - abs(position.entry_price)) / abs(position.entry_price) if position.entry_price != 0 else 0
        
        should_exit = False
        exit_reason = ""
        
        # Stop loss
        if pnl_pct <= -self.config.stop_loss_pct:
            should_exit = True
            exit_reason = "stop_loss"
        
        # Take profit
        elif pnl_pct >= self.config.take_profit_pct:
            should_exit = True
            exit_reason = "take_profit"
        
        # Close before expiration
        elif dte_remaining <= self.config.close_at_dte:
            should_exit = True
            exit_reason = "dte_exit"
        
        # Signal reversal
        elif signals['sentiment'].overall_sentiment * (1 if 'CALL' in position.strategy.value or 'BULL' in position.strategy.value else -1) < -0.5:
            should_exit = True
            exit_reason = "signal_reversal"
        
        if should_exit:
            self._close_position(symbol, current_bar, date, exit_reason, current_value)
    
    def _close_position(
        self,
        symbol: str,
        current_bar: pd.Series,
        date: datetime,
        exit_reason: str,
        exit_value: Optional[float] = None,
    ):
        """Close a position."""
        
        if symbol not in self.positions:
            return
        
        position = self.positions.pop(symbol)
        
        # Calculate exit value if not provided
        if exit_value is None:
            dte_remaining = max(1, self.config.default_dte - (date - position.entry_date).days)
            exit_value = self.strategy_selector.calculate_exit_value(
                position,
                current_bar['close'],
                dte_remaining,
                0.20,  # Assume 20% vol if not available
            )
        
        position.exit_date = date
        position.exit_price = exit_value
        position.exit_underlying = current_bar['close']
        
        # Calculate P&L
        if position.entry_price < 0:  # Credit strategy
            position.gross_pnl = (-position.entry_price - exit_value) * 100 * position.contracts
        else:  # Debit strategy
            position.gross_pnl = (exit_value - position.entry_price) * 100 * position.contracts
        
        position.net_pnl = position.gross_pnl * 0.98  # 2% for commissions/slippage
        position.pnl_pct = position.net_pnl / (abs(position.entry_price) * 100 * position.contracts) if position.entry_price != 0 else 0
        
        # Return capital
        self.capital += abs(position.entry_price) * 100 + position.net_pnl
        
        self.trades.append(position)
        
        logger.debug(
            f"CLOSE {position.strategy.value} on {symbol} | "
            f"P&L: ${position.net_pnl:,.2f} ({position.pnl_pct:.1%}) | "
            f"Reason: {exit_reason}"
        )
    
    def _record_equity(self, date: datetime, all_data: Dict, current_date: datetime):
        """Record equity curve point."""
        
        position_value = 0
        for symbol, position in self.positions.items():
            if symbol in all_data:
                df = all_data[symbol]
                mask = df['timestamp'] <= current_date
                if mask.sum() > 0:
                    current_price = df[mask].iloc[-1]['close']
                    dte = max(1, self.config.default_dte - (current_date - position.entry_date).days)
                    
                    value = self.strategy_selector.calculate_exit_value(
                        position, current_price, dte, 0.20
                    )
                    position_value += value * 100 * position.contracts
        
        total_equity = self.capital + position_value
        
        self.equity_curve.append({
            'date': date,
            'equity': total_equity,
            'positions': len(self.positions),
        })
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate comprehensive results."""
        
        results = {
            'config': {
                'symbols': self.config.symbols,
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'initial_capital': self.config.initial_capital,
            },
            'summary': {},
            'trades': [],
            'by_strategy': {},
            'by_regime': {},
            'by_symbol': {},
        }
        
        if not self.trades:
            return results
        
        # Basic stats
        winners = [t for t in self.trades if t.net_pnl > 0]
        losers = [t for t in self.trades if t.net_pnl <= 0]
        
        total_pnl = sum(t.net_pnl for t in self.trades)
        
        results['summary'] = {
            'initial_capital': self.config.initial_capital,
            'final_capital': self.capital,
            'total_pnl': total_pnl,
            'return_pct': total_pnl / self.config.initial_capital,
            'total_trades': len(self.trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(self.trades) if self.trades else 0,
            'avg_win': np.mean([t.net_pnl for t in winners]) if winners else 0,
            'avg_loss': np.mean([t.net_pnl for t in losers]) if losers else 0,
            'largest_win': max([t.net_pnl for t in winners]) if winners else 0,
            'largest_loss': min([t.net_pnl for t in losers]) if losers else 0,
            'profit_factor': abs(sum(t.net_pnl for t in winners) / sum(t.net_pnl for t in losers)) if losers and sum(t.net_pnl for t in losers) != 0 else float('inf'),
        }
        
        # By strategy
        for trade in self.trades:
            strategy = trade.strategy.value
            if strategy not in results['by_strategy']:
                results['by_strategy'][strategy] = {'trades': 0, 'pnl': 0, 'winners': 0}
            results['by_strategy'][strategy]['trades'] += 1
            results['by_strategy'][strategy]['pnl'] += trade.net_pnl
            if trade.net_pnl > 0:
                results['by_strategy'][strategy]['winners'] += 1
        
        # By regime
        for trade in self.trades:
            regime = trade.regime.value
            if regime not in results['by_regime']:
                results['by_regime'][regime] = {'trades': 0, 'pnl': 0, 'winners': 0}
            results['by_regime'][regime]['trades'] += 1
            results['by_regime'][regime]['pnl'] += trade.net_pnl
            if trade.net_pnl > 0:
                results['by_regime'][regime]['winners'] += 1
        
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
            
            results['summary']['max_drawdown_pct'] = max_dd
            
            if len(equities) > 1:
                daily_returns = pd.Series(equities).pct_change().dropna()
                if len(daily_returns) > 0 and daily_returns.std() > 0:
                    results['summary']['sharpe_ratio'] = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        
        return results
    
    def _print_results(self, results: Dict[str, Any]):
        """Print formatted results."""
        
        print("\n" + "="*70)
        print("  GNOSIS OPTIONS BACKTEST RESULTS")
        print("="*70)
        
        s = results['summary']
        
        print(f"\n{'CAPITAL & RETURNS':=^60}")
        print(f"  Initial Capital:    ${s.get('initial_capital', 0):>12,.2f}")
        print(f"  Final Capital:      ${s.get('final_capital', 0):>12,.2f}")
        print(f"  Total P&L:          ${s.get('total_pnl', 0):>12,.2f} ({s.get('return_pct', 0):.2%})")
        
        print(f"\n{'TRADE STATISTICS':=^60}")
        print(f"  Total Trades:       {s.get('total_trades', 0):>12}")
        print(f"  Winners:            {s.get('winners', 0):>12} ({s.get('win_rate', 0):.1%})")
        print(f"  Losers:             {s.get('losers', 0):>12}")
        print(f"  Profit Factor:      {s.get('profit_factor', 0):>12.2f}")
        print(f"  Avg Win:            ${s.get('avg_win', 0):>11,.2f}")
        print(f"  Avg Loss:           ${s.get('avg_loss', 0):>11,.2f}")
        
        if 'max_drawdown_pct' in s:
            print(f"\n{'RISK METRICS':=^60}")
            print(f"  Max Drawdown:       {s.get('max_drawdown_pct', 0):>12.2%}")
            if 'sharpe_ratio' in s:
                print(f"  Sharpe Ratio:       {s.get('sharpe_ratio', 0):>12.2f}")
        
        if results['by_strategy']:
            print(f"\n{'PERFORMANCE BY STRATEGY':=^60}")
            print(f"  {'Strategy':<25} {'Trades':>8} {'Win Rate':>10} {'P&L':>12}")
            print(f"  {'-'*55}")
            for strategy, data in sorted(results['by_strategy'].items(), key=lambda x: -x[1]['pnl']):
                wr = data['winners'] / data['trades'] if data['trades'] > 0 else 0
                print(f"  {strategy:<25} {data['trades']:>8} {wr:>9.1%} ${data['pnl']:>11,.2f}")
        
        if results['by_regime']:
            print(f"\n{'PERFORMANCE BY REGIME':=^60}")
            print(f"  {'Regime':<20} {'Trades':>8} {'Win Rate':>10} {'P&L':>12}")
            print(f"  {'-'*50}")
            for regime, data in sorted(results['by_regime'].items(), key=lambda x: -x[1]['pnl']):
                wr = data['winners'] / data['trades'] if data['trades'] > 0 else 0
                print(f"  {regime:<20} {data['trades']:>8} {wr:>9.1%} ${data['pnl']:>11,.2f}")
        
        print("\n" + "="*70)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gnosis_options_{timestamp}.json"
        
        # Convert trades to serializable format
        trades_data = []
        for t in self.trades:
            trades_data.append({
                'strategy': t.strategy.value,
                'entry_date': t.entry_date.isoformat() if t.entry_date else None,
                'exit_date': t.exit_date.isoformat() if t.exit_date else None,
                'underlying_price': t.underlying_price,
                'exit_underlying': t.exit_underlying,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'contracts': t.contracts,
                'net_pnl': t.net_pnl,
                'pnl_pct': t.pnl_pct,
                'regime': t.regime.value,
                'confidence': t.confidence,
            })
        
        output = {
            'summary': results['summary'],
            'by_strategy': results['by_strategy'],
            'by_regime': results['by_regime'],
            'trades': trades_data,
        }
        
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run GNOSIS options backtest."""
    
    config = GnosisOptionsBacktestConfig(
        symbols=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "META", "AMZN", "GOOGL"],
        start_date="2020-01-01",
        end_date="2024-12-01",
        initial_capital=100_000.0,
        max_position_pct=0.05,
        max_positions=5,
        default_dte=30,
        min_confidence=0.4,
        min_signal_strength=2,
    )
    
    backtester = GnosisOptionsBacktester(config)
    results = backtester.run_backtest()
    
    return results


if __name__ == "__main__":
    main()
