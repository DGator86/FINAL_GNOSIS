"""
Volatility Surface Modeling

Models implied volatility across strikes and expirations:
- IV surface construction
- Term structure analysis
- Volatility smile/skew modeling
- Surface interpolation
- Arbitrage detection

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import statistics

from loguru import logger

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("NumPy not available - using pure Python implementations")


class InterpolationMethod(str, Enum):
    """Surface interpolation methods."""
    LINEAR = "linear"
    CUBIC = "cubic"
    SVI = "svi"  # Stochastic Volatility Inspired
    SABR = "sabr"


class VolatilityRegime(str, Enum):
    """Market volatility regime."""
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class IVPoint:
    """Single IV data point."""
    strike: float
    expiration: date
    iv: float
    option_type: str  # 'call' or 'put'
    
    # Optional market data
    bid_iv: Optional[float] = None
    ask_iv: Optional[float] = None
    mid_iv: Optional[float] = None
    
    volume: int = 0
    open_interest: int = 0
    
    # Moneyness
    underlying_price: float = 0.0
    
    @property
    def moneyness(self) -> float:
        """Strike / Spot ratio."""
        if self.underlying_price == 0:
            return 1.0
        return self.strike / self.underlying_price
    
    @property
    def log_moneyness(self) -> float:
        """Log moneyness (ln(K/S))."""
        if self.underlying_price == 0:
            return 0.0
        return math.log(self.strike / self.underlying_price)
    
    @property
    def days_to_expiry(self) -> int:
        """Days until expiration."""
        return (self.expiration - date.today()).days
    
    @property
    def time_to_expiry(self) -> float:
        """Time to expiry in years."""
        return max(1, self.days_to_expiry) / 365.0


@dataclass
class TermStructure:
    """Volatility term structure."""
    underlying: str
    as_of: datetime
    
    # ATM IV by expiration
    expirations: List[date] = field(default_factory=list)
    atm_ivs: List[float] = field(default_factory=list)
    
    # Forward volatilities
    forward_vols: List[float] = field(default_factory=list)
    
    @property
    def is_contango(self) -> bool:
        """Check if term structure is in contango (upward sloping)."""
        if len(self.atm_ivs) < 2:
            return False
        return self.atm_ivs[-1] > self.atm_ivs[0]
    
    @property
    def is_backwardation(self) -> bool:
        """Check if term structure is in backwardation (downward sloping)."""
        if len(self.atm_ivs) < 2:
            return False
        return self.atm_ivs[-1] < self.atm_ivs[0]
    
    @property
    def slope(self) -> float:
        """Term structure slope (IV change per month)."""
        if len(self.atm_ivs) < 2 or len(self.expirations) < 2:
            return 0.0
        
        days_diff = (self.expirations[-1] - self.expirations[0]).days
        if days_diff == 0:
            return 0.0
        
        iv_diff = self.atm_ivs[-1] - self.atm_ivs[0]
        return iv_diff / (days_diff / 30)  # Per month
    
    def get_atm_iv(self, target_dte: int) -> float:
        """Interpolate ATM IV for target DTE."""
        if not self.expirations:
            return 0.0
        
        target_date = date.today() + timedelta(days=target_dte)
        
        # Find bracketing expirations
        for i, exp in enumerate(self.expirations):
            if exp >= target_date:
                if i == 0:
                    return self.atm_ivs[0]
                
                # Linear interpolation
                prev_exp = self.expirations[i-1]
                prev_iv = self.atm_ivs[i-1]
                curr_iv = self.atm_ivs[i]
                
                total_days = (exp - prev_exp).days
                target_days = (target_date - prev_exp).days
                
                if total_days == 0:
                    return curr_iv
                
                return prev_iv + (curr_iv - prev_iv) * (target_days / total_days)
        
        return self.atm_ivs[-1] if self.atm_ivs else 0.0


@dataclass
class VolatilitySmile:
    """Volatility smile for single expiration."""
    underlying: str
    expiration: date
    underlying_price: float
    
    # Smile data
    strikes: List[float] = field(default_factory=list)
    ivs: List[float] = field(default_factory=list)
    
    # Smile parameters
    atm_strike: float = 0.0
    atm_iv: float = 0.0
    
    @property
    def skew(self) -> float:
        """
        25-delta skew (put IV - call IV).
        Positive = puts more expensive (bearish sentiment).
        """
        if len(self.strikes) < 3:
            return 0.0
        
        # Find approximate 25-delta strikes
        otm_put_idx = 0
        otm_call_idx = len(self.strikes) - 1
        
        # Rough approximation: 25-delta is about 5-7% OTM
        for i, strike in enumerate(self.strikes):
            moneyness = strike / self.underlying_price
            if 0.93 <= moneyness <= 0.95:
                otm_put_idx = i
            elif 1.05 <= moneyness <= 1.07:
                otm_call_idx = i
        
        return self.ivs[otm_put_idx] - self.ivs[otm_call_idx]
    
    @property
    def wings(self) -> float:
        """Wing premium (average OTM IV vs ATM)."""
        if len(self.strikes) < 3:
            return 0.0
        
        atm_idx = len(self.strikes) // 2
        wing_ivs = [self.ivs[0], self.ivs[-1]]
        return statistics.mean(wing_ivs) - self.ivs[atm_idx]
    
    @property
    def curvature(self) -> float:
        """Smile curvature (butterfly spread IV)."""
        if len(self.strikes) < 3:
            return 0.0
        
        # Use 3 equally spaced strikes
        n = len(self.strikes)
        left = self.ivs[n//4] if n >= 4 else self.ivs[0]
        center = self.ivs[n//2]
        right = self.ivs[3*n//4] if n >= 4 else self.ivs[-1]
        
        return (left + right) / 2 - center
    
    def get_iv(self, strike: float) -> float:
        """Interpolate IV for given strike."""
        if not self.strikes:
            return 0.0
        
        if strike <= self.strikes[0]:
            return self.ivs[0]
        if strike >= self.strikes[-1]:
            return self.ivs[-1]
        
        # Linear interpolation
        for i in range(len(self.strikes) - 1):
            if self.strikes[i] <= strike <= self.strikes[i+1]:
                t = (strike - self.strikes[i]) / (self.strikes[i+1] - self.strikes[i])
                return self.ivs[i] + t * (self.ivs[i+1] - self.ivs[i])
        
        return self.atm_iv


@dataclass
class VolatilitySurface:
    """Full volatility surface."""
    underlying: str
    underlying_price: float
    as_of: datetime
    
    # Raw data points
    points: List[IVPoint] = field(default_factory=list)
    
    # Structured views
    smiles: Dict[date, VolatilitySmile] = field(default_factory=dict)
    term_structure: Optional[TermStructure] = None
    
    # Surface grid (for visualization)
    strike_grid: List[float] = field(default_factory=list)
    expiry_grid: List[date] = field(default_factory=list)
    iv_grid: List[List[float]] = field(default_factory=list)
    
    # Analytics
    regime: VolatilityRegime = VolatilityRegime.NORMAL
    
    def get_iv(self, strike: float, expiration: date) -> float:
        """Get interpolated IV for strike/expiration."""
        # Try exact smile first
        if expiration in self.smiles:
            return self.smiles[expiration].get_iv(strike)
        
        # Interpolate between expirations
        expirations = sorted(self.smiles.keys())
        
        if not expirations:
            return 0.0
        
        if expiration <= expirations[0]:
            return self.smiles[expirations[0]].get_iv(strike)
        if expiration >= expirations[-1]:
            return self.smiles[expirations[-1]].get_iv(strike)
        
        # Find bracketing expirations
        for i, exp in enumerate(expirations):
            if exp > expiration:
                prev_exp = expirations[i-1]
                prev_iv = self.smiles[prev_exp].get_iv(strike)
                curr_iv = self.smiles[exp].get_iv(strike)
                
                # Time-weighted interpolation
                total_days = (exp - prev_exp).days
                target_days = (expiration - prev_exp).days
                
                if total_days == 0:
                    return curr_iv
                
                return prev_iv + (curr_iv - prev_iv) * (target_days / total_days)
        
        return 0.0
    
    def get_atm_iv(self, expiration: date) -> float:
        """Get ATM IV for expiration."""
        return self.get_iv(self.underlying_price, expiration)
    
    def get_skew(self, expiration: date) -> float:
        """Get skew for expiration."""
        if expiration in self.smiles:
            return self.smiles[expiration].skew
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "underlying": self.underlying,
            "underlying_price": self.underlying_price,
            "as_of": self.as_of.isoformat(),
            "regime": self.regime.value,
            "num_points": len(self.points),
            "expirations": [exp.isoformat() for exp in self.smiles.keys()],
            "term_structure": {
                "is_contango": self.term_structure.is_contango if self.term_structure else None,
                "slope": self.term_structure.slope if self.term_structure else None,
            },
        }


class VolatilitySurfaceBuilder:
    """
    Builds and analyzes volatility surfaces.
    
    Features:
    - Surface construction from option chain
    - Smile fitting (SVI, SABR)
    - Term structure analysis
    - Arbitrage detection
    """
    
    def __init__(self, interpolation: InterpolationMethod = InterpolationMethod.LINEAR):
        """Initialize surface builder."""
        self.interpolation = interpolation
        self._surfaces: Dict[str, VolatilitySurface] = {}
        
        logger.info(f"VolatilitySurfaceBuilder initialized with {interpolation.value} interpolation")
    
    def build_surface(
        self,
        underlying: str,
        underlying_price: float,
        iv_points: List[IVPoint],
    ) -> VolatilitySurface:
        """
        Build volatility surface from IV points.
        
        Args:
            underlying: Underlying symbol
            underlying_price: Current underlying price
            iv_points: List of IV data points
        """
        surface = VolatilitySurface(
            underlying=underlying,
            underlying_price=underlying_price,
            as_of=datetime.now(),
            points=iv_points,
        )
        
        # Group by expiration
        by_expiration = defaultdict(list)
        for point in iv_points:
            point.underlying_price = underlying_price
            by_expiration[point.expiration].append(point)
        
        # Build smiles
        for exp, points in sorted(by_expiration.items()):
            smile = self._build_smile(underlying, exp, underlying_price, points)
            surface.smiles[exp] = smile
        
        # Build term structure
        surface.term_structure = self._build_term_structure(underlying, surface.smiles)
        
        # Determine regime
        surface.regime = self._determine_regime(surface)
        
        # Build grid for visualization
        self._build_grid(surface)
        
        # Store
        self._surfaces[underlying] = surface
        
        logger.info(f"Built volatility surface for {underlying} with {len(surface.smiles)} expirations")
        return surface
    
    def _build_smile(
        self,
        underlying: str,
        expiration: date,
        underlying_price: float,
        points: List[IVPoint],
    ) -> VolatilitySmile:
        """Build volatility smile for single expiration."""
        # Sort by strike
        points = sorted(points, key=lambda p: p.strike)
        
        smile = VolatilitySmile(
            underlying=underlying,
            expiration=expiration,
            underlying_price=underlying_price,
            strikes=[p.strike for p in points],
            ivs=[p.iv for p in points],
        )
        
        # Find ATM
        atm_idx = min(
            range(len(points)),
            key=lambda i: abs(points[i].strike - underlying_price)
        )
        smile.atm_strike = points[atm_idx].strike
        smile.atm_iv = points[atm_idx].iv
        
        return smile
    
    def _build_term_structure(
        self,
        underlying: str,
        smiles: Dict[date, VolatilitySmile],
    ) -> TermStructure:
        """Build term structure from smiles."""
        term = TermStructure(
            underlying=underlying,
            as_of=datetime.now(),
        )
        
        for exp in sorted(smiles.keys()):
            smile = smiles[exp]
            term.expirations.append(exp)
            term.atm_ivs.append(smile.atm_iv)
        
        # Calculate forward volatilities
        if len(term.atm_ivs) >= 2:
            for i in range(len(term.atm_ivs) - 1):
                t1 = (term.expirations[i] - date.today()).days / 365
                t2 = (term.expirations[i+1] - date.today()).days / 365
                v1 = term.atm_ivs[i]
                v2 = term.atm_ivs[i+1]
                
                if t2 > t1 and t1 > 0:
                    # Forward variance
                    var1 = v1 ** 2 * t1
                    var2 = v2 ** 2 * t2
                    fwd_var = (var2 - var1) / (t2 - t1)
                    fwd_vol = math.sqrt(max(0, fwd_var))
                    term.forward_vols.append(fwd_vol)
        
        return term
    
    def _determine_regime(self, surface: VolatilitySurface) -> VolatilityRegime:
        """Determine volatility regime."""
        if not surface.term_structure or not surface.term_structure.atm_ivs:
            return VolatilityRegime.NORMAL
        
        avg_iv = statistics.mean(surface.term_structure.atm_ivs)
        
        # Regime thresholds (can be calibrated)
        if avg_iv < 0.15:
            return VolatilityRegime.LOW
        elif avg_iv < 0.25:
            return VolatilityRegime.NORMAL
        elif avg_iv < 0.40:
            return VolatilityRegime.ELEVATED
        elif avg_iv < 0.60:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def _build_grid(self, surface: VolatilitySurface) -> None:
        """Build regular grid for visualization."""
        if not surface.smiles:
            return
        
        # Strike grid: 80% to 120% of spot
        spot = surface.underlying_price
        surface.strike_grid = [
            spot * (0.80 + 0.05 * i) for i in range(9)
        ]
        
        # Expiry grid: all available
        surface.expiry_grid = sorted(surface.smiles.keys())
        
        # IV grid
        surface.iv_grid = []
        for exp in surface.expiry_grid:
            row = []
            for strike in surface.strike_grid:
                iv = surface.get_iv(strike, exp)
                row.append(iv)
            surface.iv_grid.append(row)
    
    def get_surface(self, underlying: str) -> Optional[VolatilitySurface]:
        """Get stored surface."""
        return self._surfaces.get(underlying)
    
    def analyze_surface(self, surface: VolatilitySurface) -> Dict[str, Any]:
        """
        Comprehensive surface analysis.
        
        Returns analytics including:
        - Regime classification
        - Term structure shape
        - Skew analysis
        - Potential arbitrage
        """
        analysis = {
            "underlying": surface.underlying,
            "as_of": surface.as_of.isoformat(),
            "regime": surface.regime.value,
            "num_expirations": len(surface.smiles),
            "num_points": len(surface.points),
        }
        
        # Term structure
        if surface.term_structure:
            ts = surface.term_structure
            analysis["term_structure"] = {
                "shape": "contango" if ts.is_contango else "backwardation" if ts.is_backwardation else "flat",
                "slope_per_month": ts.slope,
                "front_iv": ts.atm_ivs[0] if ts.atm_ivs else None,
                "back_iv": ts.atm_ivs[-1] if ts.atm_ivs else None,
            }
        
        # Skew analysis
        skews = {}
        for exp, smile in surface.smiles.items():
            skews[exp.isoformat()] = {
                "skew_25d": smile.skew,
                "wings": smile.wings,
                "curvature": smile.curvature,
                "atm_iv": smile.atm_iv,
            }
        analysis["skew_by_expiration"] = skews
        
        # Arbitrage check
        arbitrage = self._check_arbitrage(surface)
        analysis["arbitrage_warnings"] = arbitrage
        
        # Trading opportunities
        opportunities = self._find_opportunities(surface)
        analysis["opportunities"] = opportunities
        
        return analysis
    
    def _check_arbitrage(self, surface: VolatilitySurface) -> List[str]:
        """Check for potential arbitrage conditions."""
        warnings = []
        
        # Calendar spread arbitrage (negative forward variance)
        if surface.term_structure and surface.term_structure.forward_vols:
            for i, fwd in enumerate(surface.term_structure.forward_vols):
                if fwd < 0.05:  # Very low forward vol
                    warnings.append(
                        f"Low forward vol ({fwd:.2%}) between "
                        f"{surface.term_structure.expirations[i]} and "
                        f"{surface.term_structure.expirations[i+1]}"
                    )
        
        # Butterfly arbitrage (negative curvature)
        for exp, smile in surface.smiles.items():
            if smile.curvature < -0.01:
                warnings.append(f"Negative butterfly on {exp} (curvature: {smile.curvature:.4f})")
        
        return warnings
    
    def _find_opportunities(self, surface: VolatilitySurface) -> List[Dict[str, Any]]:
        """Find potential trading opportunities."""
        opportunities = []
        
        # High skew opportunity
        for exp, smile in surface.smiles.items():
            if smile.skew > 0.10:  # Very steep skew
                opportunities.append({
                    "type": "skew_trade",
                    "description": "Sell put spread vs buy call spread",
                    "expiration": exp.isoformat(),
                    "skew": smile.skew,
                    "rationale": "Elevated put skew suggests potential mean reversion",
                })
            elif smile.skew < -0.05:  # Inverted skew
                opportunities.append({
                    "type": "skew_trade",
                    "description": "Buy put spread vs sell call spread",
                    "expiration": exp.isoformat(),
                    "skew": smile.skew,
                    "rationale": "Depressed put skew suggests potential increase",
                })
        
        # Term structure opportunity
        if surface.term_structure:
            if surface.term_structure.slope > 0.02:  # Steep contango
                opportunities.append({
                    "type": "calendar_spread",
                    "description": "Sell front-month, buy back-month",
                    "slope": surface.term_structure.slope,
                    "rationale": "Steep term structure suggests front IV will fall",
                })
            elif surface.term_structure.slope < -0.02:  # Steep backwardation
                opportunities.append({
                    "type": "calendar_spread",
                    "description": "Buy front-month, sell back-month",
                    "slope": surface.term_structure.slope,
                    "rationale": "Inverted term structure suggests front IV will rise",
                })
        
        # Regime-based opportunities
        if surface.regime == VolatilityRegime.HIGH:
            opportunities.append({
                "type": "vol_selling",
                "description": "Iron condors, strangles for premium",
                "regime": surface.regime.value,
                "rationale": "High IV regime favors premium selling",
            })
        elif surface.regime == VolatilityRegime.LOW:
            opportunities.append({
                "type": "vol_buying",
                "description": "Long straddles, calendars",
                "regime": surface.regime.value,
                "rationale": "Low IV regime favors long volatility",
            })
        
        return opportunities
    
    def fit_svi(
        self,
        smile: VolatilitySmile,
    ) -> Dict[str, float]:
        """
        Fit SVI (Stochastic Volatility Inspired) model to smile.
        
        SVI parameterization:
        w(k) = a + b * (ρ*(k-m) + sqrt((k-m)^2 + σ^2))
        
        where k = log(K/F) is log-moneyness.
        """
        # Simplified SVI fit (would use optimization in production)
        atm_var = smile.atm_iv ** 2
        
        # Estimate parameters from smile characteristics
        a = atm_var  # ATM total variance
        b = smile.curvature * 2  # Curvature controls wings
        rho = -smile.skew / (smile.atm_iv * 2)  # Skew controls asymmetry
        rho = max(-0.99, min(0.99, rho))  # Bound rho
        m = 0.0  # Centered at ATM
        sigma = abs(smile.wings) + 0.1  # Wing width
        
        return {
            "a": a,
            "b": max(0.01, b),
            "rho": rho,
            "m": m,
            "sigma": sigma,
            "model": "svi",
        }
    
    def get_surface_metrics(self, underlying: str) -> Dict[str, Any]:
        """Get key surface metrics for trading."""
        surface = self.get_surface(underlying)
        if not surface:
            return {}
        
        metrics = {
            "underlying": underlying,
            "regime": surface.regime.value,
        }
        
        if surface.term_structure and surface.term_structure.atm_ivs:
            metrics["atm_30d"] = surface.term_structure.get_atm_iv(30)
            metrics["atm_60d"] = surface.term_structure.get_atm_iv(60)
            metrics["atm_90d"] = surface.term_structure.get_atm_iv(90)
            metrics["term_slope"] = surface.term_structure.slope
        
        # Average skew
        if surface.smiles:
            skews = [s.skew for s in surface.smiles.values()]
            metrics["avg_skew"] = statistics.mean(skews)
            metrics["max_skew"] = max(skews)
            metrics["min_skew"] = min(skews)
        
        return metrics


# Singleton instance
surface_builder = VolatilitySurfaceBuilder()


# Convenience functions
def build_vol_surface(
    underlying: str,
    underlying_price: float,
    iv_points: List[IVPoint],
) -> VolatilitySurface:
    """Build volatility surface."""
    return surface_builder.build_surface(underlying, underlying_price, iv_points)


def get_vol_surface(underlying: str) -> Optional[VolatilitySurface]:
    """Get stored surface."""
    return surface_builder.get_surface(underlying)


def analyze_vol_surface(surface: VolatilitySurface) -> Dict[str, Any]:
    """Analyze volatility surface."""
    return surface_builder.analyze_surface(surface)
