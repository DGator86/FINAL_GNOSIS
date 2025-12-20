"""
Portfolio Greeks Risk Management - Institutional Grade

Monitors and limits portfolio-level Greek exposures for professional risk management.

GREEK EXPOSURES:
├── Delta: Directional risk ($ change per $1 move in underlying)
├── Gamma: Acceleration risk (delta change per $1 move)
├── Theta: Time decay ($ lost/gained per day)
├── Vega: Volatility risk ($ change per 1% IV change)
└── Rho: Interest rate risk (usually minimal for short-dated options)

RISK LIMITS:
- Portfolio Delta: Max net directional exposure as % of portfolio
- Portfolio Gamma: Max acceleration risk
- Portfolio Theta: Max daily time decay
- Portfolio Vega: Max volatility exposure
- Single Position: Max Greeks contribution per position

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class GreekLimitBreach(str, Enum):
    """Types of Greek limit breaches."""
    NONE = "none"
    DELTA = "delta"
    GAMMA = "gamma"
    THETA = "theta"
    VEGA = "vega"
    CONCENTRATION = "concentration"


@dataclass
class PositionGreeks:
    """Greeks for a single position."""
    symbol: str
    underlying: str
    quantity: int
    
    # Raw Greeks (per contract)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    
    # Position Greeks (quantity-adjusted, in dollars)
    position_delta: float = 0.0  # Dollar delta
    position_gamma: float = 0.0  # Dollar gamma
    position_theta: float = 0.0  # Daily theta P&L
    position_vega: float = 0.0   # Dollar vega
    
    # Position metadata
    is_option: bool = True
    option_type: str = ""  # "call" or "put"
    strike: float = 0.0
    expiration: Optional[datetime] = None
    dte: int = 0
    underlying_price: float = 0.0
    
    # Risk metrics
    notional_value: float = 0.0
    max_loss: float = 0.0
    beta: float = 1.0  # Beta to SPY


@dataclass
class PortfolioGreeks:
    """Aggregate portfolio-level Greeks."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Net portfolio Greeks (dollar-weighted)
    net_delta: float = 0.0          # Net directional exposure in $
    net_gamma: float = 0.0          # Net gamma exposure
    net_theta: float = 0.0          # Net daily theta (positive = collecting)
    net_vega: float = 0.0           # Net vega exposure
    
    # Absolute exposures (for limit checking)
    gross_delta: float = 0.0        # Sum of absolute deltas
    gross_gamma: float = 0.0
    gross_theta: float = 0.0
    gross_vega: float = 0.0
    
    # Normalized metrics (as % of portfolio)
    delta_pct: float = 0.0          # Delta as % of portfolio value
    gamma_pct: float = 0.0          # Gamma as % (1% underlying move impact)
    theta_pct: float = 0.0          # Daily theta as % of portfolio
    vega_pct: float = 0.0           # Vega as % (1% IV move impact)
    
    # Beta-adjusted delta (normalized to SPY)
    beta_adjusted_delta: float = 0.0
    
    # Portfolio value
    portfolio_value: float = 0.0
    
    # Position breakdown
    positions: List[PositionGreeks] = field(default_factory=list)
    position_count: int = 0
    
    # Concentration metrics
    largest_delta_position: str = ""
    largest_delta_pct: float = 0.0
    sector_exposures: Dict[str, float] = field(default_factory=dict)


@dataclass
class GreekLimits:
    """Risk limits for portfolio Greeks."""
    # Delta limits (as % of portfolio)
    max_net_delta_pct: float = 0.30         # 30% max net directional
    max_gross_delta_pct: float = 0.50       # 50% max gross delta
    max_single_position_delta_pct: float = 0.10  # 10% max per position
    
    # Gamma limits
    max_gamma_pct: float = 0.02             # 2% portfolio impact per 1% move
    max_single_gamma_pct: float = 0.005     # 0.5% per position
    
    # Theta limits
    max_negative_theta_pct: float = 0.005   # Max 0.5% daily theta loss
    min_positive_theta_pct: float = -0.01   # Warning if collecting too much
    
    # Vega limits
    max_vega_pct: float = 0.03              # 3% impact per 1% IV change
    max_single_vega_pct: float = 0.01       # 1% per position
    
    # Concentration limits
    max_single_underlying_pct: float = 0.15  # 15% max in one underlying
    max_sector_pct: float = 0.30             # 30% max in one sector
    max_correlated_exposure_pct: float = 0.40  # 40% in correlated assets
    
    # Warning thresholds (% of limit to trigger warning)
    warning_threshold: float = 0.75          # Warn at 75% of limit


@dataclass
class GreekRiskAssessment:
    """Result of Greek risk assessment."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Overall assessment
    is_within_limits: bool = True
    risk_score: float = 0.0  # 0-100, higher = more risk
    
    # Breaches
    breaches: List[GreekLimitBreach] = field(default_factory=list)
    breach_details: Dict[str, str] = field(default_factory=dict)
    
    # Warnings (approaching limits)
    warnings: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Can add new position?
    can_add_position: bool = True
    max_additional_delta: float = 0.0
    max_additional_contracts: int = 0


class PortfolioGreeksManager:
    """
    Manages portfolio-level Greek exposures with institutional-grade risk controls.
    
    Features:
    - Real-time Greek aggregation across all positions
    - Limit monitoring with warnings and breaches
    - Position-level and portfolio-level risk assessment
    - Pre-trade risk checks
    - Hedging recommendations
    """
    
    # Sector mappings for concentration risk
    SECTOR_MAP: Dict[str, str] = {
        # Technology
        "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
        "AMZN": "technology", "META": "technology", "NVDA": "technology",
        "AMD": "technology", "INTC": "technology", "CRM": "technology",
        "ORCL": "technology", "ADBE": "technology", "AVGO": "technology",
        "QCOM": "technology", "TSM": "technology", "MU": "technology",
        
        # Financials
        "JPM": "financials", "BAC": "financials", "WFC": "financials",
        "GS": "financials", "MS": "financials", "C": "financials",
        "V": "financials", "MA": "financials", "PYPL": "financials",
        "COIN": "financials", "HOOD": "financials",
        
        # Healthcare
        "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
        "MRK": "healthcare", "ABBV": "healthcare", "LLY": "healthcare",
        
        # Energy
        "XOM": "energy", "CVX": "energy", "COP": "energy",
        "SLB": "energy", "EOG": "energy",
        
        # Consumer
        "TSLA": "consumer", "NKE": "consumer", "MCD": "consumer",
        "SBUX": "consumer", "HD": "consumer", "LOW": "consumer",
        "TGT": "consumer", "WMT": "consumer", "COST": "consumer",
        
        # Industrial
        "BA": "industrial", "CAT": "industrial", "DE": "industrial",
        "UPS": "industrial", "FDX": "industrial",
        
        # ETFs
        "SPY": "index", "QQQ": "index", "IWM": "index", "DIA": "index",
        "XLF": "financials", "XLK": "technology", "XLE": "energy",
        "XLV": "healthcare", "XLI": "industrial", "XLP": "consumer",
        
        # Speculative
        "GME": "speculative", "AMC": "speculative", "MSTR": "speculative",
        "RIOT": "speculative", "MARA": "speculative", "SMCI": "speculative",
    }
    
    # Beta estimates (simplified - would use real data in production)
    BETA_ESTIMATES: Dict[str, float] = {
        "SPY": 1.0, "QQQ": 1.1, "IWM": 1.2,
        "NVDA": 1.8, "TSLA": 1.9, "AMD": 1.7,
        "AAPL": 1.2, "MSFT": 1.1, "GOOGL": 1.15,
        "META": 1.3, "AMZN": 1.2, "COIN": 2.5,
        "GME": 2.0, "AMC": 2.0, "SMCI": 2.2,
    }
    
    def __init__(
        self,
        limits: Optional[GreekLimits] = None,
        portfolio_value: Optional[float] = None,
    ):
        """Initialize the Greeks manager.
        
        Args:
            limits: Greek risk limits (uses defaults if not provided)
            portfolio_value: Portfolio value for % calculations
        """
        self.limits = limits or GreekLimits()
        self.portfolio_value = portfolio_value or float(
            os.getenv("DEFAULT_CAPITAL", "100000.0")
        )
        
        # Current state
        self.positions: Dict[str, PositionGreeks] = {}
        self.portfolio_greeks: Optional[PortfolioGreeks] = None
        
        logger.info(
            f"PortfolioGreeksManager initialized | "
            f"portfolio=${self.portfolio_value:,.0f} | "
            f"max_delta={self.limits.max_net_delta_pct:.0%}"
        )
    
    def add_position(
        self,
        symbol: str,
        underlying: str,
        quantity: int,
        delta: float,
        gamma: float,
        theta: float,
        vega: float,
        underlying_price: float,
        option_type: str = "",
        strike: float = 0.0,
        expiration: Optional[datetime] = None,
        is_option: bool = True,
    ) -> PositionGreeks:
        """Add or update a position with its Greeks.
        
        Args:
            symbol: Position symbol (OCC symbol for options)
            underlying: Underlying symbol
            quantity: Position quantity (negative for short)
            delta: Per-contract delta
            gamma: Per-contract gamma
            theta: Per-contract theta (daily)
            vega: Per-contract vega
            underlying_price: Current underlying price
            option_type: "call" or "put" for options
            strike: Strike price for options
            expiration: Expiration date for options
            is_option: Whether this is an options position
            
        Returns:
            PositionGreeks object
        """
        # Calculate DTE
        dte = 0
        if expiration:
            dte = max(0, (expiration - datetime.utcnow()).days)
        
        # Get beta for underlying
        beta = self.BETA_ESTIMATES.get(underlying, 1.0)
        
        # Calculate position-level Greeks (dollar terms)
        # For options: multiply by 100 (contract multiplier) and quantity
        multiplier = 100 if is_option else 1
        
        position_delta = delta * quantity * multiplier * underlying_price
        position_gamma = gamma * quantity * multiplier * underlying_price * underlying_price / 100
        position_theta = theta * quantity * multiplier  # Already in $ per day
        position_vega = vega * quantity * multiplier    # $ per 1% IV change
        
        # Calculate notional and max loss
        notional = abs(quantity) * multiplier * underlying_price
        
        # Max loss estimation (simplified)
        if is_option:
            if quantity > 0:  # Long options
                # Max loss = premium paid (approximated by delta * price for ATM)
                max_loss = abs(delta * quantity * multiplier * underlying_price * 0.1)
            else:  # Short options
                if option_type == "put":
                    max_loss = abs(quantity) * multiplier * strike  # Can go to 0
                else:  # call
                    max_loss = notional * 2  # Unlimited, but cap at 2x notional
        else:
            max_loss = notional  # Equity can go to 0
        
        position = PositionGreeks(
            symbol=symbol,
            underlying=underlying,
            quantity=quantity,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            position_delta=position_delta,
            position_gamma=position_gamma,
            position_theta=position_theta,
            position_vega=position_vega,
            is_option=is_option,
            option_type=option_type,
            strike=strike,
            expiration=expiration,
            dte=dte,
            underlying_price=underlying_price,
            notional_value=notional,
            max_loss=max_loss,
            beta=beta,
        )
        
        self.positions[symbol] = position
        
        # Recalculate portfolio Greeks
        self._recalculate_portfolio_greeks()
        
        return position
    
    def remove_position(self, symbol: str) -> None:
        """Remove a position from tracking."""
        if symbol in self.positions:
            del self.positions[symbol]
            self._recalculate_portfolio_greeks()
    
    def _recalculate_portfolio_greeks(self) -> PortfolioGreeks:
        """Recalculate aggregate portfolio Greeks."""
        
        # Initialize accumulators
        net_delta = 0.0
        net_gamma = 0.0
        net_theta = 0.0
        net_vega = 0.0
        
        gross_delta = 0.0
        gross_gamma = 0.0
        gross_theta = 0.0
        gross_vega = 0.0
        
        beta_adjusted_delta = 0.0
        
        sector_exposures: Dict[str, float] = {}
        largest_delta = 0.0
        largest_delta_symbol = ""
        
        position_list = []
        
        for symbol, pos in self.positions.items():
            # Aggregate net Greeks
            net_delta += pos.position_delta
            net_gamma += pos.position_gamma
            net_theta += pos.position_theta
            net_vega += pos.position_vega
            
            # Aggregate gross Greeks
            gross_delta += abs(pos.position_delta)
            gross_gamma += abs(pos.position_gamma)
            gross_theta += abs(pos.position_theta)
            gross_vega += abs(pos.position_vega)
            
            # Beta-adjusted delta
            beta_adjusted_delta += pos.position_delta * pos.beta
            
            # Track largest delta position
            if abs(pos.position_delta) > largest_delta:
                largest_delta = abs(pos.position_delta)
                largest_delta_symbol = pos.underlying
            
            # Sector exposure
            sector = self.SECTOR_MAP.get(pos.underlying, "other")
            sector_exposures[sector] = sector_exposures.get(sector, 0) + abs(pos.position_delta)
            
            position_list.append(pos)
        
        # Calculate percentages
        pv = max(self.portfolio_value, 1.0)
        
        self.portfolio_greeks = PortfolioGreeks(
            timestamp=datetime.utcnow(),
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_theta=net_theta,
            net_vega=net_vega,
            gross_delta=gross_delta,
            gross_gamma=gross_gamma,
            gross_theta=gross_theta,
            gross_vega=gross_vega,
            delta_pct=net_delta / pv,
            gamma_pct=net_gamma / pv,
            theta_pct=net_theta / pv,
            vega_pct=net_vega / pv,
            beta_adjusted_delta=beta_adjusted_delta,
            portfolio_value=self.portfolio_value,
            positions=position_list,
            position_count=len(position_list),
            largest_delta_position=largest_delta_symbol,
            largest_delta_pct=largest_delta / pv if pv > 0 else 0,
            sector_exposures=sector_exposures,
        )
        
        return self.portfolio_greeks
    
    def assess_risk(self) -> GreekRiskAssessment:
        """Assess current portfolio Greek risk against limits.
        
        Returns:
            GreekRiskAssessment with breaches, warnings, and recommendations
        """
        if not self.portfolio_greeks:
            self._recalculate_portfolio_greeks()
        
        pg = self.portfolio_greeks
        limits = self.limits
        
        assessment = GreekRiskAssessment()
        risk_score = 0.0
        
        # Check Delta limits
        delta_usage = abs(pg.delta_pct) / limits.max_net_delta_pct
        risk_score += delta_usage * 25  # Delta is 25% of risk score
        
        if abs(pg.delta_pct) > limits.max_net_delta_pct:
            assessment.breaches.append(GreekLimitBreach.DELTA)
            assessment.breach_details["delta"] = (
                f"Net delta {pg.delta_pct:.1%} exceeds limit {limits.max_net_delta_pct:.1%}"
            )
            assessment.is_within_limits = False
        elif delta_usage > limits.warning_threshold:
            assessment.warnings.append(
                f"Delta at {delta_usage:.0%} of limit ({pg.delta_pct:.1%}/{limits.max_net_delta_pct:.1%})"
            )
        
        # Check Gamma limits
        gamma_usage = abs(pg.gamma_pct) / limits.max_gamma_pct
        risk_score += gamma_usage * 25
        
        if abs(pg.gamma_pct) > limits.max_gamma_pct:
            assessment.breaches.append(GreekLimitBreach.GAMMA)
            assessment.breach_details["gamma"] = (
                f"Gamma {pg.gamma_pct:.2%} exceeds limit {limits.max_gamma_pct:.2%}"
            )
            assessment.is_within_limits = False
        elif gamma_usage > limits.warning_threshold:
            assessment.warnings.append(
                f"Gamma at {gamma_usage:.0%} of limit"
            )
        
        # Check Theta limits (negative theta = paying time decay)
        if pg.theta_pct < -limits.max_negative_theta_pct:
            assessment.breaches.append(GreekLimitBreach.THETA)
            assessment.breach_details["theta"] = (
                f"Daily theta loss {pg.theta_pct:.3%} exceeds limit {-limits.max_negative_theta_pct:.3%}"
            )
            assessment.is_within_limits = False
            risk_score += 20
        
        # Check Vega limits
        vega_usage = abs(pg.vega_pct) / limits.max_vega_pct
        risk_score += vega_usage * 20
        
        if abs(pg.vega_pct) > limits.max_vega_pct:
            assessment.breaches.append(GreekLimitBreach.VEGA)
            assessment.breach_details["vega"] = (
                f"Vega {pg.vega_pct:.2%} exceeds limit {limits.max_vega_pct:.2%}"
            )
            assessment.is_within_limits = False
        
        # Check concentration limits
        if pg.largest_delta_pct > limits.max_single_position_delta_pct:
            assessment.breaches.append(GreekLimitBreach.CONCENTRATION)
            assessment.breach_details["concentration"] = (
                f"{pg.largest_delta_position} has {pg.largest_delta_pct:.1%} delta "
                f"(limit: {limits.max_single_position_delta_pct:.1%})"
            )
            assessment.is_within_limits = False
            risk_score += 10
        
        # Check sector concentration
        for sector, exposure in pg.sector_exposures.items():
            sector_pct = exposure / max(self.portfolio_value, 1)
            if sector_pct > limits.max_sector_pct:
                assessment.warnings.append(
                    f"Sector {sector} exposure {sector_pct:.1%} exceeds {limits.max_sector_pct:.1%}"
                )
                risk_score += 5
        
        # Calculate max additional delta allowed
        remaining_delta_capacity = limits.max_net_delta_pct - abs(pg.delta_pct)
        assessment.max_additional_delta = remaining_delta_capacity * self.portfolio_value
        assessment.can_add_position = remaining_delta_capacity > 0.01  # 1% minimum
        
        # Generate recommendations
        if pg.net_delta > self.portfolio_value * 0.2:
            assessment.recommendations.append(
                "Consider hedging long delta with SPY puts or bear spreads"
            )
        elif pg.net_delta < -self.portfolio_value * 0.2:
            assessment.recommendations.append(
                "Consider hedging short delta with SPY calls or bull spreads"
            )
        
        if pg.net_gamma < -self.portfolio_value * 0.01:
            assessment.recommendations.append(
                "Negative gamma exposure - consider buying options to hedge"
            )
        
        if pg.net_theta < -self.portfolio_value * 0.003:
            assessment.recommendations.append(
                "High theta decay - consider selling premium or closing debit positions"
            )
        
        assessment.risk_score = min(100, risk_score)
        
        logger.debug(
            f"Greek Risk Assessment: score={assessment.risk_score:.0f} | "
            f"delta={pg.delta_pct:.1%} | gamma={pg.gamma_pct:.2%} | "
            f"theta={pg.theta_pct:.3%} | vega={pg.vega_pct:.2%}"
        )
        
        return assessment
    
    def check_new_position(
        self,
        underlying: str,
        delta: float,
        gamma: float,
        theta: float,
        vega: float,
        quantity: int,
        underlying_price: float,
    ) -> Tuple[bool, str]:
        """Check if a new position would breach Greek limits.
        
        Args:
            underlying: Underlying symbol
            delta: Per-contract delta
            gamma: Per-contract gamma
            theta: Per-contract theta
            vega: Per-contract vega
            quantity: Number of contracts
            underlying_price: Current underlying price
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        if not self.portfolio_greeks:
            self._recalculate_portfolio_greeks()
        
        pg = self.portfolio_greeks
        limits = self.limits
        
        # Calculate impact of new position
        multiplier = 100
        new_delta = delta * quantity * multiplier * underlying_price
        new_gamma = gamma * quantity * multiplier * underlying_price * underlying_price / 100
        new_theta = theta * quantity * multiplier
        new_vega = vega * quantity * multiplier
        
        # Check delta impact
        projected_delta_pct = (pg.net_delta + new_delta) / self.portfolio_value
        if abs(projected_delta_pct) > limits.max_net_delta_pct:
            return False, f"Would breach delta limit ({projected_delta_pct:.1%} > {limits.max_net_delta_pct:.1%})"
        
        # Check single position delta
        single_delta_pct = abs(new_delta) / self.portfolio_value
        if single_delta_pct > limits.max_single_position_delta_pct:
            return False, f"Position delta {single_delta_pct:.1%} exceeds single position limit"
        
        # Check gamma impact
        projected_gamma_pct = (pg.net_gamma + new_gamma) / self.portfolio_value
        if abs(projected_gamma_pct) > limits.max_gamma_pct:
            return False, f"Would breach gamma limit ({projected_gamma_pct:.2%} > {limits.max_gamma_pct:.2%})"
        
        # Check concentration
        underlying_exposure = sum(
            abs(p.position_delta) for p in pg.positions 
            if p.underlying == underlying
        ) + abs(new_delta)
        
        underlying_pct = underlying_exposure / self.portfolio_value
        if underlying_pct > limits.max_single_underlying_pct:
            return False, f"Would exceed {underlying} concentration limit ({underlying_pct:.1%})"
        
        return True, "Position within limits"
    
    def get_hedging_recommendation(self) -> Dict[str, Any]:
        """Get recommendations for hedging current Greek exposures.
        
        Returns:
            Dict with hedging recommendations
        """
        if not self.portfolio_greeks:
            self._recalculate_portfolio_greeks()
        
        pg = self.portfolio_greeks
        recommendations = {
            "delta_hedge": None,
            "gamma_hedge": None,
            "vega_hedge": None,
            "overall_action": "none",
        }
        
        # Delta hedging
        if abs(pg.net_delta) > self.portfolio_value * 0.1:
            if pg.net_delta > 0:
                # Long delta - hedge with short SPY or puts
                spy_shares = int(pg.net_delta / 500)  # Assuming SPY ~$500
                recommendations["delta_hedge"] = {
                    "action": "reduce_long_delta",
                    "instrument": "SPY",
                    "strategy": "sell_shares_or_buy_puts",
                    "quantity": spy_shares,
                    "rationale": f"Net delta ${pg.net_delta:,.0f} is too long"
                }
            else:
                # Short delta - hedge with long SPY or calls
                spy_shares = int(abs(pg.net_delta) / 500)
                recommendations["delta_hedge"] = {
                    "action": "reduce_short_delta",
                    "instrument": "SPY",
                    "strategy": "buy_shares_or_buy_calls",
                    "quantity": spy_shares,
                    "rationale": f"Net delta ${pg.net_delta:,.0f} is too short"
                }
        
        # Gamma hedging (if significantly negative)
        if pg.net_gamma < -self.portfolio_value * 0.005:
            recommendations["gamma_hedge"] = {
                "action": "buy_gamma",
                "strategy": "buy_straddles_or_strangles",
                "rationale": f"Negative gamma ${pg.net_gamma:,.0f} creates acceleration risk"
            }
        
        # Vega hedging
        if abs(pg.net_vega) > self.portfolio_value * 0.02:
            if pg.net_vega > 0:
                recommendations["vega_hedge"] = {
                    "action": "reduce_long_vega",
                    "strategy": "sell_premium_or_close_long_options",
                    "rationale": f"Long vega ${pg.net_vega:,.0f} vulnerable to IV crush"
                }
            else:
                recommendations["vega_hedge"] = {
                    "action": "reduce_short_vega", 
                    "strategy": "buy_options_or_close_short_premium",
                    "rationale": f"Short vega ${pg.net_vega:,.0f} vulnerable to IV spike"
                }
        
        # Overall action
        if any(v for k, v in recommendations.items() if k != "overall_action" and v):
            recommendations["overall_action"] = "hedge_recommended"
        
        return recommendations
    
    def update_portfolio_value(self, value: float) -> None:
        """Update portfolio value for percentage calculations."""
        self.portfolio_value = value
        self._recalculate_portfolio_greeks()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current portfolio Greeks."""
        if not self.portfolio_greeks:
            self._recalculate_portfolio_greeks()
        
        pg = self.portfolio_greeks
        assessment = self.assess_risk()
        
        return {
            "timestamp": pg.timestamp.isoformat(),
            "portfolio_value": self.portfolio_value,
            "position_count": pg.position_count,
            "greeks": {
                "net_delta": pg.net_delta,
                "net_gamma": pg.net_gamma,
                "net_theta": pg.net_theta,
                "net_vega": pg.net_vega,
                "delta_pct": pg.delta_pct,
                "gamma_pct": pg.gamma_pct,
                "theta_pct": pg.theta_pct,
                "vega_pct": pg.vega_pct,
            },
            "risk_assessment": {
                "score": assessment.risk_score,
                "is_within_limits": assessment.is_within_limits,
                "breaches": [b.value for b in assessment.breaches],
                "warnings": assessment.warnings,
            },
            "largest_position": {
                "symbol": pg.largest_delta_position,
                "delta_pct": pg.largest_delta_pct,
            },
            "sector_exposures": pg.sector_exposures,
        }


# Factory function
def create_portfolio_greeks_manager(
    portfolio_value: Optional[float] = None,
    custom_limits: Optional[Dict[str, float]] = None,
) -> PortfolioGreeksManager:
    """Create a PortfolioGreeksManager with optional custom limits.
    
    Args:
        portfolio_value: Portfolio value for calculations
        custom_limits: Dict of limit overrides
        
    Returns:
        Configured PortfolioGreeksManager
    """
    limits = GreekLimits()
    
    if custom_limits:
        for key, value in custom_limits.items():
            if hasattr(limits, key):
                setattr(limits, key, value)
    
    return PortfolioGreeksManager(
        limits=limits,
        portfolio_value=portfolio_value,
    )
