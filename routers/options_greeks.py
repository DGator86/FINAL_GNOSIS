"""
Options Greeks Calculator API.

REST API endpoints for real-time options Greeks calculation:
- Black-Scholes Greeks (delta, gamma, theta, vega, rho)
- Position-level Greeks aggregation
- Portfolio Greeks summary
- Strategy analysis (breakevens, profit zones, probability of profit)
- Implied volatility calculation

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from datetime import datetime, date
from typing import Any, Dict, List, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, validator
import math

from gnosis.utils.greeks_calculator import GreeksCalculator

router = APIRouter(prefix="/options/greeks", tags=["options-greeks"])

# Initialize calculator
_calculator = GreeksCalculator()


# ============================================================================
# Request/Response Models
# ============================================================================

class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class GreeksRequest(BaseModel):
    """Request model for Greeks calculation."""
    option_type: OptionType = Field(..., description="Option type: 'call' or 'put'")
    spot_price: float = Field(..., gt=0, description="Current underlying price")
    strike: float = Field(..., gt=0, description="Strike price")
    time_to_expiration: float = Field(
        ..., 
        ge=0, 
        description="Time to expiration in years (e.g., 0.25 for 3 months)"
    )
    volatility: float = Field(
        ..., 
        gt=0, 
        le=5.0, 
        description="Implied volatility (annual, e.g., 0.25 for 25%)"
    )
    risk_free_rate: float = Field(
        default=0.05, 
        ge=0, 
        le=1.0, 
        description="Risk-free rate (annual, e.g., 0.05 for 5%)"
    )
    dividend_yield: float = Field(
        default=0.0, 
        ge=0, 
        le=1.0, 
        description="Continuous dividend yield (annual)"
    )


class GreeksResponse(BaseModel):
    """Response model for Greeks calculation."""
    delta: float = Field(..., description="Price sensitivity (-1 to 1)")
    gamma: float = Field(..., description="Delta sensitivity")
    theta: float = Field(..., description="Daily time decay")
    vega: float = Field(..., description="Volatility sensitivity (per 1%)")
    rho: float = Field(..., description="Interest rate sensitivity (per 1%)")
    
    # Additional computed values
    option_price: Optional[float] = Field(None, description="Theoretical option price")
    intrinsic_value: Optional[float] = Field(None, description="Intrinsic value")
    time_value: Optional[float] = Field(None, description="Time/extrinsic value")


class OptionLeg(BaseModel):
    """Single option leg for strategy analysis."""
    option_type: OptionType
    strike: float = Field(..., gt=0)
    premium: float = Field(..., description="Premium per share (positive=paid, negative=received)")
    quantity: int = Field(..., description="Contracts (positive=long, negative=short)")
    
    # Optional Greeks (if known)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None


class StrategyAnalysisRequest(BaseModel):
    """Request for strategy analysis."""
    legs: List[OptionLeg] = Field(..., min_length=1, max_length=8)
    spot_price: float = Field(..., gt=0, description="Current underlying price")
    volatility: float = Field(
        default=0.25, 
        gt=0, 
        le=5.0, 
        description="Implied volatility for PoP calculation"
    )
    days_to_expiration: int = Field(..., ge=0, le=730, description="Days until expiration")
    risk_free_rate: float = Field(default=0.05, ge=0, le=1.0)


class StrategyAnalysisResponse(BaseModel):
    """Response for strategy analysis."""
    # Profit zones
    profit_zones: List[tuple] = Field(..., description="Price ranges where strategy is profitable")
    breakevens: List[float] = Field(..., description="Breakeven prices")
    
    # Max profit/loss
    max_profit: float
    max_profit_at_price: float
    max_loss: float
    max_loss_at_price: float
    
    # Probability metrics
    probability_of_profit: float = Field(..., ge=0, le=1)
    expected_profit: float
    expected_loss: float
    average_pnl: float
    
    # Current P&L
    pnl_at_current_price: float
    
    # Aggregate Greeks
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float


class PositionGreeksRequest(BaseModel):
    """Request for position-level Greeks."""
    positions: List[Dict[str, Any]] = Field(
        ..., 
        description="List of positions with symbol, quantity, and greeks"
    )


class ImpliedVolRequest(BaseModel):
    """Request for implied volatility calculation."""
    option_type: OptionType
    option_price: float = Field(..., gt=0, description="Market price of the option")
    spot_price: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    time_to_expiration: float = Field(..., ge=0)
    risk_free_rate: float = Field(default=0.05, ge=0, le=1.0)
    dividend_yield: float = Field(default=0.0, ge=0, le=1.0)


# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "/calculate",
    response_model=GreeksResponse,
    summary="Calculate Black-Scholes Greeks",
    description="""
    Calculate option Greeks using the Black-Scholes-Merton model.
    
    Returns:
    - **delta**: Price sensitivity (0 to 1 for calls, -1 to 0 for puts)
    - **gamma**: Rate of change of delta
    - **theta**: Daily time decay (negative for long options)
    - **vega**: Sensitivity to 1% change in IV
    - **rho**: Sensitivity to 1% change in interest rate
    
    Also returns theoretical option price and value breakdown.
    """,
)
def calculate_greeks(request: GreeksRequest) -> GreeksResponse:
    """Calculate Black-Scholes Greeks for a single option."""
    greeks = _calculator.calculate_black_scholes_greeks(
        option_type=request.option_type.value,
        spot_price=request.spot_price,
        strike=request.strike,
        time_to_expiration=request.time_to_expiration,
        risk_free_rate=request.risk_free_rate,
        volatility=request.volatility,
        dividend_yield=request.dividend_yield,
    )
    
    # Calculate option price using Black-Scholes
    option_price = _calculate_bs_price(
        option_type=request.option_type.value,
        spot_price=request.spot_price,
        strike=request.strike,
        time_to_expiration=request.time_to_expiration,
        risk_free_rate=request.risk_free_rate,
        volatility=request.volatility,
        dividend_yield=request.dividend_yield,
    )
    
    # Calculate intrinsic and time value
    if request.option_type == OptionType.CALL:
        intrinsic = max(0, request.spot_price - request.strike)
    else:
        intrinsic = max(0, request.strike - request.spot_price)
    
    time_value = max(0, option_price - intrinsic)
    
    return GreeksResponse(
        delta=greeks["delta"],
        gamma=greeks["gamma"],
        theta=greeks["theta"],
        vega=greeks["vega"],
        rho=greeks["rho"],
        option_price=round(option_price, 4),
        intrinsic_value=round(intrinsic, 4),
        time_value=round(time_value, 4),
    )


@router.get(
    "/quick",
    response_model=GreeksResponse,
    summary="Quick Greeks calculation (GET)",
    description="Calculate Greeks using query parameters for quick lookups.",
)
def quick_greeks(
    option_type: OptionType = Query(..., description="call or put"),
    spot: float = Query(..., gt=0, description="Spot price"),
    strike: float = Query(..., gt=0, description="Strike price"),
    dte: int = Query(..., ge=0, description="Days to expiration"),
    iv: float = Query(..., gt=0, le=500, description="IV as percentage (e.g., 25 for 25%)"),
    rate: float = Query(default=5.0, ge=0, description="Risk-free rate as percentage"),
) -> GreeksResponse:
    """Quick Greeks calculation via GET request."""
    request = GreeksRequest(
        option_type=option_type,
        spot_price=spot,
        strike=strike,
        time_to_expiration=dte / 365.0,
        volatility=iv / 100.0,
        risk_free_rate=rate / 100.0,
    )
    return calculate_greeks(request)


@router.post(
    "/strategy/analyze",
    response_model=StrategyAnalysisResponse,
    summary="Analyze options strategy",
    description="""
    Comprehensive analysis of an options strategy including:
    - Profit zones and breakeven points
    - Maximum profit and loss
    - Probability of profit (Monte Carlo simulation)
    - Aggregate position Greeks
    
    Supports multi-leg strategies like spreads, iron condors, butterflies, etc.
    """,
)
def analyze_strategy(request: StrategyAnalysisRequest) -> StrategyAnalysisResponse:
    """Analyze an options strategy."""
    # Convert legs to calculator format
    legs = [
        {
            "type": leg.option_type.value,
            "strike": leg.strike,
            "premium": leg.premium,
            "quantity": leg.quantity,
        }
        for leg in request.legs
    ]
    
    # Analyze profit zones
    zone_analysis = _calculator.analyze_profit_zones(
        legs=legs,
        spot_price=request.spot_price,
        price_range_pct=0.50,
    )
    
    # Calculate probability of profit
    pop_analysis = _calculator.calculate_probability_of_profit(
        legs=legs,
        spot_price=request.spot_price,
        volatility=request.volatility,
        days_to_expiration=request.days_to_expiration,
        risk_free_rate=request.risk_free_rate,
        simulations=10000,
    )
    
    # Calculate aggregate Greeks
    # First, compute Greeks for each leg if not provided
    positions = []
    time_to_exp = request.days_to_expiration / 365.0
    
    for leg in request.legs:
        if leg.delta is not None:
            greeks = {
                "delta": leg.delta,
                "gamma": leg.gamma or 0,
                "theta": leg.theta or 0,
                "vega": leg.vega or 0,
            }
        else:
            # Calculate Greeks for this leg
            greeks = _calculator.calculate_black_scholes_greeks(
                option_type=leg.option_type.value,
                spot_price=request.spot_price,
                strike=leg.strike,
                time_to_expiration=time_to_exp,
                risk_free_rate=request.risk_free_rate,
                volatility=request.volatility,
            )
        
        positions.append({
            "symbol": f"{leg.option_type.value}_{leg.strike}",
            "quantity": leg.quantity,
            "greeks": greeks,
        })
    
    position_greeks = _calculator.calculate_position_greeks(positions)
    
    return StrategyAnalysisResponse(
        profit_zones=zone_analysis["profit_zones"],
        breakevens=zone_analysis["breakevens"],
        max_profit=zone_analysis["max_profit"],
        max_profit_at_price=zone_analysis["max_profit_at"],
        max_loss=zone_analysis["max_loss"],
        max_loss_at_price=zone_analysis["max_loss_at"],
        probability_of_profit=pop_analysis["probability_of_profit"],
        expected_profit=pop_analysis["expected_profit"],
        expected_loss=pop_analysis["expected_loss"],
        average_pnl=pop_analysis["average_pnl"],
        pnl_at_current_price=zone_analysis["pnl_at_current"],
        net_delta=position_greeks["net_delta"],
        net_gamma=position_greeks["net_gamma"],
        net_theta=position_greeks["net_theta"],
        net_vega=position_greeks["net_vega"],
    )


@router.post(
    "/position/aggregate",
    summary="Aggregate position Greeks",
    description="Calculate aggregate Greeks for a list of option positions.",
)
def aggregate_position_greeks(request: PositionGreeksRequest) -> Dict[str, float]:
    """Aggregate Greeks across multiple positions."""
    return _calculator.calculate_position_greeks(request.positions)


@router.post(
    "/implied-volatility",
    summary="Calculate implied volatility",
    description="""
    Calculate implied volatility from option market price using Newton-Raphson iteration.
    
    Returns the IV that makes the Black-Scholes price equal to the market price.
    """,
)
def calculate_implied_volatility(request: ImpliedVolRequest) -> Dict[str, Any]:
    """Calculate implied volatility from option price."""
    iv = _calculate_implied_vol(
        option_type=request.option_type.value,
        option_price=request.option_price,
        spot_price=request.spot_price,
        strike=request.strike,
        time_to_expiration=request.time_to_expiration,
        risk_free_rate=request.risk_free_rate,
        dividend_yield=request.dividend_yield,
    )
    
    if iv is None:
        raise HTTPException(
            status_code=400,
            detail="Could not calculate implied volatility. Check input parameters."
        )
    
    # Also return Greeks at this IV
    greeks = _calculator.calculate_black_scholes_greeks(
        option_type=request.option_type.value,
        spot_price=request.spot_price,
        strike=request.strike,
        time_to_expiration=request.time_to_expiration,
        risk_free_rate=request.risk_free_rate,
        volatility=iv,
        dividend_yield=request.dividend_yield,
    )
    
    return {
        "implied_volatility": round(iv, 4),
        "implied_volatility_pct": round(iv * 100, 2),
        "greeks": greeks,
    }


@router.get(
    "/chain/greeks",
    summary="Calculate Greeks for option chain",
    description="Calculate Greeks for multiple strikes at once.",
)
def chain_greeks(
    spot: float = Query(..., gt=0, description="Spot price"),
    strikes: str = Query(..., description="Comma-separated strikes (e.g., '100,105,110')"),
    dte: int = Query(..., ge=0, description="Days to expiration"),
    iv: float = Query(..., gt=0, description="IV as percentage"),
    rate: float = Query(default=5.0, description="Risk-free rate as percentage"),
) -> Dict[str, Any]:
    """Calculate Greeks for an option chain."""
    strike_list = [float(s.strip()) for s in strikes.split(",")]
    time_to_exp = dte / 365.0
    vol = iv / 100.0
    r = rate / 100.0
    
    results = {"spot_price": spot, "dte": dte, "iv": iv, "calls": [], "puts": []}
    
    for strike in strike_list:
        # Call Greeks
        call_greeks = _calculator.calculate_black_scholes_greeks(
            option_type="call",
            spot_price=spot,
            strike=strike,
            time_to_expiration=time_to_exp,
            risk_free_rate=r,
            volatility=vol,
        )
        call_price = _calculate_bs_price("call", spot, strike, time_to_exp, r, vol)
        
        results["calls"].append({
            "strike": strike,
            "price": round(call_price, 4),
            **call_greeks,
        })
        
        # Put Greeks
        put_greeks = _calculator.calculate_black_scholes_greeks(
            option_type="put",
            spot_price=spot,
            strike=strike,
            time_to_expiration=time_to_exp,
            risk_free_rate=r,
            volatility=vol,
        )
        put_price = _calculate_bs_price("put", spot, strike, time_to_exp, r, vol)
        
        results["puts"].append({
            "strike": strike,
            "price": round(put_price, 4),
            **put_greeks,
        })
    
    return results


# ============================================================================
# Helper Functions
# ============================================================================

def _calculate_bs_price(
    option_type: str,
    spot_price: float,
    strike: float,
    time_to_expiration: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Calculate Black-Scholes option price."""
    from scipy import stats
    
    if time_to_expiration <= 0:
        if option_type == "call":
            return max(0, spot_price - strike)
        else:
            return max(0, strike - spot_price)
    
    d1 = (math.log(spot_price / strike) + 
          (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiration) / \
         (volatility * math.sqrt(time_to_expiration))
    d2 = d1 - volatility * math.sqrt(time_to_expiration)
    
    if option_type == "call":
        price = (spot_price * math.exp(-dividend_yield * time_to_expiration) * stats.norm.cdf(d1) - 
                 strike * math.exp(-risk_free_rate * time_to_expiration) * stats.norm.cdf(d2))
    else:
        price = (strike * math.exp(-risk_free_rate * time_to_expiration) * stats.norm.cdf(-d2) - 
                 spot_price * math.exp(-dividend_yield * time_to_expiration) * stats.norm.cdf(-d1))
    
    return price


def _calculate_implied_vol(
    option_type: str,
    option_price: float,
    spot_price: float,
    strike: float,
    time_to_expiration: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    max_iterations: int = 100,
    precision: float = 1e-6,
) -> Optional[float]:
    """Calculate implied volatility using Newton-Raphson method."""
    from scipy import stats
    
    if time_to_expiration <= 0:
        return None
    
    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2 * math.pi / time_to_expiration) * option_price / spot_price
    sigma = max(0.01, min(sigma, 5.0))  # Clamp to reasonable range
    
    for _ in range(max_iterations):
        # Calculate price at current sigma
        price = _calculate_bs_price(
            option_type, spot_price, strike, time_to_expiration,
            risk_free_rate, sigma, dividend_yield
        )
        
        # Calculate vega
        d1 = (math.log(spot_price / strike) + 
              (risk_free_rate - dividend_yield + 0.5 * sigma**2) * time_to_expiration) / \
             (sigma * math.sqrt(time_to_expiration))
        vega = spot_price * math.exp(-dividend_yield * time_to_expiration) * \
               stats.norm.pdf(d1) * math.sqrt(time_to_expiration)
        
        if abs(vega) < 1e-10:
            break
        
        # Newton-Raphson update
        diff = option_price - price
        if abs(diff) < precision:
            return sigma
        
        sigma = sigma + diff / vega
        sigma = max(0.001, min(sigma, 10.0))  # Keep in reasonable range
    
    return sigma if abs(option_price - _calculate_bs_price(
        option_type, spot_price, strike, time_to_expiration,
        risk_free_rate, sigma, dividend_yield
    )) < 0.01 else None
