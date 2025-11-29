"""
Unified data contracts for GNOSIS V2.0
Fixes the dict vs list inconsistencies identified in feedback
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Literal
from datetime import datetime, date
import numpy as np


class OptionQuote(BaseModel):
    """Single option contract - enforces consistent schema"""

    symbol: str
    type: Literal["call", "put"]
    strike: float = Field(gt=0)
    expiration: date
    bid: float = Field(ge=0)
    ask: float = Field(ge=0)
    iv: float = Field(ge=0, le=5.0)
    volume: int = Field(ge=0)
    open_interest: int = Field(ge=0, alias="oi")
    delta: float = Field(ge=-1.0, le=1.0)
    gamma: float = Field(ge=0)
    vega: float = Field(ge=0)
    theta: float

    @property
    def mid_price(self) -> float:
        """Consistent mid calculation"""
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as % of mid"""
        mid = self.mid_price
        return (self.ask - self.bid) / mid if mid > 0 else 1.0


class OptionsChain(BaseModel):
    """Container for option quotes - fixes list vs dict confusion"""

    quotes: List[OptionQuote]

    def by_strike(self) -> Dict[float, List[OptionQuote]]:
        """Group by strike for easy access"""
        result = {}
        for quote in self.quotes:
            result.setdefault(quote.strike, []).append(quote)
        return result

    def calls_only(self) -> List[OptionQuote]:
        return [q for q in self.quotes if q.type == "call"]

    def puts_only(self) -> List[OptionQuote]:
        return [q for q in self.quotes if q.type == "put"]


class VolatilityMetrics(BaseModel):
    """Pre-computed volatility surface metrics"""

    atm_iv: float = Field(ge=0, le=5.0)
    iv_rank: float = Field(ge=0, le=100)  # Percentile 0-100
    iv_percentile: float = Field(ge=0, le=100)
    hv_20: float = Field(ge=0)
    hv_60: float = Field(ge=0)


class VolatilityStructure(BaseModel):
    """Term structure and skew data"""

    front_month_iv: float = Field(ge=0)
    back_month_iv: float = Field(ge=0)
    put_skew_25d: float
    call_skew_25d: float

    @property
    def term_slope(self) -> float:
        """Contango/backwardation"""
        if self.front_month_iv <= 0:
            return 0.0
        return (self.back_month_iv - self.front_month_iv) / self.front_month_iv


class MacroVolatilityData(BaseModel):
    """Cross-asset volatility context with pre-computed z-scores"""

    vix: float = Field(ge=0)
    vvix: float = Field(ge=0)
    move_index: float = Field(ge=0)
    credit_spreads: float = Field(ge=0)
    dxy_volatility: float = Field(ge=0)

    # Z-scores computed upstream to avoid history dependency issues
    move_z_score: Optional[float] = None
    credit_z_score: Optional[float] = None
    dxy_z_score: Optional[float] = None


class EnhancedMarketData(BaseModel):
    """Complete options market data package - THE SINGLE SOURCE OF TRUTH"""

    ticker: str
    timestamp: datetime
    spot_price: float = Field(gt=0)
    options_chain: OptionsChain
    volatility_metrics: VolatilityMetrics
    vol_structure: VolatilityStructure
    macro_vol_data: MacroVolatilityData


class OptionsIntelligenceOutput(BaseModel):
    """Standardized output from engines - THE CENTRAL BUS OBJECT"""

    # Volatility Intelligence
    regime_classification: Literal["R1", "R2", "R3", "R4", "R5"]
    regime_confidence: float = Field(ge=0, le=1.0)
    vol_edge: float
    macro_stress_score: float

    # Risk Context
    portfolio_vega_effective: float
    portfolio_gamma_effective: float
    portfolio_delta_effective: float
    vega_utilization: float = Field(ge=0, le=1.0)

    # Execution Context
    liquidity_tier: Literal["tier_1", "tier_2", "tier_3", "tier_4"]
    execution_cost_bps: float
    tradeable_strikes: List[float]
    execution_feasibility: Literal["excellent", "good", "fair", "poor"]

    # Market Intelligence
    edge_confidence: float = Field(ge=0, le=1.0)
    flow_bias: Literal["bullish", "bearish", "neutral"]
    dealer_positioning: Literal["short_gamma", "long_gamma", "neutral"]

    # Unified Scores
    opportunity_score: float = Field(ge=0, le=100)
    risk_score: float = Field(ge=0, le=100)
