from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal
from datetime import datetime

Symbol = str

class QuoteL1(BaseModel):
    ts: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float

class Bar1m(BaseModel):
    ts: datetime  # minute close timestamp
    open: float
    high: float
    low: float
    close: float
    vwap: Optional[float] = None
    volume: float

class FlowAgg1m(BaseModel):
    ts: datetime
    buy_vol: float
    sell_vol: float
    buy_count: int = 0
    sell_count: int = 0
    dp_buy_vol: float = 0.0      # dark/off-exchange inferred buy
    dp_sell_vol: float = 0.0     # dark/off-exchange inferred sell
    # optional: aggressive vs passive splits if you have them
    passive_bid_add: float = 0.0
    passive_bid_cancel: float = 0.0
    passive_ask_add: float = 0.0
    passive_ask_cancel: float = 0.0

class L2DepthSnapshot(BaseModel):
    ts: datetime
    # aggregated depth by level: sorted best->worse
    bid_prices: List[float]
    bid_sizes: List[float]
    ask_prices: List[float]
    ask_sizes: List[float]

class GreeksField1m(BaseModel):
    ts: datetime
    gex: float = 0.0            # net gamma exposure proxy (signed)
    charm_ex: float = 0.0       # net charm exposure proxy (signed)
    theta_near: float = 0.0     # near-dated theta concentration proxy
    # strike magnets: levels where potential wells exist
    strike_levels: List[float] = Field(default_factory=list)
    strike_strength: List[float] = Field(default_factory=list)  # same length, >=0

WyckoffPhase = Literal["accumulation", "markup", "distribution", "markdown", "unknown"]

class WyckoffState1m(BaseModel):
    ts: datetime
    phase: WyckoffPhase = "unknown"
    trend_bias: float = 0.0     # [-1..+1] drift preference
    temperature_mult: float = 1.0  # >1 = more diffusive
    key_levels: List[float] = Field(default_factory=list)
    key_strength: List[float] = Field(default_factory=list)

class LiquidityField1m(BaseModel):
    ts: datetime
    # nodes / magnets derived from depth + volume profile (HVN/VWAP/POC etc.)
    node_levels: List[float] = Field(default_factory=list)
    node_strength: List[float] = Field(default_factory=list)
    # stiffness proxies (if you can slope-fit from L2, put it here)
    beta: float = 0.0           # stiffness (>=0)
    spread: float = 0.0
    top_depth: float = 0.0      # bid+ask size at touch

class MicroState1m(BaseModel):
    ts: datetime
    anchor_price: float         # microprice or vwap/close
    I: float                    # intercept gap proxy (signed)
    beta: float                 # stiffness (>=0)
    p_star: float               # implied intersection center
    rv: float                   # realized vol proxy
    spread: float

class GaussianComponent(BaseModel):
    w: float
    mu: float
    var: float  # sigma^2

class GMMState(BaseModel):
    ts: datetime
    components: List[GaussianComponent]
    # diagnostics
    entropy: float = 0.0
    log_score: float = 0.0

class Forecast1m(BaseModel):
    ts: datetime
    horizon_min: int
    mean: float
    var: float
    p_up: float
    p_tail: float  # P(|Δp| > threshold)
    modes: List[float] = Field(default_factory=list)     # top mode means
    mode_weights: List[float] = Field(default_factory=list)

class ProfitScore(BaseModel):
    ts: datetime
    symbol: Symbol
    horizon_min: int
    expected_move: float
    cost: float
    edge: float
    risk: float
    score: float
    churn_penalty: float = 0.0
