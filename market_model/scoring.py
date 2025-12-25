from math import sqrt
from typing import Tuple
from engines.physics.internal_schemas import MicroState1m, Forecast1m, ProfitScore, Symbol
from engines.physics.gmm_config import ModelConfig
from datetime import datetime

def estimate_cost(micro: MicroState1m, cfg: ModelConfig) -> float:
    """Estimate transaction cost including spread and slippage."""
    spread_cost = cfg.cost.spread_mult * micro.spread
    slippage = cfg.cost.slippage_mult_rv * micro.rv * micro.anchor_price
    # Add per-share fee logic here if needed, usually negligible for analysis relative to spread
    return spread_cost + slippage

def profit_score(symbol: Symbol, 
                 ts: datetime, 
                 forecast: Forecast1m, 
                 anchor: float, 
                 micro: MicroState1m, 
                 cfg: ModelConfig) -> Tuple[float, ProfitScore]:
    """Calculate profit score for a forecast."""
    
    exp_move = abs(forecast.mean - anchor)
    cost = estimate_cost(micro, cfg)
    edge = max(exp_move - cost, 0.0)
    risk = sqrt(max(forecast.var, 1e-12))
    
    # Must-move threshold delta: 1.5 spreads or cost
    delta = max(1.5 * micro.spread, cost)
    p_tail = forecast.p_tail # Assumed this was calculated with the relevant delta in forecast step
    
    # Alternatively, recalculate p_tail if the forecast one used a different delta
    # But for now assuming forecast.p_tail is sufficient or passed correctly
    
    raw_score = (edge / max(risk, 1e-9)) * p_tail
    
    ps = ProfitScore(
        ts=ts,
        symbol=symbol,
        horizon_min=forecast.horizon_min,
        expected_move=exp_move,
        cost=cost,
        edge=edge,
        risk=risk,
        score=raw_score
    )
    
    return raw_score, ps
