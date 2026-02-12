from math import sqrt
from typing import Optional, Dict, Any
from engines.physics.internal_schemas import MicroState1m, Forecast1m, Symbol
from engines.physics.gmm_config import ModelConfig
from .scoring import estimate_cost

def execution_gate(sym: Symbol, micro: MicroState1m, fc: Forecast1m, cfg: ModelConfig) -> Optional[Dict[str, Any]]:
    """Determine if a trade should be executed."""
    
    exp_move = fc.mean - micro.anchor_price
    cost = estimate_cost(micro, cfg)
    risk = sqrt(max(fc.var, 1e-12))
    
    # Require move > cost + risk buffer
    # "Expected profit must exceed transaction cost plus a risk premium"
    if abs(exp_move) <= cost + 0.25 * risk:
        return None
        
    # Direction confidence gate
    # P_up must be significant for LONG, low for SHORT
    if exp_move > 0 and fc.p_up < 0.55:
        return None
    if exp_move < 0 and fc.p_up > 0.45:
        return None
        
    side = "BUY" if exp_move > 0 else "SELL"
    
    # Strength signal: Edge relative to risk (Sharpe-like)
    strength = abs(exp_move) / max(risk, 1e-9)
    
    return {
        "symbol": sym,
        "side": side,
        "strength": strength,
        "expected_move": exp_move,
        "p_up": fc.p_up,
        "confidence": fc.p_up if side == "BUY" else (1.0 - fc.p_up)
    }
