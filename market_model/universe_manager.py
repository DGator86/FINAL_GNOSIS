from typing import List, Dict, Set
from engines.physics.internal_schemas import Symbol, ProfitScore
from engines.physics.gmm_config import ModelConfig

def update_universe(prev_active: List[Symbol],
                    scored: List[ProfitScore],
                    cfg: ModelConfig,
                    hold_age: Dict[Symbol, int]) -> List[Symbol]:
    """Update the active universe based on scores and churn control."""
    
    # Sort by score desc
    scored_sorted = sorted(scored, key=lambda x: x.score, reverse=True)
    
    # Determine thresholds
    # Score of the Nth best candidate
    cutoff_idx = min(len(scored_sorted), cfg.uni.active_N) - 1
    cutoff_score = scored_sorted[cutoff_idx].score if cutoff_idx >= 0 else 0.0
    
    add_threshold = cutoff_score * (1.0 + cfg.uni.add_margin)
    drop_threshold = cutoff_score * cfg.uni.drop_threshold_frac
    
    active_set = set(prev_active)
    
    # Drops: only if below drop threshold and min hold satisfied
    drops = []
    # Check each currently active symbol
    for sym in list(active_set):
        # Find current score
        s = 0.0
        for ps in scored_sorted:
            if ps.symbol == sym:
                s = ps.score
                break
        
        # If held long enough and score is weak
        if hold_age.get(sym, 0) >= cfg.uni.min_hold_minutes and s < drop_threshold:
            drops.append(sym)
            
    # Apply max turnover constraint for drops
    for sym in drops[:cfg.uni.max_turnover_per_minute]:
        if sym in active_set:
            active_set.remove(sym)
            
    # Adds: only if above add threshold
    adds = []
    for ps in scored_sorted:
        if ps.symbol in active_set:
            continue
            
        if ps.score >= add_threshold:
            adds.append(ps.symbol)
            
        # Stop if we have enough adds to potentially fill the universe
        if len(active_set) + len(adds) >= cfg.uni.active_N + cfg.uni.max_turnover_per_minute:
            break
            
    # Apply max turnover constraint for adds
    for sym in adds[:cfg.uni.max_turnover_per_minute]:
        active_set.add(sym)
        if len(active_set) >= cfg.uni.active_N * 1.5: # Safety cap
            break
            
    # Fill logic: If below target N, fill with best available regardless of add_threshold
    if len(active_set) < cfg.uni.active_N:
        for ps in scored_sorted:
            if ps.symbol not in active_set:
                active_set.add(ps.symbol)
            if len(active_set) >= cfg.uni.active_N:
                break
                
    return list(active_set)
