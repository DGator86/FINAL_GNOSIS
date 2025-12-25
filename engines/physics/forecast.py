from math import sqrt
from typing import List, Tuple
from .internal_schemas import GMMState, Forecast1m, GaussianComponent
from .utils_math import normal_cdf

def mixture_moments(comps: List[GaussianComponent]) -> Tuple[float, float]:
    """Calculate mean and variance of the mixture."""
    if not comps:
        return 0.0, 0.0
        
    mean = sum(c.w * c.mu for c in comps)
    second = sum(c.w * (c.var + c.mu * c.mu) for c in comps)
    var = max(second - mean * mean, 1e-12)
    return mean, var

def p_up_probability(comps: List[GaussianComponent], anchor: float) -> float:
    """Calculate P(price > anchor)."""
    # P(p > anchor) = Σ w_k * (1 - Φ((anchor-mu)/sigma))
    p = 0.0
    for c in comps:
        sigma = sqrt(max(c.var, 1e-12))
        p += c.w * (1.0 - normal_cdf((anchor - c.mu) / sigma))
    return min(max(p, 0.0), 1.0)

def p_tail_probability(comps: List[GaussianComponent], anchor: float, delta: float) -> float:
    """Calculate P(|price - anchor| > delta)."""
    p = 0.0
    for c in comps:
        sigma = sqrt(max(c.var, 1e-12))
        z1 = (anchor - delta - c.mu) / sigma
        z2 = (anchor + delta - c.mu) / sigma
        
        # Prob inside interval [anchor-delta, anchor+delta]
        # CDF(z2) - CDF(z1)
        inside = normal_cdf(z2) - normal_cdf(z1)
        
        # Prob outside is 1 - inside
        p += c.w * (1.0 - inside)
    return min(max(p, 0.0), 1.0)

def top_modes(comps: List[GaussianComponent], m: int = 5) -> Tuple[List[float], List[float]]:
    """Return top m modes by weight density."""
    # Rank by w/sigma (peak height)
    ranked = sorted(comps, key=lambda c: c.w / max(sqrt(c.var), 1e-9), reverse=True)
    mus = [c.mu for c in ranked[:m]]
    ws = [c.w for c in ranked[:m]]
    return mus, ws

def make_forecast(ts, horizon_min: int, comps: List[GaussianComponent], anchor: float, delta_tail: float) -> Forecast1m:
    """Create a forecast object from components."""
    mean, var = mixture_moments(comps)
    p_up = p_up_probability(comps, anchor)
    p_tail = p_tail_probability(comps, anchor, delta_tail)
    modes, weights = top_modes(comps)
    
    return Forecast1m(
        ts=ts,
        horizon_min=horizon_min,
        mean=mean,
        var=var,
        p_up=p_up,
        p_tail=p_tail,
        modes=modes,
        mode_weights=weights
    )
