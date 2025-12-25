from math import exp, sqrt, log, isinf, isnan
from typing import List, Tuple
from .internal_schemas import GMMState, MicroState1m, LiquidityField1m, GreeksField1m, WyckoffState1m, GaussianComponent
from .gmm_config import ModelConfig
from .utils_math import normal_pdf

def node_width(p: float) -> float:
    return max(0.01, p * 0.0005)

def strike_width(p: float) -> float:
    return max(0.05, p * 0.001)

def wy_width(p: float) -> float:
    return max(0.02, p * 0.0008)

def measurement_noise(micro: MicroState1m, liq: LiquidityField1m, cfg: ModelConfig) -> float:
    # Noise scales with spread and realized vol
    r = 0.5 * micro.spread + 0.2 * micro.rv * micro.anchor_price
    return max(r, 1e-4)

def tick_size_estimate(micro: MicroState1m) -> float:
    return 0.01

def select_top_magnets(liq: LiquidityField1m, greeks: GreeksField1m, wyck: WyckoffState1m, max_magnets: int) -> List[float]:
    candidates = []
    for lvl, strength in zip(liq.node_levels, liq.node_strength):
        candidates.append((lvl, strength))
    for lvl, strength in zip(greeks.strike_levels, greeks.strike_strength):
        candidates.append((lvl, strength * 0.8))
    for lvl, strength in zip(wyck.key_levels, wyck.key_strength):
        candidates.append((lvl, strength * 0.9))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in candidates[:max_magnets]]

def already_represented(lvl: float, comps: List[GaussianComponent], threshold_std: float = 1.0) -> bool:
    for c in comps:
        sigma = sqrt(max(c.var, 1e-12))
        if abs(c.mu - lvl) < threshold_std * sigma:
            return True
    return False

def renormalize_weights(comps: List[GaussianComponent]) -> List[GaussianComponent]:
    w_sum = sum(c.w for c in comps)
    if w_sum < 1e-18:
        n = len(comps)
        for c in comps: c.w = 1.0 / n
    else:
        for c in comps: c.w /= w_sum
    return comps

def merge_close_components(comps: List[GaussianComponent], kl_thresh: float) -> List[GaussianComponent]:
    if not comps: return []
    comps.sort(key=lambda c: c.w, reverse=True)
    merged = []
    while comps:
        base = comps.pop(0)
        i = 0
        while i < len(comps):
            cand = comps[i]
            sigma_base = sqrt(max(base.var, 1e-12))
            sigma_cand = sqrt(max(cand.var, 1e-12))
            avg_sigma = (sigma_base + sigma_cand) / 2.0
            dist = abs(base.mu - cand.mu)
            if dist < 0.5 * avg_sigma:
                w_total = base.w + cand.w
                mu_new = (base.w * base.mu + cand.w * cand.mu) / w_total
                second_moment = (base.w * (base.var + base.mu**2) + cand.w * (cand.var + cand.mu**2)) / w_total
                var_new = second_moment - mu_new**2
                base.w = w_total
                base.mu = mu_new
                base.var = max(var_new, 1e-12)
                comps.pop(i)
            else:
                i += 1
        merged.append(base)
    return merged

def enforce_K_bounds(comps: List[GaussianComponent], k_min: int, k_max: int, k_target: int) -> List[GaussianComponent]:
    if len(comps) > k_max:
        comps.sort(key=lambda c: c.w, reverse=True)
        comps = comps[:k_max]
        comps = renormalize_weights(comps)
    return comps

def gmm_step(prev: GMMState,
             micro: MicroState1m,
             liq: LiquidityField1m,
             greeks: GreeksField1m,
             wyck: WyckoffState1m,
             y: float,
             cfg: ModelConfig) -> GMMState:
    
    comps = [GaussianComponent(w=c.w, mu=c.mu, var=c.var) for c in prev.components]

    # ---------------------------------------------------------
    # (1) PROPAGATE: LANGEVIN DYNAMICS
    # ---------------------------------------------------------
    # Physics: Stiffness Beta determines the spring constant
    # Restoring Force pulls to P_star (Long-term Value/VWAP)
    stiffness = max(micro.beta, 0.1)
    spring_k = stiffness * 0.05 
    
    # Drift from Flow Imbalance (Momentum)
    # Note: We now treat this as "Velocity" update, position comes next
    flow_force = (micro.I * 100.0)
    drift_momentum = (cfg.dyn.a_I_over_beta * flow_force) / stiffness

    for c in comps:
        dist_from_fair = c.mu - micro.p_star
        restoring_force = -spring_k * dist_from_fair
        total_drift = drift_momentum + restoring_force
        
        c.mu += total_drift
        q = cfg.dyn.q0 * (1.0 + micro.rv * 10.0)
        c.var += q * q

    # ---------------------------------------------------------
    # (2) FIELD REWEIGHT (Potentials)
    # ---------------------------------------------------------
    tau = cfg.field.tau0
    def V_total(p: float) -> float:
        V = 0.0
        for lvl, strength in zip(liq.node_levels, liq.node_strength):
            width = node_width(p)
            val = ((p - lvl) / width) ** 2
            V += -cfg.field.lambda_liq * strength * exp(-0.5 * min(val, 100))
        return V

    w_sum = 0.0
    for c in comps:
        V = V_total(c.mu)
        c.w = c.w * exp(-V / max(tau, 1e-9))
        w_sum += c.w
    if w_sum == 0: w_sum = 1.0
    for c in comps: c.w /= w_sum

    # ---------------------------------------------------------
    # (3) INTERCEPT PACKET SHAPING (The "Missing Link")
    # ---------------------------------------------------------
    # Calculate the Supply/Demand Intercept Price
    # P_int = P_anchor + (Imbalance / Stiffness)
    # This is the "Predicted Price" based purely on order flow physics.
    
    # Scale I/Beta to price units appropriately. 
    # In features_1m, we normalized Beta to ~1.0 for standard liquid stocks.
    # I is [-1, 1] roughly.
    # Impact should be roughly 1-5 ticks per unit of imbalance for liquid stocks.
    # Let's say Imbalance 1.0 (Huge Buy) on Beta 1.0 (Normal) moves price 0.1%?
    # We use a tunable scalar.
    
    impact_scalar = 5.0 # Ticks? Or Price Units?
    # If price is 500, 0.1% is 0.50. 
    # If using absolute price units, let's assume 1.0 means $1.00 move on SPY.
    
    p_intercept = micro.anchor_price + (micro.I / stiffness) * impact_scalar
    
    # Packet Width (Uncertainty of the intercept)
    # Determines how "Hard" we force the distribution to this intercept.
    # Low Beta (Thin) -> Wide Packet (Uncertain).
    # High Beta (Thick) -> Narrow Packet (Certain).
    sigma_packet_sq = (1.0 / stiffness) ** 2
    
    w_sum = 0.0
    for c in comps:
        # Gaussian weighting around the intercept
        dist_sq = (c.mu - p_intercept) ** 2
        # We apply this as a multiplicative likelihood (Bayesian update prior to measurement)
        packet_weight = exp(-0.5 * dist_sq / max(sigma_packet_sq, 1e-6))
        c.w *= packet_weight
        w_sum += c.w
        
    if w_sum < 1e-18: w_sum = 1.0 # Safety
    for c in comps: c.w /= w_sum

    # ---------------------------------------------------------
    # (4) MEASUREMENT UPDATE (Wavefunction Collapse)
    # ---------------------------------------------------------
    r = measurement_noise(micro, liq, cfg)
    r2 = r*r
    w_sum = 0.0
    
    for c in comps:
        pred_var = c.var + r2
        like = normal_pdf(y, mean=c.mu, var=pred_var)
        c.w *= like
        w_sum += c.w
        
        denom = c.var + r2
        if denom < 1e-12: denom = 1e-12
        kalman_gain = c.var / denom
        c.mu = c.mu + kalman_gain * (y - c.mu)
        c.var = (1.0 - kalman_gain) * c.var
        
    if w_sum < 1e-18:
        comps = [GaussianComponent(w=1.0, mu=y, var=r2)]
        w_sum = 1.0
    for c in comps: c.w /= w_sum
    
    # ---------------------------------------------------------
    # (5) MAINTENANCE
    # ---------------------------------------------------------
    magnets = select_top_magnets(liq, greeks, wyck, 2)
    for lvl in magnets:
        if not already_represented(lvl, comps):
            comps.append(GaussianComponent(w=0.05, mu=lvl, var=r2))
            
    comps = renormalize_weights(comps)
    comps = [c for c in comps if c.w > 0.001]
    comps = merge_close_components(comps, 0.1)
    comps = enforce_K_bounds(comps, 4, 12, 8)
    
    entropy = -sum(c.w * log(c.w) for c in comps if c.w > 0)
    
    return GMMState(ts=micro.ts, components=comps, entropy=entropy, log_score=log(w_sum))
