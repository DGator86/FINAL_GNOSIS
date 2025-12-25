from math import sqrt, log
from .internal_schemas import (
    Bar1m, QuoteL1, FlowAgg1m, L2DepthSnapshot, GreeksField1m, 
    WyckoffState1m, LiquidityField1m, MicroState1m
)

def build_liquidity_field(l2: L2DepthSnapshot, bar: Bar1m, quote: QuoteL1) -> LiquidityField1m:
    """
    Construct the Liquidity Field from L2 snapshot and bar data.
    """
    nodes = []
    strengths = []
    
    # 1. VWAP is the primary attractor
    vwap_val = getattr(bar, 'vwap', None)
    if vwap_val is None:
        vwap_val = (float(bar.high) + float(bar.low) + float(bar.close))/3.0
    nodes.append(float(vwap_val))
    strengths.append(1.0)
        
    # 2. L2 Nodes
    if l2.bid_prices and len(l2.bid_prices) > 0:
        nodes.append(float(l2.bid_prices[0]))
        strengths.append(0.5)
    if l2.ask_prices and len(l2.ask_prices) > 0:
        nodes.append(float(l2.ask_prices[0]))
        strengths.append(0.5)
            
    # 3. Estimate Stiffness (Beta) from OHLC Structure
    # Physics: Stiffness = Force / Displacement ~ Volume / Range
    # We smooth this over time in a real system, but here we calculate instantaneous
    # Handle both Bar1m and OHLCV schemas
    high_p = float(bar.high)
    low_p = float(bar.low)
    price_range = max(high_p - low_p, 0.01) # Avoid div by zero
    
    # Scaling factor to make beta roughly ~1.0 for "normal" liquidity
    # Assuming SPY normal vol ~ 100k, range ~ 0.50 -> beta ~ 200,000
    # We want beta to be manageable, e.g., 1.0 to 10.0
    # Let's scale by 1/100,000
    beta_raw = float(bar.volume) / price_range
    beta = beta_raw / 100000.0
    
    # Clamp for stability
    beta = max(0.1, min(beta, 50.0))
    
    return LiquidityField1m(
        ts=bar.ts if hasattr(bar, 'ts') else bar.timestamp,
        node_levels=nodes,
        node_strength=strengths,
        beta=beta,
        spread=quote.ask - quote.bid,
        top_depth=quote.bid_size + quote.ask_size
    )

def build_micro_state(quote: QuoteL1, 
                      bar: Bar1m, 
                      flow: FlowAgg1m, 
                      liq: LiquidityField1m, 
                      greeks: GreeksField1m, 
                      wyck: WyckoffState1m) -> MicroState1m:
    """
    Build the MicroState vector for the Gaussian Mixture Model.
    """
    
    # 1. Intercept (I) - Flow Imbalance
    # Normalize flow: (Buy - Sell) / (Buy + Sell) -> [-1, 1]
    total_vol = flow.buy_vol + flow.sell_vol
    if total_vol > 0:
        I = (flow.buy_vol - flow.sell_vol) / total_vol
    else:
        # Fallback to bar direction if no flow data
        I = 1.0 if float(bar.close) > float(bar.open) else -1.0
        
    # Scale I by relative volume (Significance)
    # If volume is huge, I is more significant force
    # We assume '100k' is standard bucket size
    vol_factor = min(total_vol / 100000.0, 5.0)
    I *= vol_factor
    
    # 2. Beta (Stiffness)
    beta = liq.beta
    
    # 3. P* (Fair Value) - Center of the Spring
    # Use bar.vwap if available, else Typical Price
    p_star = getattr(bar, 'vwap', None)
    if p_star is None:
        p_star = (float(bar.high) + float(bar.low) + float(bar.close)) / 3.0
    
    # 4. RV (Realized Volatility) - Temperature
    # Range / Open
    rv = (float(bar.high) - float(bar.low)) / max(float(bar.open), 0.01)
    
    return MicroState1m(
        ts=bar.ts if hasattr(bar, 'ts') else bar.timestamp,
        anchor_price=float(bar.close),
        I=I,
        beta=beta,
        p_star=float(p_star),
        rv=rv,
        spread=quote.ask - quote.bid
    )
