import os
import sys
import math
import pandas as pd
from datetime import datetime, timezone
from typing import List

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath("."))

from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
from market_model import (
    GMMState, MicroState1m, LiquidityField1m, GreeksField1m, WyckoffState1m,
    FlowAgg1m, QuoteL1, Bar1m, L2DepthSnapshot, GaussianComponent,
    gmm_step, make_forecast, build_micro_state, build_liquidity_field, ModelConfig
)

# Q2 2025 Period (The "Toxic" Period)
START = datetime(2025, 4, 1, tzinfo=timezone.utc)
END = datetime(2025, 4, 15, tzinfo=timezone.utc) # Just 2 weeks to verify physics
SYMBOL = "SPY"

def run():
    print(f"🔧 Starting Physics Remediation Check ({SYMBOL})")
    
    alpaca = AlpacaMarketDataAdapter()
    bars = alpaca.get_bars(SYMBOL, START, END, timeframe="1Hour")
    
    if not bars:
        print("❌ No data found.")
        return
        
    print(f"✅ Loaded {len(bars)} bars.")
    
    cfg = ModelConfig()
    cfg.dyn.q0 *= 7.0 
    
    state = None
    
    print("\nTime                 | Price   | Beta (Stiff) | Drift (Net) | Forecast | Error")
    print("-" * 85)
    
    mse = 0.0
    count = 0
    
    for i, b in enumerate(bars):
        ts = b.timestamp
        close = float(b.close)
        vwap = getattr(b, 'vwap', close) or close
        
        # Build Inputs
        quote = QuoteL1(ts=ts, bid=close, ask=close, bid_size=100, ask_size=100)
        flow = FlowAgg1m(ts=ts, buy_vol=b.volume/2, sell_vol=b.volume/2)
        internal_bar = Bar1m(ts=ts, open=b.open, high=b.high, low=b.low, close=close, volume=b.volume, vwap=vwap)
        l2 = L2DepthSnapshot(ts=ts, bid_prices=[], bid_sizes=[], ask_prices=[], ask_sizes=[])
        
        liq = build_liquidity_field(l2, internal_bar, quote)
        micro = build_micro_state(quote, internal_bar, flow, liq, GreeksField1m(ts=ts), WyckoffState1m(ts=ts))
        
        # Init State
        if not state:
            state = GMMState(ts=ts, components=[GaussianComponent(w=1.0, mu=close, var=close*0.001)])
            
        # Step Physics
        state = gmm_step(state, micro, liq, GreeksField1m(ts=ts), WyckoffState1m(ts=ts), close, cfg)
        
        # Forecast
        fc = make_forecast(ts, 1, state.components, close, 0.0)
        
        # Error tracking
        if i > 0:
            err = (close - prev_close)**2
            mse += err
            count += 1
            
        prev_close = close
        
        # Inspect Physics Internals roughly
        # Beta is in `liq.beta`
        # Drift is internal to GMM, but we can see the forecast deviation
        
        if i % 20 == 0:
            print(f"{ts.strftime('%Y-%m-%d %H:%M')} | {close:<7.2f} | {liq.beta:<12.2f} | {fc.mean - close:<11.2f} | {fc.mean:<8.2f} | {fc.mean - close:<5.2f}")

    rmse = math.sqrt(mse / count) if count > 0 else 0
    print("-" * 85)
    print(f"RMSE: {rmse:.4f}")
    print("Physics Check Complete.")

if __name__ == "__main__":
    run()
