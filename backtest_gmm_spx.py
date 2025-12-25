import random
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List

from market_model import (
    GMMState, MicroState1m, LiquidityField1m, GreeksField1m, WyckoffState1m,
    FlowAgg1m, QuoteL1, Bar1m, L2DepthSnapshot, GaussianComponent,
    gmm_step, make_forecast, profit_score, ModelConfig
)

# ---------------------------------------------------------
# 1. Synthetic Data Generator (SPX-like 1m bars)
# ---------------------------------------------------------
def generate_synthetic_spx(n_minutes=390, start_price=4500.0, vol_annual=0.15):
    """
    Generate synthetic 1-minute bars for SPX.
    Uses Heston-like stochastic volatility and micro-structure noise.
    """
    dt = 1 / (252 * 390)  # 1 minute in years
    prices = [start_price]
    ts = datetime(2023, 1, 1, 9, 30)
    data = []
    
    current_price = start_price
    current_vol = vol_annual
    theta = 0.15  # Long term vol
    kappa = 2.0   # Mean reversion speed of vol
    xi = 0.3      # Vol of vol
    
    for i in range(n_minutes):
        # Update Volatility (Heston)
        dw_vol = random.gauss(0, 1) * math.sqrt(dt)
        current_vol = abs(current_vol + kappa * (theta - current_vol) * dt + xi * math.sqrt(current_vol) * dw_vol)
        
        # Update Price
        dw_price = random.gauss(0, 1) * math.sqrt(dt)
        drift = 0.05 * dt # Slight upward drift
        
        # Add "microstructure noise" (mean reversion to "fair value")
        # Let's say fair value is the diffusion process, price oscillates slightly
        change = current_price * drift + current_price * math.sqrt(current_vol) * dw_price
        current_price += change
        
        # Generate Bar OHL C
        close = current_price
        # Random noise for high/low/open
        noise = current_price * 0.0005
        open_p = close + random.uniform(-noise, noise)
        high = max(open_p, close) + random.uniform(0, noise)
        low = min(open_p, close) - random.uniform(0, noise)
        vwap = (high + low + close) / 3.0
        
        # Generate Flow/Liquidity proxies
        # Volume correlates with volatility
        volume = int(1000 + 100000 * current_vol)
        
        # Quote spread
        spread = 0.25 # 25 cents on SPX
        bid = close - spread/2
        ask = close + spread/2
        bid_size = 100
        ask_size = 100
        
        data.append({
            "ts": ts,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "vwap": vwap,
            "volume": volume,
            "bid": bid,
            "ask": ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "spread": spread
        })
        
        ts += timedelta(minutes=1)
        
    return data

# ---------------------------------------------------------
# 2. Backtest Loop
# ---------------------------------------------------------
def run_backtest():
    print("Generating synthetic SPX data...")
    market_data = generate_synthetic_spx(n_minutes=120) # 2 hours
    print(f"Generated {len(market_data)} bars.")
    
    # Config
    cfg = ModelConfig()
    
    # Initial State
    start_price = market_data[0]["close"]
    initial_comps = [
        GaussianComponent(w=0.5, mu=start_price, var=1.0),
        GaussianComponent(w=0.3, mu=start_price*1.001, var=1.0),
        GaussianComponent(w=0.2, mu=start_price*0.999, var=1.0)
    ]
    state = GMMState(ts=market_data[0]["ts"], components=initial_comps)
    
    results = []
    
    print("\nStarting Backtest Loop (GMM Filter)...")
    print(f"{'Time':<20} | {'Price':<10} | {'Forecast':<10} | {'Sigma':<8} | {'P_Up':<6} | {'Entropy':<8} | {'K':<3}")
    print("-" * 95)
    
    for i, bar_dict in enumerate(market_data):
        if i == 0: continue # Need prev state
        
        # 1. Adapt Data to Schemas
        ts = bar_dict["ts"]
        bar = Bar1m(
            ts=ts, open=bar_dict["open"], high=bar_dict["high"], low=bar_dict["low"], 
            close=bar_dict["close"], vwap=bar_dict["vwap"], volume=bar_dict["volume"]
        )
        quote = QuoteL1(
            ts=ts, bid=bar_dict["bid"], ask=bar_dict["ask"], 
            bid_size=bar_dict["bid_size"], ask_size=bar_dict["ask_size"]
        )
        # Mock Flow (random imbalance)
        imb_vol = random.uniform(-0.3, 0.3) * bar.volume
        buy_vol = (bar.volume + imb_vol) / 2
        sell_vol = (bar.volume - imb_vol) / 2
        flow = FlowAgg1m(ts=ts, buy_vol=buy_vol, sell_vol=sell_vol)
        
        # Mock Greeks/Wyckoff/L2 (since we are focusing on GMM mechanics)
        greeks = GreeksField1m(ts=ts, gex=0.0) # Neutral gamma for simplicity
        wyck = WyckoffState1m(ts=ts)
        # L2 Depth Snapshot (Mock)
        l2 = L2DepthSnapshot(
            ts=ts, 
            bid_prices=[quote.bid, quote.bid-0.5, quote.bid-1.0], 
            bid_sizes=[100.0, 200.0, 500.0],
            ask_prices=[quote.ask, quote.ask+0.5, quote.ask+1.0],
            ask_sizes=[100.0, 200.0, 500.0]
        )
        
        # 2. Build Features
        from market_model import build_micro_state, build_liquidity_field
        liq = build_liquidity_field(l2, bar, quote)
        micro = build_micro_state(quote, bar, flow, liq, greeks, wyck)
        
        # 3. Step GMM
        obs_price = bar.vwap if bar.vwap else bar.close
        state = gmm_step(state, micro, liq, greeks, wyck, obs_price, cfg)
        
        # 4. Forecast (1 min horizon)
        fc = make_forecast(ts, 1, state.components, micro.anchor_price, delta_tail=1.0)
        
        # Record
        results.append({
            "ts": ts,
            "price": obs_price,
            "forecast_mean": fc.mean,
            "forecast_std": math.sqrt(fc.var),
            "p_up": fc.p_up,
            "entropy": state.entropy,
            "components": len(state.components)
        })
        
        # Print every 10 mins
        if i % 10 == 0:
            print(f"{ts.strftime('%H:%M:%S'):<20} | {obs_price:<10.2f} | {fc.mean:<10.2f} | {math.sqrt(fc.var):<8.2f} | {fc.p_up:<6.2f} | {state.entropy:<8.3f} | {len(state.components):<3}")

    # ---------------------------------------------------------
    # 3. Analysis
    # ---------------------------------------------------------
    df = pd.DataFrame(results)
    df["error"] = df["price"] - df["forecast_mean"].shift(1) # Error of predicting t from t-1
    rmse = np.sqrt((df["error"]**2).mean())
    
    print("\nBacktest Complete.")
    print(f"RMSE (1-step prediction): {rmse:.4f}")
    
    # Directional Accuracy
    df["actual_dir"] = np.sign(df["price"].diff())
    # If p_up > 0.5, we predict UP (1), else DOWN (-1)
    df["pred_dir"] = df["p_up"].apply(lambda p: 1 if p > 0.5 else -1).shift(1)
    
    acc = (df["actual_dir"] == df["pred_dir"]).mean()
    print(f"Directional Accuracy: {acc:.2%}")

if __name__ == "__main__":
    run_backtest()
