import os
import sys
import torch
import numpy as np
import pandas as pd
import gc
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from engines.ml.training.model import PhysicsAgent
from engines.ml.training.physics_wrapper import FastPhysicsWrapper
from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
from engines.inputs.synthetic_options_adapter import SyntheticOptionsAdapter
from engines.inputs.synthetic_generator import generate_heston_series
from market_model import Bar1m

# Configuration
MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
START_DATE = datetime(2025, 10, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 12, 20, tzinfo=timezone.utc)
TIMEFRAMES = ["1Hour", "15Min", "1Min"]

def evaluate_symbol_timeframe(model, bars, symbol, tf_name):
    """
    Run the ML agent on a sequence of bars and return metrics.
    """
    if not bars or len(bars) < 50:
        return None
        
    wrapper = FastPhysicsWrapper()
    wrapper.reset()
    
    equity = 10000.0
    position = 0.0
    cost_bps = 0.0005
    equity_curve = []
    
    # Warmup
    obs = np.zeros(6, dtype=np.float32)
    
    for i in range(len(bars) - 1):
        # 1. Action
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32)
            action = model(x).item()
            
        # 2. Execution
        target_pos = float(np.clip(action, -1.0, 1.0))
        
        curr_bar = bars[i]
        next_bar = bars[i+1]
        
        # Handle Bar1m object or Alpaca OHLCV object
        curr_price = float(curr_bar.close)
        next_price = float(next_bar.close)
        
        trade_size = abs(target_pos - position)
        cost = trade_size * cost_bps
        
        mkt_ret = (next_price - curr_price) / curr_price
        strat_ret = position * mkt_ret - cost
        
        equity *= (1.0 + strat_ret)
        position = target_pos
        equity_curve.append(equity)
        
        # 3. Update Physics & Obs
        # Update with NEXT bar to get state for NEXT step
        # Handle both types of bar objects
        ts_val = curr_bar.ts if hasattr(curr_bar, 'ts') else curr_bar.timestamp
        vol_val = float(curr_bar.volume)
        
        bar_dict = {
            'ts': ts_val, 'open': float(next_bar.open), 'high': float(next_bar.high),
            'low': float(next_bar.low), 'close': float(next_bar.close), 'volume': float(next_bar.volume)
        }
        phys = wrapper.update(bar_dict)
        
        # Normalize
        obs = np.array([
            min(phys['entropy'], 5.0),
            min(phys['stiffness'], 50.0)/10.0,
            (phys['p_up'] - 0.5) * 2.0,
            phys['sigma'] / next_price * 100,
            (float(next_bar.high) - float(next_bar.low)) / next_price * 100,
            (next_price - curr_price) / curr_price * 100
        ], dtype=np.float32)
        
    # Metrics
    total_ret = (equity - 10000.0) / 10000.0
    
    # Sharpe (Annualized based on timeframe)
    returns = pd.Series(equity_curve).pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        factor = 252
        if tf_name == "1Hour": factor = 252 * 7
        elif tf_name == "15Min": factor = 252 * 26
        elif tf_name == "1Min": factor = 252 * 390
        
        sharpe = returns.mean() / returns.std() * np.sqrt(factor)
    else:
        sharpe = 0.0
        
    # Max DD
    series = pd.Series(equity_curve)
    max_dd = ((series.cummax() - series) / series.cummax()).max() if len(series) > 0 else 0.0
    
    return {
        "Symbol": symbol,
        "TF": tf_name,
        "Return": total_ret,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "Bars": len(bars)
    }

def run_simulation():
    print("🤖 Starting Mag7 MTF Machine Learning Simulation")
    print("=" * 80)
    
    # 1. Load Model
    model_path = "data/models/physics_agent_best.pt"
    if not os.path.exists(model_path):
        # Fallback to creating a new one if not found (e.g. timeout previous step)
        print("⚠️  No trained model found. Initializing random agent for demo.")
        model = PhysicsAgent()
    else:
        print(f"✅ Loaded Trained Agent: {model_path}")
        model = PhysicsAgent()
        model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 2. Data Adapters
    alpaca = AlpacaMarketDataAdapter()
    
    results = []
    
    # ---------------------------------------------------------
    # PHASE A: REAL DATA (Historical)
    # ---------------------------------------------------------
    print("\n[PHASE A] REAL DATA BACKTEST")
    
    for symbol in MAG7:
        for tf in TIMEFRAMES:
            print(f"   Processing {symbol} {tf}...", end="\r")
            
            # Adjust start date for 1Min to avoid OOM (fetch smaller window instead of slicing after)
            local_start = START_DATE
            if tf == "1Min":
                # Only fetch last ~5 days for 1Min to ensure we get ~2000 bars without OOM on 2GB instance
                local_start = END_DATE - timedelta(days=5)
            
            # Fetch
            bars = alpaca.get_bars(symbol, local_start, END_DATE, timeframe=tf)
            
            # Limit 1Min bars to avoid timeout/memory issues (last 2000)
            if tf == "1Min" and len(bars) > 2000:
                bars = bars[-2000:]
                
            res = evaluate_symbol_timeframe(model, bars, symbol, tf)
            if res:
                res["Type"] = "Real"
                results.append(res)
                print(f"   {symbol} {tf}: Ret={res['Return']:.2%} | Sharpe={res['Sharpe']:.2f}")
            else:
                print(f"   {symbol} {tf}: No Data")
                
            # Explicit garbage collection to prevent OOM
            del bars
            gc.collect()

    # ---------------------------------------------------------
    # PHASE B: SYNTHETIC DATA (Stress Test)
    # ---------------------------------------------------------
    print("\n[PHASE B] SYNTHETIC STRESS TEST (Heston Volatility)")
    
    for symbol in MAG7:
        # Generate 1 Hour Synthetic (1 Year)
        synth_bars = generate_heston_series(symbol, START_DATE, START_DATE + timedelta(days=365), timeframe_minutes=60, vol=0.30) # High Vol
        
        res = evaluate_symbol_timeframe(model, synth_bars, symbol, "1Hour")
        if res:
            res["Type"] = "Synthetic"
            res["TF"] = "1Hour (HighVol)"
            results.append(res)
            print(f"   {symbol} Synthetic: Ret={res['Return']:.2%} | Sharpe={res['Sharpe']:.2f}")

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 MAG7 MTF ML SIMULATION REPORT")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    if not df.empty:
        # Group by Type/TF
        print(df.sort_values(["Type", "Symbol", "TF"]).to_string(index=False, float_format="%.2f"))
        
        print("-" * 80)
        print("Summary by Timeframe (Real Data):")
        real_df = df[df["Type"] == "Real"]
        print(real_df.groupby("TF")[["Return", "Sharpe", "MaxDD"]].mean())
        
        print("\nSummary by Data Type:")
        print(df.groupby("Type")[["Return", "Sharpe", "MaxDD"]].mean())
        
    print("=" * 80)

if __name__ == "__main__":
    run_simulation()
