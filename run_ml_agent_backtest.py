import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from engines.ml.training.model import PhysicsAgent
from engines.ml.training.physics_wrapper import FastPhysicsWrapper
from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter

# Q2 2025 Toxic Period
START = datetime(2025, 4, 1, tzinfo=timezone.utc)
END = datetime(2025, 6, 30, tzinfo=timezone.utc)
SYMBOL = "SPY"

def run():
    print(f"🤖 Testing Trained Physics Agent on {SYMBOL} ({START.date()} - {END.date()})")
    
    # Load Model
    model = PhysicsAgent()
    try:
        model.load_state_dict(torch.load("data/models/physics_agent_v1.pt"))
        model.eval()
        print("✅ Loaded trained model.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Load Data
    alpaca = AlpacaMarketDataAdapter()
    bars = alpaca.get_bars(SYMBOL, START, END, timeframe="1Hour")
    if not bars:
        print("❌ No data.")
        return
        
    print(f"✅ Loaded {len(bars)} bars.")
    
    # Run Sim
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
        
        curr_price = float(curr_bar.close)
        next_price = float(next_bar.close)
        
        trade_size = abs(target_pos - position)
        cost = trade_size * cost_bps
        
        mkt_ret = (next_price - curr_price) / curr_price
        strat_ret = position * mkt_ret - cost
        
        equity *= (1.0 + strat_ret)
        position = target_pos
        
        # 3. Update Physics & Obs
        # Update with NEXT bar to get state for NEXT step
        bar_dict = {
            'ts': next_bar.timestamp, 'open': float(next_bar.open), 'high': float(next_bar.high),
            'low': float(next_bar.low), 'close': float(next_bar.close), 'volume': float(next_bar.volume)
        }
        phys = wrapper.update(bar_dict)
        
        obs = np.array([
            phys['entropy'],
            phys['stiffness'],
            phys['p_up'],
            phys['sigma'] / next_price * 100,
            (float(next_bar.high) - float(next_bar.low)) / next_price * 100,
            (next_price - curr_price) / curr_price * 100
        ], dtype=np.float32)
        
        if i % 50 == 0:
            print(f"Step {i} | Eq: ${equity:.2f} | Act: {action:+.2f} | Ent: {phys['entropy']:.2f}")
            
    print("-" * 60)
    print(f"💰 Final Equity: ${equity:.2f}")
    print(f"📈 Return: {(equity - 10000)/10000:.2%}")

if __name__ == "__main__":
    run()
