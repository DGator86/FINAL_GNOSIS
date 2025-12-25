import os
import sys
import numpy as np
import torch
import multiprocessing as mp
import random
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath("."))

from engines.ml.training.env import PhysicsEnv
from engines.ml.training.model import PhysicsAgent
from engines.ml.training.physics_wrapper import FastPhysicsWrapper
from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter

def evaluate_agent(args):
    weights, data = args
    model = PhysicsAgent()
    model.set_weights(weights)
    wrapper = FastPhysicsWrapper()
    env = PhysicsEnv(data, wrapper)
    obs = env.reset()
    total = 0.0
    done = False
    while not done:
        with torch.no_grad():
            action = model(torch.tensor(obs, dtype=torch.float32)).item()
        obs, r, done, _ = env.step(action)
        total += r
    return total

def run():
    print("🧬 Starting Incremental Training (Fast Mode)...")
    
    # 1. Data
    alpaca = AlpacaMarketDataAdapter()
    end = datetime(2025, 12, 20, tzinfo=timezone.utc)
    start = end - timedelta(days=365)
    bars = alpaca.get_bars("SPY", start, end, timeframe="1Hour")
    if not bars: return
    
    data_records = [
        {'ts': b.timestamp, 'open': float(b.open), 'high': float(b.high), 'low': float(b.low), 'close': float(b.close), 'volume': float(b.volume)}
        for b in bars
    ]
    print(f"✅ Loaded {len(data_records)} bars.")
    
    # 2. Config
    POP_SIZE = 16 # Small batch for speed
    GENERATIONS = 5
    SIGMA = 0.1
    ALPHA = 0.02
    SLICE_LEN = 200 # Short episodes
    
    master = PhysicsAgent()
    
    # Load previous if exists
    path = "data/models/physics_agent_best.pt"
    if os.path.exists(path):
        try:
            master.load_state_dict(torch.load(path))
            print("Loaded existing checkpoint.")
        except: pass
        
    base_weights = master.get_weights()
    n_params = len(base_weights)
    
    pool = mp.Pool(mp.cpu_count())
    
    for gen in range(GENERATIONS):
        noise = np.random.randn(POP_SIZE, n_params) * SIGMA
        candidates = []
        for i in range(POP_SIZE):
            w = base_weights + noise[i]
            start_idx = random.randint(0, len(data_records) - SLICE_LEN - 1)
            candidates.append((w, data_records[start_idx : start_idx + SLICE_LEN]))
            
        rewards = np.array(pool.map(evaluate_agent, candidates))
        
        # Update
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        update = np.dot(noise.T, adv)
        base_weights += ALPHA * update / (POP_SIZE * SIGMA)
        
        # Save Best
        best_idx = np.argmax(rewards)
        if rewards[best_idx] > 0: # Only save positive progress
             master.set_weights(base_weights + noise[best_idx])
             torch.save(master.state_dict(), path)
             
        print(f"Gen {gen+1} | Avg: {rewards.mean():.2f} | Max: {rewards.max():.2f}")
        
    pool.close()
    
if __name__ == "__main__":
    run()
