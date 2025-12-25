import os
import sys
import numpy as np
import torch
import multiprocessing as mp
import random
from datetime import datetime, timedelta, timezone
from copy import deepcopy

sys.path.insert(0, os.path.abspath("."))

from engines.ml.training.env import PhysicsEnv
from engines.ml.training.model import PhysicsAgent
from engines.ml.training.physics_wrapper import FastPhysicsWrapper
from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter

def evaluate_agent(args):
    """
    Evaluates one agent on a specific data slice.
    """
    weights, data = args
    model = PhysicsAgent()
    model.set_weights(weights)
    wrapper = FastPhysicsWrapper()
    env = PhysicsEnv(data, wrapper)
    
    obs = env.reset()
    total_reward = 0.0
    done = False
    
    while not done:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32)
            action = model(x).item()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        
    return total_reward

def run_evolution():
    print("🧬 Starting Massive Evolutionary Training (Scale: 64x25)...")
    
    # 1. Load Data (Max History)
    alpaca = AlpacaMarketDataAdapter()
    symbol = "SPY"
    # Try 2 years if possible, else 1
    end = datetime(2025, 12, 20, tzinfo=timezone.utc)
    start = end - timedelta(days=730)
    
    print("📥 Fetching Training Data (2 Years)...")
    bars = alpaca.get_bars(symbol, start, end, timeframe="1Hour")
    
    if not bars:
        print("❌ No data.")
        return
        
    data_records = [
        {
            'ts': b.timestamp, 'open': float(b.open), 'high': float(b.high),
            'low': float(b.low), 'close': float(b.close), 'volume': float(b.volume)
        }
        for b in bars
    ]
    print(f"✅ Loaded {len(data_records)} bars.")
    
    # 2. ES Parameters (Scaled Up)
    POP_SIZE = 64
    GENERATIONS = 25
    SIGMA = 0.1
    ALPHA = 0.02 # Slightly higher learning rate for profit seeking
    SLICE_LEN = 1000 # 1000 Hours per episode (approx 6 months trading hours)
    
    master = PhysicsAgent()
    base_weights = master.get_weights()
    n_params = len(base_weights)
    
    # Hall of Fame
    best_reward_ever = -float('inf')
    best_weights_ever = base_weights.copy()
    
    pool = mp.Pool(mp.cpu_count())
    
    for gen in range(GENERATIONS):
        noise = np.random.randn(POP_SIZE, n_params) * SIGMA
        candidates = []
        
        # Randomize start points for robustness
        for i in range(POP_SIZE):
            w = base_weights + noise[i]
            # Ensure enough data remains
            if len(data_records) > SLICE_LEN:
                start_idx = random.randint(0, len(data_records) - SLICE_LEN - 1)
                slice_data = data_records[start_idx : start_idx + SLICE_LEN]
            else:
                slice_data = data_records
            candidates.append((w, slice_data))
            
        rewards = pool.map(evaluate_agent, candidates)
        rewards = np.array(rewards)
        
        # Track Best
        gen_best_idx = np.argmax(rewards)
        gen_best_reward = rewards[gen_best_idx]
        
        if gen_best_reward > best_reward_ever:
            best_reward_ever = gen_best_reward
            best_weights_ever = base_weights + noise[gen_best_idx]
            # Save Checkpoint
            master.set_weights(best_weights_ever)
            torch.save(master.state_dict(), "data/models/physics_agent_best.pt")
            print(f"   >>> New Record! {best_reward_ever:.2f}")
        
        # Normalize for Update
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        update_step = np.dot(noise.T, adv)
        base_weights += ALPHA * update_step / (POP_SIZE * SIGMA)
        
        print(f"Gen {gen+1}/{GENERATIONS} | Avg: {rewards.mean():.2f} | Max: {rewards.max():.2f} | Record: {best_reward_ever:.2f}")
        
    pool.close()
    print("💾 Training Complete. Best agent saved.")

if __name__ == "__main__":
    run_evolution()
