import os
import sys
import numpy as np
import torch
import multiprocessing as mp
import random
from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from engines.ml.training.env import PhysicsEnv
from engines.ml.training.model import PhysicsAgent
from engines.ml.training.physics_wrapper import FastPhysicsWrapper

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

def get_yfinance_data(symbol="SPY", days=365):
    print(f"📥 Downloading data for {symbol} via yfinance...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = yf.download(symbol, start=start_date, end=end_date, interval="1h", progress=False)
    
    if df.empty:
        print("❌ No data found.")
        return []
    
    # Flatten multi-index columns if present (yfinance update)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Reset index to get 'Date' or 'Datetime' as a column
    df = df.reset_index()
    
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    
    records = []
    for _, row in df.iterrows():
        # Handle different timestamp column names
        ts = row.get('datetime', row.get('date', None))
        if pd.isna(ts): continue
        
        try:
            record = {
                'ts': ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            }
            records.append(record)
        except Exception as e:
            continue
            
    print(f"✅ Loaded {len(records)} bars from yfinance.")
    return records

def run():
    print("🚀 Starting Monday Prep Training (Robust Mode)...")
    
    # 1. Data (Using yfinance for speed/reliability in sandbox)
    data_records = get_yfinance_data("SPY", days=365)
    
    if not data_records:
        print("❌ Aborting: No training data.")
        return

    # 2. Config
    POP_SIZE = 32
    GENERATIONS = 10
    SIGMA = 0.1
    ALPHA = 0.02
    SLICE_LEN = 200 # Short episodes for incremental adaptation
    
    master = PhysicsAgent()
    
    # Load previous best model
    path = "data/models/physics_agent_best.pt"
    if os.path.exists(path):
        try:
            master.load_state_dict(torch.load(path))
            print(f"✅ Loaded existing checkpoint from {path}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            
    base_weights = master.get_weights()
    n_params = len(base_weights)
    
    # Use fewer processes to avoid memory/timeout issues in sandbox
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(4) 
    
    print(f"🏋️ Training for {GENERATIONS} generations with pop_size={POP_SIZE}...")
    
    for gen in range(GENERATIONS):
        noise = np.random.randn(POP_SIZE, n_params) * SIGMA
        candidates = []
        
        # Focus on recent data for "Monday Prep"
        # We want the model to be attuned to the latest market regime
        # So we bias start_idx towards the end of the dataset
        
        for i in range(POP_SIZE):
            w = base_weights + noise[i]
            
            # 50% chance to train on VERY recent data (last 30 days window)
            # 50% chance to train on random slice from last year (prevent overfitting)
            if random.random() > 0.5 and len(data_records) > SLICE_LEN:
                # Recent history focus
                latest_start = len(data_records) - SLICE_LEN - 1
                earliest_start = max(0, len(data_records) - (24 * 30)) # Last 30 days approx
                if earliest_start < latest_start:
                    start_idx = random.randint(earliest_start, latest_start)
                else:
                    start_idx = max(0, len(data_records) - SLICE_LEN - 1)
            else:
                # General history
                start_idx = random.randint(0, max(0, len(data_records) - SLICE_LEN - 1))
                
            candidates.append((w, data_records[start_idx : start_idx + SLICE_LEN]))
            
        rewards = np.array(pool.map(evaluate_agent, candidates))
        
        # Update
        std = rewards.std()
        if std < 1e-8: std = 1e-8
            
        adv = (rewards - rewards.mean()) / std
        update = np.dot(noise.T, adv)
        base_weights += ALPHA * update / (POP_SIZE * SIGMA)
        
        # Save Best
        best_idx = np.argmax(rewards)
        current_best_reward = rewards[best_idx]
        
        if current_best_reward > 0:
             master.set_weights(base_weights + noise[best_idx])
             torch.save(master.state_dict(), path)
             
        print(f"   Gen {gen+1} | Avg: {rewards.mean():.2f} | Max: {rewards.max():.2f}")
        
    pool.close()
    pool.join()
    print("✅ Training Complete. Model updated for Monday.")

if __name__ == "__main__":
    run()
