import os
import sys
import numpy as np
import torch
import random
import time
from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from engines.ml.training.env import PhysicsEnv
from engines.ml.training.model import PhysicsAgent
from engines.ml.training.physics_wrapper import FastPhysicsWrapper

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
POP_SIZE = 16          # Sequential mode: keep small
GENERATIONS = 10       # Daily fine-tuning
SIGMA = 0.05           # Smaller sigma for fine-tuning (stability)
ALPHA = 0.01           # Conservative learning rate
SLICE_LEN = 100        # Short episodes (approx 2 weeks of trading hours)
SAMPLES_PER_AGENT = 3  # Evaluate each agent on 3 different random tickers/slices
REPORT_DIR = "reports"

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_env_universe():
    """Manually parse .env to get UNIVERSE_SYMBOLS"""
    symbols = []
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.strip().startswith("UNIVERSE_SYMBOLS="):
                    # Extract value after =
                    val = line.strip().split("=", 1)[1]
                    # Remove potential comments
                    val = val.split("#")[0].strip()
                    symbols = [s.strip() for s in val.split(",") if s.strip()]
                    break
    except Exception as e:
        print(f"⚠️ Error reading .env: {e}")
    
    if not symbols:
        print("⚠️ UNIVERSE_SYMBOLS not found in .env, defaulting to core list.")
        symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMD"]
        
    return symbols

def get_yfinance_data_multi(symbols, days=60):
    """Download data for multiple symbols"""
    print(f"📥 Downloading data for {len(symbols)} symbols (last {days} days)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data_map = {}
    
    # Download in chunks to be nice to API or just sequential loop if list is small
    # yfinance can handle multiple tickers in one string "AAPL MSFT ..."
    
    tickers_str = " ".join(symbols)
    try:
        # Using threads=False for safety in restricted env, though True is faster
        df_all = yf.download(tickers_str, start=start_date, end=end_date, interval="1h", progress=False, threads=False)
        
        if df_all.empty:
            print("❌ No data returned from yfinance.")
            return {}
            
        # yfinance structure depends on number of tickers
        # If 1 ticker, columns are [Open, High...]
        # If >1 ticker, columns are MultiIndex (Price, Ticker) -> need to swap or iterate
        
        if len(symbols) == 1:
            # Handle single ticker case
            sym = symbols[0]
            data_map[sym] = process_single_df(df_all)
        else:
            # Multi-ticker case
            # Stack/Unstack or iterate columns. 
            # Easiest: iterate symbols and extract cross-section if MultiIndex
            if isinstance(df_all.columns, pd.MultiIndex):
                # Columns: (PriceType, Symbol) e.g. ('Open', 'AAPL')
                # We want to slice by symbol
                for sym in symbols:
                    try:
                        # Cross section for this symbol
                        df_sym = df_all.xs(sym, level=1, axis=1)
                        records = process_single_df(df_sym)
                        if records:
                            data_map[sym] = records
                    except KeyError:
                        pass # Symbol might have failed
            else:
                # Should not happen if multiple symbols requested, but fallback
                print("⚠️ Unexpected DataFrame structure.")
                
    except Exception as e:
        print(f"⚠️ Bulk download failed: {e}. Trying sequential fallback...")
        # Fallback: Sequential download
        for sym in symbols:
            try:
                df = yf.download(sym, start=start_date, end=end_date, interval="1h", progress=False)
                records = process_single_df(df)
                if records:
                    data_map[sym] = records
            except:
                continue
                
    print(f"✅ Loaded data for {len(data_map)}/{len(symbols)} symbols.")
    return data_map

def process_single_df(df):
    """Convert DF to list of dict records"""
    if df.empty: return []
    
    # Handle potential multi-index if passed raw
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except: pass
        
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    
    records = []
    for _, row in df.iterrows():
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
        except: continue
    return records

def evaluate_agent(weights, data_map):
    """Evaluate agent on random samples from the universe"""
    model = PhysicsAgent()
    model.set_weights(weights)
    wrapper = FastPhysicsWrapper()
    
    total_score = 0
    valid_samples = 0
    
    symbols = list(data_map.keys())
    
    for _ in range(SAMPLES_PER_AGENT):
        # Pick random symbol
        sym = random.choice(symbols)
        records = data_map[sym]
        
        if len(records) <= SLICE_LEN:
            continue
            
        # Pick random slice
        start_idx = random.randint(0, len(records) - SLICE_LEN - 1)
        data_slice = records[start_idx : start_idx + SLICE_LEN]
        
        # Run Env
        env = PhysicsEnv(data_slice, wrapper)
        obs = env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            with torch.no_grad():
                action = model(torch.tensor(obs, dtype=torch.float32)).item()
            obs, r, done, _ = env.step(action)
            ep_reward += r
            
        total_score += ep_reward
        valid_samples += 1
        
    if valid_samples == 0:
        return 0.0
        
    return total_score / valid_samples

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def run():
    start_time = time.time()
    date_str = datetime.now().strftime("%Y-%m-%d")
    print(f"🚀 Starting Daily Universe Training - {date_str}")
    
    # 1. Load Universe
    symbols = load_env_universe()
    print(f"🌌 Universe: {len(symbols)} tickers ({', '.join(symbols[:5])}...)")
    
    # 2. Load Data
    data_map = get_yfinance_data_multi(symbols, days=60)
    if not data_map:
        print("❌ No data available. Aborting.")
        return
        
    # 3. Initialize Model
    master = PhysicsAgent()
    path = "data/models/physics_agent_best.pt"
    initial_loaded = False
    
    if os.path.exists(path):
        try:
            master.load_state_dict(torch.load(path))
            print(f"✅ Loaded base model from {path}")
            initial_loaded = True
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            
    base_weights = master.get_weights()
    n_params = len(base_weights)
    
    # 4. Training Loop (Sequential)
    print(f"🏋️ Training for {GENERATIONS} gens (Pop: {POP_SIZE})...")
    
    best_gen_reward = -float('inf')
    improvement = 0.0
    
    for gen in range(GENERATIONS):
        noise = np.random.randn(POP_SIZE, n_params) * SIGMA
        rewards = []
        
        for i in range(POP_SIZE):
            w = base_weights + noise[i]
            r = evaluate_agent(w, data_map)
            rewards.append(r)
            
        rewards = np.array(rewards)
        
        # Stats
        gen_max = rewards.max()
        gen_mean = rewards.mean()
        
        if gen_max > best_gen_reward:
            best_gen_reward = gen_max
            
        # Update
        std = rewards.std()
        if std < 1e-8: std = 1e-8
        adv = (rewards - gen_mean) / std
        update = np.dot(noise.T, adv)
        base_weights += ALPHA * update / (POP_SIZE * SIGMA)
        
        # Checkpoint Best of Gen if positive
        best_idx = np.argmax(rewards)
        if rewards[best_idx] > 0:
             master.set_weights(base_weights + noise[best_idx])
             torch.save(master.state_dict(), path)
             
        print(f"   Gen {gen+1} | Avg: {gen_mean:.2f} | Max: {gen_max:.2f}")
        
    duration = time.time() - start_time
    
    # 5. Reporting
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
        
    report_file = os.path.join(REPORT_DIR, f"training_report_{date_str}.txt")
    
    report_content = f"""
==================================================
GNOSIS DAILY TRAINING REPORT
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
==================================================

STATUS: SUCCESS
DURATION: {duration:.1f} seconds

CONFIG:
- Universe Size: {len(symbols)}
- Data Loaded: {len(data_map)} tickers
- Generations: {GENERATIONS}
- Pop Size: {POP_SIZE}

RESULTS:
- Best Reward Seen: {best_gen_reward:.2f}
- Final Model Saved: {path}

NOTES:
- Trained on random sampling of entire universe.
- Optimization focused on recent 60-day price action.
- Sequential execution used for stability.

==================================================
"""
    
    with open(report_file, "w") as f:
        f.write(report_content)
        
    print("\n📝 Report Generated:")
    print(report_content)
    print(f"✅ Saved to {report_file}")

if __name__ == "__main__":
    run()
