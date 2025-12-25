import os
import sys
import numpy as np
import torch
import multiprocessing as mp
import random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple

# Internal imports
from engines.ml.training.env import PhysicsEnv
from engines.ml.training.model import PhysicsAgent
from engines.ml.training.physics_wrapper import FastPhysicsWrapper
from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter

# Global worker function for multiprocessing (must be top-level)
def _worker_eval(args):
    weights, data_slice = args
    model = PhysicsAgent()
    model.set_weights(weights)
    wrapper = FastPhysicsWrapper()
    env = PhysicsEnv(data_slice, wrapper)
    
    obs = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32)
            action = model(x).item()
        obs, r, done, _ = env.step(action)
        total_reward += r
    return total_reward

class EvolutionNode:
    """
    Worker node that runs evolutionary strategies on cached market data.
    """
    def __init__(self, symbol="SPY", lookback_days=365):
        self.symbol = symbol
        self.data_records = self._load_data(lookback_days)
        self.model_path = "data/models/physics_agent_best.pt"
        self.pool = None
        
    def _load_data(self, days):
        print(f"📥 [Node] Loading {days} days of {self.symbol} data...")
        alpaca = AlpacaMarketDataAdapter()
        end = datetime(2025, 12, 20, tzinfo=timezone.utc)
        start = end - timedelta(days=days)
        bars = alpaca.get_bars(self.symbol, start, end, timeframe="1Hour")
        if not bars:
            raise ValueError("No data found for training")
            
        records = [
            {'ts': b.timestamp, 'open': float(b.open), 'high': float(b.high), 
             'low': float(b.low), 'close': float(b.close), 'volume': float(b.volume)}
            for b in bars
        ]
        print(f"✅ [Node] Data loaded: {len(records)} bars")
        return records

    def train_epoch(self, generations=5, pop_size=16, sigma=0.1, alpha=0.02, slice_len=200):
        """
        Run a training epoch with specific meta-parameters.
        Returns: (best_reward, avg_reward)
        """
        # Load latest master
        master = PhysicsAgent()
        if os.path.exists(self.model_path):
            try:
                master.load_state_dict(torch.load(self.model_path))
            except:
                pass # Start fresh if corrupt
        
        base_weights = master.get_weights()
        n_params = len(base_weights)
        
        # Init pool if needed
        if self.pool is None:
            self.pool = mp.Pool(mp.cpu_count())
            
        epoch_best_reward = -float('inf')
        
        for gen in range(generations):
            # 1. Mutate
            noise = np.random.randn(pop_size, n_params) * sigma
            candidates = []
            
            for i in range(pop_size):
                w = base_weights + noise[i]
                # Random time slice
                start_idx = random.randint(0, len(self.data_records) - slice_len - 1)
                data_slice = self.data_records[start_idx : start_idx + slice_len]
                candidates.append((w, data_slice))
                
            # 2. Evaluate
            rewards = np.array(self.pool.map(_worker_eval, candidates))
            
            # 3. Update Master (Natural Gradient)
            # Normalize rewards
            std_r = rewards.std()
            if std_r < 1e-6: std_r = 1e-6
            adv = (rewards - rewards.mean()) / std_r
            
            # w_new = w + alpha * (noise * adv)
            # Average the weighted noise vectors
            weighted_noise = np.dot(noise.T, adv)
            base_weights += alpha * weighted_noise / (pop_size * sigma)
            
            # 4. Check Records
            gen_best_idx = np.argmax(rewards)
            gen_best_val = rewards[gen_best_idx]
            gen_avg = rewards.mean()
            
            if gen_best_val > epoch_best_reward:
                epoch_best_reward = gen_best_val
                
            # Save if it's a positive outlier
            if gen_best_val > 500.0: # Arbitrary "Good" threshold
                # We assume the 'master' drift is capturing the general direction,
                # but we can also jump specifically to the best mutant if it's really good.
                # Standard ES follows gradient, but we save checkpoint of updated master.
                pass
                
            print(f"   [Gen {gen+1}] Avg: {gen_avg:.2f} | Max: {gen_best_val:.2f} | σ={sigma:.3f}")

        # Save updated master
        master.set_weights(base_weights)
        torch.save(master.state_dict(), self.model_path)
        
        return epoch_best_reward, rewards.mean()

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
