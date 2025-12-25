import time
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("."))

from engines.ml.training.evolution import EvolutionNode

def run_hivemind():
    print("🐝 HiveMind: Continuous Evolution Engine Online")
    
    # 1. Initialize Node
    try:
        node = EvolutionNode("SPY", lookback_days=730) # 2 Years
    except Exception as e:
        print(f"❌ Failed to init node: {e}")
        return

    # 2. Meta-Parameters (State)
    current_alpha = 0.02
    current_sigma = 0.1
    best_avg_score = -float('inf')
    
    epoch = 0
    
    while True:
        epoch += 1
        print(f"\n🌀 [Epoch {epoch}] Baseline: α={current_alpha:.3f}, σ={current_sigma:.3f}")
        
        # --- A. BASELINE RUN ---
        # Run 3 generations to establish baseline
        print("   Running Baseline...")
        b_best, b_avg = node.train_epoch(generations=3, pop_size=16, 
                                         sigma=current_sigma, alpha=current_alpha)
        
        print(f"   >>> Baseline Score: Avg={b_avg:.2f}, Max={b_best:.2f}")
        
        # --- B. CHALLENGER RUN (Bandit) ---
        # Mutate meta-parameters
        import random
        strategy = random.choice(["alpha_up", "alpha_down", "sigma_up", "sigma_down"])
        
        c_alpha = current_alpha
        c_sigma = current_sigma
        
        if strategy == "alpha_up": c_alpha *= 1.5
        elif strategy == "alpha_down": c_alpha *= 0.7
        elif strategy == "sigma_up": c_sigma *= 1.5
        elif strategy == "sigma_down": c_sigma *= 0.7
        
        # Clamp
        c_alpha = max(0.001, min(c_alpha, 0.1))
        c_sigma = max(0.01, min(c_sigma, 0.5))
        
        print(f"   Running Challenger ({strategy}): α={c_alpha:.3f}, σ={c_sigma:.3f}")
        c_best, c_avg = node.train_epoch(generations=3, pop_size=16, 
                                         sigma=c_sigma, alpha=c_alpha)
        
        print(f"   >>> Challenger Score: Avg={c_avg:.2f}, Max={c_best:.2f}")
        
        # --- C. SELECTION ---
        # We compare AVG score (robustness) primarily
        if c_avg > b_avg:
            print(f"✅ Challenger Won! Updating params: {strategy}")
            current_alpha = c_alpha
            current_sigma = c_sigma
            best_avg_score = c_avg
        else:
            print(f"❌ Challenger Lost. Keeping baseline.")
            
        # Log to disk
        with open("evolution_log.txt", "a") as f:
            f.write(f"{datetime.now()}|Epoch {epoch}|Base: {b_avg:.2f}|Chal: {c_avg:.2f}|Winner: {'Chal' if c_avg > b_avg else 'Base'}|Alpha: {current_alpha:.4f}|Sigma: {current_sigma:.4f}\n")
            
        # Sleep slightly to prevent CPU melt?
        # Actually continuous means continuous.
        # But we print status.

if __name__ == "__main__":
    try:
        run_hivemind()
    except KeyboardInterrupt:
        print("\n🛑 HiveMind Stopped.")
