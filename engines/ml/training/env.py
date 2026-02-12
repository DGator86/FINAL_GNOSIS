import numpy as np
import math
from dataclasses import dataclass

@dataclass
class SimState:
    equity: float = 10000.0
    position: float = 0.0 # -1.0 to 1.0
    peak_equity: float = 10000.0
    max_dd: float = 0.0
    
class PhysicsEnv:
    """Fast simulation environment for Physics Engine."""
    
    def __init__(self, market_data, gmm_engine, cost_bps=0.0005):
        self.data = market_data # List of Bars
        self.engine = gmm_engine
        self.cost_bps = cost_bps
        self.reset()
        
    def reset(self):
        self.idx = 0
        self.sim = SimState()
        self.engine.reset() 
        return self._get_obs()
        
    def step(self, action):
        if self.idx >= len(self.data) - 1:
            return self._get_obs(), 0.0, True, {}
            
        current_bar = self.data[self.idx]
        next_bar = self.data[self.idx + 1]
        
        current_price = float(current_bar['close'])
        next_price = float(next_bar['close'])
        
        # Action -> Target Position
        target_pos = float(np.clip(action, -1.0, 1.0))
        
        # Friction
        trade_size = abs(target_pos - self.sim.position)
        cost = trade_size * self.cost_bps
        
        # Return
        mkt_ret = (next_price - current_price) / current_price
        strategy_ret = self.sim.position * mkt_ret - cost
        
        # Update Equity
        self.sim.equity *= (1.0 + strategy_ret)
        self.sim.position = target_pos
        
        # Drawdown Tracking
        self.sim.peak_equity = max(self.sim.peak_equity, self.sim.equity)
        dd = (self.sim.peak_equity - self.sim.equity) / self.sim.peak_equity
        self.sim.max_dd = max(self.sim.max_dd, dd)
        
        # Physics Update
        phys_obs = self.engine.update(next_bar)
        
        # Normalize Observation
        # [Entropy, Stiffness, P_up, Sigma, Volatility, Momentum]
        obs = np.array([
            min(phys_obs['entropy'], 5.0), # Clamp entropy
            min(phys_obs['stiffness'], 50.0) / 10.0, # Scale stiffness
            (phys_obs['p_up'] - 0.5) * 2.0, # Center prob (-1 to 1)
            (phys_obs['sigma'] / next_price * 100), 
            ((next_bar['high'] - next_bar['low']) / next_price * 100),
            ((next_price - current_price) / current_price * 100)
        ], dtype=np.float32)
        
        self.idx += 1
        done = self.idx >= len(self.data) - 1
        
        # --- AGGRESSIVE REWARD FUNCTION ---
        # 1. Base Return (Basis Points)
        r_step = strategy_ret * 10000.0 
        
        # 2. Drawdown Penalty (Exponential)
        # If in drawdown, reward is heavily discounted
        dd_penalty = math.exp(dd * 10) - 1.0
        
        # 3. Inaction Penalty (Bleed)
        # If position is 0, apply small negative reward to force action
        inaction = -1.0 if abs(self.sim.position) < 0.1 else 0.0
        
        final_reward = r_step - (dd_penalty * 2.0) + inaction
        
        return obs, final_reward, done, {}

    def _get_obs(self):
        return np.zeros(6, dtype=np.float32)
