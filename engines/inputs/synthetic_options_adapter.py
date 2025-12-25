import random
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath("../../"))

from engines.inputs.options_chain_adapter import OptionContract
from engines.inputs.market_data_adapter import MarketDataAdapter
from utils_bs import black_scholes, calculate_delta

class SyntheticOptionsAdapter:
    """
    Generates realistic option chains based on historical spot price.
    Uses Black-Scholes for pricing and Greeks.
    """
    
    def __init__(self, market_adapter: MarketDataAdapter):
        self.market_adapter = market_adapter
        self.risk_free_rate = 0.05
        # Base volatility assumption (can be refined to use realized vol)
        self.base_volatility = 0.20
        
    def get_chain(self, symbol: str, timestamp: datetime) -> List[OptionContract]:
        """Generate synthetic options chain centered on historical spot."""
        
        # 1. Get Spot Price
        # Try to get bar for the specific minute/hour
        bars = self.market_adapter.get_bars(symbol, timestamp, timestamp, "1Min")
        if not bars:
            # Fallback to daily if minute missing
            bars = self.market_adapter.get_bars(symbol, timestamp, timestamp, "1Day")
            
        if not bars:
            # Last ditch: try finding *any* recent bar before timestamp
            # This is slow, so maybe just return empty if strict alignment required
            return []
            
        spot = bars[0].close
        
        # 2. Generate Chain
        contracts = []
        
        # Expirations: Weekly + Monthly ish
        # For simplicity in backtest: 7 days, 30 days
        expirations = [
            timestamp + timedelta(days=7),
            timestamp + timedelta(days=30)
        ]
        
        for expiry in expirations:
            dte_years = (expiry - timestamp).total_seconds() / (365 * 24 * 3600)
            if dte_years <= 0: continue
            
            # Strikes: +/- 5% in 1% steps
            # 1% of spot roughly
            strike_step = max(1.0, round(spot * 0.01)) 
            
            # Range: -5 to +5 steps
            center_strike = round(spot / strike_step) * strike_step
            
            for i in range(-5, 6):
                strike = center_strike + (i * strike_step)
                
                # Pricing
                for opt_type in ["call", "put"]:
                    # Price
                    price = black_scholes(spot, strike, dte_years, self.risk_free_rate, self.base_volatility, opt_type)
                    
                    # Delta
                    delta = calculate_delta(spot, strike, dte_years, self.risk_free_rate, self.base_volatility, opt_type)
                    
                    # Gamma (Same for call/put)
                    # Gamma = N'(d1) / (S * sigma * sqrt(T))
                    # Simplified approx or strict BS
                    # BS Gamma:
                    d1 = (private_log(spot/strike) + (self.risk_free_rate + 0.5*self.base_volatility**2)*dte_years) / (self.base_volatility * (dte_years**0.5))
                    gamma = private_norm_pdf(d1) / (spot * self.base_volatility * (dte_years**0.5))
                    
                    # Theta (Approx) - Option loses value as T decreases
                    # Simplified: - Price * 0.05 / 365
                    theta = -price * 0.1 * dte_years 
                    
                    # Vanna/Charm placeholders or full calc
                    # For GMM Hedge engine, Gamma/Delta/Vega matter most
                    vega = spot * (dte_years**0.5) * private_norm_pdf(d1) * 0.01 # 1% vol change impact
                    
                    # Spread noise
                    spread = max(0.01, price * 0.02) # 2% spread
                    
                    contract = OptionContract(
                        symbol=f"{symbol}_{expiry.strftime('%Y%m%d')}_{opt_type[0].upper()}_{strike}",
                        strike=strike,
                        expiration=expiry,
                        option_type=opt_type,
                        bid=max(0.0, price - spread/2),
                        ask=price + spread/2,
                        last=price,
                        volume=random.randint(100, 1000),
                        open_interest=random.randint(1000, 50000),
                        implied_volatility=self.base_volatility,
                        delta=delta,
                        gamma=gamma,
                        theta=theta,
                        vega=vega,
                        rho=0.0
                    )
                    contracts.append(contract)
                    
        return contracts

    def get_flow_summary(self, symbol: str) -> Dict[str, float]:
        # Synthetic neutral flow
        return {
            "call_volume": 5000,
            "put_volume": 5000,
            "call_premium": 100000,
            "put_premium": 100000,
            "sweep_ratio": 0.0
        }

# Helpers
import math
def private_log(x): return math.log(x)
def private_norm_pdf(x): return math.exp(-x**2/2) / math.sqrt(2*math.pi)
