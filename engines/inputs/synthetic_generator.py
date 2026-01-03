import random
import math
from datetime import datetime, timedelta
from engines.physics.internal_schemas import Bar1m

def generate_heston_series(symbol: str, start_dt: datetime, end_dt: datetime, timeframe_minutes: int = 60, start_price: float = 100.0, vol: float = 0.2):
    """
    Generate synthetic OHLCV bars using Heston Stochastic Volatility model.
    """
    minutes = int((end_dt - start_dt).total_seconds() / 60)
    steps = int(minutes / timeframe_minutes)
    if steps <= 0: return []
    
    dt = timeframe_minutes / (252 * 390) # Annualized time step
    
    data = []
    price = start_price
    variance = vol ** 2
    kappa = 2.0   # Mean reversion speed of vol
    theta = variance # Long term variance
    xi = 0.3      # Vol of vol
    
    # Correlation between price and vol (leverage effect)
    rho = -0.7
    
    curr_time = start_dt
    
    for _ in range(steps):
        # Correlated random norms
        z1 = random.gauss(0, 1)
        z2 = rho * z1 + math.sqrt(1 - rho**2) * random.gauss(0, 1)
        
        # Heston Vol Update (Euler-Maruyama)
        # dv = kappa(theta - v)dt + xi*sqrt(v)*dz2
        dw_var = z2 * math.sqrt(dt)
        variance = max(0.01**2, variance + kappa*(theta - variance)*dt + xi*math.sqrt(variance)*dw_var)
        sigma = math.sqrt(variance)
        
        # Price Update
        # dS = mu*S*dt + sqrt(v)*S*dz1
        # Drift slightly positive
        drift = 0.05 
        dw_price = z1 * math.sqrt(dt)
        
        ret = (drift - 0.5*variance)*dt + sigma*dw_price
        price *= math.exp(ret)
        
        # Bar construction (Micro-noise)
        high = price * (1 + 0.5*sigma*math.sqrt(dt))
        low = price * (1 - 0.5*sigma*math.sqrt(dt))
        open_p = (high + low)/2 
        
        # Market hours filtering (approximate)
        if 13 <= curr_time.hour < 20: # 9:30 AM - 4:00 PM ET approx in UTC (13:30-20:00)
            data.append(Bar1m(
                ts=curr_time,
                open=open_p, high=high, low=low, close=price,
                volume=int(1000000/price), # Constant dollar volume approx
                vwap=price
            ))
            
        curr_time += timedelta(minutes=timeframe_minutes)
        
    return data
