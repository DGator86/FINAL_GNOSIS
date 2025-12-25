import math
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    S: Spot price
    K: Strike price
    T: Time to maturity (years)
    r: Risk-free rate
    sigma: Volatility
    """
    if T <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
        
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
    return price

def calculate_delta(S, K, T, r, sigma, option_type="call"):
    if T <= 0: return 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0
