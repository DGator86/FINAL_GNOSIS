import math

SQRT_2_PI = math.sqrt(2 * math.pi)
SQRT_2 = math.sqrt(2)

def normal_pdf(x: float, mean: float, var: float) -> float:
    """Calculate the probability density function of a normal distribution."""
    if var <= 1e-18:
        return 0.0  # Treat as 0 probability if variance is effectively zero and we can't evaluate
    sigma = math.sqrt(var)
    denom = sigma * SQRT_2_PI
    exponent = -0.5 * ((x - mean) / sigma) ** 2
    return math.exp(exponent) / denom

def normal_cdf(x: float) -> float:
    """Standard normal CDF (mean=0, std=1)."""
    return 0.5 * (1 + math.erf(x / SQRT_2))

def normal_cdf_general(x: float, mean: float, var: float) -> float:
    """Calculate the cumulative distribution function of a normal distribution."""
    if var <= 1e-18:
        return 0.0 if x < mean else 1.0
    sigma = math.sqrt(var)
    return normal_cdf((x - mean) / sigma)
