"""Utility modules for Super Gnosis.

This package provides common utilities including:
- TTL caching for adapters
- Rate limiting for API calls
- Circuit breaker pattern for resilience
- Regime detection utilities
"""

from utils.cache import (
    TTLCache,
    RateLimiter,
    cached,
    rate_limited,
    get_options_cache,
    get_market_data_cache,
    get_quote_cache,
    get_alpaca_rate_limiter,
    get_unusual_whales_rate_limiter,
)

from utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
    CircuitStats,
    get_alpaca_circuit_breaker,
    get_unusual_whales_circuit_breaker,
)

__all__ = [
    # Cache
    "TTLCache",
    "cached",
    "get_options_cache",
    "get_market_data_cache",
    "get_quote_cache",
    # Rate limiting
    "RateLimiter",
    "rate_limited",
    "get_alpaca_rate_limiter",
    "get_unusual_whales_rate_limiter",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerOpen",
    "CircuitState",
    "CircuitStats",
    "get_alpaca_circuit_breaker",
    "get_unusual_whales_circuit_breaker",
]
