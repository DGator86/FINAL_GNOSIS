"""Caching utilities for adapters and data providers."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

from loguru import logger

T = TypeVar("T")


@dataclass
class CacheEntry:
    """Single cache entry with value and metadata."""
    
    value: Any
    timestamp: float
    ttl: float
    hits: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.timestamp > self.ttl
    
    def touch(self) -> None:
        """Record a cache hit."""
        self.hits += 1


class TTLCache:
    """Thread-safe TTL cache for adapter data.
    
    Features:
    - Configurable TTL per entry or global default
    - Thread-safe operations
    - Automatic cleanup of expired entries
    - Hit/miss statistics
    """
    
    def __init__(self, default_ttl: float = 60.0, max_size: int = 1000):
        """Initialize cache.
        
        Args:
            default_ttl: Default time-to-live in seconds
            max_size: Maximum number of entries before cleanup
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if present and not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None
            
            entry.touch()
            self._hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override in seconds
        """
        with self._lock:
            # Cleanup if at capacity
            if len(self._cache) >= self.max_size:
                self._cleanup()
            
            self._cache[key] = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl,
            )
    
    def invalidate(self, key: str) -> bool:
        """Remove entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was removed
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def invalidate_prefix(self, prefix: str) -> int:
        """Remove all entries with matching key prefix.
        
        Args:
            prefix: Key prefix to match
            
        Returns:
            Number of entries removed
        """
        with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(prefix)]
            for key in keys_to_remove:
                del self._cache[key]
            return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def _cleanup(self) -> None:
        """Remove expired entries and oldest entries if still over capacity."""
        # Remove expired entries
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired:
            del self._cache[key]
        
        # If still over capacity, remove oldest entries
        if len(self._cache) >= self.max_size:
            sorted_entries = sorted(
                self._cache.items(), 
                key=lambda x: x[1].timestamp
            )
            to_remove = len(self._cache) - self.max_size // 2
            for key, _ in sorted_entries[:to_remove]:
                del self._cache[key]
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "max_size": self.max_size,
                "default_ttl": self.default_ttl,
            }


# Global adapter cache instances
_options_cache = TTLCache(default_ttl=60.0, max_size=500)
_market_data_cache = TTLCache(default_ttl=30.0, max_size=1000)
_quote_cache = TTLCache(default_ttl=5.0, max_size=200)


def get_options_cache() -> TTLCache:
    """Get global options chain cache."""
    return _options_cache


def get_market_data_cache() -> TTLCache:
    """Get global market data cache."""
    return _market_data_cache


def get_quote_cache() -> TTLCache:
    """Get global quote cache (short TTL)."""
    return _quote_cache


def cached(
    cache: TTLCache,
    key_fn: Optional[Callable[..., str]] = None,
    ttl: Optional[float] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to cache function results.
    
    Args:
        cache: TTLCache instance to use
        key_fn: Function to generate cache key from args/kwargs
        ttl: Optional TTL override
        
    Returns:
        Decorated function with caching
        
    Example:
        @cached(get_options_cache(), key_fn=lambda symbol, ts: f"chain:{symbol}")
        def get_chain(symbol: str, timestamp: datetime) -> List[Contract]:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                # Default key from function name and args
                key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Check cache
            cached_value = cache.get(key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {key}")
                return cached_value
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            logger.debug(f"Cache miss for {key}, stored result")
            return result
        
        # Attach cache invalidation helpers
        wrapper.cache = cache
        wrapper.invalidate = lambda *args, **kwargs: cache.invalidate(
            key_fn(*args, **kwargs) if key_fn else f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
        )
        
        return wrapper
    return decorator


class RateLimiter:
    """Simple token bucket rate limiter.
    
    Thread-safe rate limiting for API calls.
    """
    
    def __init__(self, calls_per_second: float = 10.0, burst_size: int = 20):
        """Initialize rate limiter.
        
        Args:
            calls_per_second: Sustained rate limit
            burst_size: Maximum burst capacity
        """
        self.rate = calls_per_second
        self.burst_size = burst_size
        self._tokens = float(burst_size)
        self._last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self, timeout: float = 10.0) -> bool:
        """Acquire a token, waiting if necessary.
        
        Args:
            timeout: Maximum time to wait for a token
            
        Returns:
            True if token acquired, False if timeout
        """
        deadline = time.time() + timeout
        
        while True:
            with self._lock:
                now = time.time()
                # Refill tokens based on time elapsed
                elapsed = now - self._last_update
                self._tokens = min(
                    self.burst_size, 
                    self._tokens + elapsed * self.rate
                )
                self._last_update = now
                
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            
            if time.time() >= deadline:
                return False
            
            # Wait a bit before retrying
            time.sleep(0.05)
    
    def __enter__(self):
        """Context manager entry - acquire token."""
        if not self.acquire():
            raise TimeoutError("Rate limit timeout")
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        pass


def rate_limited(
    limiter: RateLimiter, 
    timeout: float = 10.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to rate limit function calls.
    
    Args:
        limiter: RateLimiter instance
        timeout: Maximum wait time for rate limit
        
    Returns:
        Decorated function with rate limiting
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not limiter.acquire(timeout):
                raise TimeoutError(f"Rate limit timeout for {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global rate limiters for different APIs
_alpaca_limiter = RateLimiter(calls_per_second=200.0, burst_size=200)
_unusual_whales_limiter = RateLimiter(calls_per_second=5.0, burst_size=10)


def get_alpaca_rate_limiter() -> RateLimiter:
    """Get Alpaca API rate limiter."""
    return _alpaca_limiter


def get_unusual_whales_rate_limiter() -> RateLimiter:
    """Get Unusual Whales API rate limiter."""
    return _unusual_whales_limiter
