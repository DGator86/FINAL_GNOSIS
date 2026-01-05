"""
Redis Caching Layer for Trading System

Provides high-performance caching for:
- Market data (quotes, bars, chains)
- Greeks calculations
- Portfolio state
- API responses
- Session data

Features:
- Async Redis client
- Automatic serialization/deserialization
- TTL-based expiration
- Key namespacing
- Cache invalidation patterns
- Fallback to in-memory cache when Redis unavailable

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, Generic

from loguru import logger


# =============================================================================
# Cache Configuration
# =============================================================================

class CacheNamespace(str, Enum):
    """Cache key namespaces."""
    MARKET_DATA = "market"
    GREEKS = "greeks"
    PORTFOLIO = "portfolio"
    OPTIONS_CHAIN = "chain"
    QUOTES = "quotes"
    BARS = "bars"
    SIGNALS = "signals"
    API = "api"
    SESSION = "session"
    ANALYTICS = "analytics"


def _get_redis_url() -> str:
    """Get Redis URL from environment with fallback."""
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


def _get_redis_password() -> Optional[str]:
    """Get Redis password from environment."""
    return os.getenv("REDIS_PASSWORD", None)


@dataclass
class CacheConfig:
    """Cache configuration."""
    # Redis connection - reads from environment by default
    redis_url: str = None  # Will be set in __post_init__
    redis_password: Optional[str] = None  # Will be set in __post_init__
    redis_db: int = 0

    def __post_init__(self):
        """Set defaults from environment after initialization."""
        if self.redis_url is None:
            self.redis_url = _get_redis_url()
        if self.redis_password is None:
            self.redis_password = _get_redis_password()
    
    # Connection settings
    max_connections: int = 10
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    
    # Default TTLs (in seconds)
    default_ttl: int = 300  # 5 minutes
    quote_ttl: int = 5      # 5 seconds for real-time quotes
    bar_ttl: int = 60       # 1 minute for bars
    chain_ttl: int = 30     # 30 seconds for options chains
    greeks_ttl: int = 10    # 10 seconds for Greeks
    portfolio_ttl: int = 5  # 5 seconds for portfolio
    api_ttl: int = 60       # 1 minute for API responses
    
    # Memory cache settings (fallback)
    memory_cache_size: int = 10000
    memory_cache_ttl: int = 300
    
    # Key prefix
    key_prefix: str = "gnosis"


# =============================================================================
# Abstract Cache Interface
# =============================================================================

T = TypeVar('T')


class CacheBackend(ABC):
    """Abstract cache backend interface."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in namespace."""
        pass
    
    @abstractmethod
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values."""
        pass
    
    @abstractmethod
    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values."""
        pass


# =============================================================================
# In-Memory Cache Backend (Fallback)
# =============================================================================

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    expires_at: float
    created_at: float = field(default_factory=time.time)


class MemoryCacheBackend(CacheBackend):
    """
    In-memory LRU cache backend.
    
    Used as fallback when Redis is unavailable.
    """
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 300):
        """Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        
        logger.info(f"MemoryCacheBackend initialized (max_size={max_size})")
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired."""
        return time.time() > entry.expires_at
    
    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired = [k for k, v in self._cache.items() if now > v.expires_at]
        for key in expired:
            del self._cache[key]
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            if self._is_expired(entry):
                del self._cache[key]
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        ttl = ttl or self.default_ttl
        
        with self._lock:
            self._evict_if_needed()
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=time.time() + ttl,
            )
            self._cache.move_to_end(key)
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
        return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if self._is_expired(entry):
                del self._cache[key]
                return False
            return True
    
    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in namespace."""
        with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(f"{namespace}:")]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values."""
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values."""
        for key, value in mapping.items():
            await self.set(key, value, ttl)
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            self._cleanup_expired()
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "utilization": len(self._cache) / self.max_size,
            }


# =============================================================================
# Redis Cache Backend
# =============================================================================

class RedisCacheBackend(CacheBackend):
    """
    Redis cache backend with async support.
    
    Uses redis-py with asyncio for non-blocking operations.
    """
    
    def __init__(self, config: CacheConfig):
        """Initialize Redis backend.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self._redis = None
        self._connected = False
        
        logger.info(f"RedisCacheBackend initialized (url={config.redis_url})")
    
    async def connect(self) -> bool:
        """Connect to Redis server."""
        try:
            import redis.asyncio as redis
            
            self._redis = redis.from_url(
                self.config.redis_url,
                password=self.config.redis_password,
                db=self.config.redis_db,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                max_connections=self.config.max_connections,
            )
            
            # Test connection
            await self._redis.ping()
            self._connected = True
            
            logger.info("Redis connection established")
            return True
            
        except ImportError:
            logger.warning("redis package not installed, using memory cache")
            return False
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._connected = False
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.config.key_prefix}:{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            return json.dumps(value).encode('utf-8')
        except (TypeError, ValueError):
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return pickle.loads(data)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._connected:
            return None
        
        try:
            data = await self._redis.get(self._make_key(key))
            if data is None:
                return None
            return self._deserialize(data)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        if not self._connected:
            return False
        
        try:
            ttl = ttl or self.config.default_ttl
            data = self._serialize(value)
            await self._redis.setex(self._make_key(key), ttl, data)
            return True
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self._connected:
            return False
        
        try:
            result = await self._redis.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self._connected:
            return False
        
        try:
            return await self._redis.exists(self._make_key(key)) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error: {e}")
            return False
    
    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in namespace."""
        if not self._connected:
            return 0
        
        try:
            pattern = f"{self.config.key_prefix}:{namespace}:*"
            cursor = 0
            deleted = 0
            
            while True:
                cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted += await self._redis.delete(*keys)
                if cursor == 0:
                    break
            
            return deleted
        except Exception as e:
            logger.error(f"Redis CLEAR error: {e}")
            return 0
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values."""
        if not self._connected or not keys:
            return {}
        
        try:
            prefixed_keys = [self._make_key(k) for k in keys]
            values = await self._redis.mget(prefixed_keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize(value)
            
            return result
        except Exception as e:
            logger.error(f"Redis MGET error: {e}")
            return {}
    
    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values."""
        if not self._connected or not mapping:
            return False
        
        try:
            ttl = ttl or self.config.default_ttl
            
            async with self._redis.pipeline() as pipe:
                for key, value in mapping.items():
                    data = self._serialize(value)
                    pipe.setex(self._make_key(key), ttl, data)
                await pipe.execute()
            
            return True
        except Exception as e:
            logger.error(f"Redis MSET error: {e}")
            return False


# =============================================================================
# Unified Cache Manager
# =============================================================================

class CacheManager:
    """
    Unified cache manager with Redis + memory fallback.
    
    Features:
    - Automatic fallback to memory cache
    - Namespace-based key organization
    - TTL management per data type
    - Cache decorators for functions
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache manager.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        
        # Initialize backends
        self._redis = RedisCacheBackend(self.config)
        self._memory = MemoryCacheBackend(
            max_size=self.config.memory_cache_size,
            default_ttl=self.config.memory_cache_ttl,
        )
        
        self._use_redis = False
        
        logger.info("CacheManager initialized")
    
    async def initialize(self) -> bool:
        """Initialize cache connections.
        
        Returns:
            True if Redis connected, False if using memory fallback
        """
        self._use_redis = await self._redis.connect()
        
        if not self._use_redis:
            logger.warning("Using in-memory cache (Redis unavailable)")
        
        return self._use_redis
    
    async def close(self) -> None:
        """Close cache connections."""
        if self._use_redis:
            await self._redis.disconnect()
    
    @property
    def backend(self) -> CacheBackend:
        """Get active cache backend."""
        return self._redis if self._use_redis else self._memory
    
    def _build_key(self, namespace: CacheNamespace, *parts: str) -> str:
        """Build cache key with namespace.
        
        Args:
            namespace: Key namespace
            parts: Key parts to join
            
        Returns:
            Full cache key
        """
        return f"{namespace.value}:{':'.join(parts)}"
    
    def _get_ttl(self, namespace: CacheNamespace) -> int:
        """Get TTL for namespace.
        
        Args:
            namespace: Key namespace
            
        Returns:
            TTL in seconds
        """
        ttl_map = {
            CacheNamespace.QUOTES: self.config.quote_ttl,
            CacheNamespace.BARS: self.config.bar_ttl,
            CacheNamespace.OPTIONS_CHAIN: self.config.chain_ttl,
            CacheNamespace.GREEKS: self.config.greeks_ttl,
            CacheNamespace.PORTFOLIO: self.config.portfolio_ttl,
            CacheNamespace.API: self.config.api_ttl,
        }
        return ttl_map.get(namespace, self.config.default_ttl)
    
    # =========================================================================
    # Generic Cache Operations
    # =========================================================================
    
    async def get(
        self,
        namespace: CacheNamespace,
        *key_parts: str,
    ) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            namespace: Key namespace
            key_parts: Parts of the key
            
        Returns:
            Cached value or None
        """
        key = self._build_key(namespace, *key_parts)
        return await self.backend.get(key)
    
    async def set(
        self,
        namespace: CacheNamespace,
        *key_parts: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache.
        
        Args:
            namespace: Key namespace
            key_parts: Parts of the key
            value: Value to cache
            ttl: Optional TTL override
            
        Returns:
            True if successful
        """
        key = self._build_key(namespace, *key_parts)
        ttl = ttl or self._get_ttl(namespace)
        return await self.backend.set(key, value, ttl)
    
    async def delete(
        self,
        namespace: CacheNamespace,
        *key_parts: str,
    ) -> bool:
        """Delete from cache.
        
        Args:
            namespace: Key namespace
            key_parts: Parts of the key
            
        Returns:
            True if deleted
        """
        key = self._build_key(namespace, *key_parts)
        return await self.backend.delete(key)
    
    async def clear_namespace(self, namespace: CacheNamespace) -> int:
        """Clear all keys in a namespace.
        
        Args:
            namespace: Namespace to clear
            
        Returns:
            Number of keys deleted
        """
        return await self.backend.clear_namespace(namespace.value)
    
    # =========================================================================
    # Specialized Cache Methods
    # =========================================================================
    
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached quote for symbol."""
        return await self.get(CacheNamespace.QUOTES, symbol)
    
    async def set_quote(self, symbol: str, quote: Dict[str, Any]) -> bool:
        """Cache quote for symbol."""
        return await self.set(CacheNamespace.QUOTES, symbol, value=quote)
    
    async def get_greeks(self, option_symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached Greeks for option."""
        return await self.get(CacheNamespace.GREEKS, option_symbol)
    
    async def set_greeks(self, option_symbol: str, greeks: Dict[str, Any]) -> bool:
        """Cache Greeks for option."""
        return await self.set(CacheNamespace.GREEKS, option_symbol, value=greeks)
    
    async def get_options_chain(
        self,
        symbol: str,
        expiration: str,
    ) -> Optional[Dict[str, Any]]:
        """Get cached options chain."""
        return await self.get(CacheNamespace.OPTIONS_CHAIN, symbol, expiration)
    
    async def set_options_chain(
        self,
        symbol: str,
        expiration: str,
        chain: Dict[str, Any],
    ) -> bool:
        """Cache options chain."""
        return await self.set(
            CacheNamespace.OPTIONS_CHAIN,
            symbol,
            expiration,
            value=chain,
        )
    
    async def get_portfolio(self) -> Optional[Dict[str, Any]]:
        """Get cached portfolio state."""
        return await self.get(CacheNamespace.PORTFOLIO, "current")
    
    async def set_portfolio(self, portfolio: Dict[str, Any]) -> bool:
        """Cache portfolio state."""
        return await self.set(CacheNamespace.PORTFOLIO, "current", value=portfolio)
    
    # =========================================================================
    # Cache Decorator
    # =========================================================================
    
    def cached(
        self,
        namespace: CacheNamespace,
        ttl: Optional[int] = None,
        key_builder: Optional[Callable[..., str]] = None,
    ):
        """Decorator to cache function results.
        
        Args:
            namespace: Cache namespace
            ttl: Optional TTL override
            key_builder: Optional function to build cache key from args
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Build cache key
                if key_builder:
                    key_suffix = key_builder(*args, **kwargs)
                else:
                    # Default: hash of args
                    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
                    key_suffix = hashlib.md5(key_data.encode()).hexdigest()[:16]
                
                key = self._build_key(namespace, func.__name__, key_suffix)
                
                # Try cache
                cached_value = await self.backend.get(key)
                if cached_value is not None:
                    return cached_value
                
                # Call function
                result = await func(*args, **kwargs)
                
                # Cache result
                cache_ttl = ttl or self._get_ttl(namespace)
                await self.backend.set(key, result, cache_ttl)
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, run in event loop
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "backend": "redis" if self._use_redis else "memory",
            "config": {
                "redis_url": self.config.redis_url,
                "default_ttl": self.config.default_ttl,
            },
        }
        
        if not self._use_redis:
            stats["memory_stats"] = self._memory.get_stats()
        
        return stats


# =============================================================================
# Global Cache Instance
# =============================================================================

cache = CacheManager()


# =============================================================================
# Initialization Helper
# =============================================================================

async def initialize_cache(config: Optional[CacheConfig] = None) -> CacheManager:
    """Initialize the global cache manager.
    
    Args:
        config: Optional cache configuration
        
    Returns:
        Initialized cache manager
    """
    global cache
    
    if config:
        cache = CacheManager(config)
    
    await cache.initialize()
    return cache


async def close_cache() -> None:
    """Close the global cache manager."""
    global cache
    await cache.close()
