"""Adapter factory with automatic fallback to stubs.

Features:
- Automatic fallback from real APIs to stubs
- TTL caching for adapter instances
- Rate limiting support
- Provider selection via environment variables
"""

from __future__ import annotations

import os
from typing import Any, Optional

from loguru import logger

from utils.cache import (
    TTLCache,
    get_options_cache,
    get_market_data_cache,
    get_alpaca_rate_limiter,
    get_unusual_whales_rate_limiter,
)

# Cache for adapter instances (long TTL - adapters are reusable)
_adapter_cache = TTLCache(default_ttl=3600.0, max_size=50)


def create_market_data_adapter(prefer_real: bool = True, provider: Optional[str] = None) -> Any:
    """
    Create market data adapter with fallback.

    Args:
        prefer_real: Try real adapter first
        provider: Preferred provider ("massive", "alpaca", or None for auto)

    Returns:
        Market data adapter instance
    """
    # Check for preferred provider from environment
    if provider is None:
        provider = os.getenv("MARKET_DATA_PROVIDER", "auto").lower()

    if prefer_real:
        # Try MASSIVE first if enabled or preferred
        if provider in ("massive", "auto") and os.getenv("MASSIVE_API_ENABLED", "false").lower() == "true":
            try:
                from engines.inputs.massive_market_adapter import MassiveMarketDataAdapter
                adapter = MassiveMarketDataAdapter()
                logger.info("Using MassiveMarketDataAdapter (real)")
                return adapter
            except Exception as e:
                logger.warning(f"Failed to initialize MASSIVE market data adapter: {e}")
                if provider == "massive":
                    logger.info("Falling back to Alpaca adapter")

        # Try Alpaca
        if provider in ("alpaca", "auto"):
            try:
                from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
                adapter = AlpacaMarketDataAdapter()
                logger.info("Using AlpacaMarketDataAdapter (real)")
                return adapter
            except Exception as e:
                logger.warning(f"Failed to initialize Alpaca market data adapter: {e}")
                logger.info("Falling back to StaticMarketDataAdapter")

    from engines.inputs.stub_adapters import StaticMarketDataAdapter
    logger.info("Using StaticMarketDataAdapter (stub)")
    return StaticMarketDataAdapter()


def create_options_adapter(prefer_real: bool = True, provider: Optional[str] = None) -> Any:
    """
    Create options chain adapter with fallback.

    Args:
        prefer_real: Try real adapter first
        provider: Preferred provider ("massive", "unusual_whales", or None for auto)

    Returns:
        Options chain adapter instance
    """
    # Check for preferred provider from environment
    if provider is None:
        provider = os.getenv("OPTIONS_DATA_PROVIDER", "auto").lower()

    if prefer_real:
        # Try MASSIVE first if enabled or preferred (comprehensive options data)
        if provider in ("massive", "auto") and os.getenv("MASSIVE_API_ENABLED", "false").lower() == "true":
            try:
                from engines.inputs.massive_options_adapter import MassiveOptionsAdapter
                adapter = MassiveOptionsAdapter()
                logger.info("Using MassiveOptionsAdapter (real)")
                return adapter
            except Exception as e:
                logger.warning(f"Failed to initialize MASSIVE options adapter: {e}")
                if provider == "massive":
                    logger.info("Falling back to Unusual Whales adapter")

        # Try Unusual Whales
        if provider in ("unusual_whales", "auto"):
            try:
                from engines.inputs.unusual_whales_adapter import UnusualWhalesAdapter
                adapter = UnusualWhalesAdapter()
                logger.info("Using UnusualWhalesAdapter (real)")
                return adapter
            except Exception as e:
                logger.warning(f"Failed to initialize Unusual Whales adapter: {e}")
                logger.info("Falling back to StaticOptionsAdapter")

    from engines.inputs.stub_adapters import StaticOptionsAdapter
    logger.info("Using StaticOptionsAdapter (stub)")
    return StaticOptionsAdapter()


def create_massive_options_adapter() -> Any:
    """
    Create MASSIVE options adapter for comprehensive historical options data.

    Provides multi-timeframe options data for ML training.

    Returns:
        MassiveOptionsAdapter instance or None if not configured
    """
    if os.getenv("MASSIVE_API_ENABLED", "false").lower() != "true":
        logger.warning("MASSIVE API not enabled. Set MASSIVE_API_ENABLED=true in .env")
        return None

    try:
        from engines.inputs.massive_options_adapter import MassiveOptionsAdapter
        adapter = MassiveOptionsAdapter()
        logger.info("MassiveOptionsAdapter created successfully")
        return adapter
    except Exception as e:
        logger.error(f"Failed to create MASSIVE options adapter: {e}")
        return None


def create_news_adapter(prefer_real: bool = True) -> Any:
    """
    Create news adapter with fallback.

    Args:
        prefer_real: Try real adapter first

    Returns:
        News adapter instance
    """
    if prefer_real and os.getenv("MASSIVE_API_ENABLED", "false").lower() == "true":
        try:
            from engines.inputs.massive_market_adapter import MassiveMarketDataAdapter
            adapter = MassiveMarketDataAdapter()
            logger.info("Using MassiveMarketDataAdapter for news (real)")
            return adapter
        except Exception as e:
            logger.warning(f"Failed to initialize MASSIVE adapter for news: {e}")

    from engines.inputs.stub_adapters import StaticNewsAdapter
    logger.info("Using StaticNewsAdapter (stub)")
    return StaticNewsAdapter()


def create_massive_adapter() -> Any:
    """
    Create MASSIVE market data adapter for comprehensive data access.

    Returns:
        MassiveMarketDataAdapter instance or None if not configured
    """
    if os.getenv("MASSIVE_API_ENABLED", "false").lower() != "true":
        logger.warning("MASSIVE API not enabled. Set MASSIVE_API_ENABLED=true in .env")
        return None

    try:
        from engines.inputs.massive_market_adapter import MassiveMarketDataAdapter
        adapter = MassiveMarketDataAdapter()
        logger.info("MassiveMarketDataAdapter created successfully")
        return adapter
    except Exception as e:
        logger.error(f"Failed to create MASSIVE adapter: {e}")
        return None


def create_broker_adapter(paper: Optional[bool] = None, prefer_real: bool = True) -> Any:
    """
    Create broker adapter with fallback.
    
    Args:
        paper: Use paper trading
        prefer_real: Try real adapter first
        
    Returns:
        Broker adapter instance
    """
    if paper is None:
        from execution.broker_adapters.settings import get_alpaca_paper_setting

        paper = get_alpaca_paper_setting()

    if prefer_real:
        try:
            from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
            adapter = AlpacaBrokerAdapter(paper=paper)
            logger.info(f"Using AlpacaBrokerAdapter (real, paper={paper})")
            return adapter
        except Exception as e:
            logger.warning(f"Failed to initialize Alpaca broker adapter: {e}")
            logger.info("No fallback broker available")
            return None
    
    logger.info("Real broker not requested")
    return None


class CachedOptionsAdapter:
    """Wrapper that adds TTL caching to any options adapter.
    
    Reduces API calls by caching options chain data for a configurable TTL.
    """
    
    def __init__(self, adapter: Any, ttl: float = 60.0):
        """Initialize cached adapter.
        
        Args:
            adapter: Underlying options adapter
            ttl: Cache TTL in seconds (default 60s)
        """
        self._adapter = adapter
        self._cache = get_options_cache()
        self._ttl = ttl
        logger.info(f"CachedOptionsAdapter wrapping {type(adapter).__name__} (TTL={ttl}s)")
    
    def get_chain(self, symbol: str, timestamp: Any = None) -> Any:
        """Get options chain with caching.
        
        Args:
            symbol: Ticker symbol
            timestamp: Optional timestamp (not used for caching key)
            
        Returns:
            Options chain data
        """
        cache_key = f"chain:{symbol}"
        
        # Check cache
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Options cache hit for {symbol}")
            return cached
        
        # Fetch from underlying adapter
        result = self._adapter.get_chain(symbol, timestamp)
        
        # Cache result
        if result:
            self._cache.set(cache_key, result, self._ttl)
            logger.debug(f"Cached options chain for {symbol}")
        
        return result
    
    def __getattr__(self, name: str) -> Any:
        """Delegate other methods to underlying adapter."""
        return getattr(self._adapter, name)
    
    @property
    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return self._cache.stats


class CachedMarketDataAdapter:
    """Wrapper that adds TTL caching to any market data adapter.
    
    Reduces API calls by caching market data for a configurable TTL.
    """
    
    def __init__(self, adapter: Any, bars_ttl: float = 30.0, quote_ttl: float = 5.0):
        """Initialize cached adapter.
        
        Args:
            adapter: Underlying market data adapter
            bars_ttl: Cache TTL for bars data (default 30s)
            quote_ttl: Cache TTL for quotes (default 5s)
        """
        self._adapter = adapter
        self._cache = get_market_data_cache()
        self._bars_ttl = bars_ttl
        self._quote_ttl = quote_ttl
        logger.info(f"CachedMarketDataAdapter wrapping {type(adapter).__name__}")
    
    def get_bars(self, symbol: str, start: Any = None, end: Any = None, timeframe: str = "1Day") -> Any:
        """Get bars data with caching.
        
        Args:
            symbol: Ticker symbol
            start: Start timestamp
            end: End timestamp
            timeframe: Bar timeframe
            
        Returns:
            Bars data
        """
        # Create cache key from parameters
        cache_key = f"bars:{symbol}:{timeframe}:{start}:{end}"
        
        # Check cache
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Market data cache hit for {symbol}")
            return cached
        
        # Fetch from underlying adapter
        result = self._adapter.get_bars(symbol, start, end, timeframe)
        
        # Cache result
        if result:
            self._cache.set(cache_key, result, self._bars_ttl)
        
        return result
    
    def get_latest_quote(self, symbol: str) -> Any:
        """Get latest quote with short-TTL caching.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Quote data
        """
        cache_key = f"quote:{symbol}"
        
        # Check cache
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Fetch from underlying adapter
        result = self._adapter.get_latest_quote(symbol)
        
        # Cache with short TTL
        if result:
            self._cache.set(cache_key, result, self._quote_ttl)
        
        return result
    
    def __getattr__(self, name: str) -> Any:
        """Delegate other methods to underlying adapter."""
        return getattr(self._adapter, name)
    
    @property
    def cache_stats(self) -> dict:
        """Get cache statistics."""
        return self._cache.stats


def create_cached_options_adapter(prefer_real: bool = True, ttl: float = 60.0) -> Any:
    """Create options adapter with TTL caching.
    
    Args:
        prefer_real: Try real adapter first
        ttl: Cache TTL in seconds
        
    Returns:
        Cached options adapter
    """
    adapter = create_options_adapter(prefer_real=prefer_real)
    return CachedOptionsAdapter(adapter, ttl=ttl)


def create_cached_market_data_adapter(prefer_real: bool = True, bars_ttl: float = 30.0) -> Any:
    """Create market data adapter with TTL caching.
    
    Args:
        prefer_real: Try real adapter first
        bars_ttl: Cache TTL for bars data in seconds
        
    Returns:
        Cached market data adapter
    """
    adapter = create_market_data_adapter(prefer_real=prefer_real)
    return CachedMarketDataAdapter(adapter, bars_ttl=bars_ttl)
