"""
Price Series Provider for ML Dataset Building.

Provides historical price data for:
- ML dataset label generation
- Backtesting
- Feature engineering

Supports multiple data sources with fallback:
1. Alpaca Historical API (primary for live trading)
2. yfinance (backup/free tier)
3. Local database/cache (fastest)

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from loguru import logger

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.debug("yfinance not available - using fallback data sources")


class PriceInterval(str, Enum):
    """Supported price intervals."""
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    WEEK_1 = "1wk"


@dataclass
class PriceBar:
    """Single price bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    
    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2
    
    @property
    def range(self) -> float:
        return self.high - self.low
    
    @property
    def body(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


@dataclass
class PriceSeries:
    """Collection of price bars."""
    symbol: str
    interval: PriceInterval
    bars: List[PriceBar]
    
    @property
    def closes(self) -> List[float]:
        return [bar.close for bar in self.bars]
    
    @property
    def opens(self) -> List[float]:
        return [bar.open for bar in self.bars]
    
    @property
    def highs(self) -> List[float]:
        return [bar.high for bar in self.bars]
    
    @property
    def lows(self) -> List[float]:
        return [bar.low for bar in self.bars]
    
    @property
    def volumes(self) -> List[int]:
        return [bar.volume for bar in self.bars]
    
    @property
    def timestamps(self) -> List[datetime]:
        return [bar.timestamp for bar in self.bars]
    
    def get_returns(self) -> List[float]:
        """Get percentage returns."""
        closes = self.closes
        if len(closes) < 2:
            return []
        return [(closes[i] / closes[i-1] - 1) * 100 for i in range(1, len(closes))]
    
    def slice(self, start: datetime, end: datetime) -> "PriceSeries":
        """Get a slice of the series."""
        filtered = [bar for bar in self.bars if start <= bar.timestamp <= end]
        return PriceSeries(
            symbol=self.symbol,
            interval=self.interval,
            bars=filtered,
        )


class PriceProviderBase(ABC):
    """Base class for price data providers."""
    
    @abstractmethod
    def get_price_series(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: PriceInterval = PriceInterval.DAY_1,
    ) -> Optional[PriceSeries]:
        """Get price series for a symbol."""
        pass
    
    @abstractmethod
    def get_closes(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: PriceInterval = PriceInterval.DAY_1,
    ) -> List[float]:
        """Get close prices for ML labeling."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available."""
        pass


class AlpacaPriceProvider(PriceProviderBase):
    """Price provider using Alpaca API."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        self._client = None
    
    def _get_client(self):
        """Lazy-load Alpaca client."""
        if self._client is None:
            try:
                from alpaca.data import StockHistoricalDataClient
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
                
                self._client = StockHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.api_secret,
                )
            except Exception as e:
                logger.debug(f"Could not create Alpaca client: {e}")
                self._client = None
        return self._client
    
    def _interval_to_timeframe(self, interval: PriceInterval):
        """Convert interval to Alpaca TimeFrame."""
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        
        mapping = {
            PriceInterval.MIN_1: TimeFrame(1, TimeFrameUnit.Minute),
            PriceInterval.MIN_5: TimeFrame(5, TimeFrameUnit.Minute),
            PriceInterval.MIN_15: TimeFrame(15, TimeFrameUnit.Minute),
            PriceInterval.MIN_30: TimeFrame(30, TimeFrameUnit.Minute),
            PriceInterval.HOUR_1: TimeFrame(1, TimeFrameUnit.Hour),
            PriceInterval.DAY_1: TimeFrame(1, TimeFrameUnit.Day),
            PriceInterval.WEEK_1: TimeFrame(1, TimeFrameUnit.Week),
        }
        return mapping.get(interval, TimeFrame(1, TimeFrameUnit.Day))
    
    def is_available(self) -> bool:
        client = self._get_client()
        return client is not None
    
    def get_price_series(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: PriceInterval = PriceInterval.DAY_1,
    ) -> Optional[PriceSeries]:
        client = self._get_client()
        if not client:
            return None
        
        try:
            from alpaca.data.requests import StockBarsRequest
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                start=start,
                end=end,
                timeframe=self._interval_to_timeframe(interval),
            )
            
            bars_response = client.get_stock_bars(request)
            
            if symbol not in bars_response:
                return None
            
            bars = []
            for bar in bars_response[symbol]:
                bars.append(PriceBar(
                    timestamp=bar.timestamp,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=int(bar.volume),
                ))
            
            return PriceSeries(
                symbol=symbol,
                interval=interval,
                bars=bars,
            )
            
        except Exception as e:
            logger.warning(f"Alpaca price fetch failed for {symbol}: {e}")
            return None
    
    def get_closes(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: PriceInterval = PriceInterval.DAY_1,
    ) -> List[float]:
        series = self.get_price_series(symbol, start, end, interval)
        return series.closes if series else []


class YFinancePriceProvider(PriceProviderBase):
    """Price provider using yfinance (free, no API key needed)."""
    
    def __init__(self):
        self._available = YFINANCE_AVAILABLE
    
    def _interval_to_yf(self, interval: PriceInterval) -> str:
        """Convert interval to yfinance format."""
        mapping = {
            PriceInterval.MIN_1: "1m",
            PriceInterval.MIN_5: "5m",
            PriceInterval.MIN_15: "15m",
            PriceInterval.MIN_30: "30m",
            PriceInterval.HOUR_1: "1h",
            PriceInterval.DAY_1: "1d",
            PriceInterval.WEEK_1: "1wk",
        }
        return mapping.get(interval, "1d")
    
    def is_available(self) -> bool:
        return self._available
    
    def get_price_series(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: PriceInterval = PriceInterval.DAY_1,
    ) -> Optional[PriceSeries]:
        if not self._available:
            return None
        
        try:
            ticker = yf.Ticker(symbol)
            
            # yfinance has limits on intraday data (7 days for 1m)
            yf_interval = self._interval_to_yf(interval)
            
            df = ticker.history(
                start=start,
                end=end,
                interval=yf_interval,
            )
            
            if df.empty:
                return None
            
            bars = []
            for idx, row in df.iterrows():
                # Handle timezone
                ts = idx
                if hasattr(ts, 'to_pydatetime'):
                    ts = ts.to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                
                bars.append(PriceBar(
                    timestamp=ts,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']),
                ))
            
            return PriceSeries(
                symbol=symbol,
                interval=interval,
                bars=bars,
            )
            
        except Exception as e:
            logger.warning(f"yfinance fetch failed for {symbol}: {e}")
            return None
    
    def get_closes(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: PriceInterval = PriceInterval.DAY_1,
    ) -> List[float]:
        series = self.get_price_series(symbol, start, end, interval)
        return series.closes if series else []


class CachedPriceProvider(PriceProviderBase):
    """Price provider with in-memory caching."""
    
    def __init__(self, providers: List[PriceProviderBase], cache_ttl_seconds: int = 300):
        self.providers = providers
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._cache: Dict[str, Tuple[datetime, PriceSeries]] = {}
    
    def _cache_key(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: PriceInterval,
    ) -> str:
        return f"{symbol}:{interval.value}:{start.isoformat()}:{end.isoformat()}"
    
    def is_available(self) -> bool:
        return any(p.is_available() for p in self.providers)
    
    def get_price_series(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: PriceInterval = PriceInterval.DAY_1,
    ) -> Optional[PriceSeries]:
        cache_key = self._cache_key(symbol, start, end, interval)
        now = datetime.now(timezone.utc)
        
        # Check cache
        if cache_key in self._cache:
            cached_time, cached_series = self._cache[cache_key]
            if now - cached_time < self.cache_ttl:
                return cached_series
        
        # Try providers in order
        for provider in self.providers:
            if not provider.is_available():
                continue
            
            series = provider.get_price_series(symbol, start, end, interval)
            if series and series.bars:
                self._cache[cache_key] = (now, series)
                return series
        
        return None
    
    def get_closes(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: PriceInterval = PriceInterval.DAY_1,
    ) -> List[float]:
        series = self.get_price_series(symbol, start, end, interval)
        return series.closes if series else []
    
    def clear_cache(self):
        """Clear the price cache."""
        self._cache.clear()


class UnifiedPriceProvider:
    """
    Unified price provider with automatic fallback.
    
    Tries providers in order:
    1. Alpaca (if API keys available)
    2. yfinance (free fallback)
    3. Returns empty if all fail
    """
    
    def __init__(
        self,
        alpaca_api_key: Optional[str] = None,
        alpaca_api_secret: Optional[str] = None,
        enable_cache: bool = True,
        cache_ttl_seconds: int = 300,
    ):
        providers: List[PriceProviderBase] = []
        
        # Add Alpaca if credentials available
        alpaca = AlpacaPriceProvider(alpaca_api_key, alpaca_api_secret)
        if alpaca.is_available():
            providers.append(alpaca)
            logger.info("Alpaca price provider enabled")
        
        # Add yfinance as fallback
        yf_provider = YFinancePriceProvider()
        if yf_provider.is_available():
            providers.append(yf_provider)
            logger.info("yfinance price provider enabled")
        
        if not providers:
            logger.warning("No price providers available")
        
        # Wrap with cache
        if enable_cache and providers:
            self.provider = CachedPriceProvider(providers, cache_ttl_seconds)
        elif providers:
            self.provider = providers[0]
        else:
            self.provider = None
    
    def get_price_series(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: PriceInterval = PriceInterval.DAY_1,
    ) -> Optional[PriceSeries]:
        """Get price series for a symbol."""
        if not self.provider:
            return None
        return self.provider.get_price_series(symbol, start, end, interval)
    
    def get_closes(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: PriceInterval = PriceInterval.DAY_1,
    ) -> List[float]:
        """Get close prices (for ML labeling)."""
        if not self.provider:
            return []
        return self.provider.get_closes(symbol, start, end, interval)
    
    def is_available(self) -> bool:
        """Check if any provider is available."""
        return self.provider is not None and self.provider.is_available()


# Global instance
_price_provider: Optional[UnifiedPriceProvider] = None


def get_price_provider() -> UnifiedPriceProvider:
    """Get or create the global price provider."""
    global _price_provider
    
    if _price_provider is None:
        _price_provider = UnifiedPriceProvider()
    
    return _price_provider


def get_price_series_for_ml(
    symbol: str,
    start: datetime,
    end: datetime,
) -> List[float]:
    """
    Get price series for ML dataset building.
    
    This is the function wired into routers/ml_trades.py
    
    Args:
        symbol: Stock symbol
        start: Start time
        end: End time
        
    Returns:
        List of close prices
    """
    provider = get_price_provider()
    return provider.get_closes(symbol, start, end)


__all__ = [
    "PriceInterval",
    "PriceBar",
    "PriceSeries",
    "PriceProviderBase",
    "AlpacaPriceProvider",
    "YFinancePriceProvider",
    "CachedPriceProvider",
    "UnifiedPriceProvider",
    "get_price_provider",
    "get_price_series_for_ml",
]
