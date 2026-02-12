"""
Polygon.io Historical Options Data Adapter

Provides access to historical options data for backtesting including:
- Options chains with full Greeks
- Historical OHLCV for individual contracts
- Expired contracts support
- Aggregated data at various timeframes

Pricing Tiers:
- Options Basic (FREE): End-of-day + minute aggregates
- Options Starter ($29/mo): 15-min delayed + minute aggregates  
- Options Developer ($79/mo): 15-min delayed + tick data + trades
- Options Advanced ($199/mo): Real-time + full historical

API Docs: https://polygon.io/docs/options

Author: GNOSIS Trading System
Version: 1.0.0
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import httpx
import numpy as np
import pandas as pd
from loguru import logger

from engines.inputs.options_chain_adapter import OptionContract, OptionsChainAdapter


# ============================================================================
# CONFIGURATION
# ============================================================================

POLYGON_DEFAULT_API_KEY = os.getenv("POLYGON_API_KEY", "")

# Rate limiting
REQUESTS_PER_MINUTE = 5  # Free tier limit
REQUEST_DELAY = 12.0  # seconds between requests for free tier


@dataclass
class PolygonConfig:
    """Configuration for Polygon.io adapter."""
    
    api_key: str = ""
    base_url: str = "https://api.polygon.io"
    timeout: float = 30.0
    rate_limit_delay: float = REQUEST_DELAY
    max_retries: int = 3
    cache_enabled: bool = True
    
    @classmethod
    def from_env(cls, api_key: Optional[str] = None) -> "PolygonConfig":
        """Build configuration from environment."""
        key = api_key or os.getenv("POLYGON_API_KEY") or POLYGON_DEFAULT_API_KEY
        
        # Adjust rate limit based on tier (can be set via env)
        tier = os.getenv("POLYGON_TIER", "basic").lower()
        if tier == "starter":
            delay = 6.0  # ~10 req/min
        elif tier in ("developer", "advanced"):
            delay = 0.5  # ~120 req/min
        else:
            delay = REQUEST_DELAY  # Free tier
        
        return cls(
            api_key=key,
            rate_limit_delay=delay,
        )


class PolygonTimespan(str, Enum):
    """Timespan for aggregates."""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PolygonOptionContract:
    """Option contract from Polygon.io."""
    
    # Contract identification
    ticker: str  # OCC format: O:SPY251219C00600000
    underlying_ticker: str
    strike_price: float
    expiration_date: date
    contract_type: str  # "call" or "put"
    
    # Contract details
    shares_per_contract: int = 100
    exercise_style: str = "american"
    primary_exchange: str = ""
    
    # Greeks (if available)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    # Implied volatility
    implied_volatility: float = 0.0
    
    # Pricing snapshot
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    mid: float = 0.0
    
    # Volume/OI
    volume: int = 0
    open_interest: int = 0
    
    # Timestamp
    last_updated: Optional[datetime] = None


@dataclass
class PolygonOptionsChain:
    """Complete options chain from Polygon."""
    
    underlying: str
    as_of_date: date
    spot_price: float = 0.0
    
    contracts: List[PolygonOptionContract] = field(default_factory=list)
    calls: List[PolygonOptionContract] = field(default_factory=list)
    puts: List[PolygonOptionContract] = field(default_factory=list)
    
    # Aggregated metrics
    total_call_volume: int = 0
    total_put_volume: int = 0
    total_call_oi: int = 0
    total_put_oi: int = 0
    put_call_ratio: float = 0.0


@dataclass
class PolygonOptionBar:
    """OHLCV bar for an option contract."""
    
    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float = 0.0
    transactions: int = 0


# ============================================================================
# MAIN ADAPTER
# ============================================================================

class PolygonOptionsAdapter(OptionsChainAdapter):
    """
    Polygon.io Options Data Adapter for historical backtesting.
    
    Features:
    - Historical options chains
    - Expired contracts lookup
    - OHLCV aggregates for individual contracts
    - Greeks and IV data
    - Rate limiting and caching
    
    Usage:
        adapter = PolygonOptionsAdapter()
        
        # Get options chain for a date
        chain = adapter.get_historical_chain("SPY", date(2023, 6, 15))
        
        # Get price history for a contract
        bars = adapter.get_option_bars(
            "O:SPY230616C00450000",
            date(2023, 6, 1),
            date(2023, 6, 15),
        )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[PolygonConfig] = None,
    ):
        """Initialize adapter with API key."""
        self.config = config or PolygonConfig.from_env(api_key)
        
        if not self.config.api_key:
            logger.warning(
                "Polygon API key not set. Set POLYGON_API_KEY env var or pass api_key. "
                "Get free key at https://polygon.io"
            )
        
        self.client = httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            params={"apiKey": self.config.api_key},
        )
        
        self._last_request_time = 0.0
        self._cache: Dict[str, Any] = {}
        
        logger.info(
            f"PolygonOptionsAdapter initialized | "
            f"rate_limit={self.config.rate_limit_delay}s"
        )
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Make API request with rate limiting and caching."""
        
        cache_key = f"{endpoint}:{params}" if use_cache else None
        
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        
        self._rate_limit()
        
        try:
            response = self.client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            if cache_key and self.config.cache_enabled:
                self._cache[cache_key] = data
            
            return data
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Polygon rate limit hit, waiting 60s...")
                time.sleep(60)
                return self._request(endpoint, params, use_cache)
            elif e.response.status_code == 403:
                logger.error(
                    "Polygon API key invalid or insufficient permissions. "
                    "Check your API key and subscription tier."
                )
                raise
            else:
                logger.error(f"Polygon API error: {e.response.status_code} - {e.response.text[:200]}")
                raise
        except Exception as e:
            logger.error(f"Polygon request failed: {e}")
            raise
    
    # ==================== OPTIONS CONTRACTS ====================
    
    def get_options_contracts(
        self,
        underlying: str,
        expiration_date: Optional[date] = None,
        expiration_date_gte: Optional[date] = None,
        expiration_date_lte: Optional[date] = None,
        strike_price: Optional[float] = None,
        strike_price_gte: Optional[float] = None,
        strike_price_lte: Optional[float] = None,
        contract_type: Optional[str] = None,  # "call" or "put"
        expired: bool = True,  # Include expired contracts
        as_of: Optional[date] = None,  # Historical snapshot date
        limit: int = 1000,
    ) -> List[PolygonOptionContract]:
        """
        Get options contracts for an underlying.
        
        Args:
            underlying: Stock symbol (e.g., "SPY")
            expiration_date: Exact expiration date
            expiration_date_gte: Expiration >= this date
            expiration_date_lte: Expiration <= this date
            strike_price: Exact strike price
            strike_price_gte: Strike >= this price
            strike_price_lte: Strike <= this price
            contract_type: "call" or "put"
            expired: Include expired contracts (required for historical)
            as_of: Get contracts as they existed on this date
            limit: Max results per request
        
        Returns:
            List of option contracts
        """
        endpoint = "/v3/reference/options/contracts"
        
        params = {
            "underlying_ticker": underlying,
            "expired": str(expired).lower(),
            "limit": limit,
            "order": "asc",
            "sort": "expiration_date",
        }
        
        if expiration_date:
            params["expiration_date"] = expiration_date.isoformat()
        if expiration_date_gte:
            params["expiration_date.gte"] = expiration_date_gte.isoformat()
        if expiration_date_lte:
            params["expiration_date.lte"] = expiration_date_lte.isoformat()
        if strike_price:
            params["strike_price"] = strike_price
        if strike_price_gte:
            params["strike_price.gte"] = strike_price_gte
        if strike_price_lte:
            params["strike_price.lte"] = strike_price_lte
        if contract_type:
            params["contract_type"] = contract_type
        if as_of:
            params["as_of"] = as_of.isoformat()
        
        contracts = []
        next_url = None
        
        while True:
            if next_url:
                # Pagination - use full URL
                response = self.client.get(next_url)
                response.raise_for_status()
                data = response.json()
            else:
                data = self._request(endpoint, params)
            
            results = data.get("results", [])
            
            for item in results:
                try:
                    exp_date = datetime.strptime(
                        item["expiration_date"], "%Y-%m-%d"
                    ).date()
                    
                    contracts.append(PolygonOptionContract(
                        ticker=item.get("ticker", ""),
                        underlying_ticker=item.get("underlying_ticker", underlying),
                        strike_price=float(item.get("strike_price", 0)),
                        expiration_date=exp_date,
                        contract_type=item.get("contract_type", ""),
                        shares_per_contract=int(item.get("shares_per_contract", 100)),
                        exercise_style=item.get("exercise_style", "american"),
                        primary_exchange=item.get("primary_exchange", ""),
                    ))
                except Exception as e:
                    logger.debug(f"Error parsing contract: {e}")
                    continue
            
            # Pagination
            next_url = data.get("next_url")
            if not next_url or len(contracts) >= limit * 10:  # Safety limit
                break
            
            # Add API key to next URL
            if "apiKey" not in next_url:
                next_url += f"&apiKey={self.config.api_key}"
        
        logger.info(f"Retrieved {len(contracts)} option contracts for {underlying}")
        return contracts
    
    # ==================== OPTIONS AGGREGATES (OHLCV) ====================
    
    def get_option_bars(
        self,
        option_ticker: str,
        from_date: date,
        to_date: date,
        timespan: PolygonTimespan = PolygonTimespan.DAY,
        multiplier: int = 1,
    ) -> List[PolygonOptionBar]:
        """
        Get OHLCV bars for an option contract.
        
        Args:
            option_ticker: OCC symbol (e.g., "O:SPY230616C00450000")
            from_date: Start date
            to_date: End date
            timespan: Bar timespan (minute, hour, day, etc.)
            multiplier: Timespan multiplier
        
        Returns:
            List of OHLCV bars
        """
        # Ensure option ticker format
        if not option_ticker.startswith("O:"):
            option_ticker = f"O:{option_ticker}"
        
        endpoint = f"/v2/aggs/ticker/{option_ticker}/range/{multiplier}/{timespan.value}/{from_date.isoformat()}/{to_date.isoformat()}"
        
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        }
        
        data = self._request(endpoint, params)
        results = data.get("results", [])
        
        bars = []
        for item in results:
            try:
                ts = datetime.fromtimestamp(item["t"] / 1000, tz=timezone.utc)
                bars.append(PolygonOptionBar(
                    ticker=option_ticker,
                    timestamp=ts,
                    open=float(item.get("o", 0)),
                    high=float(item.get("h", 0)),
                    low=float(item.get("l", 0)),
                    close=float(item.get("c", 0)),
                    volume=int(item.get("v", 0)),
                    vwap=float(item.get("vw", 0)),
                    transactions=int(item.get("n", 0)),
                ))
            except Exception as e:
                logger.debug(f"Error parsing bar: {e}")
                continue
        
        logger.debug(f"Retrieved {len(bars)} bars for {option_ticker}")
        return bars
    
    # ==================== OPTIONS SNAPSHOT ====================
    
    def get_option_snapshot(
        self,
        underlying: str,
        option_ticker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get real-time snapshot for option(s).
        
        Note: Requires Options Starter tier or above.
        
        Args:
            underlying: Underlying symbol
            option_ticker: Specific option ticker (optional)
        
        Returns:
            Snapshot data with Greeks, IV, and pricing
        """
        if option_ticker:
            endpoint = f"/v3/snapshot/options/{underlying}/{option_ticker}"
        else:
            endpoint = f"/v3/snapshot/options/{underlying}"
        
        params = {"limit": 250}
        
        data = self._request(endpoint, params)
        return data.get("results", {})
    
    # ==================== HISTORICAL CHAIN ====================
    
    def get_historical_chain(
        self,
        underlying: str,
        as_of_date: date,
        expiration_min_dte: int = 0,
        expiration_max_dte: int = 60,
        strike_range_pct: float = 0.20,  # +/- 20% from spot
        spot_price: Optional[float] = None,
    ) -> PolygonOptionsChain:
        """
        Get historical options chain for backtesting.
        
        Args:
            underlying: Stock symbol
            as_of_date: Historical date to get chain for
            expiration_min_dte: Minimum days to expiry
            expiration_max_dte: Maximum days to expiry
            strike_range_pct: Strike range as % of spot
            spot_price: Spot price (fetched if not provided)
        
        Returns:
            Complete options chain with all contracts
        """
        # Calculate expiration date range
        exp_min = as_of_date + timedelta(days=expiration_min_dte)
        exp_max = as_of_date + timedelta(days=expiration_max_dte)
        
        # Get spot price if not provided
        if spot_price is None:
            spot_price = self._get_historical_spot(underlying, as_of_date)
        
        # Calculate strike range
        strike_min = spot_price * (1 - strike_range_pct)
        strike_max = spot_price * (1 + strike_range_pct)
        
        # Get contracts
        contracts = self.get_options_contracts(
            underlying=underlying,
            expiration_date_gte=exp_min,
            expiration_date_lte=exp_max,
            strike_price_gte=strike_min,
            strike_price_lte=strike_max,
            expired=True,
            as_of=as_of_date,
        )
        
        # Enrich with historical prices
        enriched_contracts = []
        for contract in contracts:
            try:
                # Get historical price for this contract on as_of_date
                bars = self.get_option_bars(
                    contract.ticker,
                    as_of_date,
                    as_of_date,
                    PolygonTimespan.DAY,
                )
                
                if bars:
                    bar = bars[0]
                    contract.last_price = bar.close
                    contract.bid = bar.close * 0.98  # Estimate
                    contract.ask = bar.close * 1.02  # Estimate
                    contract.mid = bar.close
                    contract.volume = bar.volume
                    contract.last_updated = bar.timestamp
                
                enriched_contracts.append(contract)
                
            except Exception as e:
                logger.debug(f"Error enriching contract {contract.ticker}: {e}")
                enriched_contracts.append(contract)
        
        # Separate calls and puts
        calls = [c for c in enriched_contracts if c.contract_type == "call"]
        puts = [c for c in enriched_contracts if c.contract_type == "put"]
        
        # Calculate aggregates
        total_call_vol = sum(c.volume for c in calls)
        total_put_vol = sum(c.volume for c in puts)
        total_call_oi = sum(c.open_interest for c in calls)
        total_put_oi = sum(c.open_interest for c in puts)
        
        pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0
        
        chain = PolygonOptionsChain(
            underlying=underlying,
            as_of_date=as_of_date,
            spot_price=spot_price,
            contracts=enriched_contracts,
            calls=calls,
            puts=puts,
            total_call_volume=total_call_vol,
            total_put_volume=total_put_vol,
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            put_call_ratio=pc_ratio,
        )
        
        logger.info(
            f"Built historical chain for {underlying} @ {as_of_date}: "
            f"{len(calls)} calls, {len(puts)} puts"
        )
        
        return chain
    
    def _get_historical_spot(self, symbol: str, as_of_date: date) -> float:
        """Get historical spot price for underlying."""
        endpoint = f"/v2/aggs/ticker/{symbol}/range/1/day/{as_of_date.isoformat()}/{as_of_date.isoformat()}"
        
        params = {"adjusted": "true"}
        
        try:
            data = self._request(endpoint, params)
            results = data.get("results", [])
            if results:
                return float(results[0].get("c", 0))
        except Exception as e:
            logger.warning(f"Could not get spot price for {symbol}: {e}")
        
        return 0.0
    
    # ==================== PROTOCOL IMPLEMENTATION ====================
    
    def get_chain(
        self,
        symbol: str,
        timestamp: datetime,
    ) -> List[OptionContract]:
        """
        Protocol implementation for OptionsChainAdapter.
        
        Returns contracts in the standard OptionContract format.
        """
        as_of_date = timestamp.date() if isinstance(timestamp, datetime) else timestamp
        
        try:
            chain = self.get_historical_chain(symbol, as_of_date)
            
            # Convert to standard format
            contracts = []
            for pc in chain.contracts:
                contracts.append(OptionContract(
                    symbol=pc.ticker,
                    strike=pc.strike_price,
                    expiration=datetime.combine(pc.expiration_date, datetime.min.time()),
                    option_type=pc.contract_type,
                    bid=pc.bid,
                    ask=pc.ask,
                    last=pc.last_price,
                    volume=pc.volume,
                    open_interest=pc.open_interest,
                    implied_volatility=pc.implied_volatility,
                    delta=pc.delta,
                    gamma=pc.gamma,
                    theta=pc.theta,
                    vega=pc.vega,
                ))
            
            return contracts
            
        except Exception as e:
            logger.error(f"Error getting chain for {symbol}: {e}")
            return []
    
    # ==================== BULK HISTORICAL DATA ====================
    
    def get_historical_data_range(
        self,
        underlying: str,
        start_date: date,
        end_date: date,
        sample_frequency: int = 5,  # Every N days
    ) -> Dict[date, PolygonOptionsChain]:
        """
        Get historical options chains for a date range.
        
        Useful for backtesting over extended periods.
        
        Args:
            underlying: Stock symbol
            start_date: Start of range
            end_date: End of range
            sample_frequency: Sample every N trading days
        
        Returns:
            Dictionary mapping dates to options chains
        """
        # Generate trading days (approximate)
        current = start_date
        dates = []
        day_count = 0
        
        while current <= end_date:
            # Skip weekends
            if current.weekday() < 5:
                if day_count % sample_frequency == 0:
                    dates.append(current)
                day_count += 1
            current += timedelta(days=1)
        
        logger.info(
            f"Fetching {len(dates)} historical chains for {underlying} "
            f"from {start_date} to {end_date}"
        )
        
        chains = {}
        for i, dt in enumerate(dates):
            try:
                chain = self.get_historical_chain(underlying, dt)
                chains[dt] = chain
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(dates)} chains fetched")
                    
            except Exception as e:
                logger.warning(f"Failed to get chain for {dt}: {e}")
                continue
        
        return chains
    
    # ==================== CLEANUP ====================
    
    def close(self):
        """Close HTTP client."""
        if self.client:
            self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_polygon_api_key() -> str:
    """Get Polygon API key from environment."""
    return os.getenv("POLYGON_API_KEY", "")


def set_polygon_api_key(key: str):
    """Set Polygon API key in environment."""
    os.environ["POLYGON_API_KEY"] = key


# ============================================================================
# DEMO
# ============================================================================

def demo_polygon_adapter():
    """Demonstrate Polygon.io adapter usage."""
    
    print("\n" + "="*70)
    print("  POLYGON.IO OPTIONS ADAPTER - DEMO")
    print("="*70)
    
    api_key = get_polygon_api_key()
    
    if not api_key:
        print("\n⚠️  POLYGON_API_KEY not set!")
        print("   Get a free API key at: https://polygon.io")
        print("   Then: export POLYGON_API_KEY=your_key_here")
        print("\n   Showing adapter structure without live data...\n")
        
        # Show what the adapter can do
        print("Available methods:")
        print("  - get_options_contracts(): Get contract definitions")
        print("  - get_option_bars(): Get OHLCV history for a contract")
        print("  - get_historical_chain(): Get full chain for a date")
        print("  - get_historical_data_range(): Bulk historical data")
        print("  - get_chain(): Protocol implementation")
        return
    
    adapter = PolygonOptionsAdapter(api_key=api_key)
    
    try:
        # Test 1: Get contracts
        print("\n[1] Getting SPY options contracts...")
        contracts = adapter.get_options_contracts(
            underlying="SPY",
            expiration_date_gte=date.today(),
            expiration_date_lte=date.today() + timedelta(days=30),
            contract_type="call",
            limit=10,
        )
        print(f"  Found {len(contracts)} contracts")
        
        if contracts:
            c = contracts[0]
            print(f"  Sample: {c.ticker} | Strike ${c.strike_price} | Exp {c.expiration_date}")
        
        # Test 2: Get bars for a contract
        if contracts:
            print("\n[2] Getting price history for first contract...")
            bars = adapter.get_option_bars(
                contracts[0].ticker,
                date.today() - timedelta(days=30),
                date.today(),
            )
            print(f"  Found {len(bars)} bars")
            
            if bars:
                b = bars[-1]
                print(f"  Latest: {b.timestamp.date()} | Close ${b.close:.2f} | Vol {b.volume}")
        
        # Test 3: Get historical chain
        print("\n[3] Getting historical chain...")
        historical_date = date.today() - timedelta(days=30)
        chain = adapter.get_historical_chain(
            "SPY",
            historical_date,
            expiration_max_dte=14,
        )
        print(f"  Chain for {chain.as_of_date}: {len(chain.calls)} calls, {len(chain.puts)} puts")
        print(f"  Spot price: ${chain.spot_price:.2f}")
        print(f"  P/C ratio: {chain.put_call_ratio:.2f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Check your API key and subscription tier.")
    
    finally:
        adapter.close()
    
    print("\n" + "="*70)
    print("  DEMO COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_polygon_adapter()
