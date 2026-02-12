"""
Historical Options Data Manager

Unified interface for historical options data with multiple providers:
1. Polygon.io - Real historical data (preferred)
2. Tradier - Alternative real data source
3. Synthetic Generator - Fallback when real data unavailable

Features:
- Automatic fallback between providers
- Caching for performance
- Greeks calculation/estimation
- Integration with backtest engines

Author: GNOSIS Trading System
Version: 1.0.0
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Protocol
from enum import Enum
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class HistoricalOptionContract:
    """Unified option contract with full data."""
    
    # Identification
    symbol: str  # OCC format
    underlying: str
    strike: float
    expiration: datetime
    option_type: str  # "call" or "put"
    
    # Pricing
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    mid: float = 0.0
    
    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    
    # Volatility
    implied_volatility: float = 0.0
    iv_percentile: float = 0.0
    
    # Activity
    volume: int = 0
    open_interest: int = 0
    
    # Calculated
    intrinsic_value: float = 0.0
    extrinsic_value: float = 0.0
    days_to_expiry: int = 0
    moneyness: float = 0.0
    
    # Source tracking
    data_source: str = "unknown"  # "polygon", "tradier", "synthetic"


@dataclass
class HistoricalOptionsChain:
    """Complete historical options chain."""
    
    underlying: str
    as_of_date: datetime
    spot_price: float
    
    # Contracts
    contracts: List[HistoricalOptionContract] = field(default_factory=list)
    calls: List[HistoricalOptionContract] = field(default_factory=list)
    puts: List[HistoricalOptionContract] = field(default_factory=list)
    
    # Expirations and strikes
    expirations: List[datetime] = field(default_factory=list)
    strikes: List[float] = field(default_factory=list)
    
    # Aggregated metrics
    total_call_oi: int = 0
    total_put_oi: int = 0
    total_call_volume: int = 0
    total_put_volume: int = 0
    put_call_ratio: float = 0.0
    put_call_oi_ratio: float = 0.0
    
    # Key levels
    max_pain: float = 0.0
    gamma_flip: float = 0.0
    call_wall: float = 0.0
    put_wall: float = 0.0
    
    # Dealer exposure
    net_gamma_exposure: float = 0.0
    dealer_positioning: str = "neutral"
    
    # Data quality
    data_source: str = "unknown"
    completeness: float = 1.0  # 0-1, % of expected data present


@dataclass
class HistoricalOptionsConfig:
    """Configuration for historical options data."""
    
    # Provider priority (massive = Polygon.io rebranded)
    providers: List[str] = field(default_factory=lambda: ["massive", "polygon", "synthetic"])
    
    # API keys (loaded from env if not provided)
    polygon_api_key: Optional[str] = None
    tradier_api_key: Optional[str] = None
    massive_api_key: Optional[str] = None
    
    # Caching
    cache_enabled: bool = True
    cache_dir: str = "cache/options_data"
    cache_ttl_days: int = 365  # Historical data doesn't change
    
    # Data parameters
    default_dte_min: int = 0
    default_dte_max: int = 60
    default_strike_range_pct: float = 0.20
    
    # Greeks calculation
    risk_free_rate: float = 0.05
    calculate_greeks: bool = True
    
    # Synthetic data parameters
    synthetic_base_iv: float = 0.25
    synthetic_seed: Optional[int] = None


class OptionsDataProvider(ABC):
    """Abstract base for options data providers."""
    
    @abstractmethod
    def get_chain(
        self,
        underlying: str,
        as_of_date: date,
        spot_price: Optional[float] = None,
        **kwargs,
    ) -> Optional[HistoricalOptionsChain]:
        """Get options chain for a date."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available/configured."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass


# ============================================================================
# POLYGON PROVIDER
# ============================================================================

class PolygonProvider(OptionsDataProvider):
    """Polygon.io data provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self._adapter = None
    
    @property
    def name(self) -> str:
        return "polygon"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def _get_adapter(self):
        """Lazy load adapter."""
        if self._adapter is None:
            from engines.inputs.polygon_options_adapter import PolygonOptionsAdapter
            self._adapter = PolygonOptionsAdapter(api_key=self.api_key)
        return self._adapter
    
    def get_chain(
        self,
        underlying: str,
        as_of_date: date,
        spot_price: Optional[float] = None,
        dte_min: int = 0,
        dte_max: int = 60,
        strike_range_pct: float = 0.20,
        **kwargs,
    ) -> Optional[HistoricalOptionsChain]:
        """Get chain from Polygon."""
        
        if not self.is_available():
            return None
        
        try:
            adapter = self._get_adapter()
            
            polygon_chain = adapter.get_historical_chain(
                underlying=underlying,
                as_of_date=as_of_date,
                expiration_min_dte=dte_min,
                expiration_max_dte=dte_max,
                strike_range_pct=strike_range_pct,
                spot_price=spot_price,
            )
            
            # Convert to unified format
            return self._convert_chain(polygon_chain, as_of_date)
            
        except Exception as e:
            logger.warning(f"Polygon provider error: {e}")
            return None
    
    def _convert_chain(
        self,
        polygon_chain,
        as_of_date: date,
    ) -> HistoricalOptionsChain:
        """Convert Polygon chain to unified format."""
        
        as_of_dt = datetime.combine(as_of_date, datetime.min.time())
        
        contracts = []
        for pc in polygon_chain.contracts:
            exp_dt = datetime.combine(pc.expiration_date, datetime.min.time())
            dte = (pc.expiration_date - as_of_date).days
            
            contract = HistoricalOptionContract(
                symbol=pc.ticker,
                underlying=pc.underlying_ticker,
                strike=pc.strike_price,
                expiration=exp_dt,
                option_type=pc.contract_type,
                bid=pc.bid,
                ask=pc.ask,
                last=pc.last_price,
                mid=pc.mid or (pc.bid + pc.ask) / 2,
                delta=pc.delta,
                gamma=pc.gamma,
                theta=pc.theta,
                vega=pc.vega,
                implied_volatility=pc.implied_volatility,
                volume=pc.volume,
                open_interest=pc.open_interest,
                days_to_expiry=dte,
                moneyness=pc.strike_price / polygon_chain.spot_price if polygon_chain.spot_price > 0 else 1,
                data_source="polygon",
            )
            contracts.append(contract)
        
        calls = [c for c in contracts if c.option_type == "call"]
        puts = [c for c in contracts if c.option_type == "put"]
        
        expirations = sorted(set(c.expiration for c in contracts))
        strikes = sorted(set(c.strike for c in contracts))
        
        return HistoricalOptionsChain(
            underlying=polygon_chain.underlying,
            as_of_date=as_of_dt,
            spot_price=polygon_chain.spot_price,
            contracts=contracts,
            calls=calls,
            puts=puts,
            expirations=expirations,
            strikes=strikes,
            total_call_oi=polygon_chain.total_call_oi,
            total_put_oi=polygon_chain.total_put_oi,
            total_call_volume=polygon_chain.total_call_volume,
            total_put_volume=polygon_chain.total_put_volume,
            put_call_ratio=polygon_chain.put_call_ratio,
            data_source="polygon",
        )


# ============================================================================
# MASSIVE.COM PROVIDER (Polygon.io rebranded)
# ============================================================================

class MassiveProvider(OptionsDataProvider):
    """Massive.com data provider (formerly Polygon.io).
    
    Massive.com is the rebranded Polygon.io service providing:
    - Historical tick-level options data back to 2004
    - Full options chains with Greeks
    - Multi-timeframe aggregations
    - Real-time and delayed data tiers
    """
    
    def __init__(self, api_key: Optional[str] = None):
        from config.credentials import get_massive_api_keys
        primary, secondary = get_massive_api_keys(primary=api_key)
        self.api_key = primary or secondary
        self._adapter = None
    
    @property
    def name(self) -> str:
        return "massive"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def _get_adapter(self):
        """Lazy load adapter."""
        if self._adapter is None:
            from engines.inputs.massive_options_adapter import MassiveOptionsAdapter
            self._adapter = MassiveOptionsAdapter(api_key=self.api_key)
        return self._adapter
    
    def get_chain(
        self,
        underlying: str,
        as_of_date: date,
        spot_price: Optional[float] = None,
        dte_min: int = 0,
        dte_max: int = 60,
        strike_range_pct: float = 0.20,
        **kwargs,
    ) -> Optional[HistoricalOptionsChain]:
        """Get chain from Massive.com."""
        
        if not self.is_available():
            return None
        
        try:
            adapter = self._get_adapter()
            
            # Get historical chain data
            as_of_dt = datetime.combine(as_of_date, datetime.min.time())
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=timezone.utc)
            
            massive_chain = adapter.get_historical_chain(
                underlying=underlying,
                as_of_date=as_of_date,
                spot_price=spot_price,
                dte_min=dte_min,
                dte_max=dte_max,
                strike_range_pct=strike_range_pct,
            )
            
            if not massive_chain:
                return None
            
            # Convert to unified format
            return self._convert_chain(massive_chain, as_of_dt, spot_price or 0)
            
        except Exception as e:
            logger.warning(f"Massive provider error: {e}")
            return None
    
    def _convert_chain(
        self,
        massive_chain,
        as_of_dt: datetime,
        spot_price: float,
    ) -> HistoricalOptionsChain:
        """Convert Massive chain to unified format."""
        
        contracts = []
        
        for mc in massive_chain.contracts:
            exp_dt = mc.expiration if isinstance(mc.expiration, datetime) else datetime.combine(mc.expiration, datetime.min.time())
            dte = (exp_dt.date() - as_of_dt.date()).days if isinstance(exp_dt, datetime) else 0
            
            contract = HistoricalOptionContract(
                symbol=mc.symbol,
                underlying=massive_chain.underlying,
                strike=mc.strike,
                expiration=exp_dt,
                option_type=mc.option_type,
                bid=mc.bid,
                ask=mc.ask,
                last=mc.last,
                mid=(mc.bid + mc.ask) / 2 if mc.bid and mc.ask else mc.last,
                delta=mc.delta or 0,
                gamma=mc.gamma or 0,
                theta=mc.theta or 0,
                vega=mc.vega or 0,
                rho=mc.rho or 0,
                implied_volatility=mc.implied_volatility or 0,
                volume=mc.volume or 0,
                open_interest=mc.open_interest or 0,
                days_to_expiry=dte,
                moneyness=mc.strike / spot_price if spot_price > 0 else 1,
                data_source="massive",
            )
            contracts.append(contract)
        
        calls = [c for c in contracts if c.option_type == "call"]
        puts = [c for c in contracts if c.option_type == "put"]
        
        expirations = sorted(set(c.expiration for c in contracts))
        strikes = sorted(set(c.strike for c in contracts))
        
        total_call_oi = sum(c.open_interest for c in calls)
        total_put_oi = sum(c.open_interest for c in puts)
        total_call_vol = sum(c.volume for c in calls)
        total_put_vol = sum(c.volume for c in puts)
        
        return HistoricalOptionsChain(
            underlying=massive_chain.underlying,
            as_of_date=as_of_dt,
            spot_price=massive_chain.spot_price or spot_price,
            contracts=contracts,
            calls=calls,
            puts=puts,
            expirations=expirations,
            strikes=strikes,
            total_call_oi=total_call_oi,
            total_put_oi=total_put_oi,
            total_call_volume=total_call_vol,
            total_put_volume=total_put_vol,
            put_call_ratio=total_put_vol / max(total_call_vol, 1),
            put_call_oi_ratio=total_put_oi / max(total_call_oi, 1),
            data_source="massive",
        )


# ============================================================================
# SYNTHETIC PROVIDER
# ============================================================================

class SyntheticProvider(OptionsDataProvider):
    """Synthetic data provider using realistic simulation."""
    
    def __init__(
        self,
        base_iv: float = 0.25,
        seed: Optional[int] = None,
        risk_free_rate: float = 0.05,
    ):
        self.base_iv = base_iv
        self.seed = seed
        self.risk_free_rate = risk_free_rate
        self._generator = None
    
    @property
    def name(self) -> str:
        return "synthetic"
    
    def is_available(self) -> bool:
        return True  # Always available
    
    def _get_generator(self):
        """Lazy load generator."""
        if self._generator is None:
            from backtesting.synthetic_options_data import SyntheticOptionsGenerator
            self._generator = SyntheticOptionsGenerator(
                risk_free_rate=self.risk_free_rate,
                base_iv=self.base_iv,
                seed=self.seed or 42,
            )
        return self._generator
    
    def get_chain(
        self,
        underlying: str,
        as_of_date: date,
        spot_price: Optional[float] = None,
        market_regime: str = "neutral",
        days_to_earnings: Optional[int] = None,
        dte_min: int = 0,
        dte_max: int = 60,
        strike_range_pct: float = 0.20,
        **kwargs,
    ) -> Optional[HistoricalOptionsChain]:
        """Generate synthetic chain.
        
        Note: dte_min, dte_max, strike_range_pct are accepted for API
        compatibility but synthetic generator uses its own defaults.
        """
        
        try:
            generator = self._get_generator()
            
            # Use provided spot or estimate
            if spot_price is None:
                spot_price = self._estimate_spot(underlying, as_of_date)
            
            as_of_dt = datetime.combine(as_of_date, datetime.min.time())
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=timezone.utc)
            
            synthetic_chain = generator.generate_options_chain(
                underlying=underlying,
                spot=spot_price,
                current_date=as_of_dt,
                market_regime=market_regime,
                days_to_earnings=days_to_earnings,
            )
            
            # Convert to unified format
            return self._convert_chain(synthetic_chain, as_of_dt, spot_price)
            
        except Exception as e:
            logger.error(f"Synthetic provider error: {e}")
            return None
    
    def _estimate_spot(self, underlying: str, as_of_date: date) -> float:
        """Estimate spot price from historical data."""
        # Default prices for common symbols
        defaults = {
            "SPY": 450.0,
            "QQQ": 380.0,
            "IWM": 200.0,
            "AAPL": 180.0,
            "MSFT": 380.0,
            "GOOGL": 140.0,
            "AMZN": 150.0,
            "NVDA": 500.0,
            "TSLA": 250.0,
            "META": 350.0,
        }
        return defaults.get(underlying, 100.0)
    
    def _convert_chain(
        self,
        synthetic_chain,
        as_of_dt: datetime,
        spot_price: float,
    ) -> HistoricalOptionsChain:
        """Convert synthetic chain to unified format."""
        
        contracts = []
        
        for sc in synthetic_chain.calls + synthetic_chain.puts:
            contract = HistoricalOptionContract(
                symbol=sc.symbol,
                underlying=synthetic_chain.underlying,
                strike=sc.strike,
                expiration=sc.expiration,
                option_type=sc.option_type,
                bid=sc.bid,
                ask=sc.ask,
                last=sc.last,
                mid=sc.mid,
                delta=sc.delta,
                gamma=sc.gamma,
                theta=sc.theta,
                vega=sc.vega,
                rho=sc.rho,
                implied_volatility=sc.implied_volatility,
                iv_percentile=sc.iv_percentile,
                volume=sc.volume,
                open_interest=sc.open_interest,
                intrinsic_value=sc.intrinsic_value,
                extrinsic_value=sc.extrinsic_value,
                days_to_expiry=sc.days_to_expiry,
                moneyness=sc.moneyness,
                data_source="synthetic",
            )
            contracts.append(contract)
        
        calls = [c for c in contracts if c.option_type == "call"]
        puts = [c for c in contracts if c.option_type == "put"]
        
        expirations = [datetime.combine(e, datetime.min.time()) if isinstance(e, date) else e 
                       for e in synthetic_chain.expirations]
        
        return HistoricalOptionsChain(
            underlying=synthetic_chain.underlying,
            as_of_date=as_of_dt,
            spot_price=spot_price,
            contracts=contracts,
            calls=calls,
            puts=puts,
            expirations=expirations,
            strikes=synthetic_chain.strikes,
            total_call_oi=synthetic_chain.total_call_oi,
            total_put_oi=synthetic_chain.total_put_oi,
            total_call_volume=synthetic_chain.total_call_volume,
            total_put_volume=synthetic_chain.total_put_volume,
            put_call_ratio=synthetic_chain.put_call_ratio,
            put_call_oi_ratio=synthetic_chain.put_call_oi_ratio,
            max_pain=synthetic_chain.max_pain,
            gamma_flip=synthetic_chain.gamma_flip,
            call_wall=synthetic_chain.call_wall,
            put_wall=synthetic_chain.put_wall,
            net_gamma_exposure=synthetic_chain.net_gamma_exposure,
            dealer_positioning=synthetic_chain.dealer_positioning,
            data_source="synthetic",
        )


# ============================================================================
# MAIN MANAGER
# ============================================================================

class HistoricalOptionsManager:
    """
    Unified manager for historical options data.
    
    Automatically selects the best available data source:
    1. Polygon.io (if API key configured)
    2. Synthetic data (always available as fallback)
    
    Features:
    - Automatic provider fallback
    - Caching for performance
    - Bulk data retrieval
    - Greeks calculation
    
    Usage:
        manager = HistoricalOptionsManager()
        
        # Get single chain
        chain = manager.get_chain("SPY", date(2023, 6, 15))
        
        # Get chains for date range
        chains = manager.get_chains_range(
            "SPY",
            date(2023, 1, 1),
            date(2023, 12, 31),
        )
    """
    
    def __init__(self, config: Optional[HistoricalOptionsConfig] = None):
        self.config = config or HistoricalOptionsConfig()
        self._providers: Dict[str, OptionsDataProvider] = {}
        self._cache: Dict[str, HistoricalOptionsChain] = {}
        
        # Initialize providers
        self._init_providers()
        
        # Setup cache directory
        if self.config.cache_enabled:
            cache_path = Path(self.config.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"HistoricalOptionsManager initialized | "
            f"providers={list(self._providers.keys())}"
        )
    
    def _init_providers(self):
        """Initialize configured providers."""
        
        for provider_name in self.config.providers:
            provider = None
            
            if provider_name == "massive":
                api_key = self.config.massive_api_key
                provider = MassiveProvider(api_key=api_key)
            
            elif provider_name == "polygon":
                api_key = self.config.polygon_api_key or os.getenv("POLYGON_API_KEY", "")
                provider = PolygonProvider(api_key=api_key)
                
            elif provider_name == "synthetic":
                provider = SyntheticProvider(
                    base_iv=self.config.synthetic_base_iv,
                    seed=self.config.synthetic_seed,
                    risk_free_rate=self.config.risk_free_rate,
                )
            
            if provider and provider.is_available():
                self._providers[provider_name] = provider
                logger.debug(f"Provider '{provider_name}' initialized")
            elif provider:
                logger.debug(f"Provider '{provider_name}' not available")
    
    def _get_cache_key(self, underlying: str, as_of_date: date) -> str:
        """Generate cache key."""
        return f"{underlying}_{as_of_date.isoformat()}"
    
    def _load_from_cache(self, underlying: str, as_of_date: date) -> Optional[HistoricalOptionsChain]:
        """Load chain from cache."""
        
        if not self.config.cache_enabled:
            return None
        
        cache_key = self._get_cache_key(underlying, as_of_date)
        
        # Memory cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Disk cache
        cache_file = Path(self.config.cache_dir) / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    chain = pickle.load(f)
                self._cache[cache_key] = chain
                return chain
            except Exception as e:
                logger.debug(f"Cache load error: {e}")
        
        return None
    
    def _save_to_cache(self, chain: HistoricalOptionsChain, underlying: str, as_of_date: date):
        """Save chain to cache."""
        
        if not self.config.cache_enabled:
            return
        
        cache_key = self._get_cache_key(underlying, as_of_date)
        
        # Memory cache
        self._cache[cache_key] = chain
        
        # Disk cache
        cache_file = Path(self.config.cache_dir) / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(chain, f)
        except Exception as e:
            logger.debug(f"Cache save error: {e}")
    
    def get_chain(
        self,
        underlying: str,
        as_of_date: date,
        spot_price: Optional[float] = None,
        force_provider: Optional[str] = None,
        **kwargs,
    ) -> Optional[HistoricalOptionsChain]:
        """
        Get historical options chain for a specific date.
        
        Args:
            underlying: Stock symbol (e.g., "SPY")
            as_of_date: Historical date
            spot_price: Spot price (fetched/estimated if not provided)
            force_provider: Force specific provider (bypass priority)
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Historical options chain or None
        """
        # Check cache first
        cached = self._load_from_cache(underlying, as_of_date)
        if cached:
            logger.debug(f"Cache hit for {underlying} @ {as_of_date}")
            return cached
        
        # Determine provider order
        if force_provider:
            providers = [force_provider] if force_provider in self._providers else []
        else:
            providers = [p for p in self.config.providers if p in self._providers]
        
        # Try providers in order
        for provider_name in providers:
            provider = self._providers.get(provider_name)
            if not provider:
                continue
            
            try:
                chain = provider.get_chain(
                    underlying=underlying,
                    as_of_date=as_of_date,
                    spot_price=spot_price,
                    dte_min=kwargs.get('dte_min', self.config.default_dte_min),
                    dte_max=kwargs.get('dte_max', self.config.default_dte_max),
                    strike_range_pct=kwargs.get('strike_range_pct', self.config.default_strike_range_pct),
                    **kwargs,
                )
                
                if chain and chain.contracts:
                    logger.info(
                        f"Got {len(chain.contracts)} contracts for {underlying} @ {as_of_date} "
                        f"from {provider_name}"
                    )
                    
                    # Calculate additional metrics
                    self._enrich_chain(chain)
                    
                    # Cache result
                    self._save_to_cache(chain, underlying, as_of_date)
                    
                    return chain
                    
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue
        
        logger.warning(f"No data available for {underlying} @ {as_of_date}")
        return None
    
    def get_chains_range(
        self,
        underlying: str,
        start_date: date,
        end_date: date,
        sample_frequency: int = 1,  # Every N trading days
        spot_prices: Optional[Dict[date, float]] = None,
        progress_callback: Optional[callable] = None,
        **kwargs,
    ) -> Dict[date, HistoricalOptionsChain]:
        """
        Get historical options chains for a date range.
        
        Args:
            underlying: Stock symbol
            start_date: Start of range
            end_date: End of range
            sample_frequency: Sample every N trading days
            spot_prices: Dict mapping dates to spot prices
            progress_callback: Callback for progress updates
            **kwargs: Additional parameters
        
        Returns:
            Dictionary mapping dates to options chains
        """
        # Generate trading days
        current = start_date
        dates = []
        day_count = 0
        
        while current <= end_date:
            if current.weekday() < 5:  # Skip weekends
                if day_count % sample_frequency == 0:
                    dates.append(current)
                day_count += 1
            current += timedelta(days=1)
        
        logger.info(
            f"Fetching {len(dates)} chains for {underlying} "
            f"from {start_date} to {end_date}"
        )
        
        chains = {}
        for i, dt in enumerate(dates):
            spot = spot_prices.get(dt) if spot_prices else None
            
            chain = self.get_chain(underlying, dt, spot_price=spot, **kwargs)
            
            if chain:
                chains[dt] = chain
            
            if progress_callback:
                progress_callback(i + 1, len(dates), underlying, dt)
            
            if (i + 1) % 20 == 0:
                logger.info(f"Progress: {i + 1}/{len(dates)} chains")
        
        return chains
    
    def _enrich_chain(self, chain: HistoricalOptionsChain):
        """Calculate additional metrics for chain."""
        
        if not chain.contracts:
            return
        
        # Calculate max pain
        if chain.strikes:
            chain.max_pain = self._calculate_max_pain(chain)
        
        # Calculate gamma flip
        chain.gamma_flip = self._calculate_gamma_flip(chain)
        
        # Find call/put walls
        chain.call_wall, chain.put_wall = self._find_walls(chain)
        
        # Calculate net gamma exposure
        chain.net_gamma_exposure = self._calculate_net_gamma(chain)
        
        # Determine dealer positioning
        if chain.net_gamma_exposure > 1e6:
            chain.dealer_positioning = "long_gamma"
        elif chain.net_gamma_exposure < -1e6:
            chain.dealer_positioning = "short_gamma"
        else:
            chain.dealer_positioning = "neutral"
    
    def _calculate_max_pain(self, chain: HistoricalOptionsChain) -> float:
        """Calculate max pain strike."""
        
        strike_pain = {}
        
        for strike in chain.strikes:
            pain = 0
            
            for contract in chain.contracts:
                if contract.open_interest > 0:
                    if contract.option_type == "call":
                        # Calls are worthless below strike
                        if strike <= contract.strike:
                            pain += 0
                        else:
                            pain += (strike - contract.strike) * contract.open_interest * 100
                    else:  # put
                        # Puts are worthless above strike
                        if strike >= contract.strike:
                            pain += 0
                        else:
                            pain += (contract.strike - strike) * contract.open_interest * 100
            
            strike_pain[strike] = pain
        
        if strike_pain:
            return min(strike_pain.keys(), key=lambda k: strike_pain[k])
        return chain.spot_price
    
    def _calculate_gamma_flip(self, chain: HistoricalOptionsChain) -> float:
        """Calculate gamma flip level."""
        # Simplified: near max pain
        return chain.max_pain if chain.max_pain else chain.spot_price
    
    def _find_walls(self, chain: HistoricalOptionsChain) -> Tuple[float, float]:
        """Find call and put walls (highest OI strikes)."""
        
        call_oi = {}
        put_oi = {}
        
        for contract in chain.contracts:
            if contract.option_type == "call" and contract.strike > chain.spot_price:
                call_oi[contract.strike] = call_oi.get(contract.strike, 0) + contract.open_interest
            elif contract.option_type == "put" and contract.strike < chain.spot_price:
                put_oi[contract.strike] = put_oi.get(contract.strike, 0) + contract.open_interest
        
        call_wall = max(call_oi.keys(), key=lambda k: call_oi[k]) if call_oi else chain.spot_price * 1.05
        put_wall = max(put_oi.keys(), key=lambda k: put_oi[k]) if put_oi else chain.spot_price * 0.95
        
        return call_wall, put_wall
    
    def _calculate_net_gamma(self, chain: HistoricalOptionsChain) -> float:
        """Calculate net gamma exposure."""
        
        net_gamma = 0
        
        for contract in chain.contracts:
            gamma_dollars = contract.gamma * contract.open_interest * 100 * chain.spot_price
            
            if contract.option_type == "call":
                net_gamma -= gamma_dollars  # Dealers short calls
            else:
                net_gamma += gamma_dollars  # Dealers long puts (from retail)
        
        return net_gamma
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self._providers.keys())
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
        
        if self.config.cache_enabled:
            cache_path = Path(self.config.cache_dir)
            for f in cache_path.glob("*.pkl"):
                f.unlink()
        
        logger.info("Cache cleared")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_historical_options(
    underlying: str,
    as_of_date: date,
    **kwargs,
) -> Optional[HistoricalOptionsChain]:
    """
    Quick access to historical options data.
    
    Usage:
        chain = get_historical_options("SPY", date(2023, 6, 15))
    """
    manager = HistoricalOptionsManager()
    return manager.get_chain(underlying, as_of_date, **kwargs)


def get_historical_options_range(
    underlying: str,
    start_date: date,
    end_date: date,
    **kwargs,
) -> Dict[date, HistoricalOptionsChain]:
    """
    Quick access to historical options data range.
    
    Usage:
        chains = get_historical_options_range(
            "SPY",
            date(2023, 1, 1),
            date(2023, 12, 31),
        )
    """
    manager = HistoricalOptionsManager()
    return manager.get_chains_range(underlying, start_date, end_date, **kwargs)


# ============================================================================
# DEMO
# ============================================================================

def demo_historical_options_manager():
    """Demonstrate historical options manager."""
    
    print("\n" + "="*70)
    print("  HISTORICAL OPTIONS MANAGER - DEMO")
    print("="*70)
    
    # Initialize manager
    manager = HistoricalOptionsManager()
    
    print(f"\nAvailable providers: {manager.get_available_providers()}")
    
    # Get a historical chain
    test_date = date(2023, 6, 15)
    print(f"\nFetching SPY options chain for {test_date}...")
    
    chain = manager.get_chain("SPY", test_date, spot_price=440.0)
    
    if chain:
        print(f"\n📊 Chain Summary ({chain.data_source}):")
        print(f"  Underlying: {chain.underlying}")
        print(f"  As of: {chain.as_of_date}")
        print(f"  Spot Price: ${chain.spot_price:.2f}")
        print(f"  Total Contracts: {len(chain.contracts)}")
        print(f"  Calls: {len(chain.calls)}")
        print(f"  Puts: {len(chain.puts)}")
        
        print(f"\n📈 Open Interest:")
        print(f"  Total Call OI: {chain.total_call_oi:,}")
        print(f"  Total Put OI: {chain.total_put_oi:,}")
        print(f"  Put/Call OI Ratio: {chain.put_call_oi_ratio:.2f}")
        
        print(f"\n🎯 Key Levels:")
        print(f"  Max Pain: ${chain.max_pain:.2f}")
        print(f"  Call Wall: ${chain.call_wall:.2f}")
        print(f"  Put Wall: ${chain.put_wall:.2f}")
        print(f"  Gamma Flip: ${chain.gamma_flip:.2f}")
        print(f"  Dealer Position: {chain.dealer_positioning}")
        
        # Sample contracts
        if chain.calls:
            print(f"\n📋 Sample ATM Call:")
            atm_calls = sorted(chain.calls, key=lambda c: abs(c.strike - chain.spot_price))
            if atm_calls:
                c = atm_calls[0]
                print(f"  {c.symbol}")
                print(f"  Strike: ${c.strike:.2f}")
                print(f"  Exp: {c.expiration.date()}")
                print(f"  Bid/Ask: ${c.bid:.2f} / ${c.ask:.2f}")
                print(f"  IV: {c.implied_volatility:.1%}")
                print(f"  Delta: {c.delta:.3f}")
                print(f"  OI: {c.open_interest:,}")
    else:
        print("  No data available")
    
    print("\n" + "="*70)
    print("  DEMO COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_historical_options_manager()
