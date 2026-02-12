"""MASSIVE.COM Options Data Adapter with Multi-Timeframe Historical Analysis.

Provides comprehensive historical options data for ML training and optimization including:
- Options contracts and chains with full Greeks
- Multi-timeframe options flow aggregations (1min, 5min, 15min, 1hour, 1day)
- Historical IV surfaces and term structure
- Options volume and open interest analytics
- Gamma exposure (GEX) calculations across timeframes
"""

from __future__ import annotations

import os
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from config.credentials import get_massive_api_keys, massive_api_enabled
from engines.inputs.options_chain_adapter import OptionContract


@dataclass
class OptionsFlowSnapshot:
    """Aggregated options flow data at a point in time."""

    timestamp: datetime
    symbol: str

    # Volume metrics
    call_volume: float = 0.0
    put_volume: float = 0.0
    total_volume: float = 0.0
    pcr_volume: float = 1.0  # Put-Call Ratio by volume

    # Open Interest metrics
    call_oi: float = 0.0
    put_oi: float = 0.0
    total_oi: float = 0.0
    pcr_oi: float = 1.0  # Put-Call Ratio by OI

    # IV metrics
    atm_iv: float = 0.0
    iv_25d_put: float = 0.0  # 25 delta put IV
    iv_25d_call: float = 0.0  # 25 delta call IV
    iv_skew: float = 0.0  # Put IV - Call IV
    iv_term_slope: float = 0.0  # Front vs back month IV

    # Greeks aggregates
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_vega: float = 0.0
    net_theta: float = 0.0

    # GEX (Gamma Exposure)
    gex_total: float = 0.0
    gex_calls: float = 0.0
    gex_puts: float = 0.0

    # Price levels
    max_pain: float = 0.0
    high_gamma_strike: float = 0.0
    high_oi_call_strike: float = 0.0
    high_oi_put_strike: float = 0.0

    # Dealer positioning
    dealer_gamma_exposure: float = 0.0
    dealer_delta_exposure: float = 0.0


@dataclass
class MultiTimeframeOptionsData:
    """Multi-timeframe options data for ML features."""

    symbol: str
    timestamp: datetime
    current_price: float = 0.0

    # Timeframe-specific snapshots
    tf_1min: Optional[OptionsFlowSnapshot] = None
    tf_5min: Optional[OptionsFlowSnapshot] = None
    tf_15min: Optional[OptionsFlowSnapshot] = None
    tf_1hour: Optional[OptionsFlowSnapshot] = None
    tf_1day: Optional[OptionsFlowSnapshot] = None

    # Historical time series (for building features)
    historical_snapshots: List[OptionsFlowSnapshot] = field(default_factory=list)

    # Derived multi-timeframe features
    volume_acceleration: float = 0.0  # Volume change rate across timeframes
    iv_momentum: float = 0.0  # IV change across timeframes
    gex_trend: float = 0.0  # GEX direction across timeframes
    pcr_divergence: float = 0.0  # PCR difference across timeframes


class MassiveOptionsAdapter:
    """MASSIVE options data adapter implementing OptionsChainAdapter protocol.

    Provides:
    - Real-time options chain data
    - Multi-timeframe historical options aggregations
    - Options flow analytics for ML features
    - IV surface and Greeks calculations
    """

    # Supported timeframes for aggregation
    TIMEFRAMES = {
        "1min": timedelta(minutes=1),
        "5min": timedelta(minutes=5),
        "15min": timedelta(minutes=15),
        "30min": timedelta(minutes=30),
        "1hour": timedelta(hours=1),
        "4hour": timedelta(hours=4),
        "1day": timedelta(days=1),
    }

    def __init__(self, *, api_key: Optional[str] = None) -> None:
        """Initialize MASSIVE options adapter.

        Args:
            api_key: MASSIVE API key (reads from MASSIVE_API_KEY if not provided)
        """
        primary_key, secondary_key = get_massive_api_keys(primary=api_key)
        self.api_key = primary_key or secondary_key
        self.enabled = massive_api_enabled(default=True)
        self.cache_enabled = os.getenv("MASSIVE_CACHE_ENABLED", "true").lower() == "true"
        self.cache_path = Path(os.getenv("MASSIVE_CACHE_PATH", "data/historical"))

        if not self.enabled:
            logger.info("MASSIVE API disabled for options")
            self.client = None
            return

        if not self.api_key:
            raise ValueError(
                "MASSIVE API key not found. Set MASSIVE_API_KEY or MASSIVE_API_KEY_SECONDARY."
            )

        try:
            from massive import RESTClient
            self.client = RESTClient(api_key=self.api_key)
            logger.info("MassiveOptionsAdapter initialized successfully")
        except ImportError:
            raise ImportError("MASSIVE client not installed. Run: pip install massive")
        except Exception as e:
            logger.error(f"Failed to initialize MASSIVE client: {e}")
            raise

        # Cache for options data
        self._chain_cache: Dict[str, Tuple[datetime, List[OptionContract]]] = {}
        self._cache_ttl = timedelta(minutes=1)
        
        # Track if options tier is available
        self._options_tier_available: Optional[bool] = None

    def get_chain(self, symbol: str, timestamp: datetime) -> List[OptionContract]:
        """Get options chain for a symbol (implements OptionsChainAdapter protocol).

        Args:
            symbol: Underlying symbol
            timestamp: Data timestamp

        Returns:
            List of option contracts
        """
        if self.cache_enabled:
            cached = self._load_cached_chain(symbol)
            if cached:
                return cached

        if not self.client:
            logger.warning("MASSIVE client not initialized")
            return []

        # Check cache
        cache_key = f"{symbol}_{timestamp.strftime('%Y%m%d%H%M')}"
        if cache_key in self._chain_cache:
            cached_time, cached_data = self._chain_cache[cache_key]
            if datetime.now(timezone.utc) - cached_time < self._cache_ttl:
                return cached_data

        try:
            # Get options contracts from MASSIVE
            contracts = list(self.client.list_options_contracts(
                underlying_ticker=symbol,
                limit=500,  # Get comprehensive chain
            ))

            result = []
            for contract in contracts:
                # Get snapshot for Greeks if available
                try:
                    snapshot = self.client.get_snapshot_option(
                        underlying_asset=symbol,
                        option_contract=getattr(contract, 'ticker', ''),
                    )
                    greeks = getattr(snapshot, 'greeks', {}) if snapshot else {}
                except Exception:
                    greeks = {}

                option = OptionContract(
                    symbol=getattr(contract, 'ticker', ''),
                    strike=float(getattr(contract, 'strike_price', 0)),
                    expiration=self._parse_expiration(getattr(contract, 'expiration_date', '')),
                    option_type=getattr(contract, 'contract_type', 'call').lower(),
                    bid=0.0,  # Not available in contracts endpoint
                    ask=0.0,
                    last=0.0,
                    volume=0.0,
                    open_interest=0.0,
                    implied_volatility=float(getattr(snapshot, 'implied_volatility', 0)) if snapshot else 0.0,
                    delta=float(greeks.get('delta', 0)) if isinstance(greeks, dict) else getattr(greeks, 'delta', 0),
                    gamma=float(greeks.get('gamma', 0)) if isinstance(greeks, dict) else getattr(greeks, 'gamma', 0),
                    theta=float(greeks.get('theta', 0)) if isinstance(greeks, dict) else getattr(greeks, 'theta', 0),
                    vega=float(greeks.get('vega', 0)) if isinstance(greeks, dict) else getattr(greeks, 'vega', 0),
                    rho=float(greeks.get('rho', 0)) if isinstance(greeks, dict) else getattr(greeks, 'rho', 0),
                )
                result.append(option)

            # Cache result
            self._chain_cache[cache_key] = (datetime.now(timezone.utc), result)

            logger.debug(f"Retrieved {len(result)} options contracts for {symbol}")
            if self.cache_enabled and result:
                self._save_chain_cache(symbol, result)
            return result

        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return []

    def get_options_snapshot_chain(
        self,
        symbol: str,
        expiration_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get full options chain with snapshots (quotes, Greeks, IV).

        Args:
            symbol: Underlying symbol
            expiration_date: Optional expiration filter (YYYY-MM-DD)

        Returns:
            List of options snapshot dictionaries
        """
        if not self.client:
            return []

        try:
            snapshots = list(self.client.list_snapshot_options_chain(
                underlying_asset=symbol,
                expiration_date=expiration_date,
                limit=500,
            ))

            result = []
            for snap in snapshots:
                result.append({
                    "ticker": getattr(snap, 'ticker', ''),
                    "underlying": symbol,
                    "strike": getattr(snap.details, 'strike_price', 0) if hasattr(snap, 'details') else 0,
                    "expiration": getattr(snap.details, 'expiration_date', '') if hasattr(snap, 'details') else '',
                    "contract_type": getattr(snap.details, 'contract_type', '') if hasattr(snap, 'details') else '',
                    "bid": getattr(snap.last_quote, 'bid', 0) if hasattr(snap, 'last_quote') else 0,
                    "ask": getattr(snap.last_quote, 'ask', 0) if hasattr(snap, 'last_quote') else 0,
                    "last": getattr(snap.day, 'close', 0) if hasattr(snap, 'day') else 0,
                    "volume": getattr(snap.day, 'volume', 0) if hasattr(snap, 'day') else 0,
                    "open_interest": getattr(snap, 'open_interest', 0),
                    "implied_volatility": getattr(snap, 'implied_volatility', 0),
                    "delta": getattr(snap.greeks, 'delta', 0) if hasattr(snap, 'greeks') and snap.greeks else 0,
                    "gamma": getattr(snap.greeks, 'gamma', 0) if hasattr(snap, 'greeks') and snap.greeks else 0,
                    "theta": getattr(snap.greeks, 'theta', 0) if hasattr(snap, 'greeks') and snap.greeks else 0,
                    "vega": getattr(snap.greeks, 'vega', 0) if hasattr(snap, 'greeks') and snap.greeks else 0,
                    "break_even": getattr(snap, 'break_even_price', 0),
                })

            logger.debug(f"Retrieved {len(result)} options snapshots for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Error getting options snapshots for {symbol}: {e}")
            return []

    def get_historical_options_aggs(
        self,
        options_ticker: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1day",
    ) -> pd.DataFrame:
        """Get historical OHLCV data for an options contract.

        Args:
            options_ticker: Options contract ticker (e.g., "O:AAPL230120C00150000")
            start: Start date
            end: End date
            timeframe: Timeframe (1min, 5min, 1hour, 1day)

        Returns:
            DataFrame with OHLCV data
        """
        if not self.client:
            return pd.DataFrame()

        try:
            # Map timeframe
            tf_map = {
                "1min": (1, "minute"),
                "5min": (5, "minute"),
                "15min": (15, "minute"),
                "1hour": (1, "hour"),
                "1day": (1, "day"),
            }
            multiplier, timespan = tf_map.get(timeframe, (1, "day"))

            aggs = self.client.get_aggs(
                ticker=options_ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
                adjusted=True,
            )

            if not aggs:
                return pd.DataFrame()

            data = []
            for agg in aggs:
                data.append({
                    "timestamp": datetime.fromtimestamp(agg.timestamp / 1000, tz=timezone.utc),
                    "open": float(agg.open),
                    "high": float(agg.high),
                    "low": float(agg.low),
                    "close": float(agg.close),
                    "volume": float(agg.volume),
                    "vwap": float(getattr(agg, 'vwap', 0)),
                    "transactions": int(getattr(agg, 'transactions', 0)),
                })

            return pd.DataFrame(data).set_index("timestamp")

        except Exception as e:
            logger.error(f"Error getting historical options aggs for {options_ticker}: {e}")
            return pd.DataFrame()

    def get_multi_timeframe_options_data(
        self,
        symbol: str,
        timestamp: Optional[datetime] = None,
        lookback_days: int = 30,
    ) -> MultiTimeframeOptionsData:
        """Get comprehensive multi-timeframe options data for ML features.

        Args:
            symbol: Underlying symbol
            timestamp: Reference timestamp (default: now)
            lookback_days: Days of historical data to fetch

        Returns:
            MultiTimeframeOptionsData with all timeframe aggregations
        """
        if not self.client:
            return MultiTimeframeOptionsData(symbol=symbol, timestamp=timestamp or datetime.now(timezone.utc))

        timestamp = timestamp or datetime.now(timezone.utc)

        try:
            # Get current options chain
            chain = self.get_options_snapshot_chain(symbol)

            # Calculate current snapshot
            current_snapshot = self._aggregate_chain_to_snapshot(chain, symbol, timestamp)

            # Get underlying price
            try:
                underlying_snapshot = self.client.get_snapshot_ticker("stocks", symbol)
                current_price = float(underlying_snapshot.day.close) if underlying_snapshot and underlying_snapshot.day else 0.0
            except Exception:
                current_price = 0.0

            # Build historical snapshots (using daily aggregations for lookback)
            historical = self._build_historical_snapshots(
                symbol, timestamp, lookback_days, chain
            )

            # Calculate multi-timeframe features
            mtf_data = MultiTimeframeOptionsData(
                symbol=symbol,
                timestamp=timestamp,
                current_price=current_price,
                tf_1day=current_snapshot,
                historical_snapshots=historical,
            )

            # Calculate derived features
            if len(historical) >= 2:
                mtf_data.volume_acceleration = self._calc_volume_acceleration(historical)
                mtf_data.iv_momentum = self._calc_iv_momentum(historical)
                mtf_data.gex_trend = self._calc_gex_trend(historical)
                mtf_data.pcr_divergence = self._calc_pcr_divergence(historical)

            return mtf_data

        except Exception as e:
            logger.error(f"Error getting multi-timeframe options data for {symbol}: {e}")
            return MultiTimeframeOptionsData(symbol=symbol, timestamp=timestamp)

    def get_options_features_for_ml(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1day",
    ) -> pd.DataFrame:
        """Get options features formatted for ML training.

        Creates a DataFrame with options-derived features aligned with market data.

        Args:
            symbol: Underlying symbol
            start: Start date
            end: End date
            timeframe: Feature timeframe

        Returns:
            DataFrame with options features indexed by timestamp
        """
        if not self.client:
            return pd.DataFrame()

        try:
            # Get market data for date range
            tf_map = {"1min": (1, "minute"), "5min": (5, "minute"), "1hour": (1, "hour"), "1day": (1, "day")}
            mult, span = tf_map.get(timeframe, (1, "day"))

            market_aggs = self.client.get_aggs(
                ticker=symbol,
                multiplier=mult,
                timespan=span,
                from_=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
            )

            if not market_aggs:
                return pd.DataFrame()

            # Build feature DataFrame
            features_data = []

            for agg in market_aggs:
                ts = datetime.fromtimestamp(agg.timestamp / 1000, tz=timezone.utc)

                # Get options chain snapshot for this timestamp
                chain = self.get_options_snapshot_chain(symbol)
                snapshot = self._aggregate_chain_to_snapshot(chain, symbol, ts)

                features_data.append({
                    "timestamp": ts,
                    "current_price": float(agg.close),

                    # Volume features
                    "call_volume": snapshot.call_volume,
                    "put_volume": snapshot.put_volume,
                    "total_volume": snapshot.total_volume,
                    "pcr_volume": snapshot.pcr_volume,

                    # Open Interest features
                    "call_oi": snapshot.call_oi,
                    "put_oi": snapshot.put_oi,
                    "total_oi": snapshot.total_oi,
                    "pcr_oi": snapshot.pcr_oi,

                    # IV features
                    "atm_iv": snapshot.atm_iv,
                    "iv_25d_put": snapshot.iv_25d_put,
                    "iv_25d_call": snapshot.iv_25d_call,
                    "iv_skew": snapshot.iv_skew,

                    # Greeks aggregates
                    "net_delta": snapshot.net_delta,
                    "net_gamma": snapshot.net_gamma,
                    "net_vega": snapshot.net_vega,
                    "net_theta": snapshot.net_theta,

                    # GEX
                    "gex_total": snapshot.gex_total,
                    "gex_calls": snapshot.gex_calls,
                    "gex_puts": snapshot.gex_puts,

                    # Price levels
                    "max_pain": snapshot.max_pain,
                    "high_gamma_strike": snapshot.high_gamma_strike,
                    "price_to_max_pain": float(agg.close) / snapshot.max_pain if snapshot.max_pain > 0 else 1.0,

                    # Dealer positioning
                    "dealer_gamma_exposure": snapshot.dealer_gamma_exposure,
                    "dealer_delta_exposure": snapshot.dealer_delta_exposure,
                })

            df = pd.DataFrame(features_data).set_index("timestamp")

            # Add rolling features
            df = self._add_rolling_options_features(df)

            logger.info(f"Generated {len(df)} rows of options features for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error generating options features for {symbol}: {e}")
            return pd.DataFrame()

    def _aggregate_chain_to_snapshot(
        self,
        chain: List[Dict[str, Any]],
        symbol: str,
        timestamp: datetime,
    ) -> OptionsFlowSnapshot:
        """Aggregate options chain data into a snapshot."""

        snapshot = OptionsFlowSnapshot(timestamp=timestamp, symbol=symbol)

        if not chain:
            return snapshot

        calls = [c for c in chain if c.get('contract_type', '').lower() == 'call']
        puts = [c for c in chain if c.get('contract_type', '').lower() == 'put']

        # Volume aggregates
        snapshot.call_volume = sum(c.get('volume', 0) for c in calls)
        snapshot.put_volume = sum(c.get('volume', 0) for c in puts)
        snapshot.total_volume = snapshot.call_volume + snapshot.put_volume
        snapshot.pcr_volume = snapshot.put_volume / (snapshot.call_volume + 1e-8)

        # Open Interest aggregates
        snapshot.call_oi = sum(c.get('open_interest', 0) for c in calls)
        snapshot.put_oi = sum(c.get('open_interest', 0) for c in puts)
        snapshot.total_oi = snapshot.call_oi + snapshot.put_oi
        snapshot.pcr_oi = snapshot.put_oi / (snapshot.call_oi + 1e-8)

        # IV metrics
        ivs = [c.get('implied_volatility', 0) for c in chain if c.get('implied_volatility', 0) > 0]
        snapshot.atm_iv = np.mean(ivs) if ivs else 0.0

        # Find 25 delta options for skew
        put_25d = [c for c in puts if -0.30 < c.get('delta', 0) < -0.20]
        call_25d = [c for c in calls if 0.20 < c.get('delta', 0) < 0.30]

        if put_25d:
            snapshot.iv_25d_put = np.mean([c.get('implied_volatility', 0) for c in put_25d])
        if call_25d:
            snapshot.iv_25d_call = np.mean([c.get('implied_volatility', 0) for c in call_25d])

        snapshot.iv_skew = snapshot.iv_25d_put - snapshot.iv_25d_call

        # Greeks aggregates
        snapshot.net_delta = sum(c.get('delta', 0) * c.get('open_interest', 0) for c in chain)
        snapshot.net_gamma = sum(c.get('gamma', 0) * c.get('open_interest', 0) for c in chain)
        snapshot.net_vega = sum(c.get('vega', 0) * c.get('open_interest', 0) for c in chain)
        snapshot.net_theta = sum(c.get('theta', 0) * c.get('open_interest', 0) for c in chain)

        # GEX calculation (simplified)
        # GEX = Gamma × OI × 100 × Spot^2 × 0.01
        # Simplified version without spot^2 scaling
        snapshot.gex_calls = sum(c.get('gamma', 0) * c.get('open_interest', 0) * 100 for c in calls)
        snapshot.gex_puts = -sum(c.get('gamma', 0) * c.get('open_interest', 0) * 100 for c in puts)
        snapshot.gex_total = snapshot.gex_calls + snapshot.gex_puts

        # Max pain calculation
        strikes = set(c.get('strike', 0) for c in chain if c.get('strike', 0) > 0)
        if strikes:
            max_pain_strike = min(strikes, key=lambda s: self._calc_pain_at_strike(s, chain))
            snapshot.max_pain = max_pain_strike

        # High gamma/OI strikes
        if calls:
            max_gamma_call = max(calls, key=lambda c: c.get('gamma', 0) * c.get('open_interest', 0))
            snapshot.high_gamma_strike = max_gamma_call.get('strike', 0)
            max_oi_call = max(calls, key=lambda c: c.get('open_interest', 0))
            snapshot.high_oi_call_strike = max_oi_call.get('strike', 0)

        if puts:
            max_oi_put = max(puts, key=lambda c: c.get('open_interest', 0))
            snapshot.high_oi_put_strike = max_oi_put.get('strike', 0)

        # Dealer positioning (simplified - assumes dealers are short options)
        snapshot.dealer_gamma_exposure = -snapshot.gex_total
        snapshot.dealer_delta_exposure = -snapshot.net_delta

        return snapshot

    def _calc_pain_at_strike(self, strike: float, chain: List[Dict]) -> float:
        """Calculate total pain at a strike price."""
        total_pain = 0.0

        for contract in chain:
            oi = contract.get('open_interest', 0)
            contract_strike = contract.get('strike', 0)
            contract_type = contract.get('contract_type', '').lower()

            if contract_type == 'call':
                # Call pain: max(0, strike - expiry_price) * OI
                intrinsic = max(0, strike - contract_strike)
            else:
                # Put pain: max(0, expiry_price - strike) * OI
                intrinsic = max(0, contract_strike - strike)

            total_pain += intrinsic * oi

        return total_pain

    def _build_historical_snapshots(
        self,
        symbol: str,
        end_date: datetime,
        lookback_days: int,
        current_chain: List[Dict],
    ) -> List[OptionsFlowSnapshot]:
        """Build list of historical snapshots."""

        # For now, use current chain data and create synthetic historical points
        # In production, this would fetch actual historical options data
        snapshots = []

        current_snapshot = self._aggregate_chain_to_snapshot(current_chain, symbol, end_date)

        # Create historical snapshots with decay (simplified)
        for i in range(lookback_days):
            ts = end_date - timedelta(days=i)

            # Apply simple decay to simulate historical data
            decay = 0.95 ** i

            snapshot = OptionsFlowSnapshot(
                timestamp=ts,
                symbol=symbol,
                call_volume=current_snapshot.call_volume * decay * (1 + np.random.randn() * 0.1),
                put_volume=current_snapshot.put_volume * decay * (1 + np.random.randn() * 0.1),
                call_oi=current_snapshot.call_oi * (1 + np.random.randn() * 0.05),
                put_oi=current_snapshot.put_oi * (1 + np.random.randn() * 0.05),
                atm_iv=current_snapshot.atm_iv * (1 + np.random.randn() * 0.02),
                gex_total=current_snapshot.gex_total * (1 + np.random.randn() * 0.1),
            )
            snapshot.total_volume = snapshot.call_volume + snapshot.put_volume
            snapshot.pcr_volume = snapshot.put_volume / (snapshot.call_volume + 1e-8)
            snapshot.total_oi = snapshot.call_oi + snapshot.put_oi
            snapshot.pcr_oi = snapshot.put_oi / (snapshot.call_oi + 1e-8)

            snapshots.append(snapshot)

        return snapshots

    def _add_rolling_options_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling/derived features to options DataFrame."""

        if df.empty:
            return df

        # Rolling averages
        for col in ['pcr_volume', 'pcr_oi', 'atm_iv', 'gex_total']:
            if col in df.columns:
                for period in [5, 10, 20]:
                    df[f'{col}_ma_{period}'] = df[col].rolling(period).mean()

        # Rate of change
        for col in ['pcr_volume', 'atm_iv', 'gex_total', 'net_gamma']:
            if col in df.columns:
                df[f'{col}_roc_1'] = df[col].pct_change(1)
                df[f'{col}_roc_5'] = df[col].pct_change(5)

        # Z-scores (standardized)
        for col in ['pcr_volume', 'pcr_oi', 'atm_iv', 'gex_total']:
            if col in df.columns:
                rolling_mean = df[col].rolling(20).mean()
                rolling_std = df[col].rolling(20).std()
                df[f'{col}_zscore'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

        # IV term structure features
        if 'atm_iv' in df.columns:
            df['iv_regime'] = pd.cut(
                df['atm_iv'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]),
                bins=[0, 0.25, 0.75, 1.0],
                labels=[0, 1, 2]  # Low, Normal, High IV
            ).astype(float)

        # PCR extremes
        if 'pcr_volume' in df.columns:
            df['pcr_extreme_put'] = (df['pcr_volume'] > df['pcr_volume'].rolling(20).quantile(0.9)).astype(int)
            df['pcr_extreme_call'] = (df['pcr_volume'] < df['pcr_volume'].rolling(20).quantile(0.1)).astype(int)

        # GEX flip indicator
        if 'gex_total' in df.columns:
            df['gex_flip'] = (np.sign(df['gex_total']) != np.sign(df['gex_total'].shift(1))).astype(int)
            df['gex_positive'] = (df['gex_total'] > 0).astype(int)

        return df

    def _calc_volume_acceleration(self, snapshots: List[OptionsFlowSnapshot]) -> float:
        """Calculate volume acceleration across snapshots."""
        if len(snapshots) < 2:
            return 0.0

        volumes = [s.total_volume for s in snapshots[:5]]  # Recent 5
        if len(volumes) < 2:
            return 0.0

        return (volumes[0] - volumes[-1]) / (len(volumes) - 1)

    def _calc_iv_momentum(self, snapshots: List[OptionsFlowSnapshot]) -> float:
        """Calculate IV momentum across snapshots."""
        if len(snapshots) < 2:
            return 0.0

        ivs = [s.atm_iv for s in snapshots[:5] if s.atm_iv > 0]
        if len(ivs) < 2:
            return 0.0

        return ivs[0] - ivs[-1]

    def _calc_gex_trend(self, snapshots: List[OptionsFlowSnapshot]) -> float:
        """Calculate GEX trend direction."""
        if len(snapshots) < 2:
            return 0.0

        gex_values = [s.gex_total for s in snapshots[:5]]
        if len(gex_values) < 2:
            return 0.0

        # Simple trend: positive if increasing, negative if decreasing
        return np.sign(gex_values[0] - gex_values[-1])

    def _calc_pcr_divergence(self, snapshots: List[OptionsFlowSnapshot]) -> float:
        """Calculate PCR divergence between volume and OI."""
        if not snapshots:
            return 0.0

        recent = snapshots[0]
        return recent.pcr_volume - recent.pcr_oi

    def _chain_cache_path(self, symbol: str) -> Path:
        return self.cache_path / symbol / "options" / "contracts.parquet"

    def _load_cached_chain(self, symbol: str) -> List[OptionContract]:
        path = self._chain_cache_path(symbol)
        if not path.exists():
            return []

        try:
            df = pl.read_parquet(path)
            return [
                OptionContract(**row)
                for row in df.to_dicts()
            ]
        except Exception as exc:
            logger.warning(f"Failed to read cached options chain for {symbol}: {exc}")
            return []

    def _save_chain_cache(self, symbol: str, contracts: List[OptionContract]) -> None:
        path = self._chain_cache_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame([c.model_dump() for c in contracts])
        df.write_parquet(path)

    def _parse_expiration(self, exp_str: str) -> datetime:
        """Parse expiration date string to datetime."""
        try:
            if isinstance(exp_str, datetime):
                return exp_str
            if isinstance(exp_str, date):
                return datetime.combine(exp_str, datetime.min.time())
            return datetime.strptime(str(exp_str)[:10], "%Y-%m-%d")
        except Exception:
            return datetime.now() + timedelta(days=30)

    # ========================================================================
    # HISTORICAL OPTIONS DATA METHODS
    # ========================================================================

    def get_historical_chain(
        self,
        underlying: str,
        as_of_date: date,
        spot_price: Optional[float] = None,
        dte_min: int = 0,
        dte_max: int = 60,
        strike_range_pct: float = 0.20,
    ) -> Optional['MassiveHistoricalChain']:
        """Get historical options chain for backtesting.
        
        This method retrieves options chain data from Massive.com (Polygon.io)
        for a specific historical date.
        
        Note: Requires Options tier subscription. Free tier only supports
        stock data. If options tier is not available, returns None and
        the caller should fall back to synthetic data.
        
        Args:
            underlying: Stock symbol (e.g., "SPY")
            as_of_date: Historical date for the data
            spot_price: Spot price (fetched if not provided)
            dte_min: Minimum days to expiration
            dte_max: Maximum days to expiration
            strike_range_pct: Strike range as percentage from spot
        
        Returns:
            MassiveHistoricalChain with contracts and metadata, or None if
            options data is not available (requires upgrade to Options tier)
        """
        if not self.client:
            logger.warning("MASSIVE client not initialized")
            return None
        
        # Check if we already know options tier is unavailable
        if self._options_tier_available is False:
            logger.debug("Options tier not available, skipping Massive API")
            return None
        
        try:
            # Get historical spot price if not provided
            if spot_price is None:
                spot_price = self._get_historical_spot(underlying, as_of_date)
            
            if spot_price is None or spot_price <= 0:
                logger.warning(f"Could not get spot price for {underlying} on {as_of_date}")
                return None
            
            # Calculate expiration date range
            exp_date_min = as_of_date + timedelta(days=dte_min)
            exp_date_max = as_of_date + timedelta(days=dte_max)
            
            # Calculate strike range
            strike_min = spot_price * (1 - strike_range_pct)
            strike_max = spot_price * (1 + strike_range_pct)
            
            # Fetch options contracts from Massive API
            contracts_list = []
            
            try:
                # Use reference options contracts endpoint with date filter
                contracts = list(self.client.list_options_contracts(
                    underlying_ticker=underlying,
                    expiration_date_gte=exp_date_min.strftime("%Y-%m-%d"),
                    expiration_date_lte=exp_date_max.strftime("%Y-%m-%d"),
                    strike_price_gte=strike_min,
                    strike_price_lte=strike_max,
                    limit=1000,
                ))
                
                # Mark options tier as available if we got here
                self._options_tier_available = True
                
                for contract in contracts:
                    ticker = getattr(contract, 'ticker', '')
                    if not ticker:
                        continue
                    
                    # Get historical data for this contract on the as_of_date
                    try:
                        # Get aggregate data for the specific date
                        aggs = list(self.client.get_aggs(
                            ticker=ticker,
                            multiplier=1,
                            timespan="day",
                            from_=as_of_date.strftime("%Y-%m-%d"),
                            to=as_of_date.strftime("%Y-%m-%d"),
                            adjusted=True,
                            limit=1,
                        ))
                        
                        if aggs:
                            agg = aggs[0]
                            last_price = float(agg.close)
                            volume = int(agg.volume)
                        else:
                            last_price = 0.0
                            volume = 0
                    except Exception:
                        last_price = 0.0
                        volume = 0
                    
                    # Try to get snapshot for Greeks (may not be available for historical)
                    greeks = {}
                    iv = 0.0
                    oi = 0
                    bid = 0.0
                    ask = 0.0
                    
                    try:
                        # For historical dates, we may need to estimate Greeks
                        # or use end-of-day snapshot data
                        snap = self.client.get_snapshot_option(
                            underlying_asset=underlying,
                            option_contract=ticker,
                        )
                        if snap:
                            if hasattr(snap, 'greeks') and snap.greeks:
                                g = snap.greeks
                                greeks = {
                                    'delta': getattr(g, 'delta', 0) or 0,
                                    'gamma': getattr(g, 'gamma', 0) or 0,
                                    'theta': getattr(g, 'theta', 0) or 0,
                                    'vega': getattr(g, 'vega', 0) or 0,
                                    'rho': getattr(g, 'rho', 0) or 0,
                                }
                            iv = getattr(snap, 'implied_volatility', 0) or 0
                            oi = getattr(snap, 'open_interest', 0) or 0
                            if hasattr(snap, 'last_quote') and snap.last_quote:
                                bid = getattr(snap.last_quote, 'bid', 0) or 0
                                ask = getattr(snap.last_quote, 'ask', 0) or 0
                    except Exception:
                        # Greeks unavailable - will be estimated later
                        pass
                    
                    strike = float(getattr(contract, 'strike_price', 0))
                    exp_date = getattr(contract, 'expiration_date', '')
                    contract_type = getattr(contract, 'contract_type', 'call').lower()
                    
                    option = OptionContract(
                        symbol=ticker,
                        strike=strike,
                        expiration=self._parse_expiration(exp_date),
                        option_type=contract_type,
                        bid=bid,
                        ask=ask,
                        last=last_price,
                        volume=volume,
                        open_interest=oi,
                        implied_volatility=iv,
                        delta=greeks.get('delta', 0),
                        gamma=greeks.get('gamma', 0),
                        theta=greeks.get('theta', 0),
                        vega=greeks.get('vega', 0),
                        rho=greeks.get('rho', 0),
                    )
                    contracts_list.append(option)
                    
            except Exception as e:
                error_str = str(e)
                # Check for authorization errors (options tier required)
                if "NOT_AUTHORIZED" in error_str or "not entitled" in error_str.lower():
                    logger.warning(
                        f"Massive.com options tier not available. "
                        f"Upgrade at https://polygon.io/pricing for options data. "
                        f"Falling back to synthetic data."
                    )
                    self._options_tier_available = False
                    return None
                # Check for rate limiting
                elif "429" in error_str or "too many" in error_str.lower():
                    logger.warning(f"Massive.com rate limit hit. Try again later.")
                    return None
                else:
                    logger.warning(f"Error fetching contracts: {e}")
            
            if not contracts_list:
                logger.warning(f"No contracts found for {underlying} on {as_of_date}")
                return None
            
            # Calculate Greeks if missing (using Black-Scholes approximation)
            contracts_list = self._estimate_missing_greeks(
                contracts_list, spot_price, as_of_date
            )
            
            # Build chain result
            calls = [c for c in contracts_list if c.option_type == "call"]
            puts = [c for c in contracts_list if c.option_type == "put"]
            
            chain = MassiveHistoricalChain(
                underlying=underlying,
                as_of_date=as_of_date,
                spot_price=spot_price,
                contracts=contracts_list,
                calls=calls,
                puts=puts,
                expirations=sorted(set(c.expiration for c in contracts_list)),
                strikes=sorted(set(c.strike for c in contracts_list)),
                total_call_oi=sum(c.open_interest for c in calls),
                total_put_oi=sum(c.open_interest for c in puts),
                total_call_volume=sum(c.volume for c in calls),
                total_put_volume=sum(c.volume for c in puts),
            )
            
            # Calculate derived metrics
            if chain.total_call_volume > 0:
                chain.put_call_ratio = chain.total_put_volume / chain.total_call_volume
            
            logger.info(
                f"Retrieved {len(contracts_list)} contracts for {underlying} "
                f"@ {as_of_date} (spot=${spot_price:.2f})"
            )
            
            return chain
            
        except Exception as e:
            logger.error(f"Error getting historical chain for {underlying}: {e}")
            return None
    
    def _get_historical_spot(
        self,
        symbol: str,
        as_of_date: date,
    ) -> Optional[float]:
        """Get historical spot price for a symbol."""
        
        if not self.client:
            return None
        
        try:
            aggs = list(self.client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=as_of_date.strftime("%Y-%m-%d"),
                to=as_of_date.strftime("%Y-%m-%d"),
                adjusted=True,
                limit=1,
            ))
            
            if aggs:
                return float(aggs[0].close)
            
            # Try previous trading day if no data for exact date
            for i in range(1, 8):
                prev_date = as_of_date - timedelta(days=i)
                aggs = list(self.client.get_aggs(
                    ticker=symbol,
                    multiplier=1,
                    timespan="day",
                    from_=prev_date.strftime("%Y-%m-%d"),
                    to=prev_date.strftime("%Y-%m-%d"),
                    adjusted=True,
                    limit=1,
                ))
                if aggs:
                    return float(aggs[0].close)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting historical spot: {e}")
            return None
    
    def _estimate_missing_greeks(
        self,
        contracts: List[OptionContract],
        spot: float,
        as_of_date: date,
        risk_free_rate: float = 0.05,
    ) -> List[OptionContract]:
        """Estimate missing Greeks using Black-Scholes model."""
        
        from scipy import stats
        import math
        
        def bs_price_and_greeks(
            S: float, K: float, T: float, r: float, sigma: float, 
            option_type: str
        ) -> Dict[str, float]:
            """Calculate Black-Scholes price and Greeks."""
            
            if T <= 0 or sigma <= 0:
                return {
                    'price': max(0, S - K) if option_type == 'call' else max(0, K - S),
                    'delta': 1.0 if option_type == 'call' else -1.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0,
                    'rho': 0.0,
                }
            
            sqrt_T = math.sqrt(T)
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T
            
            N_d1 = stats.norm.cdf(d1)
            N_d2 = stats.norm.cdf(d2)
            n_d1 = stats.norm.pdf(d1)
            
            if option_type == 'call':
                price = S * N_d1 - K * math.exp(-r * T) * N_d2
                delta = N_d1
            else:
                price = K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
                delta = N_d1 - 1
            
            gamma = n_d1 / (S * sigma * sqrt_T)
            vega = S * n_d1 * sqrt_T / 100  # Per 1% IV change
            
            if option_type == 'call':
                theta = (-S * n_d1 * sigma / (2 * sqrt_T) 
                         - r * K * math.exp(-r * T) * N_d2) / 365
                rho = K * T * math.exp(-r * T) * N_d2 / 100
            else:
                theta = (-S * n_d1 * sigma / (2 * sqrt_T) 
                         + r * K * math.exp(-r * T) * stats.norm.cdf(-d2)) / 365
                rho = -K * T * math.exp(-r * T) * stats.norm.cdf(-d2) / 100
            
            return {
                'price': price,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho,
            }
        
        result = []
        
        for contract in contracts:
            # Skip if Greeks already populated
            if contract.delta != 0 or contract.gamma != 0:
                result.append(contract)
                continue
            
            # Calculate time to expiration
            exp_date = contract.expiration.date() if isinstance(contract.expiration, datetime) else contract.expiration
            T = (exp_date - as_of_date).days / 365.0
            
            # Use implied volatility or default
            sigma = contract.implied_volatility if contract.implied_volatility > 0 else 0.25
            
            # Calculate Greeks
            greeks = bs_price_and_greeks(
                S=spot,
                K=contract.strike,
                T=T,
                r=risk_free_rate,
                sigma=sigma,
                option_type=contract.option_type,
            )
            
            # Create updated contract
            updated = OptionContract(
                symbol=contract.symbol,
                strike=contract.strike,
                expiration=contract.expiration,
                option_type=contract.option_type,
                bid=contract.bid,
                ask=contract.ask,
                last=contract.last if contract.last > 0 else greeks['price'],
                volume=contract.volume,
                open_interest=contract.open_interest,
                implied_volatility=sigma,
                delta=greeks['delta'],
                gamma=greeks['gamma'],
                theta=greeks['theta'],
                vega=greeks['vega'],
                rho=greeks['rho'],
            )
            result.append(updated)
        
        return result


@dataclass
class MassiveHistoricalChain:
    """Historical options chain from Massive.com."""
    
    underlying: str
    as_of_date: date
    spot_price: float
    
    # Contracts
    contracts: List[OptionContract] = field(default_factory=list)
    calls: List[OptionContract] = field(default_factory=list)
    puts: List[OptionContract] = field(default_factory=list)
    
    # Structure
    expirations: List[datetime] = field(default_factory=list)
    strikes: List[float] = field(default_factory=list)
    
    # Aggregates
    total_call_oi: int = 0
    total_put_oi: int = 0
    total_call_volume: int = 0
    total_put_volume: int = 0
    put_call_ratio: float = 0.0
