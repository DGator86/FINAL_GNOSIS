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
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

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

    # Hardcoded MASSIVE API keys with environment override (primary + secondary)
    DEFAULT_API_KEYS = (
        "Jm_fqc_gtSTSXG78P67dpBpO3LX_4P6D",
        "22265906-ec01-4a42-928a-0037ccadbde3",
    )

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
        self.api_key = api_key or os.getenv("MASSIVE_API_KEY") or os.getenv("MASSIVE_API_KEY_SECONDARY")
        self.api_key = (
            api_key
            or os.getenv("MASSIVE_API_KEY")
            or os.getenv("MASSIVE_API_KEY_SECONDARY")
            or self._get_default_api_key()
        )
        self.enabled = os.getenv("MASSIVE_API_ENABLED", "true").lower() == "true"

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

    @classmethod
    def _get_default_api_key(cls) -> Optional[str]:
        """Return the first available hardcoded MASSIVE API key."""

        for key in cls.DEFAULT_API_KEYS:
            if key:
                return key
        return None

    def get_chain(self, symbol: str, timestamp: datetime) -> List[OptionContract]:
        """Get options chain for a symbol (implements OptionsChainAdapter protocol).

        Args:
            symbol: Underlying symbol
            timestamp: Data timestamp

        Returns:
            List of option contracts
        """
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
