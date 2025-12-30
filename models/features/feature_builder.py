"""
Enhanced Feature Builder for FINAL_GNOSIS
Integrates with Hedge Engine v3.0 and other engines to create comprehensive feature matrix
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature building"""
    use_technical: bool = True
    use_hedge_engine: bool = True
    use_microstructure: bool = True
    use_sentiment: bool = True
    use_temporal: bool = True
    use_regime: bool = True
    use_options: bool = True
    
    # Feature engineering params
    lookback_periods: List[int] = None  # [5, 20, 60]
    normalize: bool = True
    remove_correlated: bool = True
    correlation_threshold: float = 0.95
    
    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [5, 20, 60]


class EnhancedFeatureBuilder:
    """
    Production-grade feature builder that creates 150+ features from:
    - Market data (OHLCV)
    - Hedge Engine v3.0 outputs (elasticity, movement_energy, etc.)
    - Options chain data
    - Sentiment signals
    - Temporal patterns
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = FeatureConfig()
        self.config = config
        self.feature_names: List[str] = []
        self.scaler = None  # Will be StandardScaler
        self._feature_stats = {}  # For monitoring drift

    @staticmethod
    def build_from_ledger(ledger_path: Path) -> pd.DataFrame:
        """Create a feature dataframe from ledger.jsonl records."""

        if not ledger_path.exists():
            raise FileNotFoundError(f"Ledger not found at {ledger_path}")

        rows: List[Dict] = []
        with open(ledger_path, "r") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                hedge = payload.get("hedge_snapshot") or {}
                liquidity = payload.get("liquidity_snapshot") or {}
                sentiment = payload.get("sentiment_snapshot") or {}
                elasticity = payload.get("elasticity_snapshot") or {}
                consensus = payload.get("consensus") or {}
                trade_ideas = payload.get("trade_ideas") or []

                rows.append(
                    {
                        "timestamp": payload.get("timestamp"),
                        "symbol": payload.get("symbol"),
                        "hedge_elasticity": hedge.get("elasticity"),
                        "movement_energy": hedge.get("movement_energy"),
                        "energy_asymmetry": hedge.get("energy_asymmetry"),
                        "pressure_net": hedge.get("pressure_net"),
                        "dealer_gamma_sign": hedge.get("dealer_gamma_sign"),
                        "hedge_confidence": hedge.get("confidence"),
                        "liquidity_score": liquidity.get("liquidity_score"),
                        "bid_ask_spread": liquidity.get("bid_ask_spread"),
                        "volume": liquidity.get("volume"),
                        "sentiment_score": sentiment.get("sentiment_score"),
                        "flow_sentiment": sentiment.get("flow_sentiment"),
                        "news_sentiment": sentiment.get("news_sentiment"),
                        "technical_sentiment": sentiment.get("technical_sentiment"),
                        "elasticity_vol": elasticity.get("volatility"),
                        "volatility_regime": elasticity.get("volatility_regime"),
                        "trend_strength": elasticity.get("trend_strength"),
                        "consensus_confidence": consensus.get("confidence"),
                        "consensus_direction": consensus.get("direction"),
                        "trade_count": len(trade_ideas),
                    }
                )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df["target_regime"] = df["consensus_direction"].shift(-1)
        return df.dropna(subset=["timestamp"])
        
    def build(
        self,
        market_data: pd.DataFrame,
        hedge_output: Optional[Dict] = None,
        liquidity_output: Optional[Dict] = None,
        sentiment_output: Optional[Dict] = None,
        options_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Build comprehensive feature matrix
        
        Args:
            market_data: DataFrame with columns [open, high, low, close, volume]
            hedge_output: Dict from Hedge Engine v3.0
            liquidity_output: Dict from Liquidity Engine
            sentiment_output: Dict from Sentiment Engine
            options_data: DataFrame with options chain data
            
        Returns:
            DataFrame with shape (len(market_data), n_features)
        """
        logger.info("Building features...")
        features_list = []
        
        # 1. Technical Indicators
        if self.config.use_technical:
            technical = self._build_technical_features(market_data)
            features_list.append(technical)
            logger.debug(f"Technical features: {technical.shape[1]} features")
        
        # 2. Hedge Engine Features
        if self.config.use_hedge_engine and hedge_output:
            hedge_features = self._build_hedge_features(hedge_output, len(market_data))
            features_list.append(hedge_features)
            logger.debug(f"Hedge features: {hedge_features.shape[1]} features")
        
        # 3. Microstructure Features
        if self.config.use_microstructure:
            microstructure = self._build_microstructure_features(market_data)
            features_list.append(microstructure)
            logger.debug(f"Microstructure features: {microstructure.shape[1]} features")
        
        # 4. Sentiment Features
        if self.config.use_sentiment and sentiment_output:
            sentiment_features = self._build_sentiment_features(
                sentiment_output, len(market_data)
            )
            features_list.append(sentiment_features)
            logger.debug(f"Sentiment features: {sentiment_features.shape[1]} features")
        
        # 5. Temporal Features
        if self.config.use_temporal:
            temporal = self._build_temporal_features(market_data)
            features_list.append(temporal)
            logger.debug(f"Temporal features: {temporal.shape[1]} features")
        
        # 6. Regime Features
        if self.config.use_regime:
            regime = self._build_regime_features(market_data, hedge_output)
            features_list.append(regime)
            logger.debug(f"Regime features: {regime.shape[1]} features")
        
        # 7. Options Features
        if self.config.use_options and options_data is not None:
            options_features = self._build_options_features(options_data)
            features_list.append(options_features)
            logger.debug(f"Options features: {options_features.shape[1]} features")
        
        # Concatenate all features
        feature_matrix = pd.concat(features_list, axis=1)
        
        # Store feature names
        self.feature_names = feature_matrix.columns.tolist()
        
        # Post-processing
        feature_matrix = self._post_process_features(feature_matrix)
        
        logger.info(f"Built feature matrix: {feature_matrix.shape}")
        return feature_matrix
    
    def _build_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build 50+ technical indicator features"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['high_low_range'] = (df['high'] - df['low']) / df['close']
        features['close_open_range'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages (multiple periods)
        for period in [5, 10, 20, 50, 200]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}']
        
        # Volatility indicators
        for period in [5, 20, 60]:
            features[f'std_{period}'] = df['close'].pct_change().rolling(period).std()
            features[f'realized_vol_{period}'] = features[f'std_{period}'] * np.sqrt(252)
            
            # Bollinger Bands
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = sma + 2 * std
            features[f'bb_lower_{period}'] = sma - 2 * std
            features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - 
                                               features[f'bb_lower_{period}']) / sma
            features[f'bb_position_{period}'] = (df['close'] - features[f'bb_lower_{period}']) / \
                                                 (features[f'bb_upper_{period}'] - 
                                                  features[f'bb_lower_{period}'])
        
        # Momentum indicators
        # RSI
        for period in [14, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Stochastic Oscillator
        for period in [14, 28]:
            low_min = df['low'].rolling(period).min()
            high_max = df['high'].rolling(period).max()
            features[f'stoch_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            features[f'stoch_signal_{period}'] = features[f'stoch_{period}'].rolling(3).mean()
        
        # Williams %R
        for period in [14, 28]:
            high_max = df['high'].rolling(period).max()
            low_min = df['low'].rolling(period).min()
            features[f'williams_r_{period}'] = -100 * (high_max - df['close']) / \
                                                (high_max - low_min)
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = df['close'].pct_change(period)
        
        # CCI (Commodity Channel Index)
        for period in [20]:
            tp = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
            features[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad)
        
        # Volume indicators
        features['volume_change'] = df['volume'].pct_change()
        features['volume_ma_5'] = df['volume'].rolling(5).mean()
        features['volume_ma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_ma_20']
        
        # OBV (On-Balance Volume)
        features['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        features['obv_ma_5'] = features['obv'].rolling(5).mean()
        
        # MFI (Money Flow Index)
        for period in [14]:
            tp = (df['high'] + df['low'] + df['close']) / 3
            raw_money_flow = tp * df['volume']
            
            positive_flow = raw_money_flow.where(tp > tp.shift(1), 0).rolling(period).sum()
            negative_flow = raw_money_flow.where(tp < tp.shift(1), 0).rolling(period).sum()
            
            mfi_ratio = positive_flow / negative_flow
            features[f'mfi_{period}'] = 100 - (100 / (1 + mfi_ratio))
        
        # ATR (Average True Range)
        for period in [14, 28]:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features[f'atr_{period}'] = true_range.rolling(period).mean()
            features[f'atr_pct_{period}'] = features[f'atr_{period}'] / df['close']
        
        # ADX (Average Directional Index)
        for period in [14]:
            high_diff = df['high'].diff()
            low_diff = -df['low'].diff()
            
            pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
            
            tr = pd.concat([
                df['high'] - df['low'],
                np.abs(df['high'] - df['close'].shift()),
                np.abs(df['low'] - df['close'].shift())
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(period).mean()
            pos_di = 100 * pos_dm.rolling(period).mean() / atr
            neg_di = 100 * neg_dm.rolling(period).mean() / atr
            
            dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
            features[f'adx_{period}'] = dx.rolling(period).mean()
        
        return features
    
    def _build_hedge_features(
        self, 
        hedge_output: Dict, 
        length: int
    ) -> pd.DataFrame:
        """
        Build features from Hedge Engine v3.0 outputs
        
        Hedge Engine v3.0 provides:
        - elasticity: Market stiffness
        - movement_energy: Energy required to move price
        - energy_asymmetry: Directional bias
        - pressure_up, pressure_down, pressure_net: Dealer hedge pressure
        - gamma_pressure, vanna_pressure, charm_pressure: Greek-based pressure
        - dealer_gamma_sign: Stabilizing/destabilizing
        """
        features = pd.DataFrame(index=range(length))
        
        # Direct features from Hedge Engine
        direct_features = [
            'elasticity',
            'movement_energy',
            'energy_asymmetry',
            'pressure_up',
            'pressure_down',
            'pressure_net',
            'gamma_pressure',
            'vanna_pressure',
            'charm_pressure',
            'dealer_gamma_sign'
        ]
        
        for feat in direct_features:
            if feat in hedge_output:
                # If hedge_output contains time series, use it
                if isinstance(hedge_output[feat], (list, np.ndarray, pd.Series)):
                    features[feat] = hedge_output[feat]
                else:
                    # If single value, broadcast
                    features[feat] = hedge_output[feat]
        
        # Derived features
        if 'elasticity' in features.columns:
            # Moving averages of elasticity
            for period in [5, 20, 60]:
                features[f'elasticity_ma_{period}'] = features['elasticity'].rolling(period).mean()
            
            # Elasticity momentum
            features['elasticity_momentum'] = features['elasticity'].diff()
            features['elasticity_momentum_5'] = features['elasticity'].diff(5)
            
            # Elasticity percentile (vs recent history)
            features['elasticity_percentile'] = features['elasticity'].rolling(100).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
        
        if 'movement_energy' in features.columns:
            # Energy momentum
            features['energy_momentum'] = features['movement_energy'].diff()
            features['energy_momentum_5'] = features['movement_energy'].diff(5)
            
            # Energy regime (categorized)
            features['energy_regime'] = pd.cut(
                features['movement_energy'],
                bins=[-np.inf, 0.2, 0.5, np.inf],
                labels=[0, 1, 2]  # low, medium, high
            ).astype(float)
        
        if 'energy_asymmetry' in features.columns:
            # Asymmetry momentum
            features['asymmetry_momentum'] = features['energy_asymmetry'].diff()
            
            # Asymmetry regime
            features['asymmetry_direction'] = np.sign(features['energy_asymmetry'])
        
        if 'pressure_net' in features.columns:
            # Pressure regime
            features['pressure_regime'] = pd.cut(
                features['pressure_net'],
                bins=[-np.inf, -0.5, 0.5, np.inf],
                labels=[0, 1, 2]  # bearish, neutral, bullish
            ).astype(float)
            
            # Pressure momentum
            features['pressure_momentum'] = features['pressure_net'].diff()
        
        if 'dealer_gamma_sign' in features.columns:
            # Gamma flip indicator (sign change)
            features['gamma_flip'] = (
                features['dealer_gamma_sign'] != features['dealer_gamma_sign'].shift()
            ).astype(int)
        
        # Interaction features (elasticity * energy)
        if 'elasticity' in features.columns and 'movement_energy' in features.columns:
            features['elasticity_energy_product'] = (
                features['elasticity'] * features['movement_energy']
            )
        
        # Ratio features
        if 'pressure_up' in features.columns and 'pressure_down' in features.columns:
            features['pressure_ratio'] = (
                features['pressure_up'] / (features['pressure_down'] + 1e-8)
            )
        
        return features
    
    def _build_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build market microstructure features"""
        features = pd.DataFrame(index=df.index)
        
        # VWAP (Volume Weighted Average Price)
        features['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        features['vwap_deviation'] = (df['close'] - features['vwap']) / features['vwap']
        
        # Volume profile
        for period in [20, 60]:
            features[f'volume_profile_{period}'] = (
                df['volume'].rolling(period).apply(
                    lambda x: np.percentile(x, 50)
                )
            )
        
        # Tick rule (simplified - needs tick data for real implementation)
        # Here we use close price changes as proxy
        features['tick_direction'] = np.sign(df['close'].diff())
        features['tick_direction_ma_5'] = features['tick_direction'].rolling(5).mean()
        
        # Order flow imbalance proxy (high-low as proxy for bid-ask pressure)
        features['ofi_proxy'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8)
        features['ofi_proxy_ma_5'] = features['ofi_proxy'].rolling(5).mean()
        
        return features
    
    def _build_sentiment_features(
        self, 
        sentiment_output: Dict, 
        length: int
    ) -> pd.DataFrame:
        """Build sentiment-based features"""
        features = pd.DataFrame(index=range(length))
        
        # Direct sentiment scores
        if 'overall_sentiment' in sentiment_output:
            features['sentiment_score'] = sentiment_output['overall_sentiment']
        
        if 'news_sentiment' in sentiment_output:
            features['news_sentiment'] = sentiment_output['news_sentiment']
        
        if 'social_sentiment' in sentiment_output:
            features['social_sentiment'] = sentiment_output['social_sentiment']
        
        # Derived features
        if 'sentiment_score' in features.columns:
            # Sentiment change rate
            features['sentiment_momentum'] = features['sentiment_score'].diff()
            features['sentiment_momentum_5'] = features['sentiment_score'].diff(5)
            
            # Sentiment volatility
            features['sentiment_volatility_20'] = (
                features['sentiment_score'].rolling(20).std()
            )
        
        return features
    
    def _build_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build time-based features with cyclical encoding"""
        features = pd.DataFrame(index=df.index)
        
        # Assume df.index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            # If timestamp column exists, use it
            if 'timestamp' in df.columns:
                dt = pd.to_datetime(df['timestamp'])
            else:
                # Create dummy datetime
                dt = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
        else:
            dt = df.index
        
        # Hour of day (cyclical encoding)
        hour = dt.hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week (cyclical)
        dayofweek = dt.dayofweek
        features['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / 7)
        features['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / 7)
        
        # Day of month (cyclical)
        day = dt.day
        features['day_sin'] = np.sin(2 * np.pi * day / 31)
        features['day_cos'] = np.cos(2 * np.pi * day / 31)
        
        # Month of year (cyclical)
        month = dt.month
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        # Market session indicators
        features['is_premarket'] = (hour < 9.5).astype(int)
        features['is_open'] = ((hour >= 9.5) & (hour < 16)).astype(int)
        features['is_afterhours'] = (hour >= 16).astype(int)
        
        # Week of month
        features['week_of_month'] = (dt.day - 1) // 7
        
        return features
    
    def _build_regime_features(
        self, 
        df: pd.DataFrame, 
        hedge_output: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Build regime classification features"""
        features = pd.DataFrame(index=df.index)
        
        # Volatility regime
        returns = df['close'].pct_change()
        vol_20 = returns.rolling(20).std()
        vol_percentile = vol_20.rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )
        
        features['vol_regime'] = pd.cut(
            vol_percentile,
            bins=[0, 0.33, 0.67, 1.0],
            labels=[0, 1, 2],  # low, medium, high
            include_lowest=True
        ).astype(float)
        
        # Trend regime
        sma_50 = df['close'].rolling(50).mean()
        df['close'].rolling(200).mean()
        
        # Strong uptrend: price > SMA50 > SMA200
        # Weak uptrend: price > SMA50, SMA50 < SMA200
        # Ranging: price near SMA50
        # Downtrend: price < SMA50
        features['trend_regime'] = 0  # Default: ranging
        features.loc[df['close'] > sma_50 * 1.02, 'trend_regime'] = 1  # Uptrend
        features.loc[df['close'] < sma_50 * 0.98, 'trend_regime'] = -1  # Downtrend
        
        # Use Hedge Engine regime if available
        if hedge_output and 'regime' in hedge_output:
            features['hedge_regime'] = hedge_output['regime']
        
        return features
    
    def _build_options_features(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """Build comprehensive options-specific features with multi-timeframe support.

        Supports both legacy format and enhanced MASSIVE options data format.
        """
        features = pd.DataFrame(index=range(len(options_df)))

        # Graph-derived features to capture strike/OI topology
        graph_feats = self._graph_metrics(options_df)
        for key, value in graph_feats.items():
            features[key] = value

        # =====================================================================
        # Core Put-Call Ratio Features
        # =====================================================================
        if 'put_volume' in options_df.columns and 'call_volume' in options_df.columns:
            features['pcr_volume'] = (
                options_df['put_volume'] / (options_df['call_volume'] + 1e-8)
            )
            # PCR moving averages
            for period in [5, 10, 20]:
                features[f'pcr_volume_ma_{period}'] = features['pcr_volume'].rolling(period).mean()
            # PCR momentum
            features['pcr_volume_momentum'] = features['pcr_volume'].diff()
            features['pcr_volume_momentum_5'] = features['pcr_volume'].diff(5)
            # PCR z-score
            pcr_mean = features['pcr_volume'].rolling(20).mean()
            pcr_std = features['pcr_volume'].rolling(20).std()
            features['pcr_volume_zscore'] = (features['pcr_volume'] - pcr_mean) / (pcr_std + 1e-8)

        if 'put_oi' in options_df.columns and 'call_oi' in options_df.columns:
            features['pcr_oi'] = (
                options_df['put_oi'] / (options_df['call_oi'] + 1e-8)
            )
            for period in [5, 10, 20]:
                features[f'pcr_oi_ma_{period}'] = features['pcr_oi'].rolling(period).mean()
            features['pcr_oi_momentum'] = features['pcr_oi'].diff()

        # PCR divergence (volume vs OI)
        if 'pcr_volume' in features.columns and 'pcr_oi' in features.columns:
            features['pcr_divergence'] = features['pcr_volume'] - features['pcr_oi']

        # =====================================================================
        # Implied Volatility Features
        # =====================================================================
        # IV Skew (legacy format)
        if 'iv_otm_put' in options_df.columns and 'iv_otm_call' in options_df.columns:
            features['iv_skew'] = options_df['iv_otm_put'] - options_df['iv_otm_call']

        # IV Skew (MASSIVE format)
        if 'iv_25d_put' in options_df.columns and 'iv_25d_call' in options_df.columns:
            features['iv_skew'] = options_df['iv_25d_put'] - options_df['iv_25d_call']
            features['iv_25d_put'] = options_df['iv_25d_put']
            features['iv_25d_call'] = options_df['iv_25d_call']

        # ATM IV and derivatives
        if 'atm_iv' in options_df.columns:
            features['atm_iv'] = options_df['atm_iv']
            for period in [5, 10, 20]:
                features[f'atm_iv_ma_{period}'] = features['atm_iv'].rolling(period).mean()
            features['iv_momentum'] = features['atm_iv'].diff()
            features['iv_momentum_5'] = features['atm_iv'].diff(5)
            features['iv_acceleration'] = features['iv_momentum'].diff()
            # IV percentile rank
            features['iv_percentile'] = features['atm_iv'].rolling(60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
            )
            # IV regime classification
            features['iv_regime'] = pd.cut(
                features['iv_percentile'],
                bins=[0, 0.25, 0.75, 1.0],
                labels=[0, 1, 2]  # Low, Normal, High
            ).astype(float)

        # IV Term Structure
        if 'iv_term_slope' in options_df.columns:
            features['iv_term_slope'] = options_df['iv_term_slope']
            # Contango (positive) vs backwardation (negative)
            features['iv_term_structure'] = np.sign(features['iv_term_slope'])

        # =====================================================================
        # Max Pain Features
        # =====================================================================
        if 'max_pain_strike' in options_df.columns:
            features['max_pain'] = options_df['max_pain_strike']
        elif 'max_pain' in options_df.columns:
            features['max_pain'] = options_df['max_pain']

        if 'max_pain' in features.columns:
            if 'current_price' in options_df.columns:
                features['price_to_max_pain'] = (
                    options_df['current_price'] / (features['max_pain'] + 1e-8)
                )
                # Distance to max pain (normalized)
                features['max_pain_distance'] = (
                    (options_df['current_price'] - features['max_pain']) /
                    options_df['current_price']
                )
                # Max pain attraction (inverse distance)
                features['max_pain_attraction'] = 1.0 / (
                    np.abs(features['max_pain_distance']) + 0.01
                )

        # =====================================================================
        # Gamma Exposure (GEX) Features
        # =====================================================================
        gex_col = 'total_gamma' if 'total_gamma' in options_df.columns else 'gex_total'
        if gex_col in options_df.columns:
            features['gex'] = options_df[gex_col]
            for period in [5, 10, 20]:
                features[f'gex_ma_{period}'] = features['gex'].rolling(period).mean()
            features['gex_momentum'] = features['gex'].diff()
            features['gex_momentum_5'] = features['gex'].diff(5)
            # GEX sign (positive = dealer long gamma = stabilizing)
            features['gex_sign'] = np.sign(features['gex'])
            # GEX flip indicator
            features['gex_flip'] = (
                features['gex_sign'] != features['gex_sign'].shift()
            ).astype(int)
            # GEX z-score
            gex_mean = features['gex'].rolling(20).mean()
            gex_std = features['gex'].rolling(20).std()
            features['gex_zscore'] = (features['gex'] - gex_mean) / (gex_std + 1e-8)

        # GEX breakdown (calls vs puts)
        if 'gex_calls' in options_df.columns and 'gex_puts' in options_df.columns:
            features['gex_calls'] = options_df['gex_calls']
            features['gex_puts'] = options_df['gex_puts']
            features['gex_ratio'] = features['gex_calls'] / (np.abs(features['gex_puts']) + 1e-8)

        # =====================================================================
        # Greeks Aggregates (Net Exposure)
        # =====================================================================
        if 'net_delta' in options_df.columns:
            features['net_delta'] = options_df['net_delta']
            features['net_delta_ma_5'] = features['net_delta'].rolling(5).mean()
            features['delta_momentum'] = features['net_delta'].diff()

        if 'net_gamma' in options_df.columns:
            features['net_gamma'] = options_df['net_gamma']
            features['net_gamma_ma_5'] = features['net_gamma'].rolling(5).mean()

        if 'net_vega' in options_df.columns:
            features['net_vega'] = options_df['net_vega']
            features['vega_momentum'] = features['net_vega'].diff()

        if 'net_theta' in options_df.columns:
            features['net_theta'] = options_df['net_theta']

        # =====================================================================
        # Dealer Positioning Features
        # =====================================================================
        if 'dealer_gamma_exposure' in options_df.columns:
            features['dealer_gamma'] = options_df['dealer_gamma_exposure']
            features['dealer_gamma_sign'] = np.sign(features['dealer_gamma'])
            # Dealer gamma flip
            features['dealer_gamma_flip'] = (
                features['dealer_gamma_sign'] != features['dealer_gamma_sign'].shift()
            ).astype(int)

        if 'dealer_delta_exposure' in options_df.columns:
            features['dealer_delta'] = options_df['dealer_delta_exposure']

        # =====================================================================
        # Strike Distribution Features
        # =====================================================================
        if 'high_gamma_strike' in options_df.columns:
            features['high_gamma_strike'] = options_df['high_gamma_strike']
            if 'current_price' in options_df.columns:
                features['price_to_high_gamma'] = (
                    options_df['current_price'] / (features['high_gamma_strike'] + 1e-8)
                )

        if 'high_oi_call_strike' in options_df.columns:
            features['high_oi_call_strike'] = options_df['high_oi_call_strike']

        if 'high_oi_put_strike' in options_df.columns:
            features['high_oi_put_strike'] = options_df['high_oi_put_strike']

        # Strike clustering (call vs put)
        if 'high_oi_call_strike' in features.columns and 'high_oi_put_strike' in features.columns:
            features['oi_strike_spread'] = (
                features['high_oi_call_strike'] - features['high_oi_put_strike']
            )

        # =====================================================================
        # Multi-Timeframe Features (if available from MASSIVE)
        # =====================================================================
        for tf_suffix in ['_1min', '_5min', '_15min', '_1hour', '_1day']:
            pcr_col = f'pcr_volume{tf_suffix}'
            if pcr_col in options_df.columns:
                features[pcr_col] = options_df[pcr_col]

            gex_col = f'gex{tf_suffix}'
            if gex_col in options_df.columns:
                features[gex_col] = options_df[gex_col]

            iv_col = f'atm_iv{tf_suffix}'
            if iv_col in options_df.columns:
                features[iv_col] = options_df[iv_col]

        # =====================================================================
        # Composite / Interaction Features
        # =====================================================================
        # PCR × IV (high PCR + high IV = fear)
        if 'pcr_volume' in features.columns and 'atm_iv' in features.columns:
            features['fear_index'] = features['pcr_volume'] * features['atm_iv']

        # GEX × IV interaction
        if 'gex' in features.columns and 'atm_iv' in features.columns:
            features['gex_iv_product'] = features['gex'] * features['atm_iv']

        # Options sentiment score (composite)
        sentiment_score = 0
        n_components = 0
        if 'pcr_volume_zscore' in features.columns:
            sentiment_score -= features['pcr_volume_zscore']  # High PCR = bearish
            n_components += 1
        if 'gex_zscore' in features.columns:
            sentiment_score += features['gex_zscore']  # High GEX = bullish
            n_components += 1
        if 'iv_percentile' in features.columns:
            sentiment_score -= (features['iv_percentile'] - 0.5) * 2  # High IV = bearish
            n_components += 1
        if n_components > 0:
            features['options_sentiment'] = sentiment_score / n_components

        return features

    def _graph_metrics(self, options_df: pd.DataFrame) -> Dict[str, float]:
        """Construct strike graph centrality metrics for OI diffusion."""

        if options_df.empty or 'strike' not in options_df.columns:
            return {}

        graph = nx.Graph()
        strikes = options_df['strike'].to_numpy()
        oi_series = options_df.get('open_interest', pd.Series([1.0] * len(options_df)))

        for strike, oi in zip(strikes, oi_series):
            graph.add_node(float(strike), weight=float(oi))

        sorted_nodes = sorted(zip(strikes, oi_series), key=lambda x: x[0])
        for (strike_a, oi_a), (strike_b, oi_b) in zip(sorted_nodes, sorted_nodes[1:]):
            proximity = 1.0 / (abs(float(strike_a) - float(strike_b)) + 1e-3)
            similarity = 1.0 / (abs(float(oi_a) - float(oi_b)) + 1.0)
            graph.add_edge(float(strike_a), float(strike_b), weight=proximity * similarity)

        centrality = nx.degree_centrality(graph)
        values = list(centrality.values())
        if not values:
            return {}

        return {
            "graph_centrality_mean": float(np.mean(values)),
            "graph_centrality_max": float(np.max(values)),
            "graph_density": float(nx.density(graph)),
        }
    
    def _post_process_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Post-processing: handle NaN, remove correlated, normalize"""
        
        # Handle missing values
        # Forward fill (for time series continuity)
        features = features.fillna(method='ffill')
        # Backward fill for any remaining NaN at start
        features = features.fillna(method='bfill')
        # Fill any remaining with 0
        features = features.fillna(0)
        
        # Remove highly correlated features
        if self.config.remove_correlated:
            features = self._remove_correlated_features(
                features, 
                threshold=self.config.correlation_threshold
            )
        
        # Normalization (if enabled)
        if self.config.normalize:
            features = self._normalize_features(features)
        
        # Replace inf with large numbers
        features = features.replace([np.inf, -np.inf], [1e8, -1e8])
        
        return features
    
    def _remove_correlated_features(
        self, 
        features: pd.DataFrame, 
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """Remove features with correlation > threshold"""
        
        # Compute correlation matrix
        corr_matrix = features.corr().abs()
        
        # Select upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation > threshold
        to_drop = [
            column for column in upper.columns 
            if any(upper[column] > threshold)
        ]
        
        logger.info(f"Removing {len(to_drop)} correlated features")
        return features.drop(columns=to_drop)
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using StandardScaler"""
        from sklearn.preprocessing import StandardScaler
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            normalized = self.scaler.fit_transform(features)
        else:
            normalized = self.scaler.transform(features)
        
        return pd.DataFrame(
            normalized, 
            index=features.index, 
            columns=features.columns
        )
    
    def save_scaler(self, path: str):
        """Save fitted scaler for production use"""
        import joblib
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler saved to {path}")
    
    def load_scaler(self, path: str):
        """Load pre-fitted scaler"""
        import joblib
        self.scaler = joblib.load(path)
        logger.info(f"Scaler loaded from {path}")


# Example usage
if __name__ == "__main__":
    # Configure feature builder
    config = FeatureConfig(
        use_technical=True,
        use_hedge_engine=True,
        use_microstructure=True,
        use_sentiment=True,
        use_temporal=True,
        use_regime=True,
        normalize=True
    )
    
    builder = EnhancedFeatureBuilder(config)
    
    # Mock data for testing
    np.random.seed(42)
    n = 1000
    
    market_data = pd.DataFrame({
        'open': np.random.randn(n).cumsum() + 100,
        'high': np.random.randn(n).cumsum() + 102,
        'low': np.random.randn(n).cumsum() + 98,
        'close': np.random.randn(n).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    hedge_output = {
        'elasticity': np.random.rand(n) * 0.5 + 0.1,
        'movement_energy': np.random.rand(n) * 0.3,
        'energy_asymmetry': np.random.randn(n) * 0.2,
        'pressure_net': np.random.randn(n) * 0.5,
        'dealer_gamma_sign': np.random.choice([-1, 1], n)
    }
    
    # Build features
    features = builder.build(
        market_data=market_data,
        hedge_output=hedge_output
    )
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature names: {builder.feature_names[:10]}...")
    print(f"\nSample features:\n{features.head()}")
