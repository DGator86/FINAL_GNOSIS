"""
Gnosis Alpha - Feature Engineering

Extracts ML-ready features from market data for short-term prediction.
Focuses on features that are predictive for 0-7 day horizons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    import pandas as pd
    YFINANCE_AVAILABLE = True
except ImportError:
    logger.warning("yfinance/pandas not available")
    YFINANCE_AVAILABLE = False


@dataclass
class FeatureSet:
    """
    Complete feature set for ML model input.
    
    Features are organized into categories:
    - Price/trend features
    - Momentum features
    - Volatility features
    - Volume features
    - Pattern features
    """
    symbol: str
    timestamp: datetime
    
    # Raw features dict for model input
    features: Dict[str, float] = field(default_factory=dict)
    
    # Feature metadata
    feature_names: List[str] = field(default_factory=list)
    
    # Data quality
    has_sufficient_data: bool = True
    missing_features: List[str] = field(default_factory=list)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array in consistent order."""
        return np.array([self.features.get(f, 0.0) for f in self.feature_names])
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "features": self.features,
            "has_sufficient_data": self.has_sufficient_data,
        }


class AlphaFeatureEngine:
    """
    Feature engineering for Alpha ML models.
    
    Extracts ~50 features optimized for short-term directional prediction:
    - Technical indicators (SMA, EMA, RSI, MACD, etc.)
    - Price action patterns
    - Volatility metrics
    - Volume analysis
    - Relative strength vs market
    """
    
    # Feature names for consistent ordering
    FEATURE_NAMES = [
        # Price/Trend (12 features)
        "price_vs_sma5", "price_vs_sma10", "price_vs_sma20", "price_vs_sma50",
        "sma5_vs_sma20", "sma10_vs_sma20", "sma20_vs_sma50",
        "price_pct_1d", "price_pct_3d", "price_pct_5d", "price_pct_10d", "price_pct_20d",
        
        # Momentum (10 features)
        "rsi_14", "rsi_7", "rsi_normalized",
        "macd", "macd_signal", "macd_histogram", "macd_histogram_slope",
        "stoch_k", "stoch_d", "stoch_signal",
        
        # Volatility (8 features)
        "atr_14", "atr_normalized", "atr_percentile",
        "bb_position", "bb_width", "bb_squeeze",
        "volatility_5d", "volatility_20d",
        
        # Volume (8 features)
        "volume_ratio_1d", "volume_ratio_5d", "volume_sma_ratio",
        "volume_trend", "obv_slope", "obv_normalized",
        "price_volume_trend", "volume_breakout",
        
        # Pattern/Structure (8 features)
        "higher_highs", "higher_lows", "trend_strength",
        "gap_up", "gap_down", "range_position",
        "support_distance", "resistance_distance",
        
        # Market relative (4 features)
        "relative_strength_spy", "beta_20d", "correlation_spy", "outperformance_5d",
    ]
    
    def __init__(self, lookback_days: int = 100):
        """
        Initialize feature engine.
        
        Args:
            lookback_days: Days of history needed for feature calculation
        """
        self.lookback_days = lookback_days
        self._spy_cache: Optional[pd.DataFrame] = None
        self._spy_cache_date: Optional[datetime] = None
    
    def extract(self, symbol: str, as_of: Optional[datetime] = None) -> FeatureSet:
        """
        Extract features for a symbol.
        
        Args:
            symbol: Stock symbol
            as_of: Point-in-time for features (default: now)
            
        Returns:
            FeatureSet with all features
        """
        timestamp = as_of or datetime.now()
        feature_set = FeatureSet(
            symbol=symbol,
            timestamp=timestamp,
            feature_names=self.FEATURE_NAMES.copy(),
        )
        
        if not YFINANCE_AVAILABLE:
            feature_set.has_sufficient_data = False
            feature_set.missing_features = self.FEATURE_NAMES.copy()
            return feature_set
        
        try:
            # Get historical data
            df = self._get_price_data(symbol, self.lookback_days)
            
            if df is None or len(df) < 50:
                feature_set.has_sufficient_data = False
                feature_set.missing_features = self.FEATURE_NAMES.copy()
                return feature_set
            
            # Get SPY for relative features
            spy_df = self._get_spy_data()
            
            # Extract all feature categories
            features = {}
            features.update(self._extract_price_trend_features(df))
            features.update(self._extract_momentum_features(df))
            features.update(self._extract_volatility_features(df))
            features.update(self._extract_volume_features(df))
            features.update(self._extract_pattern_features(df))
            features.update(self._extract_market_relative_features(df, spy_df))
            
            feature_set.features = features
            
            # Check for missing
            for name in self.FEATURE_NAMES:
                if name not in features or np.isnan(features.get(name, 0)):
                    feature_set.missing_features.append(name)
                    features[name] = 0.0  # Fill with 0 for missing
            
            feature_set.has_sufficient_data = len(feature_set.missing_features) < 10
            
        except Exception as e:
            logger.error(f"Error extracting features for {symbol}: {e}")
            feature_set.has_sufficient_data = False
            feature_set.missing_features = self.FEATURE_NAMES.copy()
        
        return feature_set
    
    def extract_batch(
        self,
        symbols: List[str],
        as_of: Optional[datetime] = None,
    ) -> Dict[str, FeatureSet]:
        """Extract features for multiple symbols."""
        return {symbol: self.extract(symbol, as_of) for symbol in symbols}
    
    def _get_price_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Get historical price data."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d")
            return df if not df.empty else None
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _get_spy_data(self) -> Optional[pd.DataFrame]:
        """Get SPY data for relative calculations (with caching)."""
        now = datetime.now()
        
        # Use cache if less than 1 hour old
        if (self._spy_cache is not None and 
            self._spy_cache_date and 
            (now - self._spy_cache_date).seconds < 3600):
            return self._spy_cache
        
        try:
            spy = yf.Ticker("SPY")
            self._spy_cache = spy.history(period=f"{self.lookback_days}d")
            self._spy_cache_date = now
            return self._spy_cache
        except Exception:
            return None
    
    def _extract_price_trend_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract price and trend features."""
        features = {}
        close = df['Close']
        current_price = close.iloc[-1]
        
        # Price vs SMAs
        for period in [5, 10, 20, 50]:
            sma = close.rolling(period).mean().iloc[-1]
            features[f"price_vs_sma{period}"] = (current_price - sma) / sma if sma else 0
        
        # SMA crossovers (normalized)
        sma5 = close.rolling(5).mean().iloc[-1]
        sma10 = close.rolling(10).mean().iloc[-1]
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        
        features["sma5_vs_sma20"] = (sma5 - sma20) / sma20 if sma20 else 0
        features["sma10_vs_sma20"] = (sma10 - sma20) / sma20 if sma20 else 0
        features["sma20_vs_sma50"] = (sma20 - sma50) / sma50 if sma50 else 0
        
        # Price changes over different periods
        for days in [1, 3, 5, 10, 20]:
            if len(close) > days:
                features[f"price_pct_{days}d"] = (
                    (current_price - close.iloc[-days-1]) / close.iloc[-days-1]
                )
            else:
                features[f"price_pct_{days}d"] = 0
        
        return features
    
    def _extract_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract momentum indicators."""
        features = {}
        close = df['Close']
        
        # RSI
        for period in [7, 14]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features[f"rsi_{period}"] = rsi.iloc[-1] / 100 if not np.isnan(rsi.iloc[-1]) else 0.5
        
        # RSI normalized (-1 to 1 scale)
        features["rsi_normalized"] = (features["rsi_14"] - 0.5) * 2
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        # Normalize MACD by price
        price = close.iloc[-1]
        features["macd"] = (macd_line.iloc[-1] / price) * 100 if price else 0
        features["macd_signal"] = (signal_line.iloc[-1] / price) * 100 if price else 0
        features["macd_histogram"] = (histogram.iloc[-1] / price) * 100 if price else 0
        
        # MACD histogram slope
        if len(histogram) >= 3:
            features["macd_histogram_slope"] = histogram.iloc[-1] - histogram.iloc[-3]
        else:
            features["macd_histogram_slope"] = 0
        
        # Stochastic
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        stoch_k = 100 * (close - low_14) / (high_14 - low_14)
        stoch_d = stoch_k.rolling(3).mean()
        
        features["stoch_k"] = stoch_k.iloc[-1] / 100 if not np.isnan(stoch_k.iloc[-1]) else 0.5
        features["stoch_d"] = stoch_d.iloc[-1] / 100 if not np.isnan(stoch_d.iloc[-1]) else 0.5
        features["stoch_signal"] = 1 if features["stoch_k"] > features["stoch_d"] else -1
        
        return features
    
    def _extract_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract volatility features."""
        features = {}
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean().iloc[-1]
        
        features["atr_14"] = atr_14
        features["atr_normalized"] = atr_14 / close.iloc[-1] if close.iloc[-1] else 0
        
        # ATR percentile (where current ATR ranks historically)
        atr_series = tr.rolling(14).mean()
        features["atr_percentile"] = (
            (atr_series < atr_14).sum() / len(atr_series)
            if len(atr_series) > 0 else 0.5
        )
        
        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        
        current_price = close.iloc[-1]
        bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / sma20.iloc[-1] if sma20.iloc[-1] else 0
        
        # BB position: -1 (at lower) to +1 (at upper)
        if bb_upper.iloc[-1] != bb_lower.iloc[-1]:
            features["bb_position"] = (
                2 * (current_price - bb_lower.iloc[-1]) / 
                (bb_upper.iloc[-1] - bb_lower.iloc[-1]) - 1
            )
        else:
            features["bb_position"] = 0
        
        features["bb_width"] = bb_width
        
        # BB squeeze (narrow bands = potential breakout)
        bb_width_20 = ((bb_upper - bb_lower) / sma20).rolling(20).mean().iloc[-1]
        features["bb_squeeze"] = 1 if bb_width < bb_width_20 * 0.8 else 0
        
        # Realized volatility
        returns = close.pct_change()
        features["volatility_5d"] = returns.tail(5).std() * np.sqrt(252)
        features["volatility_20d"] = returns.tail(20).std() * np.sqrt(252)
        
        return features
    
    def _extract_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract volume features."""
        features = {}
        volume = df['Volume']
        close = df['Close']
        
        # Volume ratios
        avg_vol_20 = volume.rolling(20).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        features["volume_ratio_1d"] = current_vol / avg_vol_20 if avg_vol_20 else 1
        features["volume_ratio_5d"] = volume.tail(5).mean() / avg_vol_20 if avg_vol_20 else 1
        features["volume_sma_ratio"] = current_vol / volume.rolling(50).mean().iloc[-1] if volume.rolling(50).mean().iloc[-1] else 1
        
        # Volume trend (increasing or decreasing)
        vol_5 = volume.tail(5).mean()
        vol_20 = volume.tail(20).mean()
        features["volume_trend"] = (vol_5 - vol_20) / vol_20 if vol_20 else 0
        
        # OBV (On-Balance Volume)
        obv = (np.sign(close.diff()) * volume).cumsum()
        obv_slope = (obv.iloc[-1] - obv.iloc[-5]) / obv.iloc[-5] if obv.iloc[-5] != 0 else 0
        features["obv_slope"] = np.clip(obv_slope, -1, 1)
        features["obv_normalized"] = (obv.iloc[-1] - obv.rolling(20).mean().iloc[-1]) / obv.rolling(20).std().iloc[-1] if obv.rolling(20).std().iloc[-1] else 0
        
        # Price-Volume trend
        price_change = close.pct_change().iloc[-1]
        vol_change = volume.pct_change().iloc[-1]
        features["price_volume_trend"] = 1 if (price_change > 0 and vol_change > 0.5) else (-1 if (price_change < 0 and vol_change > 0.5) else 0)
        
        # Volume breakout
        features["volume_breakout"] = 1 if features["volume_ratio_1d"] > 2 else 0
        
        return features
    
    def _extract_pattern_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract price pattern features."""
        features = {}
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Higher highs / higher lows (trend quality)
        highs = high.tail(10).values
        lows = low.tail(10).values
        
        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        
        features["higher_highs"] = (higher_highs - 4.5) / 4.5  # Normalize around 0
        features["higher_lows"] = (higher_lows - 4.5) / 4.5
        
        # Trend strength (ADX-like)
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean().iloc[-1]
        features["trend_strength"] = adx / 100 if not np.isnan(adx) else 0.25
        
        # Gap detection
        prev_close = close.iloc[-2] if len(close) > 1 else close.iloc[-1]
        open_price = df['Open'].iloc[-1]
        features["gap_up"] = 1 if open_price > prev_close * 1.01 else 0
        features["gap_down"] = 1 if open_price < prev_close * 0.99 else 0
        
        # Range position (where in recent range)
        high_20 = high.tail(20).max()
        low_20 = low.tail(20).min()
        current_price = close.iloc[-1]
        
        if high_20 != low_20:
            features["range_position"] = (current_price - low_20) / (high_20 - low_20)
        else:
            features["range_position"] = 0.5
        
        # Support/Resistance distance (simplified)
        features["support_distance"] = (current_price - low_20) / current_price
        features["resistance_distance"] = (high_20 - current_price) / current_price
        
        return features
    
    def _extract_market_relative_features(
        self,
        df: pd.DataFrame,
        spy_df: Optional[pd.DataFrame],
    ) -> Dict[str, float]:
        """Extract market-relative features."""
        features = {
            "relative_strength_spy": 0,
            "beta_20d": 1,
            "correlation_spy": 0,
            "outperformance_5d": 0,
        }
        
        if spy_df is None or spy_df.empty:
            return features
        
        try:
            stock_close = df['Close']
            spy_close = spy_df['Close']
            
            # Align dates
            common_dates = stock_close.index.intersection(spy_close.index)
            if len(common_dates) < 20:
                return features
            
            stock_aligned = stock_close.loc[common_dates]
            spy_aligned = spy_close.loc[common_dates]
            
            stock_returns = stock_aligned.pct_change().dropna()
            spy_returns = spy_aligned.pct_change().dropna()
            
            # Relative strength
            stock_20d_return = (stock_aligned.iloc[-1] / stock_aligned.iloc[-20] - 1)
            spy_20d_return = (spy_aligned.iloc[-1] / spy_aligned.iloc[-20] - 1)
            features["relative_strength_spy"] = stock_20d_return - spy_20d_return
            
            # Beta (20-day)
            if len(stock_returns) >= 20 and spy_returns.var() > 0:
                cov = stock_returns.tail(20).cov(spy_returns.tail(20))
                var = spy_returns.tail(20).var()
                features["beta_20d"] = cov / var if var else 1
            
            # Correlation
            if len(stock_returns) >= 20:
                features["correlation_spy"] = stock_returns.tail(20).corr(spy_returns.tail(20))
            
            # 5-day outperformance
            stock_5d = (stock_aligned.iloc[-1] / stock_aligned.iloc[-5] - 1) if len(stock_aligned) >= 5 else 0
            spy_5d = (spy_aligned.iloc[-1] / spy_aligned.iloc[-5] - 1) if len(spy_aligned) >= 5 else 0
            features["outperformance_5d"] = stock_5d - spy_5d
            
        except Exception as e:
            logger.debug(f"Error calculating market relative features: {e}")
        
        return features
    
    def build_training_dataset(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        forward_days: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
        """
        Build training dataset with features and labels.
        
        Args:
            symbol: Stock symbol
            start_date: Start of training period
            end_date: End of training period
            forward_days: Days ahead for return calculation (label)
            
        Returns:
            Tuple of (features array, labels array, dates)
        """
        if not YFINANCE_AVAILABLE:
            return np.array([]), np.array([]), []
        
        try:
            # Get extended data (need extra days for forward returns)
            ticker = yf.Ticker(symbol)
            total_days = (end_date - start_date).days + forward_days + self.lookback_days + 50
            df = ticker.history(period=f"{total_days}d")
            
            if df.empty or len(df) < self.lookback_days + forward_days:
                return np.array([]), np.array([]), []
            
            features_list = []
            labels_list = []
            dates_list = []
            
            # Generate features for each day
            for i in range(self.lookback_days, len(df) - forward_days):
                current_date = df.index[i]
                
                # Check if within date range
                if current_date.date() < start_date.date() or current_date.date() > end_date.date():
                    continue
                
                # Get features using data up to this point
                df_slice = df.iloc[:i+1]
                feature_dict = {}
                feature_dict.update(self._extract_price_trend_features(df_slice))
                feature_dict.update(self._extract_momentum_features(df_slice))
                feature_dict.update(self._extract_volatility_features(df_slice))
                feature_dict.update(self._extract_volume_features(df_slice))
                feature_dict.update(self._extract_pattern_features(df_slice))
                
                # Simplified market features (skip for speed)
                feature_dict.update({
                    "relative_strength_spy": 0,
                    "beta_20d": 1,
                    "correlation_spy": 0,
                    "outperformance_5d": 0,
                })
                
                # Build feature vector
                feature_vec = np.array([
                    feature_dict.get(f, 0.0) for f in self.FEATURE_NAMES
                ])
                
                # Replace NaN with 0
                feature_vec = np.nan_to_num(feature_vec, 0.0)
                
                # Calculate label (forward return)
                current_price = df['Close'].iloc[i]
                future_price = df['Close'].iloc[i + forward_days]
                forward_return = (future_price - current_price) / current_price
                
                features_list.append(feature_vec)
                labels_list.append(forward_return)
                dates_list.append(current_date)
            
            return (
                np.array(features_list),
                np.array(labels_list),
                dates_list,
            )
            
        except Exception as e:
            logger.error(f"Error building training dataset for {symbol}: {e}")
            return np.array([]), np.array([]), []
