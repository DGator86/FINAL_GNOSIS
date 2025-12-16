"""MASSIVE Options Data Integration for ML Training and Optimization.

Provides a unified interface for feeding multi-timeframe historical options data
from MASSIVE into the GNOSIS ML training pipeline, including:
- LSTM time series models
- XGBoost ensemble classifiers
- Feature engineering with options data
- Hyperparameter optimization with options features
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class MassiveOptionsMLIntegration:
    """Integrates MASSIVE options data with GNOSIS ML pipeline.

    Provides:
    - Historical options data fetching for training
    - Multi-timeframe feature generation
    - Integration with EnhancedFeatureBuilder
    - Training data preparation for LSTM and XGBoost
    - Options-enhanced feature importance analysis
    """

    # Feature categories for options data
    FEATURE_CATEGORIES = {
        "volume": ["call_volume", "put_volume", "total_volume", "pcr_volume"],
        "oi": ["call_oi", "put_oi", "total_oi", "pcr_oi"],
        "iv": ["atm_iv", "iv_25d_put", "iv_25d_call", "iv_skew"],
        "greeks": ["net_delta", "net_gamma", "net_vega", "net_theta"],
        "gex": ["gex_total", "gex_calls", "gex_puts"],
        "levels": ["max_pain", "high_gamma_strike", "price_to_max_pain"],
        "dealer": ["dealer_gamma_exposure", "dealer_delta_exposure"],
    }

    def __init__(
        self,
        market_adapter: Optional[Any] = None,
        options_adapter: Optional[Any] = None,
        feature_builder: Optional[Any] = None,
    ):
        """Initialize the integration.

        Args:
            market_adapter: Market data adapter (auto-created if None)
            options_adapter: MASSIVE options adapter (auto-created if None)
            feature_builder: Feature builder (auto-created if None)
        """
        self._init_adapters(market_adapter, options_adapter, feature_builder)

    def _init_adapters(
        self,
        market_adapter: Optional[Any],
        options_adapter: Optional[Any],
        feature_builder: Optional[Any],
    ) -> None:
        """Initialize adapters with fallbacks."""

        # Market data adapter
        if market_adapter is not None:
            self.market_adapter = market_adapter
        else:
            try:
                from adapters.adapter_factory import create_market_data_adapter
                self.market_adapter = create_market_data_adapter(provider="massive")
            except Exception as e:
                logger.warning(f"Failed to create market adapter: {e}")
                self.market_adapter = None

        # Options adapter
        if options_adapter is not None:
            self.options_adapter = options_adapter
        else:
            try:
                from adapters.adapter_factory import create_massive_options_adapter
                self.options_adapter = create_massive_options_adapter()
            except Exception as e:
                logger.warning(f"Failed to create options adapter: {e}")
                self.options_adapter = None

        # Feature builder
        if feature_builder is not None:
            self.feature_builder = feature_builder
        else:
            try:
                from models.features.feature_builder import EnhancedFeatureBuilder, FeatureConfig
                config = FeatureConfig(
                    use_technical=True,
                    use_hedge_engine=True,
                    use_microstructure=True,
                    use_sentiment=True,
                    use_temporal=True,
                    use_regime=True,
                    use_options=True,  # Enable options features
                    normalize=True,
                )
                self.feature_builder = EnhancedFeatureBuilder(config)
            except Exception as e:
                logger.warning(f"Failed to create feature builder: {e}")
                self.feature_builder = None

    def get_training_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1day",
        include_options: bool = True,
        target_horizons: List[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get complete training data with options features.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            include_options: Include options features
            target_horizons: Forward return horizons for targets (default: [1, 5, 15, 60])

        Returns:
            Tuple of (features_df, targets_df)
        """
        target_horizons = target_horizons or [1, 5, 15, 60]

        logger.info(f"Fetching training data for {symbol} from {start_date} to {end_date}")

        # Get market data
        market_data = self._get_market_data(symbol, start_date, end_date, timeframe)

        if market_data is None or market_data.empty:
            logger.error(f"No market data available for {symbol}")
            return pd.DataFrame(), pd.DataFrame()

        # Get options data if available and requested
        options_data = None
        if include_options and self.options_adapter:
            options_data = self._get_options_data(symbol, start_date, end_date, timeframe)

        # Build features
        features_df = self._build_ml_features(market_data, options_data)

        # Create targets (forward returns)
        targets_df = self._create_targets(market_data, target_horizons)

        # Align features and targets
        common_idx = features_df.index.intersection(targets_df.index)
        features_df = features_df.loc[common_idx]
        targets_df = targets_df.loc[common_idx]

        logger.info(
            f"Generated {len(features_df)} samples with {features_df.shape[1]} features "
            f"and {targets_df.shape[1]} targets"
        )

        return features_df, targets_df

    def get_options_features_only(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1day",
    ) -> pd.DataFrame:
        """Get only options-derived features for analysis.

        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe

        Returns:
            DataFrame with options features
        """
        if not self.options_adapter:
            logger.warning("MASSIVE options adapter not available")
            return pd.DataFrame()

        try:
            options_df = self.options_adapter.get_options_features_for_ml(
                symbol=symbol,
                start=start_date,
                end=end_date,
                timeframe=timeframe,
            )
            return options_df

        except Exception as e:
            logger.error(f"Error getting options features: {e}")
            return pd.DataFrame()

    def prepare_lstm_sequences(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        sequence_length: int = 60,
        target_col: str = "return_1",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training.

        Args:
            features_df: Feature DataFrame
            targets_df: Target DataFrame
            sequence_length: Length of input sequences
            target_col: Target column name

        Returns:
            Tuple of (X sequences, y targets)
        """
        if target_col not in targets_df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        # Align and fill NaN
        features = features_df.fillna(0).values
        targets = targets_df[target_col].fillna(0).values

        # Create sequences
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(targets[i + sequence_length])

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Created {len(X)} sequences of shape {X.shape[1:]}")

        return X, y

    def prepare_xgboost_data(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        target_col: str = "direction_1",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for XGBoost training.

        Args:
            features_df: Feature DataFrame
            targets_df: Target DataFrame
            target_col: Target column name

        Returns:
            Tuple of (features, labels)
        """
        if target_col not in targets_df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        # Align indices
        common_idx = features_df.index.intersection(targets_df.index)

        X = features_df.loc[common_idx].fillna(0)
        y = targets_df.loc[common_idx, target_col].fillna(0)

        logger.info(f"Prepared {len(X)} samples for XGBoost")

        return X, y

    def get_feature_importance_by_category(
        self,
        model: Any,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Get feature importance aggregated by category.

        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names

        Returns:
            Dict mapping category to importance score
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model doesn't have feature_importances_")
            return {}

        importances = model.feature_importances_

        # Map features to categories
        category_importance = {cat: 0.0 for cat in self.FEATURE_CATEGORIES}
        category_importance["technical"] = 0.0
        category_importance["other"] = 0.0

        for feat_name, imp in zip(feature_names, importances):
            categorized = False
            for cat, prefixes in self.FEATURE_CATEGORIES.items():
                if any(prefix in feat_name.lower() for prefix in prefixes):
                    category_importance[cat] += imp
                    categorized = True
                    break

            if not categorized:
                # Check for technical indicators
                tech_keywords = ["sma", "ema", "rsi", "macd", "bollinger", "atr", "adx"]
                if any(kw in feat_name.lower() for kw in tech_keywords):
                    category_importance["technical"] += imp
                else:
                    category_importance["other"] += imp

        # Normalize
        total = sum(category_importance.values())
        if total > 0:
            category_importance = {k: v / total for k, v in category_importance.items()}

        return category_importance

    def create_train_val_test_split(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create time-series aware train/val/test split.

        Args:
            features_df: Feature DataFrame
            targets_df: Target DataFrame
            train_ratio: Training set ratio
            val_ratio: Validation set ratio

        Returns:
            Dict with 'train', 'val', 'test' splits
        """
        n = len(features_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        splits = {
            "train": (
                features_df.iloc[:train_end],
                targets_df.iloc[:train_end],
            ),
            "val": (
                features_df.iloc[train_end:val_end],
                targets_df.iloc[train_end:val_end],
            ),
            "test": (
                features_df.iloc[val_end:],
                targets_df.iloc[val_end:],
            ),
        }

        logger.info(
            f"Split data: train={train_end}, val={val_end - train_end}, "
            f"test={n - val_end}"
        )

        return splits

    def analyze_options_feature_correlation(
        self,
        features_df: pd.DataFrame,
        returns_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Analyze correlation between options features and returns.

        Args:
            features_df: Feature DataFrame
            returns_df: Returns DataFrame

        Returns:
            DataFrame with correlation analysis
        """
        # Identify options-related columns
        options_cols = []
        for col in features_df.columns:
            for cat, keywords in self.FEATURE_CATEGORIES.items():
                if any(kw in col.lower() for kw in keywords):
                    options_cols.append(col)
                    break

        if not options_cols:
            logger.warning("No options features found in DataFrame")
            return pd.DataFrame()

        # Calculate correlations with returns
        correlations = []
        for col in options_cols:
            if col in features_df.columns:
                for ret_col in returns_df.columns:
                    corr = features_df[col].corr(returns_df[ret_col])
                    correlations.append({
                        "feature": col,
                        "return_horizon": ret_col,
                        "correlation": corr,
                        "abs_correlation": abs(corr),
                    })

        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values("abs_correlation", ascending=False)

        return corr_df

    def _get_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """Get market data from adapter."""
        if not self.market_adapter:
            return None

        try:
            bars = self.market_adapter.get_bars(
                symbol=symbol,
                start=start_date,
                end=end_date,
                timeframe=self._map_timeframe(timeframe),
            )

            if not bars:
                return None

            # Convert to DataFrame
            data = []
            for bar in bars:
                data.append({
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                })

            df = pd.DataFrame(data)
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')

            return df

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None

    def _get_options_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        """Get options data from MASSIVE adapter."""
        if not self.options_adapter:
            return None

        try:
            return self.options_adapter.get_options_features_for_ml(
                symbol=symbol,
                start=start_date,
                end=end_date,
                timeframe=timeframe,
            )
        except Exception as e:
            logger.error(f"Error getting options data: {e}")
            return None

    def _build_ml_features(
        self,
        market_data: pd.DataFrame,
        options_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Build complete feature set for ML."""
        if self.feature_builder is None:
            return market_data

        try:
            # Build features using EnhancedFeatureBuilder
            features = self.feature_builder.build(
                market_data=market_data,
                options_data=options_data,
            )
            return features
        except Exception as e:
            logger.error(f"Error building features: {e}")
            return market_data

    def _create_targets(
        self,
        market_data: pd.DataFrame,
        horizons: List[int],
    ) -> pd.DataFrame:
        """Create target variables for different horizons."""
        targets = pd.DataFrame(index=market_data.index)

        close = market_data['close'] if 'close' in market_data.columns else market_data.iloc[:, 3]

        for h in horizons:
            # Forward returns
            targets[f'return_{h}'] = close.pct_change(h).shift(-h)

            # Direction labels (up=2, neutral=1, down=0)
            threshold = 0.002  # 0.2% threshold
            targets[f'direction_{h}'] = np.where(
                targets[f'return_{h}'] > threshold, 2,
                np.where(targets[f'return_{h}'] < -threshold, 0, 1)
            )

            # Volatility target (absolute return)
            targets[f'volatility_{h}'] = np.abs(targets[f'return_{h}'])

        return targets

    def _map_timeframe(self, timeframe: str) -> str:
        """Map timeframe string to adapter format."""
        mapping = {
            "1min": "1Min",
            "5min": "5Min",
            "15min": "15Min",
            "1hour": "1Hour",
            "1day": "1Day",
        }
        return mapping.get(timeframe.lower(), timeframe)


def create_options_ml_integration() -> Optional[MassiveOptionsMLIntegration]:
    """Factory function to create MassiveOptionsMLIntegration.

    Returns:
        MassiveOptionsMLIntegration instance or None if MASSIVE not enabled
    """
    if os.getenv("MASSIVE_API_ENABLED", "false").lower() != "true":
        logger.warning("MASSIVE API not enabled. Set MASSIVE_API_ENABLED=true in .env")
        return None

    try:
        return MassiveOptionsMLIntegration()
    except Exception as e:
        logger.error(f"Failed to create options ML integration: {e}")
        return None
