"""
LSTM Engine for Real-Time Price Prediction
Integrates bidirectional LSTM lookahead model with the pipeline orchestration system
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from engines.inputs.market_data_adapter import MarketDataAdapter
from models.features.feature_builder import EnhancedFeatureBuilder
from models.lstm_lookahead import LookaheadConfig, LSTMLookaheadPredictor
from schemas.core_schemas import ForecastSnapshot, PipelineResult


class LSTMPredictionEngine:
    """
    LSTM Engine for short-term price lookahead predictions

    Integrates with the pipeline to provide:
    - Multi-horizon price forecasts (1min, 5min, 15min, 60min)
    - Uncertainty quantification
    - Direction classification (up/down/neutral)
    - Attention-weighted temporal importance
    """

    def __init__(
        self,
        market_adapter: MarketDataAdapter,
        feature_builder: Optional[EnhancedFeatureBuilder] = None,
        model_path: Optional[str] = None,
        config: Optional[LookaheadConfig] = None,
        lookback_periods: int = 200,  # Historical bars needed for feature calculation
    ):
        """
        Initialize LSTM Prediction Engine

        Args:
            market_adapter: Market data adapter for fetching historical data
            feature_builder: Feature builder for creating input features
            model_path: Path to pre-trained model (optional)
            config: LSTM configuration (optional)
            lookback_periods: Number of historical periods to fetch
        """
        self.market_adapter = market_adapter
        self.feature_builder = feature_builder or EnhancedFeatureBuilder()
        self.lookback_periods = lookback_periods

        # Initialize LSTM predictor
        self.config = config or LookaheadConfig()
        self.predictor = LSTMLookaheadPredictor(config=self.config, model_path=model_path)

        # Cache for recent features to avoid recomputation
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        self._cache_max_age_seconds = 60  # Invalidate cache after 60 seconds

        logger.info(
            f"LSTMPredictionEngine initialized with horizons: {self.config.forecast_horizons}"
        )

    def enhance(
        self,
        pipeline_result: PipelineResult,
        timestamp: datetime,
        return_forecast_snapshot: bool = True,
    ) -> Optional[ForecastSnapshot]:
        """
        Generate LSTM-based price predictions

        Args:
            pipeline_result: Current pipeline result with engine snapshots
            timestamp: Current timestamp
            return_forecast_snapshot: If True, return ForecastSnapshot; if False, return dict

        Returns:
            ForecastSnapshot with predictions or None if error
        """
        symbol = pipeline_result.symbol

        try:
            # Get recent features for prediction
            features = self._get_recent_features(symbol, timestamp)

            if features is None or len(features) < self.config.sequence_length:
                logger.warning(
                    f"Insufficient data for LSTM prediction on {symbol}. "
                    f"Need {self.config.sequence_length} periods, got {len(features) if features is not None else 0}"
                )
                return self._empty_forecast(symbol, timestamp)

            # Extract feature sequence (exclude target column if present)
            feature_cols = [col for col in features.columns if col not in ["close", "timestamp"]]
            feature_sequence = features[feature_cols].tail(self.config.sequence_length).values

            # Make prediction
            prediction = self.predictor.predict(feature_sequence)

            # Convert to ForecastSnapshot format
            if return_forecast_snapshot:
                return self._create_forecast_snapshot(symbol, timestamp, prediction)
            else:
                return prediction

        except Exception as e:
            logger.error(f"Error in LSTM prediction for {symbol}: {e}", exc_info=True)
            return self._empty_forecast(symbol, timestamp)

    def predict_raw(self, symbol: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """
        Make raw LSTM prediction without ForecastSnapshot wrapping

        Args:
            symbol: Trading symbol
            timestamp: Current timestamp

        Returns:
            Raw prediction dictionary
        """
        try:
            features = self._get_recent_features(symbol, timestamp)

            if features is None or len(features) < self.config.sequence_length:
                return None

            feature_cols = [col for col in features.columns if col not in ["close", "timestamp"]]
            feature_sequence = features[feature_cols].tail(self.config.sequence_length).values

            return self.predictor.predict(feature_sequence)

        except Exception as e:
            logger.error(f"Error in raw LSTM prediction for {symbol}: {e}")
            return None

    def _get_recent_features(self, symbol: str, timestamp: datetime) -> Optional[pd.DataFrame]:
        """
        Get recent features for the symbol, using cache when available

        Args:
            symbol: Trading symbol
            timestamp: Current timestamp

        Returns:
            DataFrame with features
        """
        # Check cache
        if symbol in self._feature_cache:
            cached_df = self._feature_cache[symbol]
            if not cached_df.empty and "timestamp" in cached_df.columns:
                latest_cached = pd.to_datetime(cached_df["timestamp"].iloc[-1])
                age_seconds = (timestamp - latest_cached).total_seconds()

                if age_seconds < self._cache_max_age_seconds:
                    logger.debug(f"Using cached features for {symbol} (age: {age_seconds:.1f}s)")
                    return cached_df

        # Fetch fresh data
        try:
            # Get historical market data
            bars = self.market_adapter.get_historical_bars(
                symbol=symbol,
                timeframe="1Min",  # 1-minute bars for intraday prediction
                end=timestamp,
                limit=self.lookback_periods,
            )

            if bars is None or len(bars) == 0:
                logger.warning(f"No historical data available for {symbol}")
                return None

            # Build features
            features_df = self.feature_builder.build_features(
                df=bars,
                symbol=symbol,
                include_hedge=True,
                include_sentiment=True,
                include_options=True,
            )

            # Cache the features
            self._feature_cache[symbol] = features_df

            return features_df

        except Exception as e:
            logger.error(f"Error fetching features for {symbol}: {e}")
            return None

    def _create_forecast_snapshot(
        self, symbol: str, timestamp: datetime, prediction: Dict[str, Any]
    ) -> ForecastSnapshot:
        """
        Convert raw prediction to ForecastSnapshot

        Args:
            symbol: Trading symbol
            timestamp: Current timestamp
            prediction: Raw prediction dictionary from LSTM model

        Returns:
            ForecastSnapshot
        """
        # Extract predictions for all horizons
        forecast_values = []
        for horizon in self.config.forecast_horizons:
            forecast_values.append(prediction["predictions"].get(horizon, 0.0))

        # Calculate overall confidence from uncertainties
        uncertainties = [prediction["uncertainties"].get(h, 1.0) for h in self.config.forecast_horizons]
        avg_uncertainty = np.mean(uncertainties)
        confidence = 1.0 / (1.0 + avg_uncertainty)  # Convert uncertainty to confidence

        # Adjust confidence with direction probability
        direction_prob = prediction["direction_probs"][prediction["direction"]]
        confidence = confidence * direction_prob

        # Metadata with additional information
        metadata = {
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "horizons": self.config.forecast_horizons,
            "direction": prediction["direction"],
            "direction_probs": prediction["direction_probs"],
            "uncertainties": {h: prediction["uncertainties"].get(h, 0.0) for h in self.config.forecast_horizons},
            "predictions_pct": {h: prediction["predictions"].get(h, 0.0) for h in self.config.forecast_horizons},
        }

        return ForecastSnapshot(
            model="lstm_lookahead",
            horizon=max(self.config.forecast_horizons),
            forecast=forecast_values,
            confidence=float(confidence),
            metadata=metadata,
        )

    def _empty_forecast(self, symbol: str, timestamp: datetime) -> ForecastSnapshot:
        """Create empty forecast when prediction fails"""
        return ForecastSnapshot(
            model="lstm_lookahead",
            horizon=max(self.config.forecast_horizons),
            forecast=[0.0] * len(self.config.forecast_horizons),
            confidence=0.0,
            metadata={
                "symbol": symbol,
                "timestamp": timestamp.isoformat(),
                "error": "insufficient_data",
            },
        )

    def train_from_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train LSTM model on historical data

        Args:
            symbol: Trading symbol
            start_date: Start date for training data
            end_date: End date for training data
            save_path: Path to save trained model

        Returns:
            Training history
        """
        logger.info(f"Training LSTM model for {symbol} from {start_date} to {end_date}")

        # Fetch historical data
        bars = self.market_adapter.get_historical_bars(
            symbol=symbol,
            timeframe="1Min",
            start=start_date,
            end=end_date,
            limit=None,  # Get all data in range
        )

        if bars is None or len(bars) < self.config.sequence_length * 2:
            raise ValueError(f"Insufficient historical data for training: {len(bars) if bars is not None else 0} bars")

        # Build features
        features_df = self.feature_builder.build_features(
            df=bars,
            symbol=symbol,
            include_hedge=True,
            include_sentiment=True,
            include_options=True,
        )

        # Prepare data
        X, y = self.predictor.prepare_data(features_df, target_col="close")

        logger.info(f"Prepared {len(X)} training samples")

        # Train model
        history = self.predictor.train(X, y)

        # Save model
        if save_path:
            self.predictor.save(save_path)
            logger.info(f"Model saved to {save_path}")

        return history

    def load_model(self, model_path: str):
        """Load pre-trained model"""
        self.predictor.load(model_path)
        logger.info(f"Loaded LSTM model from {model_path}")

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear feature cache"""
        if symbol:
            self._feature_cache.pop(symbol, None)
            logger.debug(f"Cleared cache for {symbol}")
        else:
            self._feature_cache.clear()
            logger.debug("Cleared all feature cache")
