"""
Tests for LSTM Lookahead Model and Engine
"""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from engines.ml.lstm_engine import LSTMPredictionEngine
from models.predictors.lstm_lookahead import (
    BidirectionalLSTMLookahead,
    LookaheadConfig,
    LSTMLookaheadPredictor,
)
from schemas.core_schemas import (
    ForecastSnapshot,
    HedgeSnapshot,
    PipelineResult,
)


class TestLookaheadConfig:
    """Test LSTM Lookahead configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = LookaheadConfig()
        assert config.input_dim == 150
        assert config.hidden_dim == 128
        assert config.num_layers == 2
        assert config.dropout == 0.3
        assert config.forecast_horizons == [1, 5, 15, 60]
        assert config.sequence_length == 60
        assert config.bidirectional is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = LookaheadConfig(
            input_dim=100,
            hidden_dim=256,
            forecast_horizons=[1, 5, 10],
            bidirectional=False,
        )
        assert config.input_dim == 100
        assert config.hidden_dim == 256
        assert config.forecast_horizons == [1, 5, 10]
        assert config.bidirectional is False


class TestBidirectionalLSTMLookahead:
    """Test the Bidirectional LSTM model architecture"""

    @pytest.fixture
    def config(self):
        return LookaheadConfig(
            input_dim=10,  # Small for testing
            hidden_dim=32,
            num_layers=2,
            forecast_horizons=[1, 5],
        )

    @pytest.fixture
    def model(self, config):
        return BidirectionalLSTMLookahead(config)

    def test_model_initialization(self, model, config):
        """Test model initializes correctly"""
        assert model.config == config
        assert isinstance(model.lstm, torch.nn.LSTM)
        assert model.lstm.input_size == 10
        assert model.lstm.hidden_size == 32
        assert model.lstm.bidirectional is True

    def test_forward_pass_shape(self, model):
        """Test forward pass produces correct output shapes"""
        batch_size = 8
        seq_len = 60
        input_dim = 10

        x = torch.randn(batch_size, seq_len, input_dim)
        predictions, uncertainties, directions, attention_weights = model(x)

        # Check predictions shape
        assert len(predictions) == 2  # Two horizons
        for horizon, pred in predictions.items():
            assert pred.shape == (batch_size, 1)

        # Check uncertainties shape
        assert len(uncertainties) == 2
        for horizon, unc in uncertainties.items():
            assert unc.shape == (batch_size, 1)

        # Check directions shape
        assert directions.shape == (batch_size, 3)  # up, neutral, down
        assert torch.allclose(directions.sum(dim=-1), torch.ones(batch_size))  # Probabilities sum to 1

        # Check attention weights shape
        assert attention_weights.shape == (batch_size, seq_len, 1)

    def test_uncertainty_positive(self, model):
        """Test that uncertainties are always positive"""
        batch_size = 4
        seq_len = 60
        input_dim = 10

        x = torch.randn(batch_size, seq_len, input_dim)
        _, uncertainties, _, _ = model(x)

        for horizon, unc in uncertainties.items():
            assert torch.all(unc > 0), f"Uncertainties for {horizon} should be positive"


class TestLSTMLookaheadPredictor:
    """Test the high-level LSTM predictor interface"""

    @pytest.fixture
    def config(self):
        return LookaheadConfig(
            input_dim=10,
            hidden_dim=32,
            num_layers=2,
            forecast_horizons=[1, 5, 15],
            sequence_length=20,
            batch_size=4,
            max_epochs=2,  # Short for testing
        )

    @pytest.fixture
    def predictor(self, config):
        return LSTMLookaheadPredictor(config=config)

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        np.random.seed(42)
        n_samples = 200
        n_features = 10

        # Generate synthetic price and features
        data = {
            "close": 100 + np.cumsum(np.random.randn(n_samples) * 0.5),
        }

        # Add random features
        for i in range(n_features):
            data[f"feature_{i}"] = np.random.randn(n_samples)

        return pd.DataFrame(data)

    def test_prepare_data(self, predictor, sample_data):
        """Test data preparation"""
        X, y = predictor.prepare_data(sample_data, target_col="close")

        # Check shapes
        assert X.ndim == 3  # [n_samples, seq_len, n_features]
        assert y.ndim == 2  # [n_samples, n_horizons]
        assert X.shape[1] == predictor.config.sequence_length
        assert y.shape[1] == len(predictor.config.forecast_horizons)

        # Check no NaN values
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

    def test_training(self, predictor, sample_data):
        """Test model training"""
        X, y = predictor.prepare_data(sample_data, target_col="close")

        # Train model
        history = predictor.train(X, y, validation_split=0.2)

        # Check history
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) > 0
        assert len(history["val_loss"]) > 0

        # Check losses decrease
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_prediction(self, predictor, sample_data):
        """Test making predictions"""
        X, y = predictor.prepare_data(sample_data, target_col="close")

        # Train briefly
        predictor.train(X[:100], y[:100], validation_split=0.2)

        # Make prediction on single sequence
        test_sequence = X[0]  # Shape: [seq_len, n_features]
        result = predictor.predict(test_sequence)

        # Check result structure
        assert "predictions" in result
        assert "uncertainties" in result
        assert "direction_probs" in result
        assert "direction" in result

        # Check predictions for all horizons
        for horizon in predictor.config.forecast_horizons:
            assert horizon in result["predictions"]
            assert horizon in result["uncertainties"]
            assert isinstance(result["predictions"][horizon], float)
            assert isinstance(result["uncertainties"][horizon], float)

        # Check direction
        assert result["direction"] in ["up", "down", "neutral"]
        assert "up" in result["direction_probs"]
        assert "down" in result["direction_probs"]
        assert "neutral" in result["direction_probs"]

    def test_save_load_model(self, predictor, sample_data, tmp_path):
        """Test saving and loading model"""
        # Train model
        X, y = predictor.prepare_data(sample_data, target_col="close")
        predictor.train(X[:50], y[:50], validation_split=0.2)

        # Make prediction before save
        test_sequence = X[0]
        result_before = predictor.predict(test_sequence)

        # Save model
        model_path = tmp_path / "test_model.pth"
        predictor.save(str(model_path))
        assert model_path.exists()

        # Load model in new predictor
        new_predictor = LSTMLookaheadPredictor(config=predictor.config)
        new_predictor.load(str(model_path))

        # Make prediction after load
        result_after = new_predictor.predict(test_sequence)

        # Check predictions are identical
        for horizon in predictor.config.forecast_horizons:
            assert abs(result_before["predictions"][horizon] - result_after["predictions"][horizon]) < 1e-5


class TestLSTMPredictionEngine:
    """Test the LSTM Prediction Engine integration"""

    @pytest.fixture
    def config(self):
        return LookaheadConfig(
            input_dim=10,
            hidden_dim=32,
            forecast_horizons=[1, 5],
            sequence_length=20,
        )

    @pytest.fixture
    def mock_market_adapter(self):
        """Mock market data adapter"""
        adapter = Mock()

        # Create sample historical bars
        bars = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="1Min"),
            "open": 100 + np.random.randn(100),
            "high": 101 + np.random.randn(100),
            "low": 99 + np.random.randn(100),
            "close": 100 + np.cumsum(np.random.randn(100) * 0.5),
            "volume": np.random.randint(1000, 10000, 100),
        })

        adapter.get_historical_bars = Mock(return_value=bars)
        return adapter

    @pytest.fixture
    def mock_feature_builder(self):
        """Mock feature builder"""
        builder = Mock()

        def build_features(df, **kwargs):
            # Return DataFrame with some features
            features = df.copy()
            for i in range(10):
                features[f"feature_{i}"] = np.random.randn(len(df))
            return features

        builder.build_features = Mock(side_effect=build_features)
        return builder

    @pytest.fixture
    def engine(self, mock_market_adapter, mock_feature_builder, config):
        return LSTMPredictionEngine(
            market_adapter=mock_market_adapter,
            feature_builder=mock_feature_builder,
            config=config,
        )

    def test_engine_initialization(self, engine, config):
        """Test engine initializes correctly"""
        assert engine.config == config
        assert engine.predictor is not None
        assert isinstance(engine.predictor, LSTMLookaheadPredictor)

    def test_enhance_insufficient_data(self, engine):
        """Test enhance returns empty forecast when insufficient data"""
        pipeline_result = PipelineResult(
            timestamp=datetime.now(),
            symbol="AAPL",
        )

        # Mock insufficient data
        engine.market_adapter.get_historical_bars = Mock(return_value=pd.DataFrame())

        forecast = engine.enhance(pipeline_result, datetime.now())

        assert isinstance(forecast, ForecastSnapshot)
        assert forecast.model == "lstm_lookahead"
        assert forecast.confidence == 0.0

    def test_enhance_with_data(self, engine, mock_market_adapter):
        """Test enhance generates forecast with sufficient data"""
        # Create pipeline result
        timestamp = datetime.now()
        pipeline_result = PipelineResult(
            timestamp=timestamp,
            symbol="AAPL",
            hedge_snapshot=HedgeSnapshot(
                timestamp=timestamp,
                symbol="AAPL",
                elasticity=0.5,
                movement_energy=50.0,
            ),
        )

        # Mock trained model (skip actual training)
        with patch.object(engine.predictor, 'predict') as mock_predict:
            mock_predict.return_value = {
                "predictions": {1: 0.5, 5: 1.2},
                "uncertainties": {1: 0.1, 5: 0.2},
                "direction": "up",
                "direction_probs": {"up": 0.7, "neutral": 0.2, "down": 0.1},
                "attention_weights": np.random.rand(20, 1),
            }

            forecast = engine.enhance(pipeline_result, timestamp)

            assert isinstance(forecast, ForecastSnapshot)
            assert forecast.model == "lstm_lookahead"
            assert forecast.confidence > 0.0
            assert len(forecast.forecast) == 2  # Two horizons
            assert "direction" in forecast.metadata
            assert forecast.metadata["direction"] == "up"

    def test_cache_functionality(self, engine, mock_market_adapter):
        """Test that feature caching works"""
        timestamp = datetime.now()
        symbol = "AAPL"

        # First call should fetch data
        features1 = engine._get_recent_features(symbol, timestamp)
        assert mock_market_adapter.get_historical_bars.call_count == 1

        # Second call within cache window should use cache
        features2 = engine._get_recent_features(symbol, timestamp)
        assert mock_market_adapter.get_historical_bars.call_count == 1  # No additional call

        # Check cache was used
        assert features1 is not None
        assert features2 is not None

        # Clear cache and try again
        engine.clear_cache(symbol)
        engine._get_recent_features(symbol, timestamp)
        assert mock_market_adapter.get_historical_bars.call_count == 2  # New call after cache clear


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
