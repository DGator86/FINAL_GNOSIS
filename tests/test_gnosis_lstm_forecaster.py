"""
Tests for GnosisLSTMForecaster to validate PR #17 review feedback fixes.
"""

import numpy as np
import pytest

# Only run tests if torch is available
torch = pytest.importorskip("torch")
sklearn = pytest.importorskip("sklearn")

from models.time_series.gnosis_lstm_forecaster import GnosisLSTMForecaster


class TestGnosisLSTMForecaster:
    """Test suite for GnosisLSTMForecaster."""

    def test_uncertainty_loss_weight_configurable(self):
        """Test that uncertainty_loss_weight can be configured."""
        # Test with default value
        config_default = {"sequence_length": 10, "horizons": [1]}
        model_default = GnosisLSTMForecaster(config_default)
        assert model_default.uncertainty_loss_weight == 0.1

        # Test with custom value
        config_custom = {"sequence_length": 10, "horizons": [1], "uncertainty_loss_weight": 0.5}
        model_custom = GnosisLSTMForecaster(config_custom)
        assert model_custom.uncertainty_loss_weight == 0.5

    def test_output_dim_in_model_config(self):
        """Test that output_dim is stored in model config during training."""
        config = {
            "sequence_length": 5,  # Reduced to allow more sequences
            "horizons": [1],
            "hidden_dim": 32,
            "num_layers": 1,
            "dropout": 0.1,
        }
        model = GnosisLSTMForecaster(config)

        # Create minimal training data with enough samples
        np.random.seed(42)
        X = np.random.randn(100, 5)  # 100 samples, 5 features
        y = np.random.randn(100)

        # Train the model
        model.train(X, y, epochs=2, batch_size=8)

        # Check that output_dim is in the saved config
        assert 1 in model.models
        assert "config" in model.models[1]
        assert "output_dim" in model.models[1]["config"]
        assert model.models[1]["config"]["output_dim"] == 1

    def test_train_with_2d_input(self):
        """Test that train method handles 2D input correctly."""
        config = {"sequence_length": 5, "horizons": [1], "hidden_dim": 32, "num_layers": 1}
        model = GnosisLSTMForecaster(config)

        # 2D input: (samples, features)
        np.random.seed(42)
        X = np.random.randn(100, 5)  # More samples for sufficient sequences
        y = np.random.randn(100)

        # Should not raise an error
        results = model.train(X, y, epochs=2, batch_size=8)
        assert isinstance(results, dict)

    def test_predict_with_2d_input(self):
        """Test that predict method handles 2D input correctly."""
        config = {"sequence_length": 5, "horizons": [1], "hidden_dim": 32, "num_layers": 1}
        model = GnosisLSTMForecaster(config)

        # Train first with more data
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.randn(100)
        model.train(X_train, y_train, epochs=2, batch_size=8)

        # Predict with 2D input
        X_pred = np.random.randn(20, 5)
        result = model.predict(X_pred)

        assert "predictions" in result
        assert isinstance(result["predictions"], dict)

    def test_loss_averaging_with_partial_batch(self):
        """Test that loss averaging works correctly when last batch is partial."""
        config = {"sequence_length": 5, "horizons": [1], "hidden_dim": 16, "num_layers": 1}
        model = GnosisLSTMForecaster(config)

        # Create data that will result in partial last batch
        # batch_size=8, but after creating sequences we'll have 25 samples
        # This gives us 3 full batches + 1 partial batch (8+8+8+1)
        np.random.seed(42)
        X = np.random.randn(35, 3)  # Will create ~25 sequences with sequence_length=5
        y = np.random.randn(35)

        # Train with batch_size that doesn't divide evenly
        results = model.train(X, y, epochs=2, batch_size=8, validation_split=0.2)

        # Should complete without errors
        assert isinstance(results, dict)
        assert 1 in results
        assert "train_losses" in results[1]
        assert len(results[1]["train_losses"]) > 0


def test_no_warnings_import():
    """Test that warnings module is not imported."""
    import sys
    import importlib
    
    # Clear the module cache if already loaded
    if 'models.time_series.gnosis_lstm_forecaster' in sys.modules:
        del sys.modules['models.time_series.gnosis_lstm_forecaster']
    
    # Import the module
    module = importlib.import_module('models.time_series.gnosis_lstm_forecaster')
    
    # Check that warnings is not in the module's globals
    assert 'warnings' not in dir(module)


def test_spelling_corrections():
    """Test that British spelling has been corrected to American spelling."""
    import models.time_series.base_model as base_model
    import models.time_series.attention_mechanism as attention_mechanism
    
    # Check base_model docstring
    assert 'standardize' in base_model.BaseGnosisModel.__doc__
    assert 'behavior' in base_model.BaseGnosisModel.__doc__
    assert 'standardise' not in base_model.BaseGnosisModel.__doc__
    assert 'behaviour' not in base_model.BaseGnosisModel.__doc__
    
    # Check attention_mechanism docstring
    assert 'summarization' in attention_mechanism.TemporalAttention.__doc__
    assert 'summarisation' not in attention_mechanism.TemporalAttention.__doc__
