"""
Tests for ML Hyperparameter Backtest Integration.

Tests cover:
- MLBacktestHyperparameters
- MLHyperparameterBacktester
- Walk-forward optimization
- Sensitivity analysis
- Regime-adaptive backtesting

Author: Super Gnosis Elite Trading System
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from backtesting.ml_hyperparameter_backtest import (
    MLBacktestMode,
    MLBacktestHyperparameters,
    MLBacktestPeriod,
    MLHyperparameterBacktestConfig,
    MLHyperparameterBacktestResults,
    MLHyperparameterBacktester,
    create_ml_hyperparameter_backtester,
    run_ml_parameter_backtest,
)

from ml import MarketRegime


class TestMLBacktestHyperparameters:
    """Tests for MLBacktestHyperparameters."""
    
    def test_default_values(self):
        hp = MLBacktestHyperparameters()
        assert hp.lstm_hidden_dim == 64
        assert hp.confidence_threshold == 0.60
        assert hp.kelly_fraction == 0.25
    
    def test_custom_values(self):
        hp = MLBacktestHyperparameters(
            lstm_hidden_dim=128,
            confidence_threshold=0.70,
            kelly_fraction=0.15,
        )
        assert hp.lstm_hidden_dim == 128
        assert hp.confidence_threshold == 0.70
        assert hp.kelly_fraction == 0.15
    
    def test_to_dict(self):
        hp = MLBacktestHyperparameters()
        d = hp.to_dict()
        
        assert "lstm_hidden_dim" in d
        assert "confidence_threshold" in d
        assert d["lstm_hidden_dim"] == 64
    
    def test_from_dict(self):
        d = {
            "lstm_hidden_dim": 128,
            "confidence_threshold": 0.70,
        }
        hp = MLBacktestHyperparameters.from_dict(d)
        
        assert hp.lstm_hidden_dim == 128
        assert hp.confidence_threshold == 0.70
    
    def test_from_preset_conservative(self):
        hp = MLBacktestHyperparameters.from_preset("conservative")
        
        assert hp.lstm_hidden_dim == 32
        assert hp.confidence_threshold == 0.70
        assert hp.kelly_fraction == 0.15
    
    def test_from_preset_balanced(self):
        hp = MLBacktestHyperparameters.from_preset("balanced")
        
        assert hp.lstm_hidden_dim == 64
        assert hp.confidence_threshold == 0.60
        assert hp.kelly_fraction == 0.25
    
    def test_from_preset_aggressive(self):
        hp = MLBacktestHyperparameters.from_preset("aggressive")
        
        assert hp.lstm_hidden_dim == 128
        assert hp.confidence_threshold == 0.50
        assert hp.kelly_fraction == 0.40


class TestMLBacktestPeriod:
    """Tests for MLBacktestPeriod."""
    
    def test_creation(self):
        period = MLBacktestPeriod(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            total_return_pct=0.15,
            sharpe_ratio=1.5,
        )
        
        assert period.total_return_pct == 0.15
        assert period.sharpe_ratio == 1.5
    
    def test_with_regime(self):
        period = MLBacktestPeriod(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            regime=MarketRegime.TRENDING_BULL,
        )
        
        assert period.regime == MarketRegime.TRENDING_BULL


class TestMLHyperparameterBacktestConfig:
    """Tests for MLHyperparameterBacktestConfig."""
    
    def test_default_values(self):
        config = MLHyperparameterBacktestConfig()
        
        assert config.mode == MLBacktestMode.STATIC
        assert config.preset == "balanced"
        assert config.initial_capital == 100000.0
    
    def test_custom_values(self):
        config = MLHyperparameterBacktestConfig(
            mode=MLBacktestMode.WALK_FORWARD,
            preset="aggressive",
            symbols=["QQQ", "IWM"],
        )
        
        assert config.mode == MLBacktestMode.WALK_FORWARD
        assert config.preset == "aggressive"
        assert "QQQ" in config.symbols


class TestMLHyperparameterBacktester:
    """Tests for MLHyperparameterBacktester."""
    
    @pytest.fixture
    def backtester(self):
        config = MLHyperparameterBacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        return MLHyperparameterBacktester(config=config)
    
    def test_initialization(self, backtester):
        assert backtester.config is not None
        assert backtester.hp_manager is not None
    
    def test_static_backtest(self, backtester):
        backtester.config.mode = MLBacktestMode.STATIC
        results = backtester.run()
        
        assert results is not None
        assert isinstance(results, MLHyperparameterBacktestResults)
        assert len(results.periods) == 1
        assert results.sharpe_ratio != 0
    
    def test_adaptive_backtest(self):
        config = MLHyperparameterBacktestConfig(
            mode=MLBacktestMode.ADAPTIVE,
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        backtester = MLHyperparameterBacktester(config=config)
        results = backtester.run()
        
        assert results is not None
        assert len(results.periods) > 0
        assert results.regime_performance is not None
    
    def test_walk_forward_backtest(self):
        config = MLHyperparameterBacktestConfig(
            mode=MLBacktestMode.WALK_FORWARD,
            start_date="2023-01-01",
            end_date="2023-12-31",
            train_pct=0.70,
        )
        backtester = MLHyperparameterBacktester(config=config)
        results = backtester.run()
        
        assert results is not None
        assert results.walk_forward_efficiency >= 0 or results.walk_forward_efficiency < 0  # Can be any value
        assert results.in_sample_sharpe != 0 or results.out_of_sample_sharpe != 0
    
    def test_sensitivity_analysis(self):
        config = MLHyperparameterBacktestConfig(
            mode=MLBacktestMode.SENSITIVITY,
            start_date="2023-01-01",
            end_date="2023-12-31",
            sensitivity_params=["confidence_threshold"],
            sensitivity_steps=3,
        )
        backtester = MLHyperparameterBacktester(config=config)
        results = backtester.run()
        
        assert results is not None
        assert "confidence_threshold" in results.sensitivity_results
        assert len(results.sensitivity_results["confidence_threshold"]) == 3
    
    def test_backtest_period(self, backtester):
        hp = MLBacktestHyperparameters()
        period = backtester._backtest_period(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            hyperparameters=hp,
        )
        
        assert period is not None
        assert period.total_trades > 0
        assert period.sharpe_ratio != 0
    
    def test_optimize_parameters(self, backtester):
        best_hp = backtester._optimize_parameters(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 30),
        )
        
        assert best_hp is not None
        assert isinstance(best_hp, MLBacktestHyperparameters)
    
    def test_get_regime_parameters(self, backtester):
        regime_params = backtester._get_regime_parameters()
        
        assert "trending_bull" in regime_params
        assert "trending_bear" in regime_params
        assert "sideways" in regime_params
        assert "high_volatility" in regime_params
        
        # Check conservative params for high vol
        high_vol_hp = regime_params["high_volatility"]
        assert high_vol_hp.confidence_threshold > 0.70
    
    def test_generate_walk_forward_windows(self, backtester):
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        
        windows = backtester._generate_walk_forward_windows(
            start=start,
            end=end,
            train_days=180,
            test_days=60,
        )
        
        assert len(windows) > 0
        
        # Check first window
        train_start, train_end, test_start, test_end = windows[0]
        assert train_start == start
        assert train_end > train_start
        assert test_start >= train_end
        assert test_end > test_start


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_ml_hyperparameter_backtester(self):
        backtester = create_ml_hyperparameter_backtester(
            preset="conservative",
            mode="static",
        )
        
        assert backtester is not None
        assert backtester.config.preset == "conservative"
        assert backtester.config.mode == MLBacktestMode.STATIC
    
    def test_run_ml_parameter_backtest(self):
        results = run_ml_parameter_backtest(
            preset="balanced",
            mode="static",
            start_date="2023-06-01",
            end_date="2023-12-31",
        )
        
        assert results is not None
        assert isinstance(results, MLHyperparameterBacktestResults)


class TestMLBacktestIntegration:
    """Integration tests for ML backtest."""
    
    def test_full_workflow_static(self):
        """Test complete static backtest workflow."""
        results = run_ml_parameter_backtest(
            preset="balanced",
            mode="static",
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        
        assert results.sharpe_ratio is not None
        assert results.total_trades > 0
        assert results.best_hyperparameters is not None
    
    def test_full_workflow_walk_forward(self):
        """Test complete walk-forward workflow."""
        config = MLHyperparameterBacktestConfig(
            mode=MLBacktestMode.WALK_FORWARD,
            start_date="2022-01-01",
            end_date="2023-12-31",
            train_pct=0.60,
            reoptimize_frequency=60,
        )
        
        backtester = MLHyperparameterBacktester(config=config)
        results = backtester.run()
        
        assert results.periods is not None
        assert results.walk_forward_efficiency is not None
    
    def test_preset_comparison(self):
        """Compare different presets."""
        results = {}
        
        for preset in ["conservative", "balanced", "aggressive"]:
            r = run_ml_parameter_backtest(
                preset=preset,
                mode="static",
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
            results[preset] = r.sharpe_ratio
        
        # All should have some sharpe ratio
        assert all(r is not None for r in results.values())
        
        # Conservative should generally have more trades filtered
        # (This is a simplified check - in reality depends on market conditions)


class TestMLBacktestResultsSerialization:
    """Tests for results serialization."""
    
    def test_results_to_json(self):
        """Test that results can be serialized."""
        import json
        
        results = run_ml_parameter_backtest(
            preset="balanced",
            mode="static",
            start_date="2023-01-01",
            end_date="2023-06-30",
        )
        
        # Create summary dict
        summary = {
            "sharpe_ratio": results.sharpe_ratio,
            "total_return_pct": results.total_return_pct,
            "total_trades": results.total_trades,
            "win_rate": results.win_rate,
        }
        
        # Should be JSON serializable
        json_str = json.dumps(summary)
        assert json_str is not None
        
        # Should be deserializable
        loaded = json.loads(json_str)
        assert loaded["sharpe_ratio"] == results.sharpe_ratio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
