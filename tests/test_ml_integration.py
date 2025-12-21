"""
Comprehensive tests for ML Integration module.

Tests cover:
- MLHyperparameterManager
- AdaptiveMLPipeline
- MLOptimizationEngine
- AutoMLTuner

Author: Super Gnosis Elite Trading System
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest


# Import ML components
from ml.hyperparameter_manager import (
    MarketRegime,
    ParameterScope,
    ParameterSpec,
    LSTMHyperparameters,
    FeatureHyperparameters,
    SignalHyperparameters,
    PositionSizingHyperparameters,
    RiskManagementHyperparameters,
    StrategySelectionHyperparameters,
    MLHyperparameterSet,
    MLHyperparameterManager,
    create_preset_config,
)

from ml.adaptive_pipeline import (
    SignalStrength,
    MLSignal,
    MLPositionSize,
    MLTradeDecision,
    RegimeDetector,
    AdaptiveMLPipeline,
)

from ml.optimization_engine import (
    OptimizationStage,
    OptimizationMetric,
    OptimizationConfig,
    StageOptimizationResult,
    FullOptimizationResult,
    ObjectiveFunction,
    MLOptimizationEngine,
    AutoMLTuner,
    create_optimization_engine,
)


# =============================================================================
# HYPERPARAMETER MANAGER TESTS
# =============================================================================

class TestLSTMHyperparameters:
    """Tests for LSTMHyperparameters."""
    
    def test_default_values(self):
        hp = LSTMHyperparameters()
        assert hp.hidden_dim == 128
        assert hp.num_layers == 2
        assert hp.dropout == 0.2
        assert hp.learning_rate == 0.001
        assert hp.sequence_length == 60
    
    def test_to_dict(self):
        hp = LSTMHyperparameters()
        d = hp.to_dict()
        assert "hidden_dim" in d
        assert "learning_rate" in d
        assert d["hidden_dim"] == 128
    
    def test_from_dict(self):
        data = {"hidden_dim": 256, "dropout": 0.3}
        hp = LSTMHyperparameters.from_dict(data)
        assert hp.hidden_dim == 256
        assert hp.dropout == 0.3
        assert hp.num_layers == 2  # Default preserved


class TestFeatureHyperparameters:
    """Tests for FeatureHyperparameters."""
    
    def test_default_values(self):
        hp = FeatureHyperparameters()
        assert hp.rsi_period == 14
        assert hp.atr_period == 14
        assert hp.max_features == 50
    
    def test_custom_values(self):
        hp = FeatureHyperparameters(rsi_period=21, atr_period=20)
        assert hp.rsi_period == 21
        assert hp.atr_period == 20


class TestSignalHyperparameters:
    """Tests for SignalHyperparameters."""
    
    def test_default_weights_sum_to_one(self):
        hp = SignalHyperparameters()
        total = (
            hp.hedge_weight +
            hp.sentiment_weight +
            hp.liquidity_weight +
            hp.elasticity_weight +
            hp.ml_forecast_weight
        )
        assert abs(total - 1.0) < 0.01
    
    def test_thresholds(self):
        hp = SignalHyperparameters()
        assert hp.bullish_threshold > hp.bearish_threshold
        assert hp.min_confidence < hp.high_confidence


class TestPositionSizingHyperparameters:
    """Tests for PositionSizingHyperparameters."""
    
    def test_kelly_constraints(self):
        hp = PositionSizingHyperparameters()
        # kelly_fraction (0.25) is % of Kelly to use (25% Kelly)
        # max_kelly_bet (0.10) is max bet as % of portfolio 
        # These are different concepts - kelly_fraction * edge gives bet, capped by max_kelly_bet
        assert 0 < hp.kelly_fraction <= 1.0  # Valid Kelly fraction
        assert 0 < hp.max_kelly_bet <= 1.0  # Valid max bet
        assert hp.max_position_pct < hp.max_portfolio_heat
    
    def test_regime_multipliers(self):
        hp = PositionSizingHyperparameters()
        assert "crisis" in hp.regime_size_multipliers
        assert hp.regime_size_multipliers["crisis"] < 1.0  # Should reduce size in crisis


class TestRiskManagementHyperparameters:
    """Tests for RiskManagementHyperparameters."""
    
    def test_reward_risk_ratio(self):
        hp = RiskManagementHyperparameters()
        # Take profit should be larger than stop loss
        assert hp.take_profit_atr_multiple > hp.stop_loss_atr_multiple
    
    def test_partial_profits(self):
        hp = RiskManagementHyperparameters()
        # Partial profit amounts should sum to 1
        assert abs(sum(hp.partial_profit_amounts) - 1.0) < 0.01


class TestMLHyperparameterSet:
    """Tests for MLHyperparameterSet."""
    
    def test_default_creation(self):
        params = MLHyperparameterSet()
        assert params.lstm is not None
        assert params.signals is not None
        assert params.position_sizing is not None
    
    def test_to_dict_and_back(self):
        params = MLHyperparameterSet(name="test")
        d = params.to_dict()
        restored = MLHyperparameterSet.from_dict(d)
        assert restored.name == "test"
        assert restored.lstm.hidden_dim == params.lstm.hidden_dim
    
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_params.json"
            params = MLHyperparameterSet(name="test_save")
            params.save(path)
            
            loaded = MLHyperparameterSet.load(path)
            assert loaded.name == "test_save"
    
    def test_copy(self):
        params = MLHyperparameterSet()
        params.lstm.hidden_dim = 512
        copied = params.copy()
        
        # Modify original
        params.lstm.hidden_dim = 256
        
        # Copy should be independent
        assert copied.lstm.hidden_dim == 512


class TestMLHyperparameterManager:
    """Tests for MLHyperparameterManager."""
    
    @pytest.fixture
    def manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield MLHyperparameterManager(config_dir=tmpdir)
    
    def test_initialization(self, manager):
        assert manager.current is not None
        assert isinstance(manager.current, MLHyperparameterSet)
    
    def test_set_regime(self, manager):
        params = manager.set_regime(MarketRegime.HIGH_VOLATILITY)
        # In high volatility, position sizes should be smaller
        assert params.position_sizing.kelly_fraction < manager.base_params.position_sizing.kelly_fraction
    
    def test_update_parameter(self, manager):
        manager.update_parameter("lstm.hidden_dim", 256)
        assert manager.current.lstm.hidden_dim == 256
    
    def test_update_from_dict(self, manager):
        updates = {
            "lstm.hidden_dim": 512,
            "signals.min_confidence": 0.7,
        }
        manager.update_from_dict(updates)
        assert manager.current.lstm.hidden_dim == 512
        assert manager.current.signals.min_confidence == 0.7
    
    def test_get_parameter(self, manager):
        value = manager.get_parameter("lstm.hidden_dim")
        assert value == 128  # Default
    
    def test_save_and_load_config(self, manager):
        manager.update_parameter("lstm.hidden_dim", 999)
        path = manager.save_current("test_config")
        
        # Reset and reload
        manager.update_parameter("lstm.hidden_dim", 128)
        manager.load_config("test_config")
        assert manager.current.lstm.hidden_dim == 999
    
    def test_list_configs(self, manager):
        manager.save_current("config1")
        manager.save_current("config2")
        configs = manager.list_configs()
        assert "config1" in configs
        assert "config2" in configs
    
    def test_get_optimization_space(self, manager):
        space = manager.get_optimization_space()
        assert len(space) > 0
        assert all(isinstance(s, ParameterSpec) for s in space)
    
    def test_get_optimization_space_by_scope(self, manager):
        model_space = manager.get_optimization_space([ParameterScope.MODEL])
        signal_space = manager.get_optimization_space([ParameterScope.SIGNAL])
        
        # Model space should have LSTM params
        model_names = [s.name for s in model_space]
        assert any("lstm" in n for n in model_names)
        
        # Signal space should have signal params
        signal_names = [s.name for s in signal_space]
        assert any("signal" in n for n in signal_names)


class TestPresetConfigs:
    """Tests for preset configurations."""
    
    def test_conservative_preset(self):
        params = create_preset_config("conservative")
        assert params.position_sizing.kelly_fraction < 0.25
        assert params.signals.min_confidence > 0.6
    
    def test_aggressive_preset(self):
        params = create_preset_config("aggressive")
        assert params.position_sizing.kelly_fraction > 0.25
        assert params.signals.min_confidence < 0.6
    
    def test_balanced_preset(self):
        params = create_preset_config("balanced")
        assert params.name == "balanced"
    
    def test_invalid_preset(self):
        with pytest.raises(ValueError):
            create_preset_config("invalid")


# =============================================================================
# ADAPTIVE PIPELINE TESTS
# =============================================================================

class TestMLSignal:
    """Tests for MLSignal."""
    
    def test_creation(self):
        signal = MLSignal(
            direction="bullish",
            strength=SignalStrength.STRONG_BUY,
            confidence=0.85,
            symbol="AAPL",
        )
        assert signal.direction == "bullish"
        assert signal.confidence == 0.85
        assert signal.timestamp is not None
    
    def test_default_scores(self):
        signal = MLSignal(
            direction="neutral",
            strength=SignalStrength.NEUTRAL,
            confidence=0.5,
        )
        assert signal.ml_forecast_score == 0.0
        assert signal.hedge_score == 0.0


class TestMLPositionSize:
    """Tests for MLPositionSize."""
    
    def test_final_size_respects_max(self):
        pos = MLPositionSize(
            base_size=0.10,
            adjusted_size=0.08,
            max_allowed=0.05,
        )
        assert pos.final_size == 0.05
    
    def test_final_size_uses_adjusted(self):
        pos = MLPositionSize(
            base_size=0.10,
            adjusted_size=0.03,
            max_allowed=0.05,
        )
        assert pos.final_size == 0.03


class TestRegimeDetector:
    """Tests for RegimeDetector."""
    
    def test_detection_from_market_data(self):
        detector = RegimeDetector()
        
        # Create sample market data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        market_data = pd.DataFrame({
            "close": 100 + np.cumsum(np.random.randn(100) * 0.5),
            "high": 101 + np.cumsum(np.random.randn(100) * 0.5),
            "low": 99 + np.cumsum(np.random.randn(100) * 0.5),
        }, index=dates)
        
        regime = detector.detect(market_data=market_data)
        assert isinstance(regime, MarketRegime)
    
    def test_regime_stability(self):
        detector = RegimeDetector()
        
        # Detect multiple times
        for _ in range(10):
            detector.detect()
        
        stability = detector.get_regime_stability()
        assert 0 <= stability <= 1


class TestAdaptiveMLPipeline:
    """Tests for AdaptiveMLPipeline."""
    
    @pytest.fixture
    def pipeline(self):
        return AdaptiveMLPipeline(preset="balanced")
    
    def test_initialization(self, pipeline):
        assert pipeline.hp_manager is not None
        assert pipeline.regime_detector is not None
    
    def test_process_generates_decision(self, pipeline):
        decision = pipeline.process(symbol="AAPL")
        assert isinstance(decision, MLTradeDecision)
        assert decision.action in ["buy", "sell", "hold"]
    
    def test_update_hyperparameters(self, pipeline):
        original_value = pipeline.hp_manager.current.lstm.hidden_dim
        pipeline.update_hyperparameters({"lstm.hidden_dim": 256})
        assert pipeline.hp_manager.current.lstm.hidden_dim == 256
    
    def test_regime_change_adjusts_params(self, pipeline):
        # Process to detect regime
        pipeline.process(symbol="TEST")
        
        # Force regime change
        pipeline.regime_detector._regime_history.clear()
        pipeline._current_regime = None
        
        # Process again should potentially adjust params
        decision = pipeline.process(symbol="TEST")
        assert decision is not None
    
    def test_performance_summary(self, pipeline):
        # Generate some decisions
        for _ in range(5):
            pipeline.process(symbol="TEST")
        
        summary = pipeline.get_performance_summary()
        assert "total_decisions" in summary
        assert summary["total_decisions"] >= 5


# =============================================================================
# OPTIMIZATION ENGINE TESTS
# =============================================================================

class TestOptimizationConfig:
    """Tests for OptimizationConfig."""
    
    def test_defaults(self):
        config = OptimizationConfig()
        assert config.n_trials == 100
        assert config.method == "bayesian"
        assert config.use_walk_forward is True
    
    def test_custom_config(self):
        config = OptimizationConfig(
            method="grid",
            n_trials=50,
            primary_metric=OptimizationMetric.TOTAL_RETURN,
        )
        assert config.method == "grid"
        assert config.n_trials == 50


class TestObjectiveFunction:
    """Tests for ObjectiveFunction."""
    
    @pytest.fixture
    def objective(self):
        manager = MLHyperparameterManager()
        return ObjectiveFunction(
            hp_manager=manager,
            config=OptimizationConfig(),
        )
    
    def test_evaluation_returns_score(self, objective):
        params = {"lstm.hidden_dim": 128}
        score = objective(params)
        assert isinstance(score, float)
    
    def test_evaluation_with_metrics(self, objective):
        params = {"lstm.hidden_dim": 128}
        score, metrics = objective(params, return_metrics=True)
        assert "sharpe_ratio" in metrics
        assert "total_return" in metrics
    
    def test_caching(self, objective):
        params = {"lstm.hidden_dim": 128}
        score1 = objective(params)
        score2 = objective(params)
        assert score1 == score2  # Should be cached


class TestMLOptimizationEngine:
    """Tests for MLOptimizationEngine."""
    
    @pytest.fixture
    def engine(self):
        return create_optimization_engine(preset="balanced")
    
    def test_initialization(self, engine):
        assert engine.hp_manager is not None
    
    def test_optimize_returns_result(self, engine):
        config = OptimizationConfig(
            method="random",
            n_trials=10,  # Quick test
            stages=[OptimizationStage.SIGNALS],
        )
        result = engine.optimize(config)
        assert isinstance(result, FullOptimizationResult)
        assert result.best_score is not None
    
    def test_optimize_full_pipeline(self, engine):
        config = OptimizationConfig(
            method="random",
            n_trials=10,
            stages=[OptimizationStage.FULL],
        )
        result = engine.optimize(config)
        assert result.best_params is not None
    
    def test_stage_optimization(self, engine):
        config = OptimizationConfig(
            method="random",
            n_trials=5,
            stages=[OptimizationStage.POSITION, OptimizationStage.RISK],
        )
        result = engine.optimize(config)
        assert OptimizationStage.POSITION in result.stage_results
        assert OptimizationStage.RISK in result.stage_results
    
    def test_recommendations_generated(self, engine):
        config = OptimizationConfig(method="random", n_trials=10)
        result = engine.optimize(config)
        assert isinstance(result.recommendations, list)


class TestAutoMLTuner:
    """Tests for AutoMLTuner."""
    
    @pytest.fixture
    def tuner(self):
        optimizer = create_optimization_engine()
        pipeline = AdaptiveMLPipeline()
        return AutoMLTuner(optimizer, pipeline)
    
    def test_initialization(self, tuner):
        assert tuner.optimizer is not None
        assert tuner.pipeline is not None
    
    def test_should_reoptimize_time_based(self, tuner):
        tuner._last_optimization = datetime.now() - timedelta(hours=48)
        should, reason = tuner.should_reoptimize()
        assert should is True
        assert "Time" in reason
    
    def test_record_performance(self, tuner):
        decision = MLTradeDecision(
            should_trade=True,
            action="buy",
            signal=MLSignal(
                direction="bullish",
                strength=SignalStrength.BUY,
                confidence=0.7,
            ),
            position_size=MLPositionSize(
                base_size=0.02,
                adjusted_size=0.02,
                max_allowed=0.04,
            ),
            overall_confidence=0.7,
        )
        tuner.record_performance(decision, 0.05)
        assert len(tuner._performance_history) == 1
    
    def test_run_auto_optimization(self, tuner):
        result = tuner.run_auto_optimization(
            OptimizationConfig(method="random", n_trials=5)
        )
        assert result is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestFullMLIntegration:
    """Integration tests for the complete ML pipeline."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from config to trading decision."""
        # 1. Create hyperparameter manager with preset
        preset = create_preset_config("balanced")
        manager = MLHyperparameterManager(base_params=preset)
        
        # 2. Create adaptive pipeline
        pipeline = AdaptiveMLPipeline(hyperparameter_manager=manager)
        
        # 3. Process to get decision
        decision = pipeline.process(symbol="AAPL")
        assert decision is not None
        
        # 4. Optimize parameters
        optimizer = MLOptimizationEngine(hp_manager=manager)
        config = OptimizationConfig(method="random", n_trials=5)
        result = optimizer.optimize(config)
        
        # 5. Apply optimized parameters
        if result.best_params:
            manager._current_params = result.best_params
        
        # 6. Process again with optimized params
        new_decision = pipeline.process(symbol="AAPL")
        assert new_decision is not None
    
    def test_regime_aware_optimization(self):
        """Test optimization adapts to market regimes."""
        manager = MLHyperparameterManager()
        
        # Set different regimes and check param adjustment
        for regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.LOW_VOLATILITY]:
            params = manager.set_regime(regime)
            assert params.optimized_for_regime == regime.value
    
    def test_parameter_persistence(self):
        """Test parameters can be saved and restored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            manager1 = MLHyperparameterManager(config_dir=tmpdir)
            manager1.update_parameter("lstm.hidden_dim", 512)
            manager1.save_current("test_persist")
            
            # Create new manager and load
            manager2 = MLHyperparameterManager(config_dir=tmpdir)
            manager2.load_config("test_persist")
            
            assert manager2.current.lstm.hidden_dim == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
