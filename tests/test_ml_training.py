"""
Tests for ML Training Pipelines.

Tests:
- RL Trainer
- Transformer Trainer
- Training Orchestrator
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ================== RL Trainer Tests ==================

class TestRLTrainingConfig:
    """Tests for RL training configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from ml.training.rl_trainer import RLTrainingConfig
        
        config = RLTrainingConfig()
        
        assert config.num_episodes == 1000
        assert config.max_steps_per_episode == 500
        assert config.eval_frequency == 50
        assert config.initial_capital == 100000
    
    def test_custom_config(self):
        """Test custom configuration."""
        from ml.training.rl_trainer import RLTrainingConfig
        
        config = RLTrainingConfig(
            num_episodes=500,
            initial_capital=50000
        )
        
        assert config.num_episodes == 500
        assert config.initial_capital == 50000


class TestTrainingMetrics:
    """Tests for training metrics."""
    
    def test_metrics_creation(self):
        """Test metrics initialization."""
        from ml.training.rl_trainer import TrainingMetrics
        
        metrics = TrainingMetrics(
            episode=10,
            total_reward=100.0,
            avg_reward=10.0,
            portfolio_value=105000,
            max_drawdown=0.05,
            sharpe_ratio=1.5,
            win_rate=0.6,
            num_trades=20,
            epsilon=0.5,
            loss=0.01,
            steps=100,
            duration_seconds=1.5
        )
        
        assert metrics.episode == 10
        assert metrics.total_reward == 100.0
        assert metrics.portfolio_value == 105000
    
    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        from ml.training.rl_trainer import TrainingMetrics
        
        metrics = TrainingMetrics(
            episode=10,
            total_reward=100.0,
            avg_reward=10.0,
            portfolio_value=105000,
            max_drawdown=0.05,
            sharpe_ratio=1.5,
            win_rate=0.6,
            num_trades=20,
            epsilon=0.5,
            loss=0.01,
            steps=100,
            duration_seconds=1.5
        )
        
        d = metrics.to_dict()
        assert d['episode'] == 10
        assert d['total_reward'] == 100.0


class TestMarketDataGenerator:
    """Tests for market data generator."""
    
    def test_generator_creation(self):
        """Test generator initialization."""
        from ml.training.rl_trainer import MarketDataGenerator
        
        gen = MarketDataGenerator()
        assert len(gen.symbols) > 0
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        from ml.training.rl_trainer import MarketDataGenerator
        
        gen = MarketDataGenerator()
        prices = gen.generate_synthetic_data(num_days=100)
        
        assert len(prices) == 100
        assert all(p > 0 for p in prices)
    
    def test_generate_training_episodes(self):
        """Test episode generation."""
        from ml.training.rl_trainer import MarketDataGenerator
        
        gen = MarketDataGenerator()
        episodes = gen.generate_training_episodes(num_episodes=5, episode_length=50)
        
        assert len(episodes) == 5
        assert all(len(ep) == 50 for ep in episodes)


class TestRLTrainer:
    """Tests for RL trainer."""
    
    def test_trainer_creation(self):
        """Test trainer initialization."""
        from ml.training.rl_trainer import RLTrainer, RLTrainingConfig
        
        config = RLTrainingConfig(num_episodes=10)
        trainer = RLTrainer(config)
        
        assert trainer.config == config
        assert trainer.current_episode == 0
    
    def test_training_run(self):
        """Test training execution."""
        from ml.training.rl_trainer import RLTrainer, RLTrainingConfig
        
        config = RLTrainingConfig(
            num_episodes=3,
            max_steps_per_episode=50,
            eval_frequency=100,  # Skip eval
            log_frequency=100,   # Less logging
            early_stopping_patience=1000  # No early stop
        )
        trainer = RLTrainer(config)
        
        result = trainer.train()
        
        assert 'episodes_completed' in result
        assert 'best_reward' in result
        assert 'training_time_seconds' in result


# ================== Transformer Trainer Tests ==================

class TestTransformerTrainingConfig:
    """Tests for Transformer training configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from ml.training.transformer_trainer import TransformerTrainingConfig
        
        config = TransformerTrainingConfig()
        
        assert config.d_model == 64
        assert config.n_heads == 4
        assert config.n_layers == 2
        assert config.sequence_length == 60
        assert config.learning_rate == 0.001
        assert config.batch_size == 32
    
    def test_custom_config(self):
        """Test custom configuration."""
        from ml.training.transformer_trainer import TransformerTrainingConfig
        
        config = TransformerTrainingConfig(
            d_model=128,
            n_heads=8,
            epochs=50
        )
        
        assert config.d_model == 128
        assert config.n_heads == 8
        assert config.epochs == 50


class TestDataPreprocessor:
    """Tests for data preprocessor."""
    
    def test_preprocessor_creation(self):
        """Test preprocessor initialization."""
        from ml.training.transformer_trainer import (
            DataPreprocessor, TransformerTrainingConfig
        )
        
        config = TransformerTrainingConfig()
        preprocessor = DataPreprocessor(config)
        
        assert preprocessor.config == config
        assert not preprocessor.is_fitted
    
    def test_fit_transform(self):
        """Test fit and transform."""
        from ml.training.transformer_trainer import (
            DataPreprocessor, TransformerTrainingConfig
        )
        
        config = TransformerTrainingConfig()
        preprocessor = DataPreprocessor(config)
        
        data = np.random.randn(100, 5)
        normalized = preprocessor.fit_transform(data)
        
        assert preprocessor.is_fitted
        assert normalized.shape == data.shape
        # Check normalization (approximately zero mean, unit variance)
        assert abs(np.mean(normalized)) < 0.2
    
    def test_compute_technical_indicators(self):
        """Test technical indicator computation."""
        from ml.training.transformer_trainer import (
            DataPreprocessor, TransformerTrainingConfig
        )
        
        config = TransformerTrainingConfig()
        preprocessor = DataPreprocessor(config)
        
        # OHLCV data
        prices = np.abs(np.random.randn(100, 5)) + 100
        indicators = preprocessor.compute_technical_indicators(prices)
        
        assert indicators.shape[0] == 100
        assert indicators.shape[1] >= 10  # Multiple indicators
    
    def test_compute_volatility_features(self):
        """Test volatility feature computation."""
        from ml.training.transformer_trainer import (
            DataPreprocessor, TransformerTrainingConfig
        )
        
        config = TransformerTrainingConfig()
        preprocessor = DataPreprocessor(config)
        
        prices = np.abs(np.random.randn(100, 5)) + 100
        vol_features = preprocessor.compute_volatility_features(prices)
        
        assert vol_features.shape[0] == 100
        assert vol_features.shape[1] == 6  # 6 volatility features
    
    def test_compute_time_features(self):
        """Test time feature computation."""
        from ml.training.transformer_trainer import (
            DataPreprocessor, TransformerTrainingConfig
        )
        
        config = TransformerTrainingConfig()
        preprocessor = DataPreprocessor(config)
        
        timestamps = [datetime.now() + timedelta(minutes=i) for i in range(100)]
        time_features = preprocessor.compute_time_features(timestamps)
        
        assert time_features.shape == (100, 8)
    
    def test_create_sequences(self):
        """Test sequence creation."""
        from ml.training.transformer_trainer import (
            DataPreprocessor, TransformerTrainingConfig
        )
        
        config = TransformerTrainingConfig(sequence_length=10)
        preprocessor = DataPreprocessor(config)
        
        features = np.random.randn(100, 5)
        targets = np.random.randn(100)
        
        X, y = preprocessor.create_sequences(features, targets, 10)
        
        assert X.shape[0] == 90  # 100 - 10
        assert X.shape[1] == 10  # sequence_length
        assert X.shape[2] == 5  # features


class TestSimpleTransformerModel:
    """Tests for simple transformer model."""
    
    def test_model_creation(self):
        """Test model initialization."""
        from ml.training.transformer_trainer import (
            SimpleTransformerModel, TransformerTrainingConfig
        )
        
        config = TransformerTrainingConfig()
        model = SimpleTransformerModel(config)
        
        assert not model.is_trained
        assert 'input_proj' in model.weights
        assert 'output_proj' in model.weights
    
    def test_forward_pass(self):
        """Test forward pass."""
        from ml.training.transformer_trainer import (
            SimpleTransformerModel, TransformerTrainingConfig
        )
        
        config = TransformerTrainingConfig(feature_dim=10, sequence_length=20)
        model = SimpleTransformerModel(config)
        
        x = np.random.randn(8, 20, 10)  # batch, seq, features
        output = model.forward(x)
        
        assert output.shape == (8, config.prediction_steps)
    
    def test_compute_loss(self):
        """Test loss computation."""
        from ml.training.transformer_trainer import (
            SimpleTransformerModel, TransformerTrainingConfig
        )
        
        config = TransformerTrainingConfig()
        model = SimpleTransformerModel(config)
        
        predictions = np.random.randn(16, 1)
        targets = np.random.randn(16, 1)
        
        loss, metrics = model.compute_loss(predictions, targets)
        
        assert isinstance(loss, float)
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'direction_accuracy' in metrics


class TestEarlyStopping:
    """Tests for early stopping."""
    
    def test_early_stopping_creation(self):
        """Test early stopping initialization."""
        from ml.training.transformer_trainer import EarlyStopping
        
        es = EarlyStopping(patience=10)
        
        assert es.patience == 10
        assert not es.should_stop
    
    def test_early_stopping_improvement(self):
        """Test with improving loss."""
        from ml.training.transformer_trainer import EarlyStopping
        
        es = EarlyStopping(patience=5)
        
        # Improving losses
        for loss in [1.0, 0.9, 0.8, 0.7, 0.6]:
            es.check(loss)
        
        assert not es.should_stop
        assert es.counter == 0
    
    def test_early_stopping_triggered(self):
        """Test early stopping triggered."""
        from ml.training.transformer_trainer import EarlyStopping
        
        es = EarlyStopping(patience=3)
        
        es.check(1.0)  # Best
        es.check(1.1)  # Worse
        es.check(1.2)  # Worse
        es.check(1.3)  # Worse - should trigger
        
        assert es.should_stop


class TestLearningRateScheduler:
    """Tests for learning rate scheduler."""
    
    def test_scheduler_creation(self):
        """Test scheduler initialization."""
        from ml.training.transformer_trainer import LearningRateScheduler
        
        scheduler = LearningRateScheduler(initial_lr=0.001)
        
        assert scheduler.current_lr == 0.001
    
    def test_lr_decay(self):
        """Test learning rate decay."""
        from ml.training.transformer_trainer import LearningRateScheduler
        
        scheduler = LearningRateScheduler(
            initial_lr=0.001,
            patience=2,
            decay_factor=0.5
        )
        
        scheduler.step(1.0)  # Best
        scheduler.step(1.1)  # Worse
        scheduler.step(1.2)  # Worse - decay
        
        assert scheduler.current_lr == 0.0005


class TestTransformerTrainer:
    """Tests for Transformer trainer."""
    
    def test_trainer_creation(self):
        """Test trainer initialization."""
        from ml.training.transformer_trainer import (
            TransformerTrainer, TransformerTrainingConfig
        )
        
        config = TransformerTrainingConfig()
        trainer = TransformerTrainer(config)
        
        assert trainer.config == config
    
    def test_prepare_data(self):
        """Test data preparation."""
        from ml.training.transformer_trainer import (
            TransformerTrainer, TransformerTrainingConfig, PredictionHorizon
        )
        
        config = TransformerTrainingConfig()
        trainer = TransformerTrainer(config)
        
        prices = np.abs(np.random.randn(200, 5)) + 100
        timestamps = [datetime.now() + timedelta(minutes=i) for i in range(200)]
        
        feature_set = trainer.prepare_data(
            prices, timestamps, PredictionHorizon.MIN_15
        )
        
        assert feature_set.prices.shape == prices.shape
        assert len(feature_set.timestamps) == 200
    
    def test_training_run(self):
        """Test training execution."""
        from ml.training.transformer_trainer import (
            TransformerTrainer, TransformerTrainingConfig, PredictionHorizon
        )
        
        config = TransformerTrainingConfig(
            epochs=3,
            sequence_length=20,
            batch_size=8,
            early_stopping_patience=2
        )
        trainer = TransformerTrainer(config)
        
        prices = np.abs(np.random.randn(200, 5)) + 100
        feature_set = trainer.prepare_data(prices, horizon=PredictionHorizon.MIN_15)
        
        result = trainer.train(feature_set)
        
        assert result.training_completed
        assert len(result.training_history) > 0
        assert result.training_time_seconds > 0


# ================== Orchestrator Tests ==================

class TestOrchestratorConfig:
    """Tests for orchestrator configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from ml.training.orchestrator import OrchestratorConfig
        
        config = OrchestratorConfig()
        
        assert config.models_dir == "models"
        assert config.max_parallel_jobs == 2
        assert config.keep_n_best_models == 5


class TestTrainingJob:
    """Tests for training job."""
    
    def test_job_creation(self):
        """Test job initialization."""
        from ml.training.orchestrator import TrainingJob, ModelType, TrainingStatus
        
        job = TrainingJob(
            job_id="test_job_001",
            model_type=ModelType.RL_AGENT,
            config={"epochs": 100}
        )
        
        assert job.job_id == "test_job_001"
        assert job.model_type == ModelType.RL_AGENT
        assert job.status == TrainingStatus.PENDING
    
    def test_job_to_dict(self):
        """Test job serialization."""
        from ml.training.orchestrator import TrainingJob, ModelType
        
        job = TrainingJob(
            job_id="test_job_001",
            model_type=ModelType.RL_AGENT,
            config={}
        )
        
        job_dict = job.to_dict()
        
        assert job_dict['job_id'] == "test_job_001"
        assert job_dict['model_type'] == "rl_agent"
        assert job_dict['status'] == "pending"


class TestTrainingDataset:
    """Tests for training dataset."""
    
    def test_dataset_creation(self):
        """Test dataset initialization."""
        from ml.training.orchestrator import TrainingDataset
        
        prices = np.random.randn(100, 5)
        dataset = TrainingDataset(
            name="test_dataset",
            prices=prices,
            symbols=["SPY"]
        )
        
        assert dataset.name == "test_dataset"
        assert dataset.n_samples == 100
        assert dataset.n_features == 5


class TestTrainingReport:
    """Tests for training report."""
    
    def test_report_creation(self):
        """Test report initialization."""
        from ml.training.orchestrator import TrainingReport
        
        report = TrainingReport(
            orchestrator_id="test_orch_001",
            started_at=datetime.now(),
            completed_at=datetime.now(),
            total_jobs=3,
            successful_jobs=2,
            failed_jobs=1
        )
        
        assert report.total_jobs == 3
        assert report.successful_jobs == 2
    
    def test_report_to_json(self):
        """Test report JSON serialization."""
        from ml.training.orchestrator import TrainingReport
        import json
        
        report = TrainingReport(
            orchestrator_id="test_orch_001",
            started_at=datetime.now(),
            completed_at=datetime.now(),
            total_jobs=3,
            successful_jobs=2,
            failed_jobs=1
        )
        
        json_str = report.to_json()
        parsed = json.loads(json_str)
        
        assert parsed['total_jobs'] == 3


class TestSampleDataset:
    """Tests for sample dataset generation."""
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation."""
        from ml.training.orchestrator import create_sample_dataset
        
        dataset = create_sample_dataset(n_samples=500)
        
        assert dataset.name == "sample_dataset"
        assert dataset.n_samples == 500
        assert dataset.n_features == 5  # OHLCV
        assert len(dataset.timestamps) == 500


class TestProgressCallback:
    """Tests for progress callback."""
    
    def test_callback_creation(self):
        """Test callback initialization."""
        from ml.training.orchestrator import (
            ProgressCallback, TrainingJob, ModelType
        )
        
        job = TrainingJob(
            job_id="test_001",
            model_type=ModelType.RL_AGENT,
            config={}
        )
        callback = ProgressCallback(job)
        
        assert callback.job == job
    
    def test_callback_update(self):
        """Test callback update."""
        from ml.training.orchestrator import (
            ProgressCallback, TrainingJob, ModelType
        )
        
        job = TrainingJob(
            job_id="test_001",
            model_type=ModelType.RL_AGENT,
            config={}
        )
        callback = ProgressCallback(job, update_interval=0)
        
        callback.update(0.5, {"loss": 0.1})
        
        assert job.progress == 0.5
        assert job.metrics.get("loss") == 0.1


class TestTrainingOrchestrator:
    """Tests for training orchestrator."""
    
    def test_orchestrator_creation(self):
        """Test orchestrator initialization."""
        from ml.training.orchestrator import TrainingOrchestrator, OrchestratorConfig
        
        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)
        
        assert orchestrator.config == config
        assert len(orchestrator.jobs) == 0
    
    def test_create_job(self):
        """Test job creation."""
        from ml.training.orchestrator import (
            TrainingOrchestrator, ModelType, TrainingStatus
        )
        
        orchestrator = TrainingOrchestrator()
        job = orchestrator.create_job(
            ModelType.RL_AGENT,
            {"epochs": 100}
        )
        
        assert job.model_type == ModelType.RL_AGENT
        assert job.status == TrainingStatus.PENDING
        assert job.job_id in orchestrator.jobs
    
    def test_list_jobs(self):
        """Test listing jobs."""
        from ml.training.orchestrator import (
            TrainingOrchestrator, ModelType, TrainingStatus
        )
        
        orchestrator = TrainingOrchestrator()
        orchestrator.create_job(ModelType.RL_AGENT, {})
        orchestrator.create_job(ModelType.TRANSFORMER, {})
        
        all_jobs = orchestrator.list_jobs()
        assert len(all_jobs) == 2
        
        pending_jobs = orchestrator.list_jobs(TrainingStatus.PENDING)
        assert len(pending_jobs) == 2
    
    def test_cancel_job(self):
        """Test job cancellation."""
        from ml.training.orchestrator import (
            TrainingOrchestrator, ModelType, TrainingStatus
        )
        
        orchestrator = TrainingOrchestrator()
        job = orchestrator.create_job(ModelType.RL_AGENT, {})
        
        result = orchestrator.cancel_job(job.job_id)
        
        assert result
        assert job.status == TrainingStatus.CANCELLED
    
    def test_train_transformer(self):
        """Test Transformer training via orchestrator."""
        from ml.training.orchestrator import TrainingOrchestrator, create_sample_dataset
        from ml.training.transformer_trainer import (
            TransformerTrainingConfig, PredictionHorizon
        )
        
        orchestrator = TrainingOrchestrator()
        dataset = create_sample_dataset(n_samples=200)
        
        config = TransformerTrainingConfig(
            epochs=2,
            sequence_length=20,
            batch_size=8
        )
        result = orchestrator.train_transformer(
            dataset, config, PredictionHorizon.MIN_15
        )
        
        assert result.training_completed
    
    def test_get_best_model(self):
        """Test getting best model."""
        from ml.training.orchestrator import TrainingOrchestrator, ModelType
        
        orchestrator = TrainingOrchestrator()
        
        # Initially no models
        assert orchestrator.get_best_model(ModelType.RL_AGENT) is None
        
        # Add a model version
        orchestrator.model_versions[ModelType.RL_AGENT.value] = ["model_v1", "model_v2"]
        
        best = orchestrator.get_best_model(ModelType.RL_AGENT)
        assert best == "model_v2"


class TestRLTrainingResult:
    """Tests for RL training result."""
    
    def test_result_creation(self):
        """Test result initialization."""
        from ml.training.orchestrator import RLTrainingResult
        
        result = RLTrainingResult(
            model_id="test_001",
            training_completed=True,
            episodes_completed=100,
            best_reward=500.0,
            final_epsilon=0.1,
            training_time_seconds=60.0
        )
        
        assert result.model_id == "test_001"
        assert result.training_completed
        assert result.best_reward == 500.0
    
    def test_result_from_dict(self):
        """Test creating result from dict."""
        from ml.training.orchestrator import RLTrainingResult
        
        data = {
            'episodes_completed': 100,
            'best_reward': 500.0,
            'final_epsilon': 0.1,
            'training_time_seconds': 60.0
        }
        
        result = RLTrainingResult.from_dict(data, "test_model")
        
        assert result.model_id == "test_model"
        assert result.episodes_completed == 100


class TestIntegration:
    """Integration tests for ML training pipeline."""
    
    def test_transformer_training_pipeline(self):
        """Test complete transformer training pipeline."""
        from ml.training import (
            TrainingOrchestrator,
            TransformerTrainingConfig,
            create_sample_dataset,
            PredictionHorizon
        )
        
        # Create orchestrator
        orchestrator = TrainingOrchestrator()
        
        # Create dataset
        dataset = create_sample_dataset(n_samples=200)
        
        # Train Transformer
        transformer_config = TransformerTrainingConfig(
            epochs=2, sequence_length=20, batch_size=8
        )
        transformer_result = orchestrator.train_transformer(
            dataset, transformer_config, PredictionHorizon.MIN_15
        )
        assert transformer_result.training_completed
        
        # Check jobs
        jobs = orchestrator.list_jobs()
        assert len(jobs) == 1
    
    def test_module_imports(self):
        """Test all module imports work correctly."""
        from ml.training import (
            # RL Trainer
            RLTrainer,
            RLTrainingConfig,
            RLTrainingMetrics,
            EvaluationResult,
            MarketDataGenerator,
            train_rl_agent,
            # Transformer Trainer
            TransformerTrainer,
            TransformerTrainingConfig,
            TransformerTrainingResult,
            PredictionHorizon,
            DataPreprocessor,
            EarlyStopping,
            LearningRateScheduler,
            SimpleTransformerModel,
            FeatureSet,
            TransformerTrainingMetrics,
            # Orchestrator
            TrainingOrchestrator,
            OrchestratorConfig,
            TrainingJob,
            TrainingDataset,
            TrainingReport,
            TrainingStatus,
            ModelType,
            ProgressCallback,
            RLTrainingResult,
            create_sample_dataset,
        )
        
        # All imports should succeed
        assert RLTrainer is not None
        assert TransformerTrainer is not None
        assert TrainingOrchestrator is not None
