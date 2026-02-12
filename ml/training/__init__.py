"""
ML Training Module

Provides training pipelines for:
- RL Trading Agent
- Transformer Price Predictor
- Training Orchestration

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from .rl_trainer import (
    RLTrainer,
    RLTrainingConfig,
    TrainingMetrics as RLTrainingMetrics,
    EvaluationResult,
    MarketDataGenerator,
    train_rl_agent,
)

from .transformer_trainer import (
    TransformerTrainer,
    TransformerTrainingConfig,
    TransformerTrainingResult,
    PredictionHorizon,
    DataPreprocessor,
    EarlyStopping,
    LearningRateScheduler,
    SimpleTransformerModel,
    FeatureSet,
    TrainingMetrics as TransformerTrainingMetrics,
)

from .orchestrator import (
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

__all__ = [
    # RL Trainer
    "RLTrainer",
    "RLTrainingConfig", 
    "RLTrainingMetrics",
    "EvaluationResult",
    "MarketDataGenerator",
    "train_rl_agent",
    # Transformer Trainer
    "TransformerTrainer",
    "TransformerTrainingConfig",
    "TransformerTrainingResult",
    "PredictionHorizon",
    "DataPreprocessor",
    "EarlyStopping",
    "LearningRateScheduler",
    "SimpleTransformerModel",
    "FeatureSet",
    "TransformerTrainingMetrics",
    # Orchestrator
    "TrainingOrchestrator",
    "OrchestratorConfig",
    "TrainingJob",
    "TrainingDataset",
    "TrainingReport",
    "TrainingStatus",
    "ModelType",
    "ProgressCallback",
    "RLTrainingResult",
    "create_sample_dataset",
]
