"""
ML Integration Module - Machine Learning Pipeline for Trading

This module provides comprehensive ML integration throughout the trading process:

Components:
- hyperparameter_manager: Central hyperparameter configuration and management
- adaptive_pipeline: ML-integrated trading decision pipeline
- optimization_engine: End-to-end hyperparameter optimization

Usage:
    from ml import (
        MLHyperparameterManager,
        AdaptiveMLPipeline,
        MLOptimizationEngine,
        create_preset_config,
        create_optimization_engine,
    )
    
    # Create with preset
    manager = MLHyperparameterManager(create_preset_config("balanced"))
    pipeline = AdaptiveMLPipeline(hyperparameter_manager=manager)
    
    # Run optimization
    optimizer = create_optimization_engine(preset="balanced")
    results = optimizer.optimize()

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

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

from ml.pipeline_integration import (
    MLPipelineConfig,
    TrainingResult,
    PipelineState,
    IntegratedMLPipeline,
    create_integrated_pipeline,
)


__all__ = [
    # Hyperparameter Management
    "MarketRegime",
    "ParameterScope",
    "ParameterSpec",
    "LSTMHyperparameters",
    "FeatureHyperparameters",
    "SignalHyperparameters",
    "PositionSizingHyperparameters",
    "RiskManagementHyperparameters",
    "StrategySelectionHyperparameters",
    "MLHyperparameterSet",
    "MLHyperparameterManager",
    "create_preset_config",
    
    # Adaptive Pipeline
    "SignalStrength",
    "MLSignal",
    "MLPositionSize",
    "MLTradeDecision",
    "RegimeDetector",
    "AdaptiveMLPipeline",
    
    # Optimization
    "OptimizationStage",
    "OptimizationMetric",
    "OptimizationConfig",
    "StageOptimizationResult",
    "FullOptimizationResult",
    "ObjectiveFunction",
    "MLOptimizationEngine",
    "AutoMLTuner",
    "create_optimization_engine",
    
    # Pipeline Integration
    "MLPipelineConfig",
    "TrainingResult",
    "PipelineState",
    "IntegratedMLPipeline",
    "create_integrated_pipeline",
]
