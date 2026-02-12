"""
ML Training Orchestrator for GNOSIS Platform.

Central orchestration system for training all ML models:
- RL Trading Agent (DQN-based)
- Transformer Price Predictor
- Ensemble model coordination

Features:
- Parallel training support
- Resource management
- Progress tracking
- Model versioning
- Training scheduling
- Performance monitoring
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import logging
import time
import json
import numpy as np
from pathlib import Path

from .rl_trainer import RLTrainer, RLTrainingConfig
from .transformer_trainer import (
    TransformerTrainer, TransformerTrainingConfig, 
    TransformerTrainingResult, PredictionHorizon
)

logger = logging.getLogger(__name__)


@dataclass
class RLTrainingResult:
    """Result container for RL training."""
    model_id: str
    training_completed: bool
    episodes_completed: int
    best_reward: float
    final_epsilon: float
    training_time_seconds: float
    best_metrics: Dict[str, float] = field(default_factory=dict)
    model_path: Optional[str] = None
    final_evaluation: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], model_id: str) -> 'RLTrainingResult':
        """Create from training summary dict."""
        return cls(
            model_id=model_id,
            training_completed=True,
            episodes_completed=data.get('episodes_completed', 0),
            best_reward=data.get('best_reward', 0.0),
            final_epsilon=data.get('final_epsilon', 0.0),
            training_time_seconds=data.get('training_time_seconds', 0.0),
            best_metrics={
                'best_reward': data.get('best_reward', 0.0),
                'episodes_completed': data.get('episodes_completed', 0),
            },
            model_path=None,
            final_evaluation=data.get('final_evaluation')
        )


class TrainingStatus(Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelType(Enum):
    """Supported model types."""
    RL_AGENT = "rl_agent"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"


@dataclass
class TrainingJob:
    """Training job specification."""
    job_id: str
    model_type: ModelType
    config: Dict[str, Any]
    status: TrainingStatus = TrainingStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    model_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'job_id': self.job_id,
            'model_type': self.model_type.value,
            'config': self.config,
            'status': self.status.value,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'progress': self.progress,
            'metrics': self.metrics,
            'error_message': self.error_message,
            'model_path': self.model_path
        }


@dataclass
class OrchestratorConfig:
    """Configuration for training orchestrator."""
    # Directories
    models_dir: str = "models"
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs/training"
    
    # Resource limits
    max_parallel_jobs: int = 2
    max_memory_gb: float = 8.0
    
    # Training defaults
    default_epochs: int = 100
    default_batch_size: int = 32
    early_stopping_patience: int = 10
    
    # Model versioning
    keep_n_best_models: int = 5
    model_naming_format: str = "{model_type}_{timestamp}_{version}"
    
    # Scheduling
    training_timeout_hours: float = 24.0
    retry_failed_jobs: bool = True
    max_retries: int = 3
    
    # Monitoring
    log_interval_seconds: float = 60.0
    checkpoint_interval_epochs: int = 10


@dataclass
class TrainingDataset:
    """Dataset for training."""
    name: str
    prices: np.ndarray  # OHLCV data
    timestamps: List[datetime] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_samples(self) -> int:
        return len(self.prices)
    
    @property
    def n_features(self) -> int:
        return self.prices.shape[1] if len(self.prices.shape) > 1 else 1


@dataclass
class TrainingReport:
    """Comprehensive training report."""
    orchestrator_id: str
    started_at: datetime
    completed_at: datetime
    total_jobs: int
    successful_jobs: int
    failed_jobs: int
    jobs: List[TrainingJob] = field(default_factory=list)
    best_models: Dict[str, str] = field(default_factory=dict)  # model_type -> path
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'orchestrator_id': self.orchestrator_id,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat(),
            'total_jobs': self.total_jobs,
            'successful_jobs': self.successful_jobs,
            'failed_jobs': self.failed_jobs,
            'jobs': [j.to_dict() for j in self.jobs],
            'best_models': self.best_models,
            'aggregate_metrics': self.aggregate_metrics
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ProgressCallback:
    """Progress callback for training jobs."""
    
    def __init__(self, job: TrainingJob, update_interval: float = 1.0):
        self.job = job
        self.update_interval = update_interval
        self.last_update = time.time()
        self.callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable) -> None:
        """Add a progress callback function."""
        self.callbacks.append(callback)
    
    def update(self, progress: float, metrics: Optional[Dict[str, float]] = None) -> None:
        """Update progress."""
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.job.progress = progress
            if metrics:
                self.job.metrics.update(metrics)
            
            for callback in self.callbacks:
                try:
                    callback(self.job)
                except Exception as e:
                    logger.warning(f"Callback error: {e}")
            
            self.last_update = current_time


class TrainingOrchestrator:
    """
    Central orchestrator for ML model training.
    
    Manages training jobs, resource allocation, and model versioning.
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self.jobs: Dict[str, TrainingJob] = {}
        self.rl_trainer: Optional[RLTrainer] = None
        self.transformer_trainer: Optional[TransformerTrainer] = None
        self.model_versions: Dict[str, List[str]] = {}
        self.orchestrator_id = f"orch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create directories
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        for dir_path in [
            self.config.models_dir,
            self.config.checkpoints_dir,
            self.config.logs_dir
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def create_job(
        self,
        model_type: ModelType,
        config: Optional[Dict[str, Any]] = None
    ) -> TrainingJob:
        """
        Create a new training job.
        
        Args:
            model_type: Type of model to train
            config: Training configuration
            
        Returns:
            Created TrainingJob
        """
        job_id = f"{model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        job = TrainingJob(
            job_id=job_id,
            model_type=model_type,
            config=config or {}
        )
        
        self.jobs[job_id] = job
        logger.info(f"Created training job: {job_id}")
        
        return job
    
    def train_rl_agent(
        self,
        dataset: TrainingDataset,
        config: Optional[RLTrainingConfig] = None,
        progress_callback: Optional[Callable] = None
    ) -> RLTrainingResult:
        """
        Train RL trading agent.
        
        Args:
            dataset: Training dataset
            config: RL training configuration
            progress_callback: Optional progress callback
            
        Returns:
            Training result
        """
        job = self.create_job(ModelType.RL_AGENT, {})
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.now()
        model_id = f"rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Initialize trainer with config
            self.rl_trainer = RLTrainer(config)
            
            # Create progress callback
            if progress_callback:
                callback = ProgressCallback(job)
                callback.add_callback(progress_callback)
            
            # Convert dataset to price list format expected by RLTrainer
            # The RLTrainer expects List[List[float]] for price_data
            price_list = dataset.prices.tolist() if hasattr(dataset.prices, 'tolist') else list(dataset.prices)
            price_data = [price_list]  # Wrap as list of episodes
            
            # Train - returns Dict[str, Any]
            train_result = self.rl_trainer.train(price_data=price_data)
            
            # Convert to RLTrainingResult
            result = RLTrainingResult.from_dict(train_result, model_id)
            
            # Update job
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 1.0
            job.metrics = result.best_metrics
            job.model_path = result.model_path
            
            # Version management
            self._register_model(ModelType.RL_AGENT, result.model_id, result.model_path)
            
            logger.info(f"RL training completed: {result.model_id}")
            return result
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            logger.error(f"RL training failed: {e}")
            raise
    
    def train_transformer(
        self,
        dataset: TrainingDataset,
        config: Optional[TransformerTrainingConfig] = None,
        horizon: PredictionHorizon = PredictionHorizon.MIN_15,
        progress_callback: Optional[Callable] = None
    ) -> TransformerTrainingResult:
        """
        Train transformer price predictor.
        
        Args:
            dataset: Training dataset
            config: Transformer training configuration
            horizon: Prediction horizon
            progress_callback: Optional progress callback
            
        Returns:
            Training result
        """
        job = self.create_job(
            ModelType.TRANSFORMER, 
            config.__dict__ if config else {}
        )
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.now()
        
        try:
            # Initialize trainer
            self.transformer_trainer = TransformerTrainer(config)
            
            # Prepare data
            feature_set = self.transformer_trainer.prepare_data(
                dataset.prices,
                dataset.timestamps,
                horizon
            )
            
            # Train
            result = self.transformer_trainer.train(feature_set)
            
            # Update job
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 1.0
            job.metrics = result.test_metrics
            job.model_path = result.model_path
            
            # Version management
            self._register_model(ModelType.TRANSFORMER, result.model_id, result.model_path)
            
            logger.info(f"Transformer training completed: {result.model_id}")
            return result
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            logger.error(f"Transformer training failed: {e}")
            raise
    
    def train_all(
        self,
        dataset: TrainingDataset,
        rl_config: Optional[RLTrainingConfig] = None,
        transformer_config: Optional[TransformerTrainingConfig] = None,
        progress_callback: Optional[Callable] = None
    ) -> TrainingReport:
        """
        Train all model types.
        
        Args:
            dataset: Training dataset
            rl_config: RL training configuration
            transformer_config: Transformer training configuration
            progress_callback: Optional progress callback
            
        Returns:
            Training report
        """
        start_time = datetime.now()
        successful = 0
        failed = 0
        best_models = {}
        all_jobs = []
        
        # Train RL Agent
        try:
            rl_result = self.train_rl_agent(dataset, rl_config, progress_callback)
            successful += 1
            best_models[ModelType.RL_AGENT.value] = rl_result.model_id
            all_jobs.append(self.jobs[list(self.jobs.keys())[-1]])
        except Exception as e:
            logger.error(f"RL training failed: {e}")
            failed += 1
        
        # Train Transformer for multiple horizons
        horizons = [PredictionHorizon.MIN_15, PredictionHorizon.HOUR_1]
        for horizon in horizons:
            try:
                transformer_result = self.train_transformer(
                    dataset, transformer_config, horizon, progress_callback
                )
                successful += 1
                best_models[f"{ModelType.TRANSFORMER.value}_{horizon.value}"] = transformer_result.model_id
                all_jobs.append(self.jobs[list(self.jobs.keys())[-1]])
            except Exception as e:
                logger.error(f"Transformer training for {horizon.value} failed: {e}")
                failed += 1
        
        end_time = datetime.now()
        
        # Aggregate metrics
        aggregate_metrics = self._compute_aggregate_metrics(all_jobs)
        
        report = TrainingReport(
            orchestrator_id=self.orchestrator_id,
            started_at=start_time,
            completed_at=end_time,
            total_jobs=successful + failed,
            successful_jobs=successful,
            failed_jobs=failed,
            jobs=all_jobs,
            best_models=best_models,
            aggregate_metrics=aggregate_metrics
        )
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _register_model(
        self,
        model_type: ModelType,
        model_id: str,
        model_path: Optional[str]
    ) -> None:
        """Register a trained model version."""
        if model_type.value not in self.model_versions:
            self.model_versions[model_type.value] = []
        
        self.model_versions[model_type.value].append(model_id)
        
        # Cleanup old versions
        versions = self.model_versions[model_type.value]
        if len(versions) > self.config.keep_n_best_models:
            old_versions = versions[:-self.config.keep_n_best_models]
            self.model_versions[model_type.value] = versions[-self.config.keep_n_best_models:]
            
            for old_version in old_versions:
                logger.info(f"Cleaning up old model version: {old_version}")
    
    def _compute_aggregate_metrics(
        self, jobs: List[TrainingJob]
    ) -> Dict[str, float]:
        """Compute aggregate metrics from jobs."""
        metrics = {}
        
        all_metrics = {}
        for job in jobs:
            if job.status == TrainingStatus.COMPLETED:
                for key, value in job.metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        for key, values in all_metrics.items():
            metrics[f"avg_{key}"] = float(np.mean(values))
            metrics[f"best_{key}"] = float(min(values) if 'loss' in key else max(values))
        
        metrics['total_training_time'] = sum(
            (j.completed_at - j.started_at).total_seconds()
            for j in jobs
            if j.completed_at and j.started_at
        )
        
        return metrics
    
    def _save_report(self, report: TrainingReport) -> None:
        """Save training report to file."""
        report_path = Path(self.config.logs_dir) / f"report_{report.orchestrator_id}.json"
        with open(report_path, 'w') as f:
            f.write(report.to_json())
        logger.info(f"Training report saved: {report_path}")
    
    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get status of a training job."""
        return self.jobs.get(job_id)
    
    def list_jobs(
        self,
        status: Optional[TrainingStatus] = None
    ) -> List[TrainingJob]:
        """List training jobs, optionally filtered by status."""
        jobs = list(self.jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [TrainingStatus.PENDING, TrainingStatus.RUNNING]:
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.now()
            logger.info(f"Cancelled job: {job_id}")
            return True
        
        return False
    
    def get_best_model(self, model_type: ModelType) -> Optional[str]:
        """Get the best model version for a model type."""
        versions = self.model_versions.get(model_type.value, [])
        return versions[-1] if versions else None
    
    def list_models(
        self, model_type: Optional[ModelType] = None
    ) -> Dict[str, List[str]]:
        """List all registered model versions."""
        if model_type:
            return {model_type.value: self.model_versions.get(model_type.value, [])}
        return self.model_versions.copy()


def create_sample_dataset(n_samples: int = 1000) -> TrainingDataset:
    """Create a sample dataset for testing."""
    np.random.seed(42)
    
    # Generate synthetic OHLCV data
    base_price = 100.0
    prices = []
    timestamps = []
    
    current_price = base_price
    base_time = datetime.now()
    
    for i in range(n_samples):
        # Random walk for price
        change = np.random.randn() * 0.02
        current_price *= (1 + change)
        
        # Generate OHLCV
        open_price = current_price * (1 + np.random.randn() * 0.005)
        high = max(open_price, current_price) * (1 + abs(np.random.randn()) * 0.01)
        low = min(open_price, current_price) * (1 - abs(np.random.randn()) * 0.01)
        close = current_price
        volume = 1000 * (1 + abs(np.random.randn()) * 0.5)
        
        prices.append([open_price, high, low, close, volume])
        timestamps.append(base_time.replace(hour=i % 24, minute=(i * 5) % 60))
    
    return TrainingDataset(
        name="sample_dataset",
        prices=np.array(prices),
        timestamps=timestamps,
        symbols=["SPY"],
        metadata={"source": "synthetic", "n_samples": n_samples}
    )


# Export classes
__all__ = [
    'TrainingStatus',
    'ModelType',
    'TrainingJob',
    'OrchestratorConfig',
    'TrainingDataset',
    'TrainingReport',
    'ProgressCallback',
    'TrainingOrchestrator',
    'create_sample_dataset'
]
