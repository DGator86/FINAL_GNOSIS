"""
Model Registry for ML Model Version Control.

Provides centralized model artifact management with:
- Version tracking and metadata storage
- A/B testing support
- Model promotion workflows
- Performance tracking per version
- Rollback capabilities

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

from loguru import logger


class ModelStage(str, Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelType(str, Enum):
    """Supported model types."""
    LSTM_LOOKAHEAD = "lstm_lookahead"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"
    REINFORCEMENT = "reinforcement"
    CUSTOM = "custom"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    # Training metrics
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    
    # Trading-specific metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Prediction metrics
    direction_accuracy: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    r2_score: float = 0.0
    
    # Timing
    inference_time_ms: float = 0.0
    training_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetrics":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelVersion:
    """Model version metadata."""
    version: str
    model_type: ModelType
    stage: ModelStage
    created_at: datetime
    
    # Model info
    model_path: str
    model_hash: str
    model_size_bytes: int
    
    # Training info
    training_config: Dict[str, Any] = field(default_factory=dict)
    training_data_hash: str = ""
    feature_columns: List[str] = field(default_factory=list)
    
    # Performance
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    author: str = ""
    parent_version: Optional[str] = None
    
    # A/B testing
    traffic_percentage: float = 0.0  # For canary deployments
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["model_type"] = self.model_type.value
        data["stage"] = self.stage.value
        data["metrics"] = self.metrics.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["model_type"] = ModelType(data["model_type"])
        data["stage"] = ModelStage(data["stage"])
        data["metrics"] = ModelMetrics.from_dict(data.get("metrics", {}))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    test_id: str
    name: str
    description: str
    start_time: datetime
    end_time: Optional[datetime]
    
    # Versions being tested
    control_version: str
    treatment_versions: Dict[str, float]  # version -> traffic percentage
    
    # Status
    is_active: bool = True
    winner: Optional[str] = None
    
    # Results
    version_metrics: Dict[str, ModelMetrics] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["start_time"] = self.start_time.isoformat()
        data["end_time"] = self.end_time.isoformat() if self.end_time else None
        data["version_metrics"] = {
            k: v.to_dict() for k, v in self.version_metrics.items()
        }
        return data


class ModelRegistry:
    """
    Central registry for ML model version control.
    
    Features:
    - Version tracking with semantic versioning
    - Model artifact storage and retrieval
    - A/B testing support
    - Model promotion workflows
    - Performance tracking
    - Rollback capabilities
    """
    
    def __init__(
        self,
        registry_path: str = "models/registry",
        artifacts_path: str = "models/artifacts",
    ):
        """
        Initialize Model Registry.
        
        Args:
            registry_path: Path to registry metadata storage
            artifacts_path: Path to model artifact storage
        """
        self.registry_path = Path(registry_path)
        self.artifacts_path = Path(artifacts_path)
        
        # Create directories
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._versions: Dict[str, Dict[str, ModelVersion]] = {}  # model_name -> {version -> ModelVersion}
        self._ab_tests: Dict[str, ABTestConfig] = {}
        
        # Load existing registry
        self._load_registry()
        
        logger.info(
            f"ModelRegistry initialized | "
            f"registry_path={self.registry_path} | "
            f"models={len(self._versions)}"
        )
    
    def _load_registry(self):
        """Load registry from disk."""
        registry_file = self.registry_path / "registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, "r") as f:
                    data = json.load(f)
                
                # Load versions
                for model_name, versions in data.get("versions", {}).items():
                    self._versions[model_name] = {
                        v: ModelVersion.from_dict(d) for v, d in versions.items()
                    }
                
                logger.debug(f"Loaded registry with {len(self._versions)} models")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
    
    def _save_registry(self):
        """Save registry to disk."""
        registry_file = self.registry_path / "registry.json"
        
        data = {
            "versions": {
                model_name: {v: ver.to_dict() for v, ver in versions.items()}
                for model_name, versions in self._versions.items()
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        with open(registry_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]
    
    def _get_next_version(self, model_name: str, bump: str = "patch") -> str:
        """Get next semantic version for a model."""
        versions = self._versions.get(model_name, {})
        
        if not versions:
            return "1.0.0"
        
        # Parse latest version
        latest = max(versions.keys(), key=lambda v: [int(x) for x in v.split(".")])
        parts = [int(x) for x in latest.split(".")]
        
        if bump == "major":
            parts[0] += 1
            parts[1] = 0
            parts[2] = 0
        elif bump == "minor":
            parts[1] += 1
            parts[2] = 0
        else:  # patch
            parts[2] += 1
        
        return ".".join(str(p) for p in parts)
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        model_type: ModelType = ModelType.LSTM_LOOKAHEAD,
        version: Optional[str] = None,
        version_bump: str = "patch",
        training_config: Optional[Dict[str, Any]] = None,
        feature_columns: Optional[List[str]] = None,
        metrics: Optional[ModelMetrics] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        author: str = "",
        parent_version: Optional[str] = None,
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model (e.g., "lstm_spy_1min")
            model_path: Path to the model file
            model_type: Type of model
            version: Explicit version string (auto-generated if None)
            version_bump: Version bump type if auto-generating ("major", "minor", "patch")
            training_config: Training configuration used
            feature_columns: List of feature column names
            metrics: Model performance metrics
            description: Version description
            tags: Version tags
            author: Author name
            parent_version: Parent version this was derived from
            
        Returns:
            ModelVersion object
        """
        source_path = Path(model_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Determine version
        if version is None:
            version = self._get_next_version(model_name, version_bump)
        
        # Check for duplicate
        if model_name in self._versions and version in self._versions[model_name]:
            raise ValueError(f"Version {version} already exists for model {model_name}")
        
        # Compute hash and size
        model_hash = self._compute_file_hash(source_path)
        model_size = source_path.stat().st_size
        
        # Copy to artifacts directory
        artifact_dir = self.artifacts_path / model_name / version
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / source_path.name
        shutil.copy2(source_path, artifact_path)
        
        # Create version record
        model_version = ModelVersion(
            version=version,
            model_type=model_type,
            stage=ModelStage.DEVELOPMENT,
            created_at=datetime.now(timezone.utc),
            model_path=str(artifact_path),
            model_hash=model_hash,
            model_size_bytes=model_size,
            training_config=training_config or {},
            feature_columns=feature_columns or [],
            metrics=metrics or ModelMetrics(),
            description=description,
            tags=tags or [],
            author=author,
            parent_version=parent_version,
        )
        
        # Store
        if model_name not in self._versions:
            self._versions[model_name] = {}
        self._versions[model_name][version] = model_version
        
        # Persist
        self._save_registry()
        
        logger.info(
            f"Registered model {model_name} v{version} | "
            f"hash={model_hash} | size={model_size} bytes"
        )
        
        return model_version
    
    def get_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        return self._versions.get(model_name, {}).get(version)
    
    def get_latest_version(
        self,
        model_name: str,
        stage: Optional[ModelStage] = None,
    ) -> Optional[ModelVersion]:
        """
        Get the latest version of a model.
        
        Args:
            model_name: Model name
            stage: Optional stage filter
            
        Returns:
            Latest ModelVersion or None
        """
        versions = self._versions.get(model_name, {})
        
        if not versions:
            return None
        
        # Filter by stage if specified
        if stage:
            versions = {v: ver for v, ver in versions.items() if ver.stage == stage}
        
        if not versions:
            return None
        
        # Get latest by version number
        latest_version = max(versions.keys(), key=lambda v: [int(x) for x in v.split(".")])
        return versions[latest_version]
    
    def get_production_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the current production version."""
        return self.get_latest_version(model_name, stage=ModelStage.PRODUCTION)
    
    def list_versions(
        self,
        model_name: str,
        stage: Optional[ModelStage] = None,
    ) -> List[ModelVersion]:
        """List all versions of a model."""
        versions = self._versions.get(model_name, {})
        
        if stage:
            versions = {v: ver for v, ver in versions.items() if ver.stage == stage}
        
        # Sort by version
        sorted_versions = sorted(
            versions.values(),
            key=lambda v: [int(x) for x in v.version.split(".")],
            reverse=True,
        )
        
        return sorted_versions
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._versions.keys())
    
    def promote_version(
        self,
        model_name: str,
        version: str,
        to_stage: ModelStage,
    ) -> ModelVersion:
        """
        Promote a model version to a new stage.
        
        Args:
            model_name: Model name
            version: Version to promote
            to_stage: Target stage
            
        Returns:
            Updated ModelVersion
        """
        model_version = self.get_version(model_name, version)
        if not model_version:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        old_stage = model_version.stage
        model_version.stage = to_stage
        
        # If promoting to production, demote current production version
        if to_stage == ModelStage.PRODUCTION:
            for v, ver in self._versions.get(model_name, {}).items():
                if v != version and ver.stage == ModelStage.PRODUCTION:
                    ver.stage = ModelStage.ARCHIVED
                    logger.info(f"Demoted {model_name} v{v} to ARCHIVED")
        
        self._save_registry()
        
        logger.info(
            f"Promoted {model_name} v{version} from {old_stage.value} to {to_stage.value}"
        )
        
        return model_version
    
    def rollback(
        self,
        model_name: str,
        to_version: Optional[str] = None,
    ) -> ModelVersion:
        """
        Rollback to a previous version.
        
        Args:
            model_name: Model name
            to_version: Specific version to rollback to (defaults to previous production)
            
        Returns:
            Rolled back ModelVersion now in production
        """
        if to_version:
            target = self.get_version(model_name, to_version)
            if not target:
                raise ValueError(f"Version {to_version} not found")
        else:
            # Find previous production version
            archived = self.list_versions(model_name, stage=ModelStage.ARCHIVED)
            if not archived:
                raise ValueError("No previous version available for rollback")
            target = archived[0]  # Most recent archived
        
        # Demote current production
        current_prod = self.get_production_version(model_name)
        if current_prod:
            current_prod.stage = ModelStage.ARCHIVED
            logger.info(f"Demoted {model_name} v{current_prod.version} to ARCHIVED")
        
        # Promote target to production
        target.stage = ModelStage.PRODUCTION
        self._save_registry()
        
        logger.info(f"Rolled back {model_name} to v{target.version}")
        
        return target
    
    def update_metrics(
        self,
        model_name: str,
        version: str,
        metrics: ModelMetrics,
    ) -> ModelVersion:
        """
        Update metrics for a model version.
        
        Args:
            model_name: Model name
            version: Version to update
            metrics: New metrics
            
        Returns:
            Updated ModelVersion
        """
        model_version = self.get_version(model_name, version)
        if not model_version:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        model_version.metrics = metrics
        self._save_registry()
        
        logger.debug(f"Updated metrics for {model_name} v{version}")
        
        return model_version
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: ModelStage = ModelStage.PRODUCTION,
    ) -> Tuple[Any, ModelVersion]:
        """
        Load a model from the registry.
        
        Args:
            model_name: Model name
            version: Specific version (defaults to latest in stage)
            stage: Stage to load from if version not specified
            
        Returns:
            Tuple of (loaded model, ModelVersion)
        """
        import torch
        
        if version:
            model_version = self.get_version(model_name, version)
        else:
            model_version = self.get_latest_version(model_name, stage=stage)
        
        if not model_version:
            raise ValueError(
                f"No model found for {model_name} "
                f"(version={version}, stage={stage.value if stage else 'any'})"
            )
        
        model_path = Path(model_version.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load based on model type
        if model_version.model_type == ModelType.LSTM_LOOKAHEAD:
            from models.lstm_lookahead import LSTMLookaheadPredictor
            model = LSTMLookaheadPredictor(model_path=str(model_path))
        else:
            # Generic torch load
            model = torch.load(model_path, weights_only=False)
        
        logger.info(f"Loaded {model_name} v{model_version.version}")
        
        return model, model_version
    
    def delete_version(self, model_name: str, version: str) -> bool:
        """
        Delete a model version.
        
        Args:
            model_name: Model name
            version: Version to delete
            
        Returns:
            True if deleted, False if not found
        """
        model_version = self.get_version(model_name, version)
        if not model_version:
            return False
        
        # Don't allow deleting production
        if model_version.stage == ModelStage.PRODUCTION:
            raise ValueError("Cannot delete production version. Demote first.")
        
        # Delete artifact
        artifact_path = Path(model_version.model_path)
        if artifact_path.exists():
            artifact_path.unlink()
        
        # Remove artifact directory if empty
        artifact_dir = artifact_path.parent
        if artifact_dir.exists() and not any(artifact_dir.iterdir()):
            artifact_dir.rmdir()
        
        # Remove from registry
        del self._versions[model_name][version]
        if not self._versions[model_name]:
            del self._versions[model_name]
        
        self._save_registry()
        
        logger.info(f"Deleted {model_name} v{version}")
        
        return True
    
    def compare_versions(
        self,
        model_name: str,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            model_name: Model name
            version1: First version
            version2: Second version
            
        Returns:
            Comparison dict with metrics diff
        """
        v1 = self.get_version(model_name, version1)
        v2 = self.get_version(model_name, version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        m1 = v1.metrics.to_dict()
        m2 = v2.metrics.to_dict()
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "metrics_diff": {},
            "better_version": {},
        }
        
        # Higher is better for these metrics
        higher_is_better = {
            "train_accuracy", "val_accuracy", "sharpe_ratio", "sortino_ratio",
            "win_rate", "profit_factor", "direction_accuracy", "r2_score",
        }
        
        # Lower is better for these
        lower_is_better = {
            "train_loss", "val_loss", "max_drawdown", "mae", "rmse",
            "inference_time_ms",
        }
        
        for metric in m1:
            diff = m2[metric] - m1[metric]
            comparison["metrics_diff"][metric] = diff
            
            if metric in higher_is_better:
                comparison["better_version"][metric] = version2 if diff > 0 else version1
            elif metric in lower_is_better:
                comparison["better_version"][metric] = version2 if diff < 0 else version1
        
        return comparison
    
    def get_summary(self) -> Dict[str, Any]:
        """Get registry summary."""
        summary = {
            "total_models": len(self._versions),
            "total_versions": sum(len(v) for v in self._versions.values()),
            "models": {},
        }
        
        for model_name, versions in self._versions.items():
            prod = self.get_production_version(model_name)
            latest = self.get_latest_version(model_name)
            
            summary["models"][model_name] = {
                "total_versions": len(versions),
                "production_version": prod.version if prod else None,
                "latest_version": latest.version if latest else None,
                "stages": {
                    stage.value: len([v for v in versions.values() if v.stage == stage])
                    for stage in ModelStage
                },
            }
        
        return summary


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_model_registry(
    registry_path: str = "models/registry",
    artifacts_path: str = "models/artifacts",
) -> ModelRegistry:
    """Get or create the global model registry."""
    global _registry
    
    if _registry is None:
        _registry = ModelRegistry(
            registry_path=registry_path,
            artifacts_path=artifacts_path,
        )
    
    return _registry


__all__ = [
    "ModelStage",
    "ModelType",
    "ModelMetrics",
    "ModelVersion",
    "ABTestConfig",
    "ModelRegistry",
    "get_model_registry",
]
