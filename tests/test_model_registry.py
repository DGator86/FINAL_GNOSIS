"""Tests for ML Model Registry."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ml.model_registry import (
    ModelRegistry,
    ModelVersion,
    ModelMetrics,
    ModelStage,
    ModelType,
    get_model_registry,
)


@pytest.fixture
def temp_registry():
    """Create a temporary registry for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "registry")
        artifacts_path = os.path.join(tmpdir, "artifacts")
        
        registry = ModelRegistry(
            registry_path=registry_path,
            artifacts_path=artifacts_path,
        )
        
        yield registry, tmpdir


@pytest.fixture
def sample_model_file(temp_registry):
    """Create a sample model file for testing."""
    registry, tmpdir = temp_registry
    
    model_file = Path(tmpdir) / "sample_model.pt"
    model_file.write_bytes(b"fake model data for testing " * 100)
    
    return registry, str(model_file)


class TestModelMetrics:
    """Tests for ModelMetrics dataclass."""
    
    def test_default_values(self):
        metrics = ModelMetrics()
        
        assert metrics.train_loss == 0.0
        assert metrics.val_loss == 0.0
        assert metrics.sharpe_ratio == 0.0
    
    def test_to_dict(self):
        metrics = ModelMetrics(
            train_loss=0.5,
            val_loss=0.6,
            sharpe_ratio=1.5,
            win_rate=0.55,
        )
        
        data = metrics.to_dict()
        
        assert data["train_loss"] == 0.5
        assert data["val_loss"] == 0.6
        assert data["sharpe_ratio"] == 1.5
        assert data["win_rate"] == 0.55
    
    def test_from_dict(self):
        data = {
            "train_loss": 0.3,
            "val_accuracy": 0.85,
            "direction_accuracy": 0.72,
        }
        
        metrics = ModelMetrics.from_dict(data)
        
        assert metrics.train_loss == 0.3
        assert metrics.val_accuracy == 0.85
        assert metrics.direction_accuracy == 0.72


class TestModelVersion:
    """Tests for ModelVersion dataclass."""
    
    def test_to_dict(self):
        version = ModelVersion(
            version="1.0.0",
            model_type=ModelType.LSTM_LOOKAHEAD,
            stage=ModelStage.DEVELOPMENT,
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            model_path="/path/to/model.pt",
            model_hash="abc123",
            model_size_bytes=1024,
            description="Test model",
        )
        
        data = version.to_dict()
        
        assert data["version"] == "1.0.0"
        assert data["model_type"] == "lstm_lookahead"
        assert data["stage"] == "development"
        assert "2025-01-01" in data["created_at"]
    
    def test_from_dict(self):
        data = {
            "version": "2.1.0",
            "model_type": "gradient_boosting",
            "stage": "production",
            "created_at": "2025-06-15T12:00:00+00:00",
            "model_path": "/path/model.pt",
            "model_hash": "xyz789",
            "model_size_bytes": 2048,
            "metrics": {"train_loss": 0.1},
        }
        
        version = ModelVersion.from_dict(data)
        
        assert version.version == "2.1.0"
        assert version.model_type == ModelType.GRADIENT_BOOSTING
        assert version.stage == ModelStage.PRODUCTION
        assert version.metrics.train_loss == 0.1


class TestModelRegistry:
    """Tests for ModelRegistry."""
    
    def test_initialization(self, temp_registry):
        registry, tmpdir = temp_registry
        
        assert registry.registry_path.exists()
        assert registry.artifacts_path.exists()
    
    def test_register_model(self, sample_model_file):
        registry, model_path = sample_model_file
        
        version = registry.register_model(
            model_name="test_lstm",
            model_path=model_path,
            model_type=ModelType.LSTM_LOOKAHEAD,
            description="Test LSTM model",
            tags=["test", "lstm"],
        )
        
        assert version.version == "1.0.0"
        assert version.stage == ModelStage.DEVELOPMENT
        assert version.model_type == ModelType.LSTM_LOOKAHEAD
        assert version.model_hash is not None
        assert version.model_size_bytes > 0
    
    def test_register_multiple_versions(self, sample_model_file):
        registry, model_path = sample_model_file
        
        # Register first version
        v1 = registry.register_model(
            model_name="test_model",
            model_path=model_path,
        )
        assert v1.version == "1.0.0"
        
        # Register second version (patch bump)
        v2 = registry.register_model(
            model_name="test_model",
            model_path=model_path,
        )
        assert v2.version == "1.0.1"
        
        # Register with minor bump
        v3 = registry.register_model(
            model_name="test_model",
            model_path=model_path,
            version_bump="minor",
        )
        assert v3.version == "1.1.0"
        
        # Register with major bump
        v4 = registry.register_model(
            model_name="test_model",
            model_path=model_path,
            version_bump="major",
        )
        assert v4.version == "2.0.0"
    
    def test_register_explicit_version(self, sample_model_file):
        registry, model_path = sample_model_file
        
        version = registry.register_model(
            model_name="explicit_model",
            model_path=model_path,
            version="3.5.7",
        )
        
        assert version.version == "3.5.7"
    
    def test_duplicate_version_error(self, sample_model_file):
        registry, model_path = sample_model_file
        
        registry.register_model(
            model_name="dup_model",
            model_path=model_path,
            version="1.0.0",
        )
        
        with pytest.raises(ValueError, match="already exists"):
            registry.register_model(
                model_name="dup_model",
                model_path=model_path,
                version="1.0.0",
            )
    
    def test_get_version(self, sample_model_file):
        registry, model_path = sample_model_file
        
        registry.register_model(
            model_name="get_test",
            model_path=model_path,
            version="1.0.0",
        )
        
        version = registry.get_version("get_test", "1.0.0")
        assert version is not None
        assert version.version == "1.0.0"
        
        # Non-existent version
        assert registry.get_version("get_test", "9.9.9") is None
        
        # Non-existent model
        assert registry.get_version("nonexistent", "1.0.0") is None
    
    def test_get_latest_version(self, sample_model_file):
        registry, model_path = sample_model_file
        
        registry.register_model("latest_test", model_path, version="1.0.0")
        registry.register_model("latest_test", model_path, version="1.2.0")
        registry.register_model("latest_test", model_path, version="1.1.5")
        
        latest = registry.get_latest_version("latest_test")
        assert latest.version == "1.2.0"
    
    def test_get_latest_version_by_stage(self, sample_model_file):
        registry, model_path = sample_model_file
        
        registry.register_model("stage_test", model_path, version="1.0.0")
        registry.register_model("stage_test", model_path, version="2.0.0")
        
        # Promote 1.0.0 to production
        registry.promote_version("stage_test", "1.0.0", ModelStage.PRODUCTION)
        
        # Latest overall is 2.0.0
        latest = registry.get_latest_version("stage_test")
        assert latest.version == "2.0.0"
        
        # Latest production is 1.0.0
        latest_prod = registry.get_latest_version("stage_test", stage=ModelStage.PRODUCTION)
        assert latest_prod.version == "1.0.0"
    
    def test_list_versions(self, sample_model_file):
        registry, model_path = sample_model_file
        
        registry.register_model("list_test", model_path, version="1.0.0")
        registry.register_model("list_test", model_path, version="2.0.0")
        registry.register_model("list_test", model_path, version="1.5.0")
        
        versions = registry.list_versions("list_test")
        
        assert len(versions) == 3
        # Should be sorted descending
        assert versions[0].version == "2.0.0"
        assert versions[1].version == "1.5.0"
        assert versions[2].version == "1.0.0"
    
    def test_list_models(self, sample_model_file):
        registry, model_path = sample_model_file
        
        registry.register_model("model_a", model_path)
        registry.register_model("model_b", model_path)
        registry.register_model("model_c", model_path)
        
        models = registry.list_models()
        
        assert len(models) == 3
        assert set(models) == {"model_a", "model_b", "model_c"}


class TestModelPromotion:
    """Tests for model promotion workflows."""
    
    def test_promote_to_staging(self, sample_model_file):
        registry, model_path = sample_model_file
        
        registry.register_model("promo_test", model_path)
        
        promoted = registry.promote_version("promo_test", "1.0.0", ModelStage.STAGING)
        
        assert promoted.stage == ModelStage.STAGING
    
    def test_promote_to_production(self, sample_model_file):
        registry, model_path = sample_model_file
        
        registry.register_model("prod_test", model_path)
        
        promoted = registry.promote_version("prod_test", "1.0.0", ModelStage.PRODUCTION)
        
        assert promoted.stage == ModelStage.PRODUCTION
        assert registry.get_production_version("prod_test").version == "1.0.0"
    
    def test_promote_demotes_previous_production(self, sample_model_file):
        registry, model_path = sample_model_file
        
        registry.register_model("demote_test", model_path, version="1.0.0")
        registry.register_model("demote_test", model_path, version="2.0.0")
        
        # Promote 1.0.0 to production
        registry.promote_version("demote_test", "1.0.0", ModelStage.PRODUCTION)
        assert registry.get_production_version("demote_test").version == "1.0.0"
        
        # Promote 2.0.0 to production
        registry.promote_version("demote_test", "2.0.0", ModelStage.PRODUCTION)
        
        # 2.0.0 should be production
        assert registry.get_production_version("demote_test").version == "2.0.0"
        
        # 1.0.0 should be archived
        v1 = registry.get_version("demote_test", "1.0.0")
        assert v1.stage == ModelStage.ARCHIVED
    
    def test_promote_nonexistent_raises(self, sample_model_file):
        registry, model_path = sample_model_file
        
        with pytest.raises(ValueError, match="not found"):
            registry.promote_version("nonexistent", "1.0.0", ModelStage.PRODUCTION)


class TestModelRollback:
    """Tests for model rollback functionality."""
    
    def test_rollback_to_specific_version(self, sample_model_file):
        registry, model_path = sample_model_file
        
        registry.register_model("rollback_test", model_path, version="1.0.0")
        registry.register_model("rollback_test", model_path, version="2.0.0")
        
        # Set up: 1.0.0 is production
        registry.promote_version("rollback_test", "1.0.0", ModelStage.PRODUCTION)
        
        # New version becomes production
        registry.promote_version("rollback_test", "2.0.0", ModelStage.PRODUCTION)
        
        # Rollback to 1.0.0
        rolled_back = registry.rollback("rollback_test", to_version="1.0.0")
        
        assert rolled_back.version == "1.0.0"
        assert rolled_back.stage == ModelStage.PRODUCTION
        
        # 2.0.0 should be archived
        v2 = registry.get_version("rollback_test", "2.0.0")
        assert v2.stage == ModelStage.ARCHIVED
    
    def test_rollback_to_previous(self, sample_model_file):
        registry, model_path = sample_model_file
        
        registry.register_model("auto_rollback", model_path, version="1.0.0")
        registry.register_model("auto_rollback", model_path, version="2.0.0")
        
        registry.promote_version("auto_rollback", "1.0.0", ModelStage.PRODUCTION)
        registry.promote_version("auto_rollback", "2.0.0", ModelStage.PRODUCTION)
        
        # Rollback without specifying version
        rolled_back = registry.rollback("auto_rollback")
        
        # Should rollback to 1.0.0 (the archived one)
        assert rolled_back.version == "1.0.0"
        assert rolled_back.stage == ModelStage.PRODUCTION


class TestMetricsUpdate:
    """Tests for metrics updates."""
    
    def test_update_metrics(self, sample_model_file):
        registry, model_path = sample_model_file
        
        registry.register_model("metrics_test", model_path)
        
        new_metrics = ModelMetrics(
            train_loss=0.15,
            val_loss=0.18,
            sharpe_ratio=2.1,
            win_rate=0.62,
            direction_accuracy=0.75,
        )
        
        updated = registry.update_metrics("metrics_test", "1.0.0", new_metrics)
        
        assert updated.metrics.train_loss == 0.15
        assert updated.metrics.sharpe_ratio == 2.1
        assert updated.metrics.win_rate == 0.62


class TestVersionComparison:
    """Tests for version comparison."""
    
    def test_compare_versions(self, sample_model_file):
        registry, model_path = sample_model_file
        
        registry.register_model("compare_test", model_path, version="1.0.0")
        registry.register_model("compare_test", model_path, version="2.0.0")
        
        # Update metrics
        registry.update_metrics(
            "compare_test", "1.0.0",
            ModelMetrics(sharpe_ratio=1.5, win_rate=0.55),
        )
        registry.update_metrics(
            "compare_test", "2.0.0",
            ModelMetrics(sharpe_ratio=1.8, win_rate=0.60),
        )
        
        comparison = registry.compare_versions("compare_test", "1.0.0", "2.0.0")
        
        assert comparison["version1"] == "1.0.0"
        assert comparison["version2"] == "2.0.0"
        assert comparison["metrics_diff"]["sharpe_ratio"] == pytest.approx(0.3)
        assert comparison["better_version"]["sharpe_ratio"] == "2.0.0"


class TestVersionDeletion:
    """Tests for version deletion."""
    
    def test_delete_version(self, sample_model_file):
        registry, model_path = sample_model_file
        
        registry.register_model("delete_test", model_path, version="1.0.0")
        registry.register_model("delete_test", model_path, version="2.0.0")
        
        result = registry.delete_version("delete_test", "1.0.0")
        
        assert result is True
        assert registry.get_version("delete_test", "1.0.0") is None
        assert registry.get_version("delete_test", "2.0.0") is not None
    
    def test_delete_nonexistent_returns_false(self, sample_model_file):
        registry, model_path = sample_model_file
        
        result = registry.delete_version("nonexistent", "1.0.0")
        
        assert result is False
    
    def test_cannot_delete_production(self, sample_model_file):
        registry, model_path = sample_model_file
        
        registry.register_model("prod_delete", model_path)
        registry.promote_version("prod_delete", "1.0.0", ModelStage.PRODUCTION)
        
        with pytest.raises(ValueError, match="Cannot delete production"):
            registry.delete_version("prod_delete", "1.0.0")


class TestRegistryPersistence:
    """Tests for registry persistence."""
    
    def test_registry_persists_across_instances(self, temp_registry):
        registry, tmpdir = temp_registry
        registry_path = str(registry.registry_path)
        artifacts_path = str(registry.artifacts_path)
        
        # Create a model file
        model_file = Path(tmpdir) / "persist_model.pt"
        model_file.write_bytes(b"test model data")
        
        # Register a model
        registry.register_model(
            model_name="persist_test",
            model_path=str(model_file),
            description="Persistence test",
        )
        
        # Create new registry instance
        new_registry = ModelRegistry(
            registry_path=registry_path,
            artifacts_path=artifacts_path,
        )
        
        # Should find the model
        version = new_registry.get_version("persist_test", "1.0.0")
        assert version is not None
        assert version.description == "Persistence test"


class TestRegistrySummary:
    """Tests for registry summary."""
    
    def test_get_summary(self, sample_model_file):
        registry, model_path = sample_model_file
        
        registry.register_model("summary_a", model_path, version="1.0.0")
        registry.register_model("summary_a", model_path, version="2.0.0")
        registry.register_model("summary_b", model_path, version="1.0.0")
        
        registry.promote_version("summary_a", "1.0.0", ModelStage.PRODUCTION)
        
        summary = registry.get_summary()
        
        assert summary["total_models"] == 2
        assert summary["total_versions"] == 3
        assert summary["models"]["summary_a"]["production_version"] == "1.0.0"
        assert summary["models"]["summary_a"]["latest_version"] == "2.0.0"
