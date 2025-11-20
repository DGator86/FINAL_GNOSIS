"""
MLOps Deployment Manager for GNOSIS ML Models
Production deployment, versioning, A/B testing, and monitoring.
"""

import os
import json
import pickle
import shutil
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import hashlib
import threading
from collections import defaultdict

@dataclass
class ModelVersion:
    """Model version metadata"""
    model_id: str
    version: str
    model_type: str
    created_at: datetime
    created_by: str
    metrics: Dict[str, float]
    tags: List[str]
    deployment_status: str  # 'staging', 'production', 'archived'
    model_path: str
    config_path: str
    parent_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class ModelRegistry:
    """Central registry for ML models"""
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        
        self.metadata_path = self.registry_path / "registry.json"
        self.models = self._load_registry()
        
        self.logger = self._setup_logger()
        self._lock = threading.Lock()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup registry logger"""
        logger = logging.getLogger("gnosis.model_registry")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_registry(self) -> Dict[str, Dict[str, ModelVersion]]:
        """Load registry from disk"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                
                models = {}
                for model_id, versions in data.items():
                    models[model_id] = {}
                    for version, version_data in versions.items():
                        models[model_id][version] = ModelVersion.from_dict(version_data)
                
                return models
            except Exception as e:
                print(f"Failed to load registry: {e}")
        
        return {}
    
    def _save_registry(self):
        """Save registry to disk"""
        data = {}
        for model_id, versions in self.models.items():
            data[model_id] = {}
            for version, model_version in versions.items():
                data[model_id][version] = model_version.to_dict()
        
        with open(self.metadata_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(self, model_id: str, model_path: str, config: Dict[str, Any],
                      metrics: Dict[str, float], tags: List[str] = None,
                      created_by: str = "system") -> str:
        """Register a new model version"""
        
        with self._lock:
            # Create model directory
            model_dir = self.registry_path / model_id
            model_dir.mkdir(exist_ok=True)
            
            # Generate version
            if model_id not in self.models:
                self.models[model_id] = {}
                version = "v1.0.0"
            else:
                # Increment version
                versions = list(self.models[model_id].keys())
                latest_version = sorted(versions)[-1]
                major, minor, patch = map(int, latest_version[1:].split('.'))
                version = f"v{major}.{minor}.{patch + 1}"
            
            # Copy model files
            version_dir = model_dir / version
            version_dir.mkdir(exist_ok=True)
            
            model_dest = version_dir / "model.pkl"
            shutil.copy(model_path, model_dest)
            
            config_dest = version_dir / "config.json"
            with open(config_dest, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Create model version
            model_version = ModelVersion(
                model_id=model_id,
                version=version,
                model_type=config.get('model_type', 'unknown'),
                created_at=datetime.now(),
                created_by=created_by,
                metrics=metrics,
                tags=tags or [],
                deployment_status='staging',
                model_path=str(model_dest),
                config_path=str(config_dest)
            )
            
            self.models[model_id][version] = model_version
            self._save_registry()
            
            self.logger.info(f"Registered model {model_id} version {version}")
            return version
    
    def get_model(self, model_id: str, version: Optional[str] = None) -> Optional[ModelVersion]:
        """Get model version"""
        if model_id not in self.models:
            return None
        
        if version is None:
            # Get latest version
            versions = sorted(self.models[model_id].keys())
            version = versions[-1]
        
        return self.models[model_id].get(version)
    
    def list_models(self, model_id: Optional[str] = None, 
                   deployment_status: Optional[str] = None) -> List[ModelVersion]:
        """List models with filters"""
        results = []
        
        models_to_check = {model_id: self.models[model_id]} if model_id else self.models
        
        for mid, versions in models_to_check.items():
            for version, model_version in versions.items():
                if deployment_status is None or model_version.deployment_status == deployment_status:
                    results.append(model_version)
        
        return results
    
    def promote_to_production(self, model_id: str, version: str) -> bool:
        """Promote model version to production"""
        with self._lock:
            if model_id not in self.models or version not in self.models[model_id]:
                return False
            
            # Demote current production models
            for ver, model_ver in self.models[model_id].items():
                if model_ver.deployment_status == 'production':
                    model_ver.deployment_status = 'archived'
            
            # Promote new version
            self.models[model_id][version].deployment_status = 'production'
            self._save_registry()
            
            self.logger.info(f"Promoted {model_id} {version} to production")
            return True
    
    def archive_model(self, model_id: str, version: str) -> bool:
        """Archive model version"""
        with self._lock:
            if model_id not in self.models or version not in self.models[model_id]:
                return False
            
            self.models[model_id][version].deployment_status = 'archived'
            self._save_registry()
            
            self.logger.info(f"Archived {model_id} {version}")
            return True
    
    def delete_model(self, model_id: str, version: str) -> bool:
        """Delete model version"""
        with self._lock:
            if model_id not in self.models or version not in self.models[model_id]:
                return False
            
            # Remove files
            version_dir = Path(self.models[model_id][version].model_path).parent
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            # Remove from registry
            del self.models[model_id][version]
            
            # Remove model_id if no versions left
            if not self.models[model_id]:
                del self.models[model_id]
                model_dir = self.registry_path / model_id
                if model_dir.exists():
                    shutil.rmtree(model_dir)
            
            self._save_registry()
            
            self.logger.info(f"Deleted {model_id} {version}")
            return True


class ABTestManager:
    """A/B testing manager for model deployments"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.active_tests = {}
        self.test_results = defaultdict(lambda: defaultdict(list))
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup A/B test logger"""
        logger = logging.getLogger("gnosis.ab_testing")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def create_ab_test(self, test_id: str, model_a: Tuple[str, str], 
                      model_b: Tuple[str, str], traffic_split: float = 0.5,
                      duration_hours: int = 24) -> bool:
        """
        Create A/B test
        
        Args:
            test_id: Unique test identifier
            model_a: (model_id, version) for variant A
            model_b: (model_id, version) for variant B
            traffic_split: Percentage of traffic to model_a (0-1)
            duration_hours: Test duration in hours
        """
        
        # Verify models exist
        model_a_ver = self.registry.get_model(model_a[0], model_a[1])
        model_b_ver = self.registry.get_model(model_b[0], model_b[1])
        
        if not model_a_ver or not model_b_ver:
            self.logger.error("One or both models not found in registry")
            return False
        
        # Create test configuration
        test_config = {
            'test_id': test_id,
            'model_a': model_a,
            'model_b': model_b,
            'traffic_split': traffic_split,
            'start_time': datetime.now(),
            'end_time': datetime.now() + timedelta(hours=duration_hours),
            'status': 'active'
        }
        
        self.active_tests[test_id] = test_config
        self.logger.info(f"Created A/B test {test_id}: {model_a} vs {model_b}")
        
        return True
    
    def route_traffic(self, test_id: str) -> str:
        """Route traffic to model variant"""
        if test_id not in self.active_tests:
            return 'model_a'  # Default to model_a if test not found
        
        test = self.active_tests[test_id]
        
        # Check if test has expired
        if datetime.now() > test['end_time']:
            test['status'] = 'expired'
            return 'model_a'
        
        # Random routing based on traffic split
        return 'model_a' if np.random.random() < test['traffic_split'] else 'model_b'
    
    def log_prediction(self, test_id: str, variant: str, prediction: Any, 
                      actual: Optional[Any] = None, latency_ms: Optional[float] = None):
        """Log prediction result for A/B test"""
        if test_id not in self.active_tests:
            return
        
        result = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual': actual,
            'latency_ms': latency_ms
        }
        
        self.test_results[test_id][variant].append(result)
    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get A/B test results"""
        if test_id not in self.active_tests:
            return {'error': 'Test not found'}
        
        test = self.active_tests[test_id]
        results_a = self.test_results[test_id]['model_a']
        results_b = self.test_results[test_id]['model_b']
        
        # Calculate metrics
        def calculate_metrics(results):
            if not results:
                return {}
            
            predictions = [r['prediction'] for r in results if r['prediction'] is not None]
            actuals = [r['actual'] for r in results if r['actual'] is not None]
            latencies = [r['latency_ms'] for r in results if r['latency_ms'] is not None]
            
            metrics = {
                'count': len(results),
                'avg_latency_ms': np.mean(latencies) if latencies else None
            }
            
            if len(predictions) > 0 and len(actuals) > 0 and len(predictions) == len(actuals):
                mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
                mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
                metrics['mse'] = mse
                metrics['mae'] = mae
            
            return metrics
        
        metrics_a = calculate_metrics(results_a)
        metrics_b = calculate_metrics(results_b)
        
        return {
            'test_id': test_id,
            'test_config': test,
            'model_a_metrics': metrics_a,
            'model_b_metrics': metrics_b,
            'winner': self._determine_winner(metrics_a, metrics_b)
        }
    
    def _determine_winner(self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> Optional[str]:
        """Determine winner based on metrics"""
        if not metrics_a or not metrics_b:
            return None
        
        # Compare MAE (lower is better)
        if 'mae' in metrics_a and 'mae' in metrics_b:
            if metrics_a['mae'] < metrics_b['mae']:
                return 'model_a'
            if metrics_b['mae'] < metrics_a['mae']:
                return 'model_b'
        
        return None
    
    def stop_test(self, test_id: str) -> Dict[str, Any]:
        """Stop A/B test and return results"""
        if test_id not in self.active_tests:
            return {'error': 'Test not found'}
        
        self.active_tests[test_id]['status'] = 'stopped'
        results = self.get_test_results(test_id)
        
        self.logger.info(f"Stopped A/B test {test_id}")
        return results


class DeploymentManager:
    """Main deployment manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = ModelRegistry(config.get('registry_path', './model_registry'))
        self.ab_manager = ABTestManager(self.registry)
        self.logger = self._setup_logger()
        
        # Deployment configuration
        self.enable_canary = config.get('enable_canary', True)
        self.canary_percentage = config.get('canary_percentage', 0.1)
        self.health_check_interval = config.get('health_check_interval', 300)  # 5 minutes
        
        # Monitoring
        self.prediction_logs = []
        self.error_logs = []
        
    def _setup_logger(self) -> logging.Logger:
        """Setup deployment logger"""
        logger = logging.getLogger("gnosis.deployment")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def deploy_model(self, model_id: str, version: str, 
                    deployment_strategy: str = 'direct') -> bool:
        """
        Deploy model to production
        
        Args:
            model_id: Model identifier
            version: Model version
            deployment_strategy: 'direct', 'canary', or 'blue_green'
        """
        
        model_version = self.registry.get_model(model_id, version)
        if not model_version:
            self.logger.error(f"Model {model_id} {version} not found")
            return False
        
        if deployment_strategy == 'direct':
            return self.registry.promote_to_production(model_id, version)
        
        elif deployment_strategy == 'canary':
            return self._canary_deployment(model_id, version)
        
        elif deployment_strategy == 'blue_green':
            return self._blue_green_deployment(model_id, version)
        
        else:
            self.logger.error(f"Unknown deployment strategy: {deployment_strategy}")
            return False
    
    def _canary_deployment(self, model_id: str, version: str) -> bool:
        """Canary deployment with gradual traffic shift"""
        
        # Get current production model
        prod_models = self.registry.list_models(model_id, 'production')
        if not prod_models:
            # No existing production model, do direct deployment
            return self.registry.promote_to_production(model_id, version)
        
        current_prod = prod_models[0]
        
        # Create A/B test with canary percentage
        test_id = f"canary_{model_id}_{version}_{int(time.time())}"
        
        success = self.ab_manager.create_ab_test(
            test_id,
            (current_prod.model_id, current_prod.version),
            (model_id, version),
            traffic_split=1.0 - self.canary_percentage,
            duration_hours=24
        )
        
        if success:
            self.logger.info(f"Started canary deployment: {self.canary_percentage*100}% traffic to {version}")
        
        return success
    
    def _blue_green_deployment(self, model_id: str, version: str) -> bool:
        """Blue-green deployment"""
        
        # In blue-green, we prepare the new version (green) while keeping old version (blue)
        # Then switch all traffic at once
        
        model_version = self.registry.get_model(model_id, version)
        if not model_version:
            return False
        
        # Set new version to staging
        model_version.deployment_status = 'staging'
        
        # After validation, promote to production
        # (In real implementation, you'd have validation steps here)
        return self.registry.promote_to_production(model_id, version)
    
    def rollback(self, model_id: str) -> bool:
        """Rollback to previous production version"""
        
        versions = self.registry.list_models(model_id)
        archived = [v for v in versions if v.deployment_status == 'archived']
        
        if not archived:
            self.logger.error(f"No previous version to rollback to for {model_id}")
            return False
        
        # Get most recent archived version
        previous = sorted(archived, key=lambda x: x.created_at)[-1]
        
        success = self.registry.promote_to_production(model_id, previous.version)
        if success:
            self.logger.info(f"Rolled back {model_id} to {previous.version}")
        
        return success
    
    def log_prediction(self, model_id: str, version: str, 
                      input_data: Any, prediction: Any, latency_ms: float):
        """Log prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now(),
            'model_id': model_id,
            'version': version,
            'input_hash': hashlib.md5(str(input_data).encode()).hexdigest()[:8],
            'prediction': prediction,
            'latency_ms': latency_ms
        }
        
        self.prediction_logs.append(log_entry)
    
    def log_error(self, model_id: str, version: str, error: str):
        """Log error for monitoring"""
        error_entry = {
            'timestamp': datetime.now(),
            'model_id': model_id,
            'version': version,
            'error': error
        }
        
        self.error_logs.append(error_entry)
    
    def get_deployment_status(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get deployment status"""
        
        if model_id:
            models = self.registry.list_models(model_id)
        else:
            models = self.registry.list_models()
        
        status = {
            'production': [],
            'staging': [],
            'archived': []
        }
        
        for model in models:
            status[model.deployment_status].append({
                'model_id': model.model_id,
                'version': model.version,
                'model_type': model.model_type,
                'created_at': model.created_at.isoformat(),
                'metrics': model.metrics
            })
        
        return status
    
    def get_monitoring_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get monitoring statistics"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_predictions = [
            p for p in self.prediction_logs 
            if p['timestamp'] > cutoff_time
        ]
        
        recent_errors = [
            e for e in self.error_logs
            if e['timestamp'] > cutoff_time
        ]
        
        if not recent_predictions:
            return {
                'total_predictions': 0,
                'total_errors': len(recent_errors),
                'error_rate': 0.0
            }
        
        latencies = [p['latency_ms'] for p in recent_predictions]
        
        return {
            'total_predictions': len(recent_predictions),
            'total_errors': len(recent_errors),
            'error_rate': len(recent_errors) / len(recent_predictions),
            'avg_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99)
        }
