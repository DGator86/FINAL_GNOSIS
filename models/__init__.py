try:  # Optional dependency guard
    from .base import BaseGnosisModel
except Exception:  # pragma: no cover - missing optional deps
    BaseGnosisModel = None  # type: ignore

from .hyperparameter_optimizer import BayesianOptimizer as LegacyBayesianOptimizer
from .hyperparameter_optimizer import HyperparameterOptimizer, HyperparameterSpace
from .hyperparameter_optimization import BayesianOptimizer, HyperparameterSpace as OptimizationSpace

__all__ = [
    "BaseGnosisModel",
    "BayesianOptimizer",
    "HyperparameterOptimizer",
    "HyperparameterSpace",
    "LegacyBayesianOptimizer",
    "OptimizationSpace",
]
