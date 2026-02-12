try:  # Optional dependency guard
    from .base import BaseGnosisModel
except Exception:  # pragma: no cover - missing optional deps
    BaseGnosisModel = None  # type: ignore

from .hyperparameter_optimization import BayesianOptimizer
from .hyperparameter_optimization import HyperparameterSpace as OptimizationSpace
from .hyperparameter_optimizer import BayesianOptimizer as LegacyBayesianOptimizer
from .hyperparameter_optimizer import HyperparameterOptimizer, HyperparameterSpace

__all__ = [
    "BaseGnosisModel",
    "BayesianOptimizer",
    "HyperparameterOptimizer",
    "HyperparameterSpace",
    "LegacyBayesianOptimizer",
    "OptimizationSpace",
]
