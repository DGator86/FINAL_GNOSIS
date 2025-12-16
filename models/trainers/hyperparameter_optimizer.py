"""Utilities for model hyperparameter optimization."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import optuna


@dataclass
class HyperparameterSpace:
    """Represents a single hyperparameter search dimension."""

    name: str
    param_type: str
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False

    def suggest(self, trial: optuna.Trial) -> Any:
        """Suggest a value for this hyperparameter using an Optuna trial."""

        if self.param_type == "int":
            if self.low is None or self.high is None:
                raise ValueError("Integer parameters require 'low' and 'high'.")
            return trial.suggest_int(self.name, int(self.low), int(self.high))

        if self.param_type in {"float", "log_uniform"}:
            if self.low is None or self.high is None:
                raise ValueError("Float parameters require 'low' and 'high'.")
            return trial.suggest_float(self.name, float(self.low), float(self.high), log=self.log)

        if self.param_type == "categorical":
            if not self.choices:
                raise ValueError("Categorical parameters require 'choices'.")
            return trial.suggest_categorical(self.name, self.choices)

        raise ValueError(f"Unsupported parameter type: {self.param_type}")


class BayesianOptimizer:
    """Lightweight Bayesian-style optimizer placeholder.

    This class keeps track of observed parameters and scores and uses a
    simple exploitation/exploration strategy for proposing the next set of
    parameters. It avoids heavyweight dependencies while providing a
    consistent interface for the :class:`HyperparameterOptimizer`.
    """

    def __init__(self, param_space: List[HyperparameterSpace]):
        self.param_space = param_space
        self.history: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = float("-inf")

    def suggest_next(self) -> Dict[str, Any]:
        """Suggest the next parameter set using random sampling.

        If historical data exists, the best parameters are occasionally
        returned to encourage exploitation; otherwise a random set is sampled.
        """

        if self.history and np.random.rand() < 0.2:
            return dict(self.best_params) if self.best_params else {}

        suggestion: Dict[str, Any] = {}
        for space_param in self.param_space:
            if space_param.param_type == "int" and space_param.low is not None and space_param.high is not None:
                suggestion[space_param.name] = np.random.randint(int(space_param.low), int(space_param.high) + 1)
            elif space_param.param_type in {"float", "log_uniform"} and space_param.low is not None and space_param.high is not None:
                if space_param.log:
                    log_low, log_high = np.log(space_param.low), np.log(space_param.high)
                    suggestion[space_param.name] = float(np.exp(np.random.uniform(log_low, log_high)))
                else:
                    suggestion[space_param.name] = float(np.random.uniform(space_param.low, space_param.high))
            elif space_param.param_type == "categorical" and space_param.choices:
                suggestion[space_param.name] = np.random.choice(space_param.choices)
            else:
                raise ValueError(f"Invalid parameter definition for {space_param.name}")
        return suggestion

    def update(self, params: Dict[str, Any], score: float) -> None:
        """Record the result of a parameter evaluation."""

        observation = {"params": params, "score": score, "timestamp": time.time()}
        self.history.append(observation)
        if score > self.best_score:
            self.best_score = score
            self.best_params = params


class HyperparameterOptimizer:
    """Main hyperparameter optimization orchestrator"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()

        # Optimization configuration
        self.n_trials = config.get("n_trials", 100)
        self.timeout = config.get("timeout", 3600)  # 1 hour
        self.n_jobs = config.get("n_jobs", 1)
        self.optimization_method = config.get("method", "optuna")  # 'optuna', 'bayesian', 'grid', 'random'

        # Cross-validation configuration
        self.cv_folds = config.get("cv_folds", 3)
        self.scoring_metric = config.get("scoring_metric", "neg_mean_squared_error")

        # Pruning configuration
        self.enable_pruning = config.get("enable_pruning", True)
        self.pruning_percentile = config.get("pruning_percentile", 25)

        # Results storage
        self.optimization_results: List[Dict[str, Any]] = []

    def _setup_logger(self) -> logging.Logger:
        """Setup optimizer logger"""
        logger = logging.getLogger("gnosis.hyperopt")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def optimize(
        self,
        model_class: type,
        base_config: Dict[str, Any],
        param_space: List[HyperparameterSpace],
        X: np.ndarray,
        y: np.ndarray,
        validation_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a model

        Args:
            model_class: Model class to optimize
            base_config: Base model configuration
            param_space: Hyperparameter search space
            X: Training features
            y: Training targets
            validation_fn: Custom validation function
        """

        self.logger.info(f"Starting hyperparameter optimization using {self.optimization_method}")
        self.logger.info(f"Search space: {[p.name for p in param_space]}")

        if self.optimization_method == "optuna":
            return self._optimize_optuna(model_class, base_config, param_space, X, y, validation_fn)
        if self.optimization_method == "bayesian":
            return self._optimize_bayesian(model_class, base_config, param_space, X, y, validation_fn)
        if self.optimization_method == "grid":
            return self._optimize_grid(model_class, base_config, param_space, X, y, validation_fn)
        if self.optimization_method == "random":
            return self._optimize_random(model_class, base_config, param_space, X, y, validation_fn)
        raise ValueError(f"Unknown optimization method: {self.optimization_method}")

    def _optimize_optuna(
        self,
        model_class: type,
        base_config: Dict[str, Any],
        param_space: List[HyperparameterSpace],
        X: np.ndarray,
        y: np.ndarray,
        validation_fn: Optional[Callable],
    ) -> Dict[str, Any]:
        """Optimize using Optuna"""

        def objective(trial: optuna.Trial):
            # Suggest hyperparameters
            params: Dict[str, Any] = {}
            for space_param in param_space:
                params[space_param.name] = space_param.suggest(trial)

            # Create model configuration
            model_config = base_config.copy()
            model_config.update(params)

            # Evaluate model
            try:
                if validation_fn:
                    score = validation_fn(model_class, model_config, X, y)
                else:
                    score = self._default_validation(model_class, model_config, X, y)

                # Store result
                result = {
                    "trial_number": trial.number,
                    "params": params,
                    "score": score,
                    "timestamp": time.time(),
                }
                self.optimization_results.append(result)

                return score

            except Exception as exc:  # pragma: no cover - logging path
                self.logger.error(f"Trial {trial.number} failed: {exc}")
                return float("-inf")

        # Create study
        direction = "maximize" if "accuracy" in self.scoring_metric or "r2" in self.scoring_metric else "minimize"

        pruner: Optional[optuna.pruners.BasePruner] = None
        if self.enable_pruning:
            pruner = optuna.pruners.PercentilePruner(self.pruning_percentile)

        study = optuna.create_study(
            direction=direction,
            pruner=pruner,
            study_name=f"gnosis_hyperopt_{int(time.time())}",
        )

        # Optimize
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, n_jobs=self.n_jobs)

        # Prepare results
        best_params = study.best_params
        best_score = study.best_value

        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_trial": study.best_trial.number,
            "n_trials": len(study.trials),
            "optimization_history": self.optimization_results,
            "study_summary": {
                "direction": direction,
                "n_trials": len(study.trials),
                "pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                "complete_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            },
        }

    def _optimize_bayesian(
        self,
        model_class: type,
        base_config: Dict[str, Any],
        param_space: List[HyperparameterSpace],
        X: np.ndarray,
        y: np.ndarray,
        validation_fn: Optional[Callable],
    ) -> Dict[str, Any]:
        """Optimize using Bayesian optimization"""

        optimizer = BayesianOptimizer(param_space)

        best_score = float("-inf")
        best_params: Optional[Dict[str, Any]] = None

        for trial in range(self.n_trials):
            # Suggest next parameters
            params = optimizer.suggest_next()

            # Create model configuration
            model_config = base_config.copy()
            model_config.update(params)

            # Evaluate model
            try:
                if validation_fn:
                    score = validation_fn(model_class, model_config, X, y)
                else:
                    score = self._default_validation(model_class, model_config, X, y)

                # Update optimizer
                optimizer.update(params, score)

                # Track best
                if score > best_score:
                    best_score = score
                    best_params = params

                # Store result
                result = {
                    "trial_number": trial,
                    "params": params,
                    "score": score,
                    "timestamp": time.time(),
                }
                self.optimization_results.append(result)

                self.logger.info(f"Trial {trial}: Score = {score:.6f}")

            except Exception as exc:  # pragma: no cover - logging path
                self.logger.error(f"Trial {trial} failed: {exc}")
                optimizer.update(params, float("-inf"))

        return {
            "best_params": best_params,
            "best_score": best_score,
            "n_trials": self.n_trials,
            "optimization_history": self.optimization_results,
        }

    def _optimize_grid(
        self,
        model_class: type,
        base_config: Dict[str, Any],
        param_space: List[HyperparameterSpace],
        X: np.ndarray,
        y: np.ndarray,
        validation_fn: Optional[Callable],
    ) -> Dict[str, Any]:
        """Grid search optimization"""

        # Generate grid of parameters
        param_grid = self._generate_param_grid(param_space)

        best_score = float("-inf")
        best_params: Optional[Dict[str, Any]] = None
        trial = 0

        for params in param_grid[: self.n_trials]:  # Limit to n_trials
            # Create model configuration
            model_config = base_config.copy()
            model_config.update(params)

            # Evaluate model
            try:
                if validation_fn:
                    score = validation_fn(model_class, model_config, X, y)
                else:
                    score = self._default_validation(model_class, model_config, X, y)

                # Track best
                if score > best_score:
                    best_score = score
                    best_params = params

                # Store result
                result = {
                    "trial_number": trial,
                    "params": params,
                    "score": score,
                    "timestamp": time.time(),
                }
                self.optimization_results.append(result)

                trial += 1

            except Exception as exc:  # pragma: no cover - logging path
                self.logger.error(f"Trial {trial} failed: {exc}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "n_trials": trial,
            "optimization_history": self.optimization_results,
        }

    def _optimize_random(
        self,
        model_class: type,
        base_config: Dict[str, Any],
        param_space: List[HyperparameterSpace],
        X: np.ndarray,
        y: np.ndarray,
        validation_fn: Optional[Callable],
    ) -> Dict[str, Any]:
        """Random search optimization"""

        best_score = float("-inf")
        best_params: Optional[Dict[str, Any]] = None

        for trial in range(self.n_trials):
            # Random parameter selection
            params: Dict[str, Any] = {}
            for space_param in param_space:
                if space_param.param_type == "int" and space_param.low is not None and space_param.high is not None:
                    params[space_param.name] = np.random.randint(int(space_param.low), int(space_param.high) + 1)
                elif space_param.param_type in {"float", "log_uniform"} and space_param.low is not None and space_param.high is not None:
                    if space_param.log:
                        log_low, log_high = np.log(space_param.low), np.log(space_param.high)
                        params[space_param.name] = float(np.exp(np.random.uniform(log_low, log_high)))
                    else:
                        params[space_param.name] = float(np.random.uniform(space_param.low, space_param.high))
                elif space_param.param_type == "categorical" and space_param.choices:
                    params[space_param.name] = np.random.choice(space_param.choices)
                else:
                    raise ValueError(f"Invalid parameter definition for {space_param.name}")

            # Create model configuration
            model_config = base_config.copy()
            model_config.update(params)

            # Evaluate model
            try:
                if validation_fn:
                    score = validation_fn(model_class, model_config, X, y)
                else:
                    score = self._default_validation(model_class, model_config, X, y)

                # Track best
                if score > best_score:
                    best_score = score
                    best_params = params

                # Store result
                result = {
                    "trial_number": trial,
                    "params": params,
                    "score": score,
                    "timestamp": time.time(),
                }
                self.optimization_results.append(result)

            except Exception as exc:  # pragma: no cover - logging path
                self.logger.error(f"Trial {trial} failed: {exc}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "n_trials": self.n_trials,
            "optimization_history": self.optimization_results,
        }

    def _default_validation(
        self, model_class: type, model_config: Dict[str, Any], X: np.ndarray, y: np.ndarray
    ) -> float:
        """Default validation using a simple train/validation split."""

        # Initialize model
        model = model_class(model_config)

        # Simple train-validation split for models that don't support sklearn interface
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        try:
            # Train model
            model.train(X_train, y_train)

            # Get predictions
            predictions = model.predict(X_val)

            # Handle different prediction formats
            if isinstance(predictions, dict) and "predictions" in predictions:
                # Multi-horizon predictions - use first horizon
                horizon_keys = list(predictions["predictions"].keys())
                if horizon_keys:
                    preds = predictions["predictions"][horizon_keys[0]]
                else:
                    return float("-inf")
            else:
                preds = predictions

            # Calculate score
            if len(preds) != len(y_val):
                min_len = min(len(preds), len(y_val))
                preds = preds[:min_len]
                y_val = y_val[:min_len]

            if "neg_mean_squared_error" in self.scoring_metric:
                return -float(np.mean((preds - y_val) ** 2))
            if "r2" in self.scoring_metric:
                ss_res = float(np.sum((y_val - preds) ** 2))
                ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
                return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            return -float(np.mean(np.abs(preds - y_val)))  # Negative MAE

        except Exception as exc:  # pragma: no cover - logging path
            self.logger.error(f"Validation failed: {exc}")
            return float("-inf")

    def _generate_param_grid(self, param_space: List[HyperparameterSpace]) -> List[Dict[str, Any]]:
        """Generate parameter grid for grid search"""
        import itertools

        param_values: Dict[str, List[Any]] = {}
        for space_param in param_space:
            if space_param.param_type == "categorical" and space_param.choices:
                param_values[space_param.name] = list(space_param.choices)
            elif space_param.param_type == "int" and space_param.low is not None and space_param.high is not None:
                # Create discrete values
                n_values = min(10, int(space_param.high - space_param.low) + 1)
                param_values[space_param.name] = (
                    np.linspace(space_param.low, space_param.high, n_values, dtype=int).tolist()
                )
            elif space_param.param_type in {"float", "log_uniform"} and space_param.low is not None and space_param.high is not None:
                # Create continuous values
                n_values = 10
                if space_param.log:
                    values = np.logspace(np.log10(space_param.low), np.log10(space_param.high), n_values).tolist()
                else:
                    values = np.linspace(space_param.low, space_param.high, n_values).tolist()
                param_values[space_param.name] = values
            else:
                raise ValueError(f"Invalid parameter definition for {space_param.name}")

        # Generate all combinations
        param_names = list(param_values.keys())
        param_combinations = list(itertools.product(*param_values.values()))

        grid: List[Dict[str, Any]] = []
        for combination in param_combinations:
            grid.append(dict(zip(param_names, combination)))

        return grid

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        if not self.optimization_results:
            return {"error": "No optimization results available"}

        scores = [r["score"] for r in self.optimization_results if r["score"] != float("-inf")]

        if not scores:
            return {"error": "No successful trials"}

        return {
            "total_trials": len(self.optimization_results),
            "successful_trials": len(scores),
            "best_score": max(scores),
            "worst_score": min(scores),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "optimization_method": self.optimization_method,
            "improvement_over_trials": self._calculate_improvement(),
        }

    def _calculate_improvement(self) -> List[float]:
        """Calculate improvement over trials"""
        if not self.optimization_results:
            return []

        best_so_far: List[float] = []
        current_best = float("-inf")

        for result in self.optimization_results:
            if result["score"] > current_best:
                current_best = result["score"]
            best_so_far.append(current_best)

        return best_so_far
