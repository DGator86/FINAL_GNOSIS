"""
This module provides a lightweight Bayesian optimization helper that uses a
Gaussian Process surrogate model with an Expected Improvement acquisition
function. It is designed to work alongside Optuna-defined search spaces while
remaining framework agnostic for GNOSIS models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import optuna
except Exception:  # pragma: no cover - optional dependency
    optuna = None  # type: ignore

try:
    from scipy import stats
except Exception:  # pragma: no cover - optional dependency
    stats = None  # type: ignore

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - optional dependency
    GaussianProcessRegressor = None  # type: ignore
    Matern = None  # type: ignore
    StandardScaler = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Definition of a single hyperparameter search dimension."""

    name: str
    param_type: str  # "int", "float", "categorical", "log_uniform"
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    choices: Optional[List[Any]] = None
    log: bool = False

    def suggest(self, trial: Any) -> Any:
        """Suggest a value for this space using an Optuna trial."""

        if optuna is None:
            raise ImportError("optuna is required for suggest but is not installed")

        if self.param_type == "int":
            if self.log:
                return trial.suggest_int(self.name, self.low, self.high, log=True)
            return trial.suggest_int(self.name, self.low, self.high)

        if self.param_type == "float":
            if self.log:
                return trial.suggest_float(self.name, self.low, self.high, log=True)
            return trial.suggest_float(self.name, self.low, self.high)

        if self.param_type == "categorical":
            return trial.suggest_categorical(self.name, self.choices)

        if self.param_type == "log_uniform":
            return trial.suggest_float(self.name, self.low, self.high, log=True)

        raise ValueError(f"Unknown parameter type: {self.param_type}")


_DEPENDENCIES_AVAILABLE = all([GaussianProcessRegressor, Matern, StandardScaler, stats])


class BayesianOptimizer:
    """Bayesian optimization helper for hyperparameter tuning."""

    def __init__(
        self,
        space: List[HyperparameterSpace],
        n_initial_points: int = 10,
        n_candidates: int = 200,
    ) -> None:
        self.space = space
        self.n_initial_points = n_initial_points
        self.n_candidates = n_candidates
        self.X_observed: List[np.SimpleArray] = []
        self.y_observed: List[float] = []
        self.gp_model: Any | None = None
        self.scaler = StandardScaler() if StandardScaler else None

    def suggest_next(self) -> Dict[str, Any]:
        """Suggest the next hyperparameter configuration to evaluate."""

        if len(self.X_observed) < self.n_initial_points:
            return self._random_sample()

        if not _DEPENDENCIES_AVAILABLE:
            logger.warning("GP dependencies missing; falling back to random sampling.")
            return self._random_sample()

        return self._bayesian_suggest()

    def update(self, params: Dict[str, Any], score: float) -> None:
        """Record the performance of a sampled configuration."""

        encoded_params = self._encode_params(params)
        self.X_observed.append(encoded_params)
        self.y_observed.append(score)

    def _encode_params(self, params: Dict[str, Any]) -> np.SimpleArray:
        """Encode hyperparameters into a numeric vector for GP consumption."""

        encoded: List[float] = []
        for space_param in self.space:
            value = params[space_param.name]

            if space_param.param_type in ["int", "float", "log_uniform"]:
                encoded.append(np.log(value) if space_param.log else float(value))
            elif space_param.param_type == "categorical":
                encoded.extend(
                    1.0 if value == choice else 0.0
                    for choice in (space_param.choices or [])
                )

        return np.asarray(encoded, dtype=float)

    def _decode_params(self, encoded: np.SimpleArray) -> Dict[str, Any]:
        """Decode a numeric vector back into hyperparameters."""

        params: Dict[str, Any] = {}
        idx = 0

        for space_param in self.space:
            if space_param.param_type in ["int", "float", "log_uniform"]:
                value = encoded[idx]
                if space_param.log:
                    value = np.exp(value)

                if space_param.param_type == "int":
                    value = int(round(value))

                params[space_param.name] = value
                idx += 1
            elif space_param.param_type == "categorical":
                choices = space_param.choices or []
                choice_values = encoded[idx : idx + len(choices)]
                best_choice_idx = int(np.argmax(choice_values)) if choices else 0
                params[space_param.name] = choices[best_choice_idx] if choices else None
                idx += len(choices)

        return params

    def _random_sample(self) -> Dict[str, Any]:
        """Sample hyperparameters uniformly (log-uniform when requested)."""

        params: Dict[str, Any] = {}
        for space_param in self.space:
            if space_param.param_type == "int":
                params[space_param.name] = int(
                    np.random.randint(space_param.low, space_param.high + 1)
                )
            elif space_param.param_type in ["float", "log_uniform"]:
                if space_param.log:
                    log_low = np.log(space_param.low)
                    log_high = np.log(space_param.high)
                    params[space_param.name] = float(
                        np.exp(np.random.uniform(log_low, log_high))
                    )
                else:
                    params[space_param.name] = float(
                        np.random.uniform(space_param.low, space_param.high)
                    )
            elif space_param.param_type == "categorical":
                params[space_param.name] = np.random.choice(space_param.choices)

        return params

    def _bayesian_suggest(self) -> Dict[str, Any]:
        """Suggest the next point using the GP-based acquisition function."""

        if not _DEPENDENCIES_AVAILABLE:
            return self._random_sample()

        if self.gp_model is None or len(self.X_observed) % 5 == 0:
            self._fit_gp_model()

        return self._optimize_acquisition()

    def _fit_gp_model(self) -> None:
        """Fit or refit the Gaussian Process model to observed data."""

        if not _DEPENDENCIES_AVAILABLE or not self.X_observed:
            return

        X = np.vstack(self.X_observed)
        y = np.asarray(self.y_observed, dtype=float)

        if self.scaler:
            X_normalized = self.scaler.fit_transform(X)
        else:
            X_normalized = X

        kernel = Matern(length_scale=1.0, nu=2.5) if Matern else None
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=42,
        )
        self.gp_model.fit(X_normalized, y)

    def _optimize_acquisition(self) -> Dict[str, Any]:
        """Maximize Expected Improvement over random candidates."""

        best_ei = -np.inf
        best_params: Dict[str, Any] | None = None

        for _ in range(self.n_candidates):
            candidate_params = self._random_sample()
            candidate_encoded = self._encode_params(candidate_params)
            ei = self._expected_improvement(candidate_encoded)

            if ei > best_ei:
                best_ei = ei
                best_params = candidate_params

        if best_params is None:
            logger.warning("No candidate improved EI; falling back to random sample.")
            return self._random_sample()

        return best_params

    def _expected_improvement(self, x: np.SimpleArray) -> float:
        """Compute Expected Improvement at the encoded point ``x``."""

        if not _DEPENDENCIES_AVAILABLE or self.gp_model is None or not self.y_observed:
            return 0.0

        if self.scaler:
            x_normalized = self.scaler.transform([x])
        else:
            x_normalized = [x]

        mu, sigma = self.gp_model.predict(x_normalized, return_std=True)
        mu, sigma = float(mu[0]), float(sigma[0])

        if sigma == 0:
            return 0.0

        best_y = np.max(self.y_observed)
        z = (mu - best_y) / sigma
        if stats is None:
            return 0.0
        ei = (mu - best_y) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        return float(ei)


__all__ = ["HyperparameterSpace", "BayesianOptimizer"]
