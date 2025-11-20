import numpy as np
import pytest

from models.hyperparameter_optimization import BayesianOptimizer, HyperparameterSpace


def _build_optimizer(n_initial_points: int = 2) -> BayesianOptimizer:
    space = [
        HyperparameterSpace("learning_rate", "float", low=0.001, high=0.1, log=True),
        HyperparameterSpace("num_layers", "int", low=1, high=5),
        HyperparameterSpace("optimizer", "categorical", choices=["adam", "sgd", "rmsprop"]),
    ]
    return BayesianOptimizer(space, n_initial_points=n_initial_points, n_candidates=10)


def test_random_sample_within_bounds():
    np.random.seed(0)
    optimizer = _build_optimizer()

    sample = optimizer._random_sample()

    assert 0.001 <= sample["learning_rate"] <= 0.1
    assert 1 <= sample["num_layers"] <= 5
    assert sample["optimizer"] in {"adam", "sgd", "rmsprop"}


def test_encode_decode_round_trip():
    optimizer = _build_optimizer()
    params = {
        "learning_rate": 0.01,
        "num_layers": 3,
        "optimizer": "sgd",
    }

    encoded = optimizer._encode_params(params)
    decoded = optimizer._decode_params(encoded)

    assert decoded["optimizer"] == params["optimizer"]
    assert decoded["num_layers"] == params["num_layers"]
    assert decoded["learning_rate"] == pytest.approx(params["learning_rate"])


def test_bayesian_suggest_after_observations():
    np.random.seed(1)
    optimizer = _build_optimizer(n_initial_points=1)

    # Record several observations to ensure GP fitting path is used
    for score in [0.1, 0.2, 0.15]:
        params = optimizer._random_sample()
        optimizer.update(params, score)

    suggestion = optimizer.suggest_next()

    assert set(suggestion.keys()) == {"learning_rate", "num_layers", "optimizer"}
    assert 0.001 <= suggestion["learning_rate"] <= 0.1
    assert 1 <= suggestion["num_layers"] <= 5
    assert suggestion["optimizer"] in {"adam", "sgd", "rmsprop"}
