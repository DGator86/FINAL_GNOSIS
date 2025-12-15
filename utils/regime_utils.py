"""Shared regime utilities for engines.

Provides a thin wrapper around hmmlearn where available, with a deterministic
fallback to rolling variance regimes. This keeps callers decoupled from heavy
dependencies while enabling reuse across Hedge/Liquidity agents.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from loguru import logger

try:  # pragma: no cover - optional dependency
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover
    GaussianHMM = None


class RegimeHMM:
    """Simple wrapper providing fit/predict helpers."""

    def __init__(self, n_components: int = 3) -> None:
        self.n_components = n_components
        self.model = GaussianHMM(n_components=n_components) if GaussianHMM else None

    def fit(self, observations: Sequence[Sequence[float]]) -> None:
        if not self.model:
            logger.debug("GaussianHMM unavailable; skipping fit")
            return
        self.model.fit(np.array(observations))

    def predict_proba(self, features: Sequence[float]) -> List[float]:
        if not self.model:
            # fallback: variance-based regime probabilities
            volatility = np.std(features)
            probs = np.array([max(0.0, 1 - volatility), volatility / 2, volatility / 2])
            probs = probs / probs.sum()
            return probs.tolist()
        return self.model.predict_proba([features])[0].tolist()


__all__ = ["RegimeHMM"]
