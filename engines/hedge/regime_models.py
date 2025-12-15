"""Probabilistic regime modeling utilities for Hedge Engine v3."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List, Tuple

import numpy as np
from loguru import logger
from sklearn.mixture import GaussianMixture


class MultiDimensionalRegimeDetector:
    """Lightweight Gaussian Mixture based regime identifier.

    The detector maintains a rolling feature store to adapt to new
    market conditions without requiring a full re-train. GaussianMixture
    is chosen over HMM for simplicity and to avoid long warmup times in
    intraday use.
    """

    def __init__(
        self,
        n_components: int = 3,
        history: int = 256,
        min_samples: int = 32,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.min_samples = min_samples
        self.features: Deque[List[float]] = deque(maxlen=history)
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state,
        )

    def _enough_data(self) -> bool:
        return len(self.features) >= self.min_samples

    def update(self, feature_vector: Iterable[float]) -> None:
        self.features.append(list(feature_vector))

    def infer(self, feature_vector: Iterable[float]) -> Tuple[str, Dict[str, float]]:
        """Infer regime label and probabilities.

        Returns neutral probabilities when there is insufficient data
        for mixture fitting to remain safe for live workflows.
        """

        vector = np.array(list(feature_vector)).reshape(1, -1)
        if not self._enough_data():
            logger.debug("RegimeDetector: insufficient data, returning neutral distribution")
            return "neutral", {"neutral": 1.0}

        try:
            self.model.fit(np.array(self.features))
            probs = self.model.predict_proba(vector)[0]
            label_idx = int(np.argmax(probs))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"RegimeDetector fallback due to error: {exc}")
            return "neutral", {"neutral": 1.0}

        label = self._map_label(label_idx)
        probabilities = self._format_probabilities(probs)
        return label, probabilities

    def _map_label(self, idx: int) -> str:
        labels = ["stability", "expansion", "squeeze"]
        if idx >= len(labels):
            return "neutral"
        return labels[idx]

    def _format_probabilities(self, probs: Iterable[float]) -> Dict[str, float]:
        labels = ["stability", "expansion", "squeeze"]
        return {labels[i] if i < len(labels) else f"component_{i}": float(p) for i, p in enumerate(probs)}
