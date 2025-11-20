"""Anomaly detection utilities inspired by isolation forests and autoencoders."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from loguru import logger
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """IsolationForest-based anomaly detector for market feature vectors."""

    def __init__(
        self,
        contamination: float = 0.05,
        warmup: int = 25,
        random_state: int | None = 42,
    ) -> None:
        self.contamination = contamination
        self.warmup = warmup
        self.random_state = random_state
        self._history: List[List[float]] = []
        self._model = IsolationForest(
            contamination=self.contamination, random_state=self.random_state
        )

    def score(self, vector: List[float]) -> Dict[str, float | bool]:
        """Return anomaly score and flag for the incoming vector."""

        self._history.append(vector)
        if len(self._history) < self.warmup:
            logger.debug("AnomalyDetector warming up; returning neutral score")
            return {"score": 0.5, "flagged": False}

        # Fit on all history for simplicity; lightweight enough for batch mode
        data = np.asarray(self._history, dtype=float)
        self._model.fit(data)
        score = float(-self._model.decision_function([vector])[0])
        threshold = float(np.quantile(-self._model.decision_function(data), 0.95))
        flagged = score >= threshold
        return {"score": score, "flagged": flagged}
