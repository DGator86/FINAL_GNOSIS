"""Base abstractions for GNOSIS time series models."""

from __future__ import annotations

import logging
from typing import Any, Dict, List


class BaseGnosisModel:
    """Minimal base class to standardise GNOSIS model behaviour."""

    def __init__(self, config: Dict[str, Any], model_name: str):
        self.config = config
        self.model_name = model_name
        self.logger = logging.getLogger(model_name)

        # Training metadata
        self.training_history: List[Dict[str, Any]] = []
        self.is_trained: bool = False

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.__class__.__name__}(model_name={self.model_name})"
