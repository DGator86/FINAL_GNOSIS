"""Composer agent package.

Provides consensus building from primary agents.

Versions:
- V1: Basic consensus with weighted averaging
- V2: Enhanced with backward/forward analysis
- V3: Express Lane support (0DTE, Cheap Calls)
- V4: Full GNOSIS architecture integration with PENTA methodology
"""

from agents.composer.composer_agent_v1 import ComposerAgentV1
from agents.composer.composer_agent_v4 import (
    ComposerAgentV4,
    ComposerOutput,
    ComposerMode,
    create_composer_v4,
)
from agents.composer.prediction_cone import (
    PredictionCone,
    PredictionConeConfig,
    build_prediction_cone,
    estimate_bars_per_year,
)

__all__ = [
    # V1 (canonical)
    "ComposerAgentV1",
    # V4 (full architecture)
    "ComposerAgentV4",
    "ComposerOutput",
    "ComposerMode",
    "create_composer_v4",
    # Prediction cone
    "PredictionConeConfig",
    "PredictionCone",
    "build_prediction_cone",
    "estimate_bars_per_year",
]
