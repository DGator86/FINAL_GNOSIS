from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from typing import Literal

import numpy as np

from agents.composer.prediction_cone import PredictionCone


DirectionBias = Literal["bull", "bear", "neutral"]
VolRegime = Literal["low", "medium", "high"]
UncertaintyRegime = Literal["tight", "normal", "wide"]


@dataclass
class ConeMetrics:
    """Normalized cone geometry metrics used for downstream decisions."""

    drift_pct: float
    inner_width_pct: float
    outer_width_pct: float
    up_tail_pct: float
    down_tail_pct: float
    asymmetry_pct: float
    direction_bias: str
    vol_regime: str
    uncertainty_regime: str


def compute_cone_metrics(spot: float, cone: Dict[str, Any]) -> ConeMetrics:
    """Derive cone metrics from a cone payload with sensible fallbacks."""

    drift_pct = float(cone.get("drift_pct", 0.0))
    inner_width_pct = float(cone.get("inner_width_pct", 0.0))
    outer_width_pct = float(cone.get("outer_width_pct", 0.0))
    up_tail_pct = float(cone.get("up_tail_pct", 0.0))
    down_tail_pct = float(cone.get("down_tail_pct", 0.0))

    asymmetry_pct = float(cone.get("asymmetry_pct", up_tail_pct - down_tail_pct))

    direction_bias = cone.get("direction_bias")
    if direction_bias is None:
        if drift_pct > 0.01:
            direction_bias = "bull"
        elif drift_pct < -0.01:
            direction_bias = "bear"
        else:
            direction_bias = "neutral"

    vol_regime = cone.get("vol_regime")
    if vol_regime is None:
        outer_abs = abs(outer_width_pct)
        if outer_abs >= 0.2:
            vol_regime = "high"
        elif outer_abs >= 0.1:
            vol_regime = "moderate"
        else:
            vol_regime = "low"

    uncertainty_regime = cone.get("uncertainty_regime")
    if uncertainty_regime is None:
        uncertainty_regime = "elevated" if outer_width_pct >= 0.15 else "normal"

    return ConeMetrics(
    """
    Distilled metrics from PredictionCone for the Trade Agent.

    All values are expressed as *percent of spot* to be comparable across symbols.
    """

    direction_bias: DirectionBias
    drift_pct: float  # signed total drift of center over horizon (% of spot)
    inner_width_pct: float  # width of inner band at horizon (% of spot)
    outer_width_pct: float  # width of outer band at horizon (% of spot)
    up_tail_pct: float  # (upper_2 - center) at horizon (% of spot)
    down_tail_pct: float  # (center - lower_2) at horizon (% of spot)
    asymmetry_pct: float  # up_tail_pct - down_tail_pct
    vol_regime: VolRegime
    uncertainty_regime: UncertaintyRegime


def compute_cone_metrics(
    spot: float,
    cone: PredictionCone,
    drift_neutral_epsilon: float = 0.005,  # 0.5%
    low_vol_threshold: float = 0.05,  # 5% outer width
    high_vol_threshold: float = 0.15,  # 15% outer width
) -> ConeMetrics:
    """
    Convert PredictionCone into simple, symbol-agnostic metrics.

    - All thresholds are in *percent of spot*.
    - Uses the blended_path by default (field + Donchian).
    """

    path = cone.blended_path
    center = path.center
    upper_2 = path.upper_2
    lower_2 = path.lower_2
    upper_1 = path.upper_1
    lower_1 = path.lower_1

    T = cone.steps

    # Total drift: centerline at horizon vs spot
    drift_abs = center[T] - spot
    drift_pct = float(drift_abs / spot)

    if drift_pct > drift_neutral_epsilon:
        direction_bias: DirectionBias = "bull"
    elif drift_pct < -drift_neutral_epsilon:
        direction_bias = "bear"
    else:
        direction_bias = "neutral"

    # Widths and tails at horizon
    inner_width_abs = upper_1[T] - lower_1[T]
    outer_width_abs = upper_2[T] - lower_2[T]

    inner_width_pct = float(inner_width_abs / spot)
    outer_width_pct = float(outer_width_abs / spot)

    up_tail_abs = upper_2[T] - center[T]
    down_tail_abs = center[T] - lower_2[T]

    up_tail_pct = float(up_tail_abs / spot)
    down_tail_pct = float(down_tail_abs / spot)

    asymmetry_pct = up_tail_pct - down_tail_pct

    # Vol/width regime
    if outer_width_pct < low_vol_threshold:
        vol_regime: VolRegime = "low"
    elif outer_width_pct > high_vol_threshold:
        vol_regime = "high"
    else:
        vol_regime = "medium"

    # Uncertainty regime can be same as vol_regime for now,
    # but you can later incorporate cone metadata (liquidity, Greeks).
    if vol_regime == "low":
        uncertainty_regime: UncertaintyRegime = "tight"
    elif vol_regime == "high":
        uncertainty_regime = "wide"
    else:
        uncertainty_regime = "normal"

    return ConeMetrics(
        direction_bias=direction_bias,
        drift_pct=drift_pct,
        inner_width_pct=inner_width_pct,
        outer_width_pct=outer_width_pct,
        up_tail_pct=up_tail_pct,
        down_tail_pct=down_tail_pct,
        asymmetry_pct=asymmetry_pct,
        direction_bias=str(direction_bias),
        vol_regime=str(vol_regime),
        uncertainty_regime=str(uncertainty_regime),
        vol_regime=vol_regime,
        uncertainty_regime=uncertainty_regime,
    )
