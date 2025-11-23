from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


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
        drift_pct=drift_pct,
        inner_width_pct=inner_width_pct,
        outer_width_pct=outer_width_pct,
        up_tail_pct=up_tail_pct,
        down_tail_pct=down_tail_pct,
        asymmetry_pct=asymmetry_pct,
        direction_bias=str(direction_bias),
        vol_regime=str(vol_regime),
        uncertainty_regime=str(uncertainty_regime),
    )
