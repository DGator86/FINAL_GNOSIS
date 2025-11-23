from __future__ import annotations

from dataclasses import dataclass

from trade.cone_metrics import ConeMetrics
from trade.regime_classifier import TradeRegime
from trade.structures import StructureType


@dataclass
class StructureSpec:
    """Specification for constructing an option structure."""

    structure_type: StructureType
    target_dte: int
    wing_width_pct: float
    strike_offset_pct: float
    size: float


def select_structure_from_cone(
    metrics: ConeMetrics, regime: TradeRegime, base_dte: int
) -> StructureSpec:
    """Choose a structure specification based on metrics and trade regime."""

    if regime.trend_regime == "trend_long":
        structure_type = StructureType.CALL_SPREAD
        strike_offset_pct = 0.02
    elif regime.trend_regime == "trend_short":
        structure_type = StructureType.PUT_SPREAD
        strike_offset_pct = -0.02
    elif metrics.vol_regime == "high":
        structure_type = StructureType.STRANGLE
        strike_offset_pct = 0.0
    else:
        structure_type = StructureType.STRADDLE
        strike_offset_pct = 0.0

    wing_width_pct = 0.05 if metrics.vol_regime != "low" else 0.03
    target_dte = max(7, base_dte)
    size = 1.0

    return StructureSpec(
        structure_type=structure_type,
        target_dte=target_dte,
        wing_width_pct=wing_width_pct,
        strike_offset_pct=strike_offset_pct,
        size=size,
    )
