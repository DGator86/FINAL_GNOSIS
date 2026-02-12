from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from trade.cone_metrics import ConeMetrics
from trade.regime_classifier import TradeRegime
from trade.structures import StructureType

StructureType = Literal[
    "long_call",
    "long_put",
    "call_debit_spread",
    "put_debit_spread",
    "iron_condor",
    "iron_butterfly",
    "long_strangle",
    "call_credit_spread",
    "put_credit_spread",
    "iron_condor",
    "iron_butterfly",
    "long_strangle",
    "calendar_spread",
]


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
    """Specification of an options structure selected from cone metrics."""

    structure_type: StructureType
    size: int
    target_dte: int
    strike_offset_pct: Optional[float] = None
    wing_width_pct: Optional[float] = None
    """
    Abstract options structure specification.

    The Execution Agent will translate this into actual strikes/expiries.
    """

    structure_type: StructureType
    target_dte: int
    # All widths/offsets expressed as % of spot so they are symbol-agnostic.
    wing_width_pct: Optional[float] = None
    strike_offset_pct: Optional[float] = None
    size: int = 1  # number of contracts (per leg, conventionally)


def select_structure_from_cone(
    metrics: ConeMetrics,
    regime: TradeRegime,
    base_dte: int,
    short_dte: int = 7,
    long_dte: int = 30,
) -> StructureSpec:
    """
    Map cone metrics + regime into a high-level options structure.

    Rules of thumb:
    - Strong drift, medium vol → directional debit spread.
    - Strong drift, huge vol → defined-risk vertical or long option.
    - Neutral drift, low vol → calendars / butterflies.
    - Neutral drift, high vol → iron condor (sell width of cone).
    """

    drift = metrics.drift_pct
    vol = metrics.outer_width_pct

    # Directional trend regimes
    if regime.structure_bias == "directional":
        if regime.trend_regime == "trend_long":
            # Bullish environment
            if metrics.vol_regime in ("medium", "high"):
                # Use defined risk via call debit spread
                return StructureSpec(
                    structure_type="call_debit_spread",
                    target_dte=base_dte,
                    wing_width_pct=min(0.15, vol * 0.5),  # half of cone width, capped at 15%
                    strike_offset_pct=metrics.drift_pct,  # aim near projected center
                )
            else:
                # Low vol and bullish: long call
                return StructureSpec(
                    structure_type="long_call",
                    target_dte=base_dte,
                    strike_offset_pct=metrics.drift_pct,
                )

        if regime.trend_regime == "trend_short":
            # Bearish environment
            if metrics.vol_regime in ("medium", "high"):
                return StructureSpec(
                    structure_type="put_debit_spread",
                    target_dte=base_dte,
                    wing_width_pct=min(0.15, vol * 0.5),
                    strike_offset_pct=metrics.drift_pct,
                )
            else:
                return StructureSpec(
                    structure_type="long_put",
                    target_dte=base_dte,
                    strike_offset_pct=metrics.drift_pct,
                )

    # Neutral / income regimes
    if regime.structure_bias in ("neutral", "income"):
        if metrics.vol_regime == "high":
            # Wide cone, little drift: sell the range via iron condor
            return StructureSpec(
                structure_type="iron_condor",
                target_dte=short_dte,
                wing_width_pct=min(0.25, metrics.outer_width_pct * 0.8),
                strike_offset_pct=0.0,  # centered near spot
            )

        if metrics.vol_regime == "low":
            # Tight cone: calendars / butterflies around center
            return StructureSpec(
                structure_type="iron_butterfly",
                target_dte=short_dte,
                wing_width_pct=max(0.03, metrics.inner_width_pct * 0.5),
                strike_offset_pct=0.0,
            )

        # Medium vol, modest drift: long strangle or calendar
        return StructureSpec(
            structure_type="long_strangle",
            target_dte=base_dte,
            wing_width_pct=max(0.05, metrics.outer_width_pct * 0.3),
            strike_offset_pct=0.0,
        )

    # Fallback: no strong signal, stay neutral
    return StructureSpec(
        structure_type="calendar_spread",
        target_dte=long_dte,
        wing_width_pct=max(0.05, metrics.inner_width_pct),
        strike_offset_pct=0.0,
    )
