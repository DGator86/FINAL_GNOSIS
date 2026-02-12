from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from trade.cone_metrics import ConeMetrics


@dataclass
class TradeRegime:
    """Classifies directional and structure biases from cone metrics."""

    trend_regime: str
    structure_bias: str


def classify_trade_regime(metrics: ConeMetrics) -> TradeRegime:
    """Map cone metrics into a coarse trade regime."""

    if metrics.direction_bias == "bull":
        trend_regime = "trend_long"
        structure_bias = "debit_call"
    elif metrics.direction_bias == "bear":
        trend_regime = "trend_short"
        structure_bias = "debit_put"
    elif metrics.vol_regime == "high":
        trend_regime = "mean_revert"
        structure_bias = "credit_iron_condor"
    else:
        trend_regime = "sideways"
        structure_bias = "neutral"

    return TradeRegime(trend_regime=trend_regime, structure_bias=structure_bias)
TrendRegime = Literal["trend_long", "trend_short", "mean_revert", "range"]
StructureBias = Literal["directional", "neutral", "income"]


@dataclass
class TradeRegime:
    """
    High-level regime inferred from cone + field.

    This is what the strategy selector consumes.
    """

    trend_regime: TrendRegime
    structure_bias: StructureBias


def classify_trade_regime(
    metrics: ConeMetrics,
    min_trend_drift: float = 0.02,  # 2% drift = real trend
    max_trend_vol: float = 0.25,  # if vol too insane, prefer income/neutral
) -> TradeRegime:
    """
    Classify the environment into a trading regime based on cone metrics.

    - If drift is small → more neutral/range.
    - If drift is large and vol not insane → trend following.
    - If drift is large but vol very wide → directional but via spreads.
    """

    drift = metrics.drift_pct
    vol = metrics.outer_width_pct

    # Trend vs neutral
    if abs(drift) < min_trend_drift:
        # No strong directional drift; type depends on vol
        if metrics.vol_regime == "low":
            trend_regime: TrendRegime = "range"
            structure_bias: StructureBias = "neutral"  # iron butterflies, calendars, etc.
        else:
            trend_regime = "mean_revert"  # short vol near extremes
            structure_bias = "income"
    else:
        # Clear directional drift
        if drift > 0:
            trend_regime = "trend_long"
        else:
            trend_regime = "trend_short"

        # If vol is huge, you still want directional exposure but with limited risk
        if vol > max_trend_vol:
            structure_bias = "directional"  # but via verticals / defined-risk
        else:
            structure_bias = "directional"

    return TradeRegime(
        trend_regime=trend_regime,
        structure_bias=structure_bias,
    )
