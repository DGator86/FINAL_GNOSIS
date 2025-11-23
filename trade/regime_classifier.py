from __future__ import annotations

from dataclasses import dataclass

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
