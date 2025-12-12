from __future__ import annotations

from typing import List

from trade.cone_metrics import ConeMetrics
from trade.execution_mapper import ProposedTrade
from trade.regime_classifier import TradeRegime
from trade.structure_selector import StructureSpec


def build_strategy_tags(
    metrics: ConeMetrics,
    regime: TradeRegime,
    spec: StructureSpec,
) -> tuple[str, List[str]]:
    """
    Build a strategy_id and a list of tags that describe how this trade was created.

    This is pure metadata: no side effects, just deterministic labels.
    """
    if regime.trend_regime in ("trend_long", "trend_short"):
        strategy_id = "cone_trend"
    elif regime.trend_regime in ("mean_revert",):
        strategy_id = "cone_mean_revert"
    else:
        strategy_id = "cone_neutral"

    tags: List[str] = []
    tags.append(f"dir_{metrics.direction_bias}")
    tags.append(f"trend_{regime.trend_regime}")
    tags.append(f"struct_{regime.structure_bias}")
    tags.append(f"struct_type_{spec.structure_type}")
    tags.append(f"vol_{metrics.vol_regime}")
    tags.append(f"uncert_{metrics.uncertainty_regime}")

    drift_abs = abs(metrics.drift_pct)
    outer = metrics.outer_width_pct

    if drift_abs >= 0.05:
        tags.append("drift_big")
    elif drift_abs >= 0.02:
        tags.append("drift_medium")
    else:
        tags.append("drift_small")

    if outer >= 0.20:
        tags.append("cone_very_wide")
    elif outer >= 0.10:
        tags.append("cone_wide")
    elif outer >= 0.05:
        tags.append("cone_normal")
    else:
        tags.append("cone_tight")

    if metrics.asymmetry_pct > 0.01:
        tags.append("skew_up")
    elif metrics.asymmetry_pct < -0.01:
        tags.append("skew_down")
    else:
        tags.append("skew_flat")

    return strategy_id, tags


def attach_tags_to_trade(
    trade: ProposedTrade, strategy_id: str, tags: List[str]
) -> ProposedTrade:
    """Return a copy of ProposedTrade with strategy_id and tags assigned."""

    return ProposedTrade(
        underlying=trade.underlying,
        legs=trade.legs,
        structure_type=trade.structure_type,
        target_dte=trade.target_dte,
        strategy_id=strategy_id,
        tags=tags,
        notes=trade.notes,
    )
