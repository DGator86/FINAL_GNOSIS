from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ml.features.cone_trade_features import build_cone_trade_feature_row
from trade.cone_metrics import compute_cone_metrics
from trade.execution_mapper import ProposedTrade, build_proposed_trade_from_structure
from trade.regime_classifier import classify_trade_regime
from trade.risk_analysis import TradeRiskProfile, scale_trade_to_risk_budget
from trade.strategy_tags import attach_tags_to_trade, build_strategy_tags
from trade.structure_selector import select_structure_from_cone


def build_risk_adjusted_trade_from_cone(
    symbol: str,
    timeframe: str,
    horizon_bars: int,
    spot: float,
    directive,
    cone,
    option_chain,
    base_dte: int,
    max_risk_dollars: float,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[ProposedTrade], Optional[TradeRiskProfile], Optional[Dict[str, Any]]]:
    """
    Full pipeline for ONE symbol:

    1) Distill cone → metrics.
    2) Classify TradeRegime.
    3) Select StructureSpec.
    4) Build raw ProposedTrade from chain.
    5) Scale to risk budget.
    6) Tag trade + build ML feature row.
    """
    metrics = compute_cone_metrics(spot=spot, cone=cone)
    regime = classify_trade_regime(metrics)
    spec = select_structure_from_cone(metrics=metrics, regime=regime, base_dte=base_dte)

    raw_trade = build_proposed_trade_from_structure(spec=spec, chain=option_chain, spot=spot)
    if raw_trade is None or not raw_trade.legs:
        return None, None, None

    scaled_trade, risk_profile = scale_trade_to_risk_budget(
        trade=raw_trade, spot=spot, max_risk_dollars=max_risk_dollars
    )

    if all(leg.quantity == 0 for leg in scaled_trade.legs):
        return None, risk_profile, None

    strategy_id, tags = build_strategy_tags(metrics, regime, spec)
    tagged_trade = attach_tags_to_trade(trade=scaled_trade, strategy_id=strategy_id, tags=tags)

    feature_row = build_cone_trade_feature_row(
        symbol=symbol,
        timeframe=timeframe,
        horizon_bars=horizon_bars,
        spot=spot,
        metrics=metrics,
        regime=regime,
        spec=spec,
        risk=risk_profile,
        extra_meta=extra_meta,
    )

    return tagged_trade, risk_profile, feature_row
