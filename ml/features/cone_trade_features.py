from __future__ import annotations

from typing import Any, Dict, Optional

from trade.cone_metrics import ConeMetrics
from trade.regime_classifier import TradeRegime
from trade.risk_analysis import TradeRiskProfile
from trade.strategy_tags import build_strategy_tags
from trade.structure_selector import StructureSpec


def build_cone_trade_feature_row(
    symbol: str,
    timeframe: str,
    horizon_bars: int,
    spot: float,
    metrics: ConeMetrics,
    regime: TradeRegime,
    spec: StructureSpec,
    risk: Optional[TradeRiskProfile],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a flat feature dict from cone + structure + risk for ML.

    This is what you write into your feature store as ONE row per decision.
    """
    strategy_id, tags = build_strategy_tags(metrics, regime, spec)

    row: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon_bars": horizon_bars,
        "spot": float(spot),
        "cone_drift_pct": metrics.drift_pct,
        "cone_inner_width_pct": metrics.inner_width_pct,
        "cone_outer_width_pct": metrics.outer_width_pct,
        "cone_up_tail_pct": metrics.up_tail_pct,
        "cone_down_tail_pct": metrics.down_tail_pct,
        "cone_asymmetry_pct": metrics.asymmetry_pct,
        "cone_direction_bias": metrics.direction_bias,
        "cone_vol_regime": metrics.vol_regime,
        "cone_uncertainty_regime": metrics.uncertainty_regime,
        "trade_trend_regime": regime.trend_regime,
        "trade_structure_bias": regime.structure_bias,
        "structure_type": spec.structure_type,
        "structure_target_dte": spec.target_dte,
        "structure_wing_width_pct": spec.wing_width_pct,
        "structure_strike_offset_pct": spec.strike_offset_pct,
        "structure_size_base": spec.size,
        "strategy_id": strategy_id,
        "strategy_tags": ",".join(tags),
    }

    if risk is not None:
        row.update(
            {
                "risk_max_loss": risk.max_loss,
                "risk_max_gain": risk.max_gain,
                "risk_reward_risk": risk.reward_risk,
                "risk_n_breakevens": len(risk.breakevens),
            }
        )

    if extra_meta:
        row.update(extra_meta)

    return row
