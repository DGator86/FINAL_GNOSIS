from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from trade.execution_mapper import OptionLeg, ProposedTrade


@dataclass
class TradeRiskProfile:
    """Basic risk summary for a multi-leg trade."""

    max_loss: float
    max_gain: float
    reward_risk: float
    breakevens: List[float]


def _estimate_leg_cost(leg: OptionLeg) -> float:
    return abs(leg.quantity) * leg.premium * 100


def scale_trade_to_risk_budget(
    trade: ProposedTrade, spot: float, max_risk_dollars: float
) -> Tuple[ProposedTrade, TradeRiskProfile]:
    """Scale a proposed trade to fit within a dollar risk budget."""

    base_cost = sum(_estimate_leg_cost(leg) for leg in trade.legs) or 1.0
    scale_factor = min(1.0, max_risk_dollars / base_cost)

    scaled_legs: List[OptionLeg] = []
    for leg in trade.legs:
        scaled_qty = max(0, int(leg.quantity * scale_factor)) if leg.quantity >= 0 else min(0, int(leg.quantity * scale_factor))
        scaled_qty = scaled_qty if scaled_qty != 0 else (1 if leg.quantity > 0 else -1)
        scaled_legs.append(
            OptionLeg(
                option_type=leg.option_type,
                strike=leg.strike,
                expiration=leg.expiration,
                quantity=scaled_qty,
                premium=leg.premium,
            )
        )

    scaled_cost = sum(_estimate_leg_cost(leg) for leg in scaled_legs)
    max_loss = min(scaled_cost, max_risk_dollars)
    max_gain = max_risk_dollars if scaled_cost else 0.0
    reward_risk = max_gain / max_loss if max_loss else 0.0

    risk_profile = TradeRiskProfile(
        max_loss=max_loss,
        max_gain=max_gain,
        reward_risk=reward_risk,
        breakevens=[spot],
    )

    scaled_trade = ProposedTrade(
        underlying=trade.underlying,
        legs=scaled_legs,
        structure_type=trade.structure_type,
        target_dte=trade.target_dte,
        strategy_id=trade.strategy_id,
        tags=trade.tags,
        notes=trade.notes,
    )

    return scaled_trade, risk_profile
