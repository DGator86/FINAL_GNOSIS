from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from trade.structures import StructureType
from trade.structure_selector import StructureSpec


@dataclass
class OptionLeg:
    """Represents a single option leg within a multi-leg structure."""

    option_type: str
    strike: float
    expiration: str
    quantity: int
    premium: float


@dataclass
class ProposedTrade:
    """
    Full options position to send downstream to risk engine / execution.

    - strategy_id: high-level strategy label (e.g. "cone_trend", "cone_income").
    - tags: additional granular tags for analytics & filtering.
    """

    underlying: str
    legs: List[OptionLeg]
    structure_type: StructureType
    target_dte: int
    strategy_id: Optional[str] = None
    tags: List[str] = None
    notes: Optional[str] = None


DEFAULT_EXPIRY = "T+30"
DEFAULT_PREMIUM = 1.0


def _build_leg(option_type: str, strike: float, target_dte: int, quantity: int) -> OptionLeg:
    expiration = f"{target_dte}d"
    return OptionLeg(option_type=option_type, strike=strike, expiration=expiration, quantity=quantity, premium=DEFAULT_PREMIUM)


def build_proposed_trade_from_structure(
    spec: StructureSpec, chain, spot: float
) -> Optional[ProposedTrade]:
    """Map a high-level structure spec into concrete legs using a chain snapshot."""

    legs: List[OptionLeg] = []
    base_strike = spot * (1 + spec.strike_offset_pct)
    wing_strike = base_strike * (1 + spec.wing_width_pct)

    if spec.structure_type == StructureType.CALL_SPREAD:
        legs.append(_build_leg("call", base_strike, spec.target_dte, quantity=1))
        legs.append(_build_leg("call", wing_strike, spec.target_dte, quantity=-1))
    elif spec.structure_type == StructureType.PUT_SPREAD:
        legs.append(_build_leg("put", base_strike, spec.target_dte, quantity=1))
        legs.append(_build_leg("put", wing_strike, spec.target_dte, quantity=-1))
    elif spec.structure_type == StructureType.STRADDLE:
        legs.append(_build_leg("call", spot, spec.target_dte, quantity=1))
        legs.append(_build_leg("put", spot, spec.target_dte, quantity=1))
    elif spec.structure_type == StructureType.STRANGLE:
        legs.append(_build_leg("call", wing_strike, spec.target_dte, quantity=1))
        legs.append(_build_leg("put", base_strike, spec.target_dte, quantity=1))
    else:
        legs.append(_build_leg("call", base_strike, spec.target_dte, quantity=1))

    if not legs:
        return None

    return ProposedTrade(
        underlying=str(chain.symbol if hasattr(chain, "symbol") else ""),
        legs=legs,
        structure_type=spec.structure_type,
        target_dte=spec.target_dte,
    )
