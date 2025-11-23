from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Literal, Optional, Sequence

from trade.structure_selector import StructureSpec, StructureType

OptionType = Literal["call", "put"]
OptionSide = Literal["buy", "sell"]
LegRole = Literal["long", "short", "hedge", "body", "wing"]


@dataclass
class OptionContract:
    """
    Normalized representation of a single option.

    You should populate this from your data provider (Polygon, Tradier, etc.)
    in your data ingestion layer, NOT here.
    """

    symbol: str
    underlying: str
    expiry: date
    strike: float
    option_type: OptionType  # "call" or "put"
    bid: float
    ask: float
    mid: float
    delta: float
    open_interest: int
    volume: int


@dataclass
class OptionLeg:
    """One leg of an options strategy."""

    contract: OptionContract
    side: OptionSide  # "buy" or "sell"
    quantity: int
    role: LegRole


@dataclass
class ProposedTrade:
    """
    Full options position to send downstream to risk engine / execution.

    Execution Agent can later enrich this with broker-specific fields.
    """

    underlying: str
    legs: List[OptionLeg]
    structure_type: StructureType
    target_dte: int
    notes: Optional[str] = None


def select_expiry_by_target_dte(
    chain: Sequence[OptionContract],
    target_dte: int,
    max_slippage_days: int = 5,
) -> Optional[date]:
    """
    Pick the expiry with days-to-expiry closest to target_dte within a tolerance window.

    The caller should pass contracts observed on the same reference date. If the
    environment already computes DTE per contract, prefer that value. This helper
    infers relative DTE using the earliest expiry as a baseline.
    """

    if not chain:
        return None

    expiries = sorted({c.expiry for c in chain})
    today_guess = min(expiries)

    best_exp = None
    best_diff = None

    for exp in expiries:
        relative_dte = (exp - today_guess).days
        diff = abs(relative_dte - target_dte)
        if diff > max_slippage_days:
            continue
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_exp = exp

    return best_exp


def filter_chain_by_expiry(
    chain: Sequence[OptionContract],
    expiry: date,
) -> List[OptionContract]:
    return [c for c in chain if c.expiry == expiry]


def filter_chain_by_type(
    chain: Sequence[OptionContract],
    option_type: OptionType,
) -> List[OptionContract]:
    return [c for c in chain if c.option_type == option_type]


def filter_liquid(
    chain: Sequence[OptionContract],
    min_oi: int = 50,
    min_volume: int = 10,
    max_spread_pct: float = 0.1,  # 10% of mid
) -> List[OptionContract]:
    """
    Filter out illiquid contracts using simple thresholds.
    """

    liquid = []
    for c in chain:
        if c.mid <= 0:
            continue
        spread = c.ask - c.bid
        spread_pct = spread / c.mid if c.mid > 0 else float("inf")
        if c.open_interest >= min_oi and c.volume >= min_volume and spread_pct <= max_spread_pct:
            liquid.append(c)
    return liquid


def select_strike_near_price(
    chain: Sequence[OptionContract],
    target_price: float,
) -> Optional[OptionContract]:
    """Pick the contract whose strike is closest to target_price."""

    if not chain:
        return None
    best = None
    best_diff = None
    for c in chain:
        diff = abs(c.strike - target_price)
        if best is None or diff < best_diff:
            best = c
            best_diff = diff
    return best


def build_long_call(
    spec: StructureSpec,
    chain: Sequence[OptionContract],
    spot: float,
) -> Optional[ProposedTrade]:
    exp = select_expiry_by_target_dte(chain, spec.target_dte)
    if exp is None:
        return None

    exp_chain = filter_chain_by_expiry(chain, exp)
    exp_chain = filter_chain_by_type(exp_chain, "call")
    exp_chain = filter_liquid(exp_chain)

    if not exp_chain:
        return None

    offset_pct = spec.strike_offset_pct or 0.0
    target_price = spot * (1.0 + offset_pct)
    contract = select_strike_near_price(exp_chain, target_price)
    if contract is None:
        return None

    leg = OptionLeg(
        contract=contract,
        side="buy",
        quantity=spec.size,
        role="long",
    )
    return ProposedTrade(
        underlying=contract.underlying,
        legs=[leg],
        structure_type=spec.structure_type,
        target_dte=spec.target_dte,
        notes="Auto-generated long call from cone",
    )


def build_long_put(
    spec: StructureSpec,
    chain: Sequence[OptionContract],
    spot: float,
) -> Optional[ProposedTrade]:
    exp = select_expiry_by_target_dte(chain, spec.target_dte)
    if exp is None:
        return None

    exp_chain = filter_chain_by_expiry(chain, exp)
    exp_chain = filter_chain_by_type(exp_chain, "put")
    exp_chain = filter_liquid(exp_chain)

    if not exp_chain:
        return None

    offset_pct = spec.strike_offset_pct or 0.0
    target_price = spot * (1.0 + offset_pct)
    contract = select_strike_near_price(exp_chain, target_price)
    if contract is None:
        return None

    leg = OptionLeg(
        contract=contract,
        side="buy",
        quantity=spec.size,
        role="long",
    )
    return ProposedTrade(
        underlying=contract.underlying,
        legs=[leg],
        structure_type=spec.structure_type,
        target_dte=spec.target_dte,
        notes="Auto-generated long put from cone",
    )


def build_debit_spread(
    spec: StructureSpec,
    chain: Sequence[OptionContract],
    spot: float,
    option_type: OptionType,
) -> Optional[ProposedTrade]:
    exp = select_expiry_by_target_dte(chain, spec.target_dte)
    if exp is None:
        return None

    exp_chain = filter_chain_by_expiry(chain, exp)
    exp_chain = filter_chain_by_type(exp_chain, option_type)
    exp_chain = filter_liquid(exp_chain)

    if not exp_chain:
        return None

    offset_pct = spec.strike_offset_pct or 0.0
    wing_width_pct = spec.wing_width_pct or 0.05

    base_price = spot * (1.0 + offset_pct)

    if option_type == "call":
        long_target = base_price
        short_target = base_price * (1.0 + wing_width_pct)
    else:
        long_target = base_price
        short_target = base_price * (1.0 - wing_width_pct)

    long_contract = select_strike_near_price(exp_chain, long_target)
    short_contract = select_strike_near_price(exp_chain, short_target)

    if long_contract is None or short_contract is None:
        return None

    legs = [
        OptionLeg(
            contract=long_contract,
            side="buy",
            quantity=spec.size,
            role="body",
        ),
        OptionLeg(
            contract=short_contract,
            side="sell",
            quantity=spec.size,
            role="wing",
        ),
    ]

    return ProposedTrade(
        underlying=long_contract.underlying,
        legs=legs,
        structure_type=spec.structure_type,
        target_dte=spec.target_dte,
        notes=f"Auto-generated {option_type} debit spread from cone",
    )


def build_iron_condor(
    spec: StructureSpec,
    chain: Sequence[OptionContract],
    spot: float,
) -> Optional[ProposedTrade]:
    exp = select_expiry_by_target_dte(chain, spec.target_dte)
    if exp is None:
        return None

    exp_chain = filter_chain_by_expiry(chain, exp)
    exp_chain = filter_liquid(exp_chain)

    calls = filter_chain_by_type(exp_chain, "call")
    puts = filter_chain_by_type(exp_chain, "put")

    if not calls or not puts:
        return None

    width_pct = spec.wing_width_pct or 0.10

    short_call_target = spot * (1.0 + width_pct * 0.5)
    long_call_target = spot * (1.0 + width_pct)

    short_put_target = spot * (1.0 - width_pct * 0.5)
    long_put_target = spot * (1.0 - width_pct)

    sc = select_strike_near_price(calls, short_call_target)
    lc = select_strike_near_price(calls, long_call_target)
    sp = select_strike_near_price(puts, short_put_target)
    lp = select_strike_near_price(puts, long_put_target)

    if None in (sc, lc, sp, lp):
        return None

    legs = [
        OptionLeg(contract=sc, side="sell", quantity=spec.size, role="body"),
        OptionLeg(contract=lc, side="buy", quantity=spec.size, role="wing"),
        OptionLeg(contract=sp, side="sell", quantity=spec.size, role="body"),
        OptionLeg(contract=lp, side="buy", quantity=spec.size, role="wing"),
    ]

    return ProposedTrade(
        underlying=sc.underlying,
        legs=legs,
        structure_type=spec.structure_type,
        target_dte=spec.target_dte,
        notes="Auto-generated iron condor from cone",
    )


def build_iron_butterfly(
    spec: StructureSpec,
    chain: Sequence[OptionContract],
    spot: float,
) -> Optional[ProposedTrade]:
    exp = select_expiry_by_target_dte(chain, spec.target_dte)
    if exp is None:
        return None

    exp_chain = filter_chain_by_expiry(chain, exp)
    exp_chain = filter_liquid(exp_chain)

    calls = filter_chain_by_type(exp_chain, "call")
    puts = filter_chain_by_type(exp_chain, "put")

    if not calls or not puts:
        return None

    width_pct = spec.wing_width_pct or 0.05

    body_target = spot
    upper_wing_target = spot * (1.0 + width_pct)
    lower_wing_target = spot * (1.0 - width_pct)

    body_call = select_strike_near_price(calls, body_target)
    body_put = select_strike_near_price(puts, body_target)
    wing_call = select_strike_near_price(calls, upper_wing_target)
    wing_put = select_strike_near_price(puts, lower_wing_target)

    if None in (body_call, body_put, wing_call, wing_put):
        return None

    legs = [
        OptionLeg(contract=body_call, side="sell", quantity=spec.size, role="body"),
        OptionLeg(contract=body_put, side="sell", quantity=spec.size, role="body"),
        OptionLeg(contract=wing_call, side="buy", quantity=spec.size, role="wing"),
        OptionLeg(contract=wing_put, side="buy", quantity=spec.size, role="wing"),
    ]

    return ProposedTrade(
        underlying=body_call.underlying,
        legs=legs,
        structure_type=spec.structure_type,
        target_dte=spec.target_dte,
        notes="Auto-generated iron butterfly from cone",
    )


def build_long_strangle(
    spec: StructureSpec,
    chain: Sequence[OptionContract],
    spot: float,
) -> Optional[ProposedTrade]:
    exp = select_expiry_by_target_dte(chain, spec.target_dte)
    if exp is None:
        return None

    exp_chain = filter_chain_by_expiry(chain, exp)
    exp_chain = filter_liquid(exp_chain)

    calls = filter_chain_by_type(exp_chain, "call")
    puts = filter_chain_by_type(exp_chain, "put")

    if not calls or not puts:
        return None

    width_pct = spec.wing_width_pct or 0.1

    call_target = spot * (1.0 + width_pct * 0.5)
    put_target = spot * (1.0 - width_pct * 0.5)

    call_contract = select_strike_near_price(calls, call_target)
    put_contract = select_strike_near_price(puts, put_target)

    if call_contract is None or put_contract is None:
        return None

    legs = [
        OptionLeg(contract=call_contract, side="buy", quantity=spec.size, role="wing"),
        OptionLeg(contract=put_contract, side="buy", quantity=spec.size, role="wing"),
    ]

    return ProposedTrade(
        underlying=call_contract.underlying,
        legs=legs,
        structure_type=spec.structure_type,
        target_dte=spec.target_dte,
        notes="Auto-generated long strangle from cone",
    )


STRUCTURE_BUILDERS = {
    "long_call": lambda spec, chain, spot: build_long_call(spec, chain, spot),
    "long_put": lambda spec, chain, spot: build_long_put(spec, chain, spot),
    "call_debit_spread": lambda spec, chain, spot: build_debit_spread(spec, chain, spot, "call"),
    "put_debit_spread": lambda spec, chain, spot: build_debit_spread(spec, chain, spot, "put"),
    "iron_condor": lambda spec, chain, spot: build_iron_condor(spec, chain, spot),
    "iron_butterfly": lambda spec, chain, spot: build_iron_butterfly(spec, chain, spot),
    "long_strangle": lambda spec, chain, spot: build_long_strangle(spec, chain, spot),
}


def build_proposed_trade_from_structure(
    spec: StructureSpec,
    chain: Sequence[OptionContract],
    spot: float,
) -> Optional[ProposedTrade]:
    """
    Top-level mapper from StructureSpec to ProposedTrade.

    Returns None if the chain cannot support the requested structure
    (no expiry, no liquid strikes, etc.).
    """

    builder = STRUCTURE_BUILDERS.get(spec.structure_type)
    if builder is None:
        return None
    return builder(spec, chain, spot)
