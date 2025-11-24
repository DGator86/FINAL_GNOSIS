"""Risk analysis and sizing utilities for options trades."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from trade.execution_mapper import OptionContract, OptionLeg, ProposedTrade


@dataclass
class TradeRiskProfile:
    """
    Aggregate risk metrics for a multi-leg options position.

    All PnL values are in *dollars* for the full position (all contracts).
    """

    max_loss: float
    max_gain: float
    reward_risk: float
    breakevens: List[float]
    price_grid: np.ndarray
    pnl_grid: np.ndarray


CONTRACT_MULTIPLIER = 100.0  # standard US equity/ETF options


def _mid_price(contract: OptionContract) -> float:
    """Best-effort mid-price using available quote fields."""

    if hasattr(contract, "mid") and contract.mid is not None:
        return float(contract.mid)

    bid = float(getattr(contract, "bid", 0.0) or 0.0)
    ask = float(getattr(contract, "ask", 0.0) or 0.0)

    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0

    last = float(getattr(contract, "last", 0.0) or 0.0)
    return last


def _leg_entry_cashflow(leg: OptionLeg) -> float:
    """
    Entry cashflow for one leg:

    - Buy: you PAY premium → negative cashflow.
    - Sell: you RECEIVE premium → positive cashflow.
    """

    c: OptionContract = leg.contract
    mid = _mid_price(c)
    if mid <= 0:
        return 0.0

    side_sign = 1.0 if leg.side == "sell" else -1.0
    return side_sign * mid * CONTRACT_MULTIPLIER * leg.quantity


def _leg_payoff_at_expiry(leg: OptionLeg, underlying_price: float) -> float:
    """
    Payoff at expiry for one leg (no premium, just intrinsic value).
    """

    c: OptionContract = leg.contract
    K = float(c.strike)
    S = float(underlying_price)

    if c.option_type == "call":
        intrinsic = max(S - K, 0.0)
    else:  # put
        intrinsic = max(K - S, 0.0)

    side_sign = 1.0 if leg.side == "sell" else -1.0
    return side_sign * intrinsic * CONTRACT_MULTIPLIER * leg.quantity


def compute_pnl_over_grid(trade: ProposedTrade, price_grid: np.ndarray) -> np.ndarray:
    """
    Compute total PnL of the position over a grid of underlying prices.

    PnL = sum(payoff_at_expiry) + sum(entry_cashflows)
    """

    entry_cash = sum(_leg_entry_cashflow(leg) for leg in trade.legs)

    pnl = np.zeros_like(price_grid, dtype=float)
    for i, S in enumerate(price_grid):
        payoff = 0.0
        for leg in trade.legs:
            payoff += _leg_payoff_at_expiry(leg, S)
        pnl[i] = payoff + entry_cash

    return pnl


def estimate_trade_risk(
    trade: ProposedTrade,
    spot: float,
    price_range_pct: float = 0.5,
    n_points: int = 201,
) -> TradeRiskProfile:
    """
    Estimate risk profile using a generic price grid.

    - price_range_pct: how far around spot to explore (e.g., 0.5 → ±50%).
    - n_points: resolution of the grid.

    This is structure-agnostic and works for any linear combination of options.
    """

    spot = float(spot)
    if spot <= 0:
        raise ValueError("Spot must be positive for risk estimation.")

    lower = spot * (1.0 - price_range_pct)
    upper = spot * (1.0 + price_range_pct)
    if lower <= 0:
        lower = 0.01 * spot

    price_grid = np.linspace(lower, upper, n_points)
    pnl_grid = compute_pnl_over_grid(trade, price_grid)

    max_gain = float(pnl_grid.max())
    max_loss = float(-min(pnl_grid.min(), 0.0))

    if max_loss > 0:
        rr = max_gain / max_loss if max_gain > 0 else 0.0
    else:
        rr = float("inf") if max_gain > 0 else 0.0

    breakevens: List[float] = []
    for i in range(len(price_grid) - 1):
        p0, p1 = pnl_grid[i], pnl_grid[i + 1]
        if p0 == 0.0:
            breakevens.append(float(price_grid[i]))
        elif p0 * p1 < 0:
            x0, x1 = price_grid[i], price_grid[i + 1]
            t = -p0 / (p1 - p0)
            be = x0 + t * (x1 - x0)
            breakevens.append(float(be))

    return TradeRiskProfile(
        max_loss=max_loss,
        max_gain=max_gain,
        reward_risk=rr,
        breakevens=breakevens,
        price_grid=price_grid,
        pnl_grid=pnl_grid,
    )


def scale_trade_to_risk_budget(
    trade: ProposedTrade,
    spot: float,
    max_risk_dollars: float,
    min_contracts: int = 0,
    max_contracts: int = 100,
) -> Tuple[ProposedTrade, TradeRiskProfile]:
    """
    Scale a ProposedTrade's leg quantities to fit within a dollar risk budget.

    - Assumes risk scales linearly with size (true for combinations of linear payoffs).
    - Returns the scaled trade and the recomputed risk profile.
    """

    if max_risk_dollars <= 0:
        raise ValueError("max_risk_dollars must be positive.")

    baseline_legs: List[OptionLeg] = []
    for leg in trade.legs:
        baseline_legs.append(
            OptionLeg(
                contract=leg.contract,
                side=leg.side,
                quantity=1 if leg.quantity > 0 else 0,
                role=leg.role,
            )
        )

    baseline_trade = ProposedTrade(
        underlying=trade.underlying,
        legs=baseline_legs,
        structure_type=trade.structure_type,
        target_dte=trade.target_dte,
        notes=(trade.notes or "") + " [baseline_size=1]",
    )

    baseline_risk = estimate_trade_risk(
        baseline_trade,
        spot=spot,
        price_range_pct=0.5,
        n_points=201,
    )

    if baseline_risk.max_loss <= 0:
        return trade, baseline_risk

    raw_factor = max_risk_dollars / baseline_risk.max_loss
    contracts = int(np.floor(raw_factor))

    if contracts < min_contracts:
        contracts = min_contracts
    if contracts > max_contracts:
        contracts = max_contracts

    if contracts <= 0:
        scaled_legs = [
            OptionLeg(
                contract=leg.contract,
                side=leg.side,
                quantity=0,
                role=leg.role,
            )
            for leg in trade.legs
        ]
        scaled_trade = ProposedTrade(
            underlying=trade.underlying,
            legs=scaled_legs,
            structure_type=trade.structure_type,
            target_dte=trade.target_dte,
            notes=(trade.notes or "") + " [rejected: risk too high]",
        )
        scaled_risk = estimate_trade_risk(scaled_trade, spot=spot)
        return scaled_trade, scaled_risk

    scaled_legs: List[OptionLeg] = []
    for leg in trade.legs:
        base_qty = max(leg.quantity, 1)
        scaled_qty = base_qty * contracts
        scaled_legs.append(
            OptionLeg(
                contract=leg.contract,
                side=leg.side,
                quantity=scaled_qty,
                role=leg.role,
            )
        )

    scaled_trade = ProposedTrade(
        underlying=trade.underlying,
        legs=scaled_legs,
        structure_type=trade.structure_type,
        target_dte=trade.target_dte,
        notes=(trade.notes or "") + f" [scaled_to={contracts}_contracts]",
    )

    scaled_risk = estimate_trade_risk(
        scaled_trade,
        spot=spot,
        price_range_pct=0.5,
        n_points=201,
    )

    return scaled_trade, scaled_risk


def build_risk_adjusted_trade_from_cone(
    symbol: str,
    spot: float,
    directive,
    cone,
    option_chain,
    base_dte: int,
    max_risk_dollars: float,
) -> Tuple[ProposedTrade, TradeRiskProfile] | Tuple[None, None]:
    """
    Full pipeline for ONE symbol:

    1) Distill cone → metrics.
    2) Classify TradeRegime.
    3) Select StructureSpec.
    4) Build raw ProposedTrade from chain.
    5) Scale to risk budget using risk engine.
    """

    from trade.cone_metrics import compute_cone_metrics
    from trade.regime_classifier import classify_trade_regime
    from trade.structure_selector import select_structure_from_cone
    from trade.execution_mapper import build_proposed_trade_from_structure

    metrics = compute_cone_metrics(spot=spot, cone=cone)
    regime = classify_trade_regime(metrics)
    spec = select_structure_from_cone(
        metrics=metrics,
        regime=regime,
        base_dte=base_dte,
    )

    raw_trade = build_proposed_trade_from_structure(
        spec=spec,
        chain=option_chain,
        spot=spot,
    )

    if raw_trade is None or not raw_trade.legs:
        return None, None

    scaled_trade, risk_profile = scale_trade_to_risk_budget(
        trade=raw_trade,
        spot=spot,
        max_risk_dollars=max_risk_dollars,
    )

    if all(leg.quantity == 0 for leg in scaled_trade.legs):
        return None, None

    return scaled_trade, risk_profile
