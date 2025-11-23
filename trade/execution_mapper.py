"""Utilities for mapping option structures to executable trades."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from engines.inputs.options_chain_adapter import OptionContract


@dataclass
class OptionLeg:
    """Single option leg within a multi-leg structure."""

    contract: OptionContract
    side: str  # "buy" or "sell"
    quantity: int
    role: Optional[str] = None  # e.g., "long", "short", "wing"


@dataclass
class ProposedTrade:
    """Executable multi-leg options trade."""

    underlying: str
    legs: Sequence[OptionLeg]
    structure_type: str
    target_dte: Optional[int] = None
    notes: Optional[str] = None


def build_proposed_trade_from_structure(*args, **kwargs) -> Optional[ProposedTrade]:
    """
    Placeholder for mapping a structure spec to a concrete ProposedTrade.

    This function should be implemented once structure selection is wired to
    a live options chain. It currently returns ``None`` to avoid breaking the
    pipeline until the full mapper is available.
    """

    return None
