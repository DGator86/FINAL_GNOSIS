from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

StructureType = Literal[
    "long_call",
    "long_put",
    "call_debit_spread",
    "put_debit_spread",
    "iron_condor",
    "iron_butterfly",
    "long_strangle",
]


@dataclass
class StructureSpec:
    """Specification of an options structure selected from cone metrics."""

    structure_type: StructureType
    size: int
    target_dte: int
    strike_offset_pct: Optional[float] = None
    wing_width_pct: Optional[float] = None
