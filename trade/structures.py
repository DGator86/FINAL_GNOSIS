from __future__ import annotations

from enum import Enum


class StructureType(str, Enum):
    """Enumerates supported option structure archetypes."""

    CALL_SPREAD = "call_spread"
    PUT_SPREAD = "put_spread"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    CALENDAR = "calendar"
    CUSTOM = "custom"
