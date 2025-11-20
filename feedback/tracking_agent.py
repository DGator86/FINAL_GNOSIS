"""Position and outcome tracking agent."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from loguru import logger

from schemas.core_schemas import PositionState, TrackingSnapshot


class TrackingAgent:
    """Monitors broker positions and feeds snapshots into the pipeline."""

    def __init__(self, broker_adapter: Optional[object], enable: bool = True):
        self.broker = broker_adapter
        self.enabled = enable and broker_adapter is not None
        if self.enabled:
            logger.info("TrackingAgent enabled with broker connectivity")
        else:
            logger.info("TrackingAgent initialized in passive mode (no broker)")

    def snapshot_positions(self) -> TrackingSnapshot:
        """Collect a snapshot of current broker positions."""

        timestamp = datetime.utcnow()
        notes = []

        if not self.enabled:
            notes.append("Tracking disabled or broker unavailable")
            return TrackingSnapshot(timestamp=timestamp, positions=[], notes=notes)

        positions = []
        try:
            broker_positions = self.broker.get_positions()
            for pos in broker_positions:
                positions.append(
                    PositionState(
                        symbol=pos.symbol,
                        quantity=pos.quantity,
                        avg_entry_price=pos.avg_entry_price,
                        current_price=pos.current_price,
                        market_value=pos.market_value,
                        unrealized_pnl=pos.unrealized_pnl,
                        unrealized_pnl_pct=pos.unrealized_pnl_pct,
                        side=pos.side,
                    )
                )
        except Exception as error:  # pragma: no cover - defensive
            notes.append(f"Failed to read positions: {error}")
            logger.error(f"TrackingAgent error collecting positions: {error}")

        return TrackingSnapshot(timestamp=timestamp, positions=positions, notes=notes)

