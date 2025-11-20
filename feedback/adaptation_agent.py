"""Adaptation agent that adjusts risk parameters from live feedback."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Optional

from loguru import logger

from schemas.core_schemas import AdaptationUpdate, PipelineResult, TrackingSnapshot


class AdaptationAgent:
    """Lightweight adaptive controller for hyperparameter tuning."""

    def __init__(
        self,
        state_path: Path,
        min_trades: int = 5,
        performance_lookback: int = 20,
        min_risk_per_trade: float = 0.005,
        max_risk_per_trade: float = 0.05,
    ) -> None:
        self.state_path = Path(state_path)
        self.min_trades = min_trades
        self.performance_lookback = performance_lookback
        self.min_risk_per_trade = min_risk_per_trade
        self.max_risk_per_trade = max_risk_per_trade

        self.state: Dict[str, float] = {
            "trades_seen": 0,
            "risk_per_trade": 0.02,
        }
        self._load_state()

    def _load_state(self) -> None:
        if self.state_path.exists():
            try:
                self.state.update(json.loads(self.state_path.read_text()))
                logger.info(f"Loaded adaptation state from {self.state_path}")
            except Exception as error:  # pragma: no cover - defensive
                logger.warning(f"Could not load adaptation state: {error}")

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self.state))
        except Exception as error:  # pragma: no cover - defensive
            logger.warning(f"Could not persist adaptation state: {error}")

    def update_from_feedback(
        self,
        pipeline_result: PipelineResult,
        tracking_snapshot: Optional[TrackingSnapshot],
    ) -> Optional[AdaptationUpdate]:
        """Update adaptive parameters using pipeline results and tracking."""

        self.state["trades_seen"] += len(pipeline_result.order_results)

        if self.state["trades_seen"] < self.min_trades:
            return None

        pnl_samples = []
        if tracking_snapshot:
            pnl_samples = [p.unrealized_pnl_pct for p in tracking_snapshot.positions if p.symbol == pipeline_result.symbol]

        if len(pnl_samples) < 1:
            return None

        avg_pnl = mean(pnl_samples)

        current_risk = float(self.state.get("risk_per_trade", 0.02))
        adjustment = 0.001 if avg_pnl >= 0 else -0.001
        new_risk = min(self.max_risk_per_trade, max(self.min_risk_per_trade, current_risk + adjustment))

        if abs(new_risk - current_risk) < 1e-6:
            return None

        self.state["risk_per_trade"] = new_risk
        self.state["trades_seen"] = max(0, self.state["trades_seen"] - self.performance_lookback)
        self._save_state()

        rationale = (
            "Increased risk_per_trade for positive PnL"
            if adjustment > 0
            else "Reduced risk_per_trade after drawdown"
        )

        return AdaptationUpdate(
            timestamp=datetime.utcnow(),
            changes={"risk_per_trade": new_risk},
            rationale=rationale,
        )

