"""Universe-aware TradeAgent with explicit symbol handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from config import load_config
from schemas.core_schemas import DirectionEnum, PipelineResult, TradeIdea


@dataclass
class ProposedTrade:
    """Lightweight representation of a trade ready for broker submission."""

    symbol: str
    qty: int
    side: str
    order_type: str = "market"
    time_in_force: str = "day"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None


class TradeAgent:
    """Generate broker-ready trades for a specific symbol."""

    def __init__(self, config) -> None:
        self.config = config
        self.max_position_size = self.config.agents.trade.max_position_size
        self.risk_per_trade = self.config.agents.trade.risk_per_trade
        self.position_size_pct = self.config.trading.position_size_pct
        self.max_concurrent_positions = self.config.trading.max_concurrent_positions

    @classmethod
    def from_config(cls, config_path: str) -> "TradeAgent":
        """Initialize from a YAML config path (falls back to defaults)."""
        config = load_config(config_path)
        return cls(config)

    def generate_trades(
        self,
        symbol: str,
        directive: PipelineResult,
        diagnostics: Optional[Dict] = None,
    ) -> List[ProposedTrade]:
        """
        Generate trades for a single symbol based on pipeline results.

        Enforces per-symbol sizing and ensures symbol attribution on trades.
        """
        trades: List[ProposedTrade] = []

        trade_ideas = getattr(directive, "trade_ideas", []) or []
        if not trade_ideas:
            logger.info(f"{symbol}: no trade ideas available from pipeline")
            return trades

        for idea in trade_ideas:
            if not isinstance(idea, TradeIdea):
                continue

            if idea.symbol != symbol:
                logger.debug(f"Skipping idea for {idea.symbol} while processing {symbol}")
                continue

            if idea.direction == DirectionEnum.NEUTRAL:
                logger.debug(f"Skipping neutral idea for {symbol}")
                continue

            side = "buy" if idea.direction == DirectionEnum.LONG else "sell"
            qty = self._calculate_quantity(idea)
            if qty <= 0:
                logger.debug(f"Skipping {symbol}: calculated quantity {qty} is not positive")
                continue

            trades.append(
                ProposedTrade(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    order_type="market",
                    time_in_force="day",
                )
            )

        return trades

    def _calculate_quantity(self, idea: TradeIdea) -> int:
        """Calculate position size using configured risk parameters."""
        dollar_cap = self.max_position_size
        baseline_size = max(self.position_size_pct * dollar_cap, 1)
        desired = idea.size or (self.risk_per_trade * dollar_cap)
        allowed_notional = min(dollar_cap, max(desired, baseline_size))

        price = idea.entry_price or 100.0
        qty = int(allowed_notional / max(price, 1e-6))
        return max(1, qty)
