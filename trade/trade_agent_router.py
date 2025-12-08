"""Trade Agent Router - Routes to appropriate agent."""

from __future__ import annotations

from typing import Optional, Union, Dict, Any, List
from datetime import datetime
from loguru import logger

from schemas.core_schemas import OptionsOrderRequest
from agents.composer.composer_agent_v2 import ComposerDecision
from trade.options_trade_agent import OptionsTradeAgent
from trade.trade_agent_v3 import TradeAgentV3, TradeStrategy


class TradeAgentRouter:
    """
    Routes trade generation to appropriate agent based on config.

    If config.enable_options is True:
        → Use OptionsTradeAgent (Advanced Strategies)
    Else:
        → Use TradeAgentV3 (Stock and Basic Spreads)
    """

    def __init__(self, config: Dict, options_adapter=None):
        """Initialize Router."""
        self.config = config
        self.use_options = config.get("enable_options", False) and options_adapter is not None
        self.options_adapter = options_adapter

        self.options_agent = OptionsTradeAgent(config)
        # Initialize V3 with options_adapter=None to force it to stick to equity if we want strict separation,
        # OR pass it so it can do its own simple options if OptionsTradeAgent fails.
        # For now, let's pass it.
        self.stock_agent = TradeAgentV3(
            config.get("agents", {}).get("trade_v3", {}), options_adapter
        )

        if config.get("enable_options", False) and options_adapter is None:
            logger.warning("TradeAgentRouter requested options but no adapter available; disabling options path")

        logger.info(f"TradeAgentRouter initialized (Options Enabled: {self.use_options})")

    def generate_strategy(
        self,
        composer_decision: ComposerDecision,
        current_price: float,
        available_capital: float,
        timestamp: datetime,
        **kwargs,
    ) -> Optional[Union[TradeStrategy, OptionsOrderRequest]]:
        """
        Generate trade strategy.
        Returns either TradeStrategy (V3) or OptionsOrderRequest (OptionsAgent).
        """

        # 1. Try Options Agent if enabled
        if self.use_options:
            try:
                # Map inputs
                signal = "BUY" if composer_decision.predicted_direction == "LONG" else "SELL"
                if composer_decision.predicted_direction == "NEUTRAL":
                    signal = "HOLD"

                options_order = self.options_agent.select_strategy(
                    symbol=composer_decision.symbol,
                    hedge_snapshot=kwargs.get("hedge_snapshot"),
                    composer_signal=signal,
                    composer_confidence=composer_decision.confidence,
                    current_price=current_price,
                    iv_rank=kwargs.get("iv_rank"),
                    iv_percentile=kwargs.get("iv_percentile"),
                )

                if options_order:
                    logger.info(f"Router: Selected Options Strategy: {options_order.strategy_name}")
                    return options_order
            except Exception as e:
                logger.error(f"Router: Options generation failed: {e}")

        # 2. Fallback to Stock Agent
        logger.info("Router: Delegating to Stock Agent (V3)")
        return self.stock_agent.generate_strategy(
            composer_decision, current_price, available_capital, timestamp
        )

    def validate_strategy(
        self,
        strategy: Union[TradeStrategy, OptionsOrderRequest],
        current_positions: List[str],
        total_portfolio_value: float,
    ) -> bool:
        """Validate strategy."""
        if isinstance(strategy, OptionsOrderRequest):
            # Basic validation for OptionsOrderRequest
            if strategy.symbol in current_positions:
                logger.warning(f"Already in position for {strategy.symbol}")
                return False
            if strategy.max_loss > total_portfolio_value * 0.05:  # Max 5% risk
                logger.warning(f"Options risk too high: {strategy.max_loss}")
                return False
            return True
        else:
            return self.stock_agent.validate_strategy(
                strategy, current_positions, total_portfolio_value
            )


def create_trade_agent(config: Dict = None, options_adapter=None) -> TradeAgentRouter:
    """Factory function."""
    return TradeAgentRouter(config or {}, options_adapter)
