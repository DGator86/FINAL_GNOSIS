"""
Unified Execution Orchestrator
Coordinates execution across stocks, single options, and multi-leg strategies

This is the "conductor" that executes the strategy selector's decisions.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from engines.liquidity.options_execution_v2 import OptionsExecutionModule
from engines.orchestration.strategy_selector import (
    InstrumentDecision,
    InstrumentType,
    IntelligentStrategySelector,
    StrategyType,
)
from models.options_contracts import EnhancedMarketData


@dataclass
class ExecutionResult:
    """Result of trade execution"""

    success: bool
    instrument_type: str
    strategy_type: str
    order_id: Optional[str]
    details: Dict[str, Any]
    error_message: Optional[str] = None


class UnifiedOrchestrator:
    """
    Coordinates execution across all instrument types:
    - Stocks (long/short)
    - Single options (calls/puts)
    - Multi-leg spreads (bull/bear spreads, straddles, condors)
    """

    def __init__(self, config: Dict, logger=None):
        self.config = config
        self.logger = logger

        # Initialize components
        self.strategy_selector = IntelligentStrategySelector(
            config=config.get("orchestrator", {}), logger=logger
        )

        self.options_execution = OptionsExecutionModule(config=config, logger=logger)

    def execute_optimal_trade(
        self,
        ticker: str,
        market_data: EnhancedMarketData,
        signal_direction: str,
        signal_confidence: float,
        regime: str,
        alpaca_client: Any,
        quantity: int = 1,
        portfolio_state: Optional[Dict] = None,
    ) -> ExecutionResult:
        """
        Main orchestration function:
        1. Get strategy decision from selector
        2. Execute chosen instrument
        3. Return result

        Args:
            ticker: Stock ticker
            market_data: Enhanced market data with options
            signal_direction: "bullish", "bearish", or "neutral"
            signal_confidence: 0.0 to 1.0
            regime: Market regime string
            alpaca_client: AlpacaClient instance
            quantity: Base quantity
            portfolio_state: Current portfolio state

        Returns:
            ExecutionResult with order details
        """

        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info(f"Orchestrating Trade for {ticker}")
            self.logger.info("=" * 60)

        try:
            # Step 1: Get strategy decision
            decision = self.strategy_selector.select_optimal_instrument(
                market_data=market_data,
                signal_direction=signal_direction,
                signal_confidence=signal_confidence,
                regime=regime,
                portfolio_state=portfolio_state,
            )

            # Step 2: Check if we should skip
            if decision.instrument_type == InstrumentType.SKIP:
                if self.logger:
                    self.logger.info(f"Decision: SKIP - {decision.reasoning}")

                return ExecutionResult(
                    success=True,
                    instrument_type="skip",
                    strategy_type="none",
                    order_id=None,
                    details={"reasoning": decision.reasoning},
                )

            # Step 3: Execute based on instrument type
            if decision.instrument_type == InstrumentType.STOCK:
                result = self._execute_stock_trade(
                    ticker=ticker, decision=decision, alpaca_client=alpaca_client, quantity=quantity
                )

            elif decision.instrument_type == InstrumentType.OPTION_SINGLE:
                result = self._execute_single_option(
                    ticker=ticker,
                    decision=decision,
                    market_data=market_data,
                    alpaca_client=alpaca_client,
                    quantity=quantity,
                )

            elif decision.instrument_type == InstrumentType.OPTION_SPREAD:
                result = self._execute_multi_leg_spread(
                    ticker=ticker,
                    decision=decision,
                    market_data=market_data,
                    alpaca_client=alpaca_client,
                    quantity=quantity,
                )

            else:
                raise ValueError(f"Unknown instrument type: {decision.instrument_type}")

            if self.logger:
                if result.success:
                    self.logger.success(f"✓ Trade executed: {result.strategy_type}")
                    self.logger.info(f"  Order ID: {result.order_id}")
                else:
                    self.logger.error(f"✗ Trade failed: {result.error_message}")

            return result

        except Exception as e:
            if self.logger:
                self.logger.error(f"Orchestration error: {e}")

            return ExecutionResult(
                success=False,
                instrument_type="unknown",
                strategy_type="unknown",
                order_id=None,
                details={},
                error_message=str(e),
            )

    def _execute_stock_trade(
        self, ticker: str, decision: InstrumentDecision, alpaca_client: Any, quantity: int
    ) -> ExecutionResult:
        """Execute stock trade (long or short)"""

        side = "buy" if decision.strategy_type == StrategyType.LONG_STOCK else "sell"

        if self.logger:
            self.logger.info(f"Executing STOCK trade: {side.upper()} {quantity} shares of {ticker}")

        try:
            # Use existing stock trading logic
            # This would integrate with your existing stock execution
            order = alpaca_client.api.submit_order(
                symbol=ticker, qty=quantity, side=side, type="market", time_in_force="day"
            )

            return ExecutionResult(
                success=True,
                instrument_type="stock",
                strategy_type=decision.strategy_type.value,
                order_id=str(order.id),
                details={
                    "ticker": ticker,
                    "side": side,
                    "quantity": quantity,
                    "reasoning": decision.reasoning,
                },
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                instrument_type="stock",
                strategy_type=decision.strategy_type.value,
                order_id=None,
                details={},
                error_message=f"Stock execution failed: {e}",
            )

    def _execute_single_option(
        self,
        ticker: str,
        decision: InstrumentDecision,
        market_data: EnhancedMarketData,
        alpaca_client: Any,
        quantity: int,
    ) -> ExecutionResult:
        """Execute single option trade"""

        # Determine option type and side
        if decision.strategy_type == StrategyType.LONG_CALL:
            option_type = "call"
            side = "buy"
        elif decision.strategy_type == StrategyType.LONG_PUT:
            option_type = "put"
            side = "buy"
        elif decision.strategy_type == StrategyType.SHORT_CALL:
            option_type = "call"
            side = "sell"
        elif decision.strategy_type == StrategyType.SHORT_PUT:
            option_type = "put"
            side = "sell"
        else:
            return ExecutionResult(
                success=False,
                instrument_type="option_single",
                strategy_type=decision.strategy_type.value,
                order_id=None,
                details={},
                error_message=f"Unsupported single option strategy: {decision.strategy_type}",
            )

        # Find ATM option
        option_symbol = self._find_atm_option(market_data, option_type)

        if not option_symbol:
            return ExecutionResult(
                success=False,
                instrument_type="option_single",
                strategy_type=decision.strategy_type.value,
                order_id=None,
                details={},
                error_message="No suitable option contract found",
            )

        if self.logger:
            self.logger.info(f"Executing SINGLE OPTION: {side.upper()} {option_symbol}")

        try:
            # Execute via options execution module
            legs = [{"symbol": option_symbol, "side": side, "ratio_qty": 1}]

            result = self.options_execution.execute_order(
                strategy_type="single_leg",
                legs=legs,
                alpaca_client=alpaca_client,
                quantity=quantity,
            )

            return ExecutionResult(
                success=result["success"],
                instrument_type="option_single",
                strategy_type=decision.strategy_type.value,
                order_id=result.get("order_id"),
                details={"symbol": option_symbol, "side": side, "reasoning": decision.reasoning},
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                instrument_type="option_single",
                strategy_type=decision.strategy_type.value,
                order_id=None,
                details={},
                error_message=f"Single option execution failed: {e}",
            )

    def _execute_multi_leg_spread(
        self,
        ticker: str,
        decision: InstrumentDecision,
        market_data: EnhancedMarketData,
        alpaca_client: Any,
        quantity: int,
    ) -> ExecutionResult:
        """Execute multi-leg spread strategy"""

        # Build legs based on strategy type
        legs = self._build_spread_legs(decision.strategy_type, market_data)

        if not legs:
            return ExecutionResult(
                success=False,
                instrument_type="option_spread",
                strategy_type=decision.strategy_type.value,
                order_id=None,
                details={},
                error_message="Could not construct spread legs",
            )

        if self.logger:
            self.logger.info(f"Executing MULTI-LEG SPREAD: {decision.strategy_type.value}")
            for i, leg in enumerate(legs, 1):
                self.logger.info(f"  Leg {i}: {leg['side'].upper()} {leg['symbol']}")

        try:
            # Execute via options execution module
            result = self.options_execution.execute_order(
                strategy_type="multi_leg", legs=legs, alpaca_client=alpaca_client, quantity=quantity
            )

            return ExecutionResult(
                success=result["success"],
                instrument_type="option_spread",
                strategy_type=decision.strategy_type.value,
                order_id=result.get("order_id"),
                details={"legs": legs, "reasoning": decision.reasoning},
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                instrument_type="option_spread",
                strategy_type=decision.strategy_type.value,
                order_id=None,
                details={},
                error_message=f"Multi-leg execution failed: {e}",
            )

    def _find_atm_option(self, market_data: EnhancedMarketData, option_type: str) -> Optional[str]:
        """Find at-the-money option contract"""
        if not market_data.options_chain or not market_data.options_chain.quotes:
            return None

        current_price = market_data.current_price

        # Filter by option type
        type_filter = "C" if option_type == "call" else "P"
        candidates = [q for q in market_data.options_chain.quotes if type_filter in q.symbol]

        if not candidates:
            return None

        # Find closest to ATM
        atm_option = min(candidates, key=lambda q: abs(q.strike - current_price))
        return atm_option.symbol

    def _build_spread_legs(
        self, strategy_type: StrategyType, market_data: EnhancedMarketData
    ) -> Optional[list]:
        """Build legs for spread strategies"""
        current_price = market_data.current_price

        if strategy_type == StrategyType.BULL_CALL_SPREAD:
            # Buy ATM call, sell OTM call
            atm_call = self._find_atm_option(market_data, "call")
            otm_call = self._find_option_by_strike(market_data, "call", current_price * 1.05)

            if atm_call and otm_call:
                return [
                    {"symbol": atm_call, "side": "buy", "ratio_qty": 1},
                    {"symbol": otm_call, "side": "sell", "ratio_qty": 1},
                ]

        elif strategy_type == StrategyType.BEAR_PUT_SPREAD:
            # Buy ATM put, sell OTM put
            atm_put = self._find_atm_option(market_data, "put")
            otm_put = self._find_option_by_strike(market_data, "put", current_price * 0.95)

            if atm_put and otm_put:
                return [
                    {"symbol": atm_put, "side": "buy", "ratio_qty": 1},
                    {"symbol": otm_put, "side": "sell", "ratio_qty": 1},
                ]

        elif strategy_type == StrategyType.LONG_STRADDLE:
            # Buy ATM call and put
            atm_call = self._find_atm_option(market_data, "call")
            atm_put = self._find_atm_option(market_data, "put")

            if atm_call and atm_put:
                return [
                    {"symbol": atm_call, "side": "buy", "ratio_qty": 1},
                    {"symbol": atm_put, "side": "buy", "ratio_qty": 1},
                ]

        elif strategy_type == StrategyType.IRON_CONDOR:
            # Sell OTM call spread + sell OTM put spread
            otm_call_sell = self._find_option_by_strike(market_data, "call", current_price * 1.05)
            otm_call_buy = self._find_option_by_strike(market_data, "call", current_price * 1.10)
            otm_put_sell = self._find_option_by_strike(market_data, "put", current_price * 0.95)
            otm_put_buy = self._find_option_by_strike(market_data, "put", current_price * 0.90)

            if all([otm_call_sell, otm_call_buy, otm_put_sell, otm_put_buy]):
                return [
                    {"symbol": otm_call_sell, "side": "sell", "ratio_qty": 1},
                    {"symbol": otm_call_buy, "side": "buy", "ratio_qty": 1},
                    {"symbol": otm_put_sell, "side": "sell", "ratio_qty": 1},
                    {"symbol": otm_put_buy, "side": "buy", "ratio_qty": 1},
                ]

        return None

    def _find_option_by_strike(
        self, market_data: EnhancedMarketData, option_type: str, target_strike: float
    ) -> Optional[str]:
        """Find option contract closest to target strike"""
        if not market_data.options_chain or not market_data.options_chain.quotes:
            return None

        type_filter = "C" if option_type == "call" else "P"
        candidates = [q for q in market_data.options_chain.quotes if type_filter in q.symbol]

        if not candidates:
            return None

        closest = min(candidates, key=lambda q: abs(q.strike - target_strike))
        return closest.symbol
