"""Trade Agent v1 - Trade idea generation with MTF, PPF, and time-to-profit analysis."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from schemas.core_schemas import (
    DirectionEnum,
    OrderResult,
    OrderStatus,
    PipelineResult,
    StrategyType,
    TradeIdea,
    TimeToProfitEstimate,
)
from trade.expected_move_calculator import ExpectedMoveCalculator


class TradeAgentV1:
    """
    Trade Agent v1 for generating trade ideas from consensus.

    Uses:
    - MTF alignment and strategy recommendations from Composer
    - PPF analysis (Past/Present/Future) from aggregated agent data
    - LSTM projections for time-to-profit estimation
    - Trailing stop logic based on projected moves
    """

    def __init__(
        self,
        options_adapter: Any,
        market_adapter: Any,
        config: Dict[str, Any],
        broker: Optional[Any] = None,
    ):
        self.options_adapter = options_adapter
        self.market_adapter = market_adapter
        self.config = config
        self.broker = broker

        # Initialize expected move calculator
        self.expected_move_calculator = ExpectedMoveCalculator(
            options_adapter=options_adapter,
            market_adapter=market_adapter,
            config=config,
        )

        mode = "enabled" if self.broker else "disabled"
        logger.info(f"TradeAgentV1 initialized (execution {mode})")

    def generate_ideas(
        self,
        pipeline_result: PipelineResult,
        timestamp: datetime
    ) -> List[TradeIdea]:
        """
        Generate trade ideas from pipeline results with MTF and time-to-profit analysis.

        Args:
            pipeline_result: Complete pipeline result (with MTF, PPF in consensus)
            timestamp: Generation timestamp

        Returns:
            List of trade ideas ranked by profit confidence and ROI
        """
        if not pipeline_result.consensus:
            return []

        consensus = pipeline_result.consensus
        direction_str = consensus.get("direction", "neutral")
        confidence = consensus.get("confidence", 0.0)

        # Convert string to enum
        direction = DirectionEnum(direction_str) if direction_str else DirectionEnum.NEUTRAL

        # Check minimum threshold
        min_confidence = self.config.get("min_trade_confidence", 0.1)
        if direction == DirectionEnum.NEUTRAL or confidence < min_confidence:
            return []

        # Extract MTF data from consensus
        mtf_alignment = consensus.get("mtf_alignment", 0.0)
        mtf_direction = consensus.get("mtf_direction", "neutral")
        strongest_timeframe = consensus.get("strongest_timeframe", "")
        suggested_strategy = consensus.get("suggested_strategy", "")
        suggested_expiry = consensus.get("suggested_expiry", "")
        mtf_agrees = consensus.get("mtf_agrees_with_consensus", False)

        # Extract aggregated PPF from consensus
        aggregated_ppf = consensus.get("aggregated_ppf", {})

        # Get current price
        spot_price = self._get_current_price(pipeline_result.symbol)

        # Calculate expected move
        expected_move = self.expected_move_calculator.calculate(
            symbol=pipeline_result.symbol,
            spot_price=spot_price,
            pipeline_result=pipeline_result,
            dte=1,
        )

        # Calculate time-to-profit from LSTM projections
        time_to_profit = self._calculate_time_to_profit(
            aggregated_ppf,
            direction,
            spot_price,
            expected_move,
        )

        # Select strategy - use MTF recommendation or fall back to engine-based selection
        if suggested_strategy:
            strategy_type = self._map_options_strategy_to_type(suggested_strategy)
        else:
            strategy_type = self._select_strategy(pipeline_result)

        # Calculate position size with MTF boost/penalty
        max_size = self.config.get("max_position_size", 10000.0)
        risk_per_trade = self.config.get("risk_per_trade", 0.02)

        # MTF alignment affects position sizing
        mtf_size_factor = 1.0
        if mtf_alignment > 0.7 and mtf_agrees:
            mtf_size_factor = 1.3  # Increase size when all timeframes agree
        elif mtf_alignment < 0.3:
            mtf_size_factor = 0.6  # Reduce size when timeframes conflict

        size = max_size * risk_per_trade * confidence * mtf_size_factor

        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_stops(
            direction,
            spot_price,
            expected_move,
            time_to_profit,
        )

        # Calculate expected ROI
        expected_roi = 0.0
        if take_profit and spot_price > 0:
            if direction == DirectionEnum.LONG:
                expected_roi = (take_profit - spot_price) / spot_price * 100
            else:
                expected_roi = (spot_price - take_profit) / spot_price * 100

        # Calculate profit confidence based on MTF alignment and LSTM
        profit_confidence = self._calculate_profit_confidence(
            confidence,
            mtf_alignment,
            mtf_agrees,
            time_to_profit,
        )

        # Build reasoning with full context
        reasoning = self._build_reasoning(
            strategy_type,
            confidence,
            mtf_alignment,
            mtf_direction,
            strongest_timeframe,
            suggested_strategy,
            time_to_profit,
        )

        trade_idea = TradeIdea(
            timestamp=timestamp,
            symbol=pipeline_result.symbol,
            strategy_type=strategy_type,
            direction=direction,
            confidence=confidence,
            size=size,
            entry_price=spot_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            expected_move=expected_move,

            # NEW: Time-to-profit analysis
            time_to_profit=time_to_profit,

            # NEW: Options strategy from MTF
            options_strategy=suggested_strategy,
            options_strategy_details=suggested_expiry,
            options_expiry_suggestion=self._get_expiry_from_timeframe(strongest_timeframe),

            # NEW: MTF source
            source_timeframe=strongest_timeframe,
            mtf_alignment=mtf_alignment,

            # NEW: Profit projections
            expected_roi_pct=expected_roi,
            profit_confidence=profit_confidence,
        )

        return [trade_idea]

    def _calculate_time_to_profit(
        self,
        aggregated_ppf: Dict[str, Any],
        direction: DirectionEnum,
        spot_price: float,
        expected_move: Any,
    ) -> Optional[TimeToProfitEstimate]:
        """
        Calculate estimated time to reach profit target using LSTM projections.

        Uses future projections from PPF to estimate when price will reach target.
        """
        if not aggregated_ppf:
            return None

        future = aggregated_ppf.get("future", {})
        if not future:
            return None

        # Get LSTM projections
        move_1m = future.get("projected_move_1m", 0.0)
        move_5m = future.get("projected_move_5m", 0.0)
        move_15m = future.get("projected_move_15m", 0.0)
        move_60m = future.get("projected_move_60m", 0.0)
        move_confidence = future.get("move_confidence", 0.0)

        # Determine projected direction
        avg_move = (move_1m + move_5m + move_15m + move_60m) / 4
        projected_direction = "up" if avg_move > 0 else "down" if avg_move < 0 else "neutral"

        # Check if projection agrees with trade direction
        direction_agrees = (
            (projected_direction == "up" and direction == DirectionEnum.LONG) or
            (projected_direction == "down" and direction == DirectionEnum.SHORT)
        )

        if not direction_agrees:
            # Projection disagrees - lower confidence, longer time
            return TimeToProfitEstimate(
                estimated_minutes=120,  # 2 hours pessimistic
                confidence=0.2,
                based_on="lstm_disagreement",
                initial_stop_pct=1.0,
                trailing_stop_pct=0.5,
                max_loss_pct=2.0,
                max_loss_prob=0.4,
            )

        # Estimate time based on move magnitudes
        # Find first horizon where expected move exceeds 1% (typical target)
        target_pct = 1.0
        estimated_minutes = 60  # Default

        if abs(move_1m) >= target_pct:
            estimated_minutes = 1
        elif abs(move_5m) >= target_pct:
            estimated_minutes = 5
        elif abs(move_15m) >= target_pct:
            estimated_minutes = 15
        elif abs(move_60m) >= target_pct:
            estimated_minutes = 60
        else:
            # Extrapolate based on 60m move
            if abs(move_60m) > 0:
                estimated_minutes = int(60 * target_pct / abs(move_60m))
                estimated_minutes = min(estimated_minutes, 240)  # Cap at 4 hours

        # Calculate stop percentages based on volatility
        volatility_factor = max(0.5, min(2.0, abs(move_60m) * 2))
        initial_stop = 0.5 * volatility_factor
        trailing_stop = 0.3 * volatility_factor

        # Profit targets
        target_1_pct = 0.5  # First target: 0.5%
        target_2_pct = 1.0  # Second target: 1.0%

        # Estimate probabilities based on move confidence
        target_1_prob = min(0.8, move_confidence + 0.2)
        target_2_prob = min(0.6, move_confidence)

        return TimeToProfitEstimate(
            estimated_minutes=estimated_minutes,
            confidence=move_confidence,
            based_on="lstm",
            initial_stop_pct=initial_stop,
            trailing_stop_pct=trailing_stop,
            breakeven_minutes=estimated_minutes // 2 if estimated_minutes > 2 else 1,
            target_1_pct=target_1_pct,
            target_1_prob=target_1_prob,
            target_1_minutes=estimated_minutes // 2,
            target_2_pct=target_2_pct,
            target_2_prob=target_2_prob,
            target_2_minutes=estimated_minutes,
            max_loss_pct=initial_stop * 2,
            max_loss_prob=1.0 - target_1_prob,
        )

    def _calculate_stops(
        self,
        direction: DirectionEnum,
        spot_price: float,
        expected_move: Any,
        time_to_profit: Optional[TimeToProfitEstimate],
    ) -> tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels."""
        stop_loss = None
        take_profit = None

        # Use time-to-profit stops if available
        if time_to_profit and spot_price > 0:
            if direction == DirectionEnum.LONG:
                stop_loss = spot_price * (1 - time_to_profit.initial_stop_pct / 100)
                take_profit = spot_price * (1 + time_to_profit.target_2_pct / 100)
            elif direction == DirectionEnum.SHORT:
                stop_loss = spot_price * (1 + time_to_profit.initial_stop_pct / 100)
                take_profit = spot_price * (1 - time_to_profit.target_2_pct / 100)

        # Fall back to expected move if no time-to-profit
        elif expected_move and expected_move.one_sigma:
            if direction == DirectionEnum.LONG:
                stop_loss = expected_move.one_sigma.lower
                take_profit = expected_move.two_sigma.upper if expected_move.two_sigma else expected_move.one_sigma.upper
            elif direction == DirectionEnum.SHORT:
                stop_loss = expected_move.one_sigma.upper
                take_profit = expected_move.two_sigma.lower if expected_move.two_sigma else expected_move.one_sigma.lower

        return stop_loss, take_profit

    def _calculate_profit_confidence(
        self,
        base_confidence: float,
        mtf_alignment: float,
        mtf_agrees: bool,
        time_to_profit: Optional[TimeToProfitEstimate],
    ) -> float:
        """Calculate overall profit confidence score."""
        profit_conf = base_confidence

        # Boost for MTF alignment
        if mtf_alignment > 0.7:
            profit_conf *= 1.2
        elif mtf_alignment < 0.3:
            profit_conf *= 0.7

        # Boost for MTF agreement
        if mtf_agrees:
            profit_conf *= 1.1

        # Factor in LSTM confidence
        if time_to_profit:
            profit_conf = (profit_conf + time_to_profit.confidence) / 2

        return min(1.0, profit_conf)

    def _build_reasoning(
        self,
        strategy_type: StrategyType,
        confidence: float,
        mtf_alignment: float,
        mtf_direction: str,
        strongest_timeframe: str,
        suggested_strategy: str,
        time_to_profit: Optional[TimeToProfitEstimate],
    ) -> str:
        """Build comprehensive reasoning string."""
        parts = [f"{strategy_type.value} strategy (conf: {confidence:.0%})"]

        if mtf_alignment > 0:
            parts.append(f"MTF {mtf_alignment:.0%} aligned ({mtf_direction})")

        if strongest_timeframe:
            parts.append(f"strongest: {strongest_timeframe}")

        if suggested_strategy:
            parts.append(f"options: {suggested_strategy}")

        if time_to_profit:
            parts.append(f"est. {time_to_profit.estimated_minutes}min to profit")

        return " | ".join(parts)

    def _map_options_strategy_to_type(self, options_strategy: str) -> StrategyType:
        """Map options strategy name to StrategyType enum."""
        strategy_lower = options_strategy.lower()

        if "call" in strategy_lower and "spread" not in strategy_lower:
            return StrategyType.DIRECTIONAL
        elif "put" in strategy_lower and "spread" not in strategy_lower:
            return StrategyType.DIRECTIONAL
        elif "spread" in strategy_lower:
            return StrategyType.OPTIONS_SPREAD
        elif "condor" in strategy_lower or "butterfly" in strategy_lower:
            return StrategyType.MEAN_REVERSION
        elif "straddle" in strategy_lower or "strangle" in strategy_lower:
            return StrategyType.BREAKOUT
        else:
            return StrategyType.DIRECTIONAL

    def _get_expiry_from_timeframe(self, timeframe: str) -> str:
        """Map timeframe to suggested expiry."""
        expiry_map = {
            "1Min": "0DTE",
            "5Min": "0DTE",
            "15Min": "0DTE/1DTE",
            "30Min": "1-3 DTE",
            "1Hour": "3-5 DTE",
            "4Hour": "1-2 weeks",
            "1Day": "2-4 weeks",
        }
        return expiry_map.get(timeframe, "1-2 weeks")

    def execute_trades(
        self, trade_ideas: List[TradeIdea], timestamp: datetime
    ) -> List[OrderResult]:
        """Place orders for generated trade ideas using the broker adapter."""

        if not self.broker:
            logger.info("No broker configured - execution skipped")
            return []

        order_results: List[OrderResult] = []

        for idea in trade_ideas:
            try:
                side = "buy" if idea.direction == DirectionEnum.LONG else "sell"

                quote = self.broker.get_latest_quote(idea.symbol)
                price = None
                if quote:
                    bid = quote.get("bid") or 0
                    ask = quote.get("ask") or 0
                    if bid and ask:
                        price = (bid + ask) / 2
                if price is None:
                    price = self._fallback_price(idea.symbol)

                dollars = idea.size or self.config.get("max_position_size", 10_000.0)
                quantity = max(1, round(dollars / max(price, 1e-6), 2))

                order_id = self.broker.place_order(
                    idea.symbol,
                    quantity=quantity,
                    side=side,
                )

                status = OrderStatus.SUBMITTED if order_id else OrderStatus.REJECTED
                order_results.append(
                    OrderResult(
                        timestamp=timestamp,
                        symbol=idea.symbol,
                        status=status,
                        order_id=order_id,
                        filled_qty=0.0,
                        message=f"{side} {quantity} at ~{price:.2f}" if price else "market order submitted",
                    )
                )
            except Exception as error:
                logger.error(f"Failed to execute trade for {idea.symbol}: {error}")
                order_results.append(
                    OrderResult(
                        timestamp=timestamp,
                        symbol=idea.symbol,
                        status=OrderStatus.REJECTED,
                        message=str(error),
                    )
                )

        return order_results

    def update_risk_per_trade(self, risk_per_trade: float) -> None:
        """Update risk_per_trade in-place when adaptation is active."""
        self.config["risk_per_trade"] = risk_per_trade
        logger.info(f"TradeAgent risk_per_trade updated to {risk_per_trade:.3f}")

    def _get_current_price(self, symbol: str) -> float:
        """Get current price from broker quote or fallback sources."""
        if self.broker:
            try:
                quote = self.broker.get_latest_quote(symbol)
                if quote:
                    bid = quote.get("bid") or 0
                    ask = quote.get("ask") or 0
                    if bid and ask:
                        return (bid + ask) / 2
            except Exception:
                pass

        return self._fallback_price(symbol)

    def _fallback_price(self, symbol: str) -> float:
        """Use market adapter to obtain a last known price as sizing fallback."""
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=2)
            bars = self.market_adapter.get_bars(symbol, start=start, end=end, timeframe="1Day")
            if bars:
                return float(bars[-1].close)
        except Exception:
            logger.debug("Fallback price lookup failed", exc_info=True)

        fallback_prices = {
            "SPY": 600.0,
            "QQQ": 500.0,
            "IWM": 230.0,
            "NVDA": 145.0,
            "TSLA": 350.0,
            "AAPL": 230.0,
            "MSFT": 430.0,
            "GOOGL": 175.0,
            "AMZN": 210.0,
            "META": 560.0,
        }
        return fallback_prices.get(symbol, 100.0)

    def _select_strategy(self, pipeline_result: PipelineResult) -> StrategyType:
        """Select appropriate strategy based on pipeline results."""
        if pipeline_result.elasticity_snapshot:
            volatility = pipeline_result.elasticity_snapshot.volatility
            if volatility > 0.3:
                return StrategyType.BREAKOUT
            elif volatility < 0.15:
                return StrategyType.MEAN_REVERSION

        return StrategyType.DIRECTIONAL
