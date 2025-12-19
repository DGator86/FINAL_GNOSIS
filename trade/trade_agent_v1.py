"""Trade Agent v1 - Trade idea generation with MTF, PPF, time-to-profit, and options selection."""

from __future__ import annotations

from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from schemas.core_schemas import (
    DirectionEnum,
    OrderResult,
    OrderStatus,
    PipelineResult,
    StrategyType,
    TradeIdea,
    TimeToProfitEstimate,
    OptionsOrderRequest,
    OptionsLeg,
)
from engines.inputs.options_chain_adapter import OptionContract
from trade.expected_move_calculator import ExpectedMoveCalculator


class TradeAgentV1:
    """
    Trade Agent v1 for generating trade ideas from consensus.

    Uses:
    - MTF alignment and strategy recommendations from Composer
    - PPF analysis (Past/Present/Future) from aggregated agent data
    - LSTM projections for time-to-profit estimation
    - Trailing stop logic based on projected moves
    - Specific options contract selection with profit targets
    """

    # DTE mapping from timeframe to target days-to-expiration
    TIMEFRAME_TO_DTE = {
        "1Min": 0,      # 0DTE
        "5Min": 0,      # 0DTE
        "15Min": 1,     # 0-1 DTE
        "30Min": 3,     # 1-3 DTE
        "1Hour": 5,     # 3-5 DTE
        "4Hour": 10,    # 1-2 weeks
        "1Day": 21,     # 2-4 weeks
    }

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

        # SELECT SPECIFIC OPTIONS CONTRACTS
        options_request = self._select_options_for_strategy(
            symbol=pipeline_result.symbol,
            strategy_name=suggested_strategy,
            direction=direction,
            spot_price=spot_price,
            timeframe=strongest_timeframe,
            expected_move=expected_move,
            confidence=confidence,
            timestamp=timestamp,
        )

        # Calculate options-specific profit target if we have contracts
        options_profit_target = None
        if options_request and options_request.max_profit > 0:
            options_profit_target = options_request.max_profit
            # Update expected ROI based on actual options potential
            if options_request.max_loss > 0:
                expected_roi = (options_request.max_profit / options_request.max_loss) * 100

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

            # Time-to-profit analysis
            time_to_profit=time_to_profit,

            # Options strategy from MTF
            options_strategy=suggested_strategy,
            options_strategy_details=suggested_expiry,
            options_expiry_suggestion=self._get_expiry_from_timeframe(strongest_timeframe),

            # MTF source
            source_timeframe=strongest_timeframe,
            mtf_alignment=mtf_alignment,

            # Profit projections
            expected_roi_pct=expected_roi,
            profit_confidence=profit_confidence,

            # ACTIONABLE OPTIONS - specific contracts to trade
            options_request=options_request,
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

    # =========================================================================
    # OPTIONS CONTRACT SELECTION - Actionable Trades
    # =========================================================================

    def _select_options_for_strategy(
        self,
        symbol: str,
        strategy_name: str,
        direction: DirectionEnum,
        spot_price: float,
        timeframe: str,
        expected_move: Any,
        confidence: float,
        timestamp: datetime,
    ) -> Optional[OptionsOrderRequest]:
        """
        Select specific options contracts for the suggested strategy.

        Maps timeframe to DTE, selects appropriate strikes, and calculates
        profit targets for each leg.

        Returns:
            OptionsOrderRequest with specific contracts and profit targets
        """
        if not strategy_name or not self.options_adapter:
            return None

        try:
            # Get target DTE from timeframe
            target_dte = self.TIMEFRAME_TO_DTE.get(timeframe, 7)

            # Fetch options chain
            chain = self.options_adapter.get_chain(symbol, timestamp)
            if not chain:
                logger.warning(f"No options chain available for {symbol}")
                return None

            # Filter chain by target DTE
            filtered_chain = self._filter_chain_by_dte(chain, target_dte, max_slippage=5)
            if not filtered_chain:
                logger.warning(f"No options found for {symbol} near {target_dte} DTE")
                return None

            # Select contracts based on strategy type
            strategy_lower = strategy_name.lower()

            if "long call" in strategy_lower:
                return self._build_long_call(
                    symbol, filtered_chain, spot_price, direction,
                    expected_move, confidence, timestamp, target_dte
                )
            elif "long put" in strategy_lower:
                return self._build_long_put(
                    symbol, filtered_chain, spot_price, direction,
                    expected_move, confidence, timestamp, target_dte
                )
            elif "bull call spread" in strategy_lower:
                return self._build_bull_call_spread(
                    symbol, filtered_chain, spot_price, direction,
                    expected_move, confidence, timestamp, target_dte
                )
            elif "bear put spread" in strategy_lower:
                return self._build_bear_put_spread(
                    symbol, filtered_chain, spot_price, direction,
                    expected_move, confidence, timestamp, target_dte
                )
            elif "bull put spread" in strategy_lower:
                return self._build_bull_put_spread(
                    symbol, filtered_chain, spot_price, direction,
                    expected_move, confidence, timestamp, target_dte
                )
            elif "bear call spread" in strategy_lower:
                return self._build_bear_call_spread(
                    symbol, filtered_chain, spot_price, direction,
                    expected_move, confidence, timestamp, target_dte
                )
            elif "iron condor" in strategy_lower:
                return self._build_iron_condor(
                    symbol, filtered_chain, spot_price, direction,
                    expected_move, confidence, timestamp, target_dte
                )
            elif "straddle" in strategy_lower:
                return self._build_straddle(
                    symbol, filtered_chain, spot_price, direction,
                    expected_move, confidence, timestamp, target_dte
                )
            else:
                # Default to directional based on direction
                if direction == DirectionEnum.LONG:
                    return self._build_long_call(
                        symbol, filtered_chain, spot_price, direction,
                        expected_move, confidence, timestamp, target_dte
                    )
                elif direction == DirectionEnum.SHORT:
                    return self._build_long_put(
                        symbol, filtered_chain, spot_price, direction,
                        expected_move, confidence, timestamp, target_dte
                    )

            return None

        except Exception as e:
            logger.error(f"Error selecting options for {symbol}: {e}")
            return None

    def _filter_chain_by_dte(
        self,
        chain: List[OptionContract],
        target_dte: int,
        max_slippage: int = 5,
    ) -> List[OptionContract]:
        """Filter chain to contracts near target DTE."""
        today = date.today()
        filtered = []

        for contract in chain:
            # Handle both datetime and date expiration
            if isinstance(contract.expiration, datetime):
                exp_date = contract.expiration.date()
            else:
                exp_date = contract.expiration

            dte = (exp_date - today).days
            if abs(dte - target_dte) <= max_slippage:
                filtered.append(contract)

        return filtered

    def _filter_by_type(
        self,
        chain: List[OptionContract],
        option_type: str,
    ) -> List[OptionContract]:
        """Filter chain by option type (call or put)."""
        return [c for c in chain if c.option_type.lower() == option_type.lower()]

    def _filter_liquid(
        self,
        chain: List[OptionContract],
        min_oi: int = 50,
        min_volume: int = 10,
    ) -> List[OptionContract]:
        """Filter to liquid contracts only."""
        liquid = []
        for c in chain:
            if c.open_interest >= min_oi and c.volume >= min_volume:
                liquid.append(c)
        # If no liquid options found, return all (fallback)
        return liquid if liquid else chain

    def _select_strike_near_price(
        self,
        chain: List[OptionContract],
        target_price: float,
    ) -> Optional[OptionContract]:
        """Select contract with strike nearest to target price."""
        if not chain:
            return None

        best = None
        best_diff = float('inf')

        for c in chain:
            diff = abs(c.strike - target_price)
            if diff < best_diff:
                best = c
                best_diff = diff

        return best

    def _calculate_option_profit_target(
        self,
        contract: OptionContract,
        expected_move: Any,
        direction: DirectionEnum,
        spot_price: float,
    ) -> Tuple[float, float]:
        """
        Calculate max profit and max loss for a single option.

        Returns:
            (max_profit, max_loss) tuple
        """
        entry_cost = (contract.bid + contract.ask) / 2 if contract.ask > 0 else contract.last
        max_loss = entry_cost * 100  # Per contract

        # Estimate profit based on expected move
        if expected_move and expected_move.one_sigma:
            if contract.option_type.lower() == "call":
                target_price = expected_move.one_sigma.upper
                intrinsic_at_target = max(0, target_price - contract.strike)
            else:  # put
                target_price = expected_move.one_sigma.lower
                intrinsic_at_target = max(0, contract.strike - target_price)

            # Profit = intrinsic value at target - entry cost
            max_profit = (intrinsic_at_target - entry_cost) * 100
            max_profit = max(0, max_profit)
        else:
            # Fallback: estimate 50% gain potential
            max_profit = max_loss * 0.5

        return max_profit, max_loss

    def _build_long_call(
        self,
        symbol: str,
        chain: List[OptionContract],
        spot_price: float,
        direction: DirectionEnum,
        expected_move: Any,
        confidence: float,
        timestamp: datetime,
        target_dte: int,
    ) -> Optional[OptionsOrderRequest]:
        """Build a Long Call order (directional bullish)."""
        calls = self._filter_by_type(chain, "call")
        calls = self._filter_liquid(calls)

        if not calls:
            return None

        # Select slightly OTM call (2% above spot)
        target_strike = spot_price * 1.02
        contract = self._select_strike_near_price(calls, target_strike)

        if not contract:
            return None

        max_profit, max_loss = self._calculate_option_profit_target(
            contract, expected_move, direction, spot_price
        )

        # Get expiration as string
        if isinstance(contract.expiration, datetime):
            exp_str = contract.expiration.strftime("%Y-%m-%d")
        else:
            exp_str = str(contract.expiration)

        leg = OptionsLeg(
            symbol=symbol,
            ratio=1,
            side="buy",
            type="call",
            strike=contract.strike,
            expiration=exp_str,
            action="buy_to_open",
        )

        entry_price = (contract.bid + contract.ask) / 2

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Long Call",
            legs=[leg],
            max_loss=max_loss,
            max_profit=max_profit,
            bpr=max_loss,  # BPR = cost of option
            rationale=f"Long ${contract.strike} Call @ ${entry_price:.2f}, {target_dte}DTE. Target: ${expected_move.one_sigma.upper if expected_move and expected_move.one_sigma else spot_price * 1.02:.2f}",
            confidence=confidence,
            timestamp=timestamp,
        )

    def _build_long_put(
        self,
        symbol: str,
        chain: List[OptionContract],
        spot_price: float,
        direction: DirectionEnum,
        expected_move: Any,
        confidence: float,
        timestamp: datetime,
        target_dte: int,
    ) -> Optional[OptionsOrderRequest]:
        """Build a Long Put order (directional bearish)."""
        puts = self._filter_by_type(chain, "put")
        puts = self._filter_liquid(puts)

        if not puts:
            return None

        # Select slightly OTM put (2% below spot)
        target_strike = spot_price * 0.98
        contract = self._select_strike_near_price(puts, target_strike)

        if not contract:
            return None

        max_profit, max_loss = self._calculate_option_profit_target(
            contract, expected_move, direction, spot_price
        )

        if isinstance(contract.expiration, datetime):
            exp_str = contract.expiration.strftime("%Y-%m-%d")
        else:
            exp_str = str(contract.expiration)

        leg = OptionsLeg(
            symbol=symbol,
            ratio=1,
            side="buy",
            type="put",
            strike=contract.strike,
            expiration=exp_str,
            action="buy_to_open",
        )

        entry_price = (contract.bid + contract.ask) / 2

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Long Put",
            legs=[leg],
            max_loss=max_loss,
            max_profit=max_profit,
            bpr=max_loss,
            rationale=f"Long ${contract.strike} Put @ ${entry_price:.2f}, {target_dte}DTE. Target: ${expected_move.one_sigma.lower if expected_move and expected_move.one_sigma else spot_price * 0.98:.2f}",
            confidence=confidence,
            timestamp=timestamp,
        )

    def _build_bull_call_spread(
        self,
        symbol: str,
        chain: List[OptionContract],
        spot_price: float,
        direction: DirectionEnum,
        expected_move: Any,
        confidence: float,
        timestamp: datetime,
        target_dte: int,
    ) -> Optional[OptionsOrderRequest]:
        """Build a Bull Call Spread (debit spread)."""
        calls = self._filter_by_type(chain, "call")
        calls = self._filter_liquid(calls)

        if len(calls) < 2:
            return None

        # Buy ATM call, sell OTM call
        long_strike = spot_price
        short_strike = spot_price * 1.03  # 3% OTM

        long_contract = self._select_strike_near_price(calls, long_strike)
        short_contract = self._select_strike_near_price(calls, short_strike)

        if not long_contract or not short_contract:
            return None

        if isinstance(long_contract.expiration, datetime):
            exp_str = long_contract.expiration.strftime("%Y-%m-%d")
        else:
            exp_str = str(long_contract.expiration)

        long_leg = OptionsLeg(
            symbol=symbol, ratio=1, side="buy", type="call",
            strike=long_contract.strike, expiration=exp_str, action="buy_to_open",
        )
        short_leg = OptionsLeg(
            symbol=symbol, ratio=1, side="sell", type="call",
            strike=short_contract.strike, expiration=exp_str, action="sell_to_open",
        )

        # Calculate spread P&L
        long_price = (long_contract.bid + long_contract.ask) / 2
        short_price = (short_contract.bid + short_contract.ask) / 2
        net_debit = (long_price - short_price) * 100

        spread_width = short_contract.strike - long_contract.strike
        max_profit = (spread_width * 100) - net_debit
        max_loss = net_debit

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Bull Call Spread",
            legs=[long_leg, short_leg],
            max_loss=max_loss,
            max_profit=max_profit,
            bpr=max_loss,
            rationale=f"Buy ${long_contract.strike}C / Sell ${short_contract.strike}C for ${net_debit/100:.2f} debit. Max profit: ${max_profit:.0f}",
            confidence=confidence,
            timestamp=timestamp,
        )

    def _build_bear_put_spread(
        self,
        symbol: str,
        chain: List[OptionContract],
        spot_price: float,
        direction: DirectionEnum,
        expected_move: Any,
        confidence: float,
        timestamp: datetime,
        target_dte: int,
    ) -> Optional[OptionsOrderRequest]:
        """Build a Bear Put Spread (debit spread)."""
        puts = self._filter_by_type(chain, "put")
        puts = self._filter_liquid(puts)

        if len(puts) < 2:
            return None

        # Buy ATM put, sell OTM put
        long_strike = spot_price
        short_strike = spot_price * 0.97  # 3% OTM

        long_contract = self._select_strike_near_price(puts, long_strike)
        short_contract = self._select_strike_near_price(puts, short_strike)

        if not long_contract or not short_contract:
            return None

        if isinstance(long_contract.expiration, datetime):
            exp_str = long_contract.expiration.strftime("%Y-%m-%d")
        else:
            exp_str = str(long_contract.expiration)

        long_leg = OptionsLeg(
            symbol=symbol, ratio=1, side="buy", type="put",
            strike=long_contract.strike, expiration=exp_str, action="buy_to_open",
        )
        short_leg = OptionsLeg(
            symbol=symbol, ratio=1, side="sell", type="put",
            strike=short_contract.strike, expiration=exp_str, action="sell_to_open",
        )

        long_price = (long_contract.bid + long_contract.ask) / 2
        short_price = (short_contract.bid + short_contract.ask) / 2
        net_debit = (long_price - short_price) * 100

        spread_width = long_contract.strike - short_contract.strike
        max_profit = (spread_width * 100) - net_debit
        max_loss = net_debit

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Bear Put Spread",
            legs=[long_leg, short_leg],
            max_loss=max_loss,
            max_profit=max_profit,
            bpr=max_loss,
            rationale=f"Buy ${long_contract.strike}P / Sell ${short_contract.strike}P for ${net_debit/100:.2f} debit. Max profit: ${max_profit:.0f}",
            confidence=confidence,
            timestamp=timestamp,
        )

    def _build_bull_put_spread(
        self,
        symbol: str,
        chain: List[OptionContract],
        spot_price: float,
        direction: DirectionEnum,
        expected_move: Any,
        confidence: float,
        timestamp: datetime,
        target_dte: int,
    ) -> Optional[OptionsOrderRequest]:
        """Build a Bull Put Spread (credit spread)."""
        puts = self._filter_by_type(chain, "put")
        puts = self._filter_liquid(puts)

        if len(puts) < 2:
            return None

        # Sell OTM put, buy further OTM put
        short_strike = spot_price * 0.97  # 3% OTM
        long_strike = spot_price * 0.94   # 6% OTM

        short_contract = self._select_strike_near_price(puts, short_strike)
        long_contract = self._select_strike_near_price(puts, long_strike)

        if not long_contract or not short_contract:
            return None

        if isinstance(short_contract.expiration, datetime):
            exp_str = short_contract.expiration.strftime("%Y-%m-%d")
        else:
            exp_str = str(short_contract.expiration)

        short_leg = OptionsLeg(
            symbol=symbol, ratio=1, side="sell", type="put",
            strike=short_contract.strike, expiration=exp_str, action="sell_to_open",
        )
        long_leg = OptionsLeg(
            symbol=symbol, ratio=1, side="buy", type="put",
            strike=long_contract.strike, expiration=exp_str, action="buy_to_open",
        )

        short_price = (short_contract.bid + short_contract.ask) / 2
        long_price = (long_contract.bid + long_contract.ask) / 2
        net_credit = (short_price - long_price) * 100

        spread_width = short_contract.strike - long_contract.strike
        max_loss = (spread_width * 100) - net_credit
        max_profit = net_credit

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Bull Put Spread",
            legs=[short_leg, long_leg],
            max_loss=max_loss,
            max_profit=max_profit,
            bpr=max_loss,
            rationale=f"Sell ${short_contract.strike}P / Buy ${long_contract.strike}P for ${net_credit/100:.2f} credit. Max profit: ${max_profit:.0f}",
            confidence=confidence,
            timestamp=timestamp,
        )

    def _build_bear_call_spread(
        self,
        symbol: str,
        chain: List[OptionContract],
        spot_price: float,
        direction: DirectionEnum,
        expected_move: Any,
        confidence: float,
        timestamp: datetime,
        target_dte: int,
    ) -> Optional[OptionsOrderRequest]:
        """Build a Bear Call Spread (credit spread)."""
        calls = self._filter_by_type(chain, "call")
        calls = self._filter_liquid(calls)

        if len(calls) < 2:
            return None

        # Sell OTM call, buy further OTM call
        short_strike = spot_price * 1.03  # 3% OTM
        long_strike = spot_price * 1.06   # 6% OTM

        short_contract = self._select_strike_near_price(calls, short_strike)
        long_contract = self._select_strike_near_price(calls, long_strike)

        if not long_contract or not short_contract:
            return None

        if isinstance(short_contract.expiration, datetime):
            exp_str = short_contract.expiration.strftime("%Y-%m-%d")
        else:
            exp_str = str(short_contract.expiration)

        short_leg = OptionsLeg(
            symbol=symbol, ratio=1, side="sell", type="call",
            strike=short_contract.strike, expiration=exp_str, action="sell_to_open",
        )
        long_leg = OptionsLeg(
            symbol=symbol, ratio=1, side="buy", type="call",
            strike=long_contract.strike, expiration=exp_str, action="buy_to_open",
        )

        short_price = (short_contract.bid + short_contract.ask) / 2
        long_price = (long_contract.bid + long_contract.ask) / 2
        net_credit = (short_price - long_price) * 100

        spread_width = long_contract.strike - short_contract.strike
        max_loss = (spread_width * 100) - net_credit
        max_profit = net_credit

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Bear Call Spread",
            legs=[short_leg, long_leg],
            max_loss=max_loss,
            max_profit=max_profit,
            bpr=max_loss,
            rationale=f"Sell ${short_contract.strike}C / Buy ${long_contract.strike}C for ${net_credit/100:.2f} credit. Max profit: ${max_profit:.0f}",
            confidence=confidence,
            timestamp=timestamp,
        )

    def _build_iron_condor(
        self,
        symbol: str,
        chain: List[OptionContract],
        spot_price: float,
        direction: DirectionEnum,
        expected_move: Any,
        confidence: float,
        timestamp: datetime,
        target_dte: int,
    ) -> Optional[OptionsOrderRequest]:
        """Build an Iron Condor (neutral strategy)."""
        calls = self._filter_by_type(chain, "call")
        puts = self._filter_by_type(chain, "put")
        calls = self._filter_liquid(calls)
        puts = self._filter_liquid(puts)

        if len(calls) < 2 or len(puts) < 2:
            return None

        # Iron Condor: Sell OTM put spread + Sell OTM call spread
        put_short_strike = spot_price * 0.97
        put_long_strike = spot_price * 0.94
        call_short_strike = spot_price * 1.03
        call_long_strike = spot_price * 1.06

        put_short = self._select_strike_near_price(puts, put_short_strike)
        put_long = self._select_strike_near_price(puts, put_long_strike)
        call_short = self._select_strike_near_price(calls, call_short_strike)
        call_long = self._select_strike_near_price(calls, call_long_strike)

        if not all([put_short, put_long, call_short, call_long]):
            return None

        if isinstance(put_short.expiration, datetime):
            exp_str = put_short.expiration.strftime("%Y-%m-%d")
        else:
            exp_str = str(put_short.expiration)

        legs = [
            OptionsLeg(symbol=symbol, ratio=1, side="sell", type="put",
                      strike=put_short.strike, expiration=exp_str, action="sell_to_open"),
            OptionsLeg(symbol=symbol, ratio=1, side="buy", type="put",
                      strike=put_long.strike, expiration=exp_str, action="buy_to_open"),
            OptionsLeg(symbol=symbol, ratio=1, side="sell", type="call",
                      strike=call_short.strike, expiration=exp_str, action="sell_to_open"),
            OptionsLeg(symbol=symbol, ratio=1, side="buy", type="call",
                      strike=call_long.strike, expiration=exp_str, action="buy_to_open"),
        ]

        # Calculate P&L
        put_credit = ((put_short.bid + put_short.ask)/2 - (put_long.bid + put_long.ask)/2) * 100
        call_credit = ((call_short.bid + call_short.ask)/2 - (call_long.bid + call_long.ask)/2) * 100
        total_credit = put_credit + call_credit

        put_width = put_short.strike - put_long.strike
        call_width = call_long.strike - call_short.strike
        max_risk = max(put_width, call_width) * 100 - total_credit

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Iron Condor",
            legs=legs,
            max_loss=max_risk,
            max_profit=total_credit,
            bpr=max_risk,
            rationale=f"IC: ${put_long.strike}P/${put_short.strike}P | ${call_short.strike}C/${call_long.strike}C for ${total_credit/100:.2f} credit",
            confidence=confidence,
            timestamp=timestamp,
        )

    def _build_straddle(
        self,
        symbol: str,
        chain: List[OptionContract],
        spot_price: float,
        direction: DirectionEnum,
        expected_move: Any,
        confidence: float,
        timestamp: datetime,
        target_dte: int,
    ) -> Optional[OptionsOrderRequest]:
        """Build a Long Straddle (volatility play)."""
        calls = self._filter_by_type(chain, "call")
        puts = self._filter_by_type(chain, "put")
        calls = self._filter_liquid(calls)
        puts = self._filter_liquid(puts)

        if not calls or not puts:
            return None

        # ATM straddle
        call_contract = self._select_strike_near_price(calls, spot_price)
        put_contract = self._select_strike_near_price(puts, spot_price)

        if not call_contract or not put_contract:
            return None

        if isinstance(call_contract.expiration, datetime):
            exp_str = call_contract.expiration.strftime("%Y-%m-%d")
        else:
            exp_str = str(call_contract.expiration)

        legs = [
            OptionsLeg(symbol=symbol, ratio=1, side="buy", type="call",
                      strike=call_contract.strike, expiration=exp_str, action="buy_to_open"),
            OptionsLeg(symbol=symbol, ratio=1, side="buy", type="put",
                      strike=put_contract.strike, expiration=exp_str, action="buy_to_open"),
        ]

        call_price = (call_contract.bid + call_contract.ask) / 2
        put_price = (put_contract.bid + put_contract.ask) / 2
        total_cost = (call_price + put_price) * 100

        # Breakeven requires move of total_cost from strike
        breakeven_move_pct = (call_price + put_price) / spot_price * 100

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Long Straddle",
            legs=legs,
            max_loss=total_cost,
            max_profit=total_cost * 3,  # Estimate 3x potential on big move
            bpr=total_cost,
            rationale=f"ATM ${call_contract.strike} Straddle for ${total_cost/100:.2f}. Need {breakeven_move_pct:.1f}% move to breakeven",
            confidence=confidence,
            timestamp=timestamp,
        )
