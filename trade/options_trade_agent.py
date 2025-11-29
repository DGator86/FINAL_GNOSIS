"""Options Trade Agent - Intelligent Strategy Selection."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from loguru import logger

from gnosis.utils.option_utils import OptionUtils
from schemas.core_schemas import OptionsLeg, OptionsOrderRequest


class OptionsTradeAgent:
    """
    Intelligent options strategy selector.
    Maps market conditions (Hedge/Composer) to specific option strategies.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config
        self.risk_config = config.get("risk_management", {})
        self.options_config = config.get("agents", {}).get("trade_v3", {})

        logger.info("OptionsTradeAgent initialized")

    def select_strategy(
        self,
        symbol: str,
        hedge_snapshot: Optional[Dict[str, Any]],
        composer_signal: str,  # BUY, SELL, HOLD
        composer_confidence: float,
        current_price: float,
        iv_rank: Optional[float] = None,
        iv_percentile: Optional[float] = None,
    ) -> Optional[OptionsOrderRequest]:
        """
        Select the best options strategy based on inputs.

        Strategy Selection Logic:
        - Neutral (low confidence or HOLD) + High IV → Iron Condor
        - Neutral + Low IV + High expected vol → Straddle/Strangle
        - Directional + High IV → Credit Spreads
        - Directional + Low IV → Debit Spreads or Long Options
        """
        # 1. Determine Volatility Environment
        is_high_iv = False
        if iv_rank and iv_rank > 50:
            is_high_iv = True

        # 2. Detect Neutral Signal
        # Neutral if: HOLD signal OR low confidence (<0.5)
        is_neutral = composer_signal == "HOLD" or composer_confidence < 0.5

        # 3. Check for High Volatility Expectation
        # Use hedge snapshot to detect volatility regime changes
        expect_high_volatility = False
        if hedge_snapshot:
            # If movement energy is high, expect continued volatility
            movement_energy = hedge_snapshot.get("movement_energy", 0.0)
            if movement_energy > 50:  # Threshold for high volatility expectation
                expect_high_volatility = True

        # 4. Strategy Selection

        # NEUTRAL STRATEGIES
        if is_neutral:
            if is_high_iv:
                # High IV + Neutral → Iron Condor (sell premium in range)
                logger.info(f"Neutral signal with high IV → Iron Condor for {symbol}")
                return self._build_iron_condor(symbol, current_price, composer_confidence)
            elif expect_high_volatility:
                # Low IV + Expect Big Move → Straddle or Strangle
                if composer_confidence > 0.4:
                    # Higher confidence in volatility → Straddle (more expensive but better)
                    logger.info(f"Neutral + high vol expectation → Straddle for {symbol}")
                    return self._build_straddle(symbol, current_price, composer_confidence)
                else:
                    # Lower confidence → Strangle (cheaper)
                    logger.info(f"Neutral + high vol expectation → Strangle for {symbol}")
                    return self._build_strangle(symbol, current_price, composer_confidence)
            else:
                # Neutral + Low IV + No vol expectation → Skip
                logger.info(f"Neutral signal with low IV, no clear opportunity for {symbol}")
                return None

        # DIRECTIONAL STRATEGIES
        direction = "bullish" if composer_signal == "BUY" else "bearish"

        if direction == "bullish":
            if is_high_iv:
                # High IV Bull → Bull Put Spread (Credit)
                return self._build_bull_put_spread(symbol, current_price, composer_confidence)
            else:
                # Low IV Bull → Long Call or Bull Call Spread
                if composer_confidence > 0.8:
                    return self._build_long_call(symbol, current_price, composer_confidence)
                else:
                    return self._build_bull_call_spread(symbol, current_price, composer_confidence)

        elif direction == "bearish":
            if is_high_iv:
                # High IV Bear → Bear Call Spread (Credit)
                return self._build_bear_call_spread(symbol, current_price, composer_confidence)
            else:
                # Low IV Bear → Long Put or Bear Put Spread
                if composer_confidence > 0.8:
                    return self._build_long_put(symbol, current_price, composer_confidence)
                else:
                    return self._build_bear_put_spread(symbol, current_price, composer_confidence)

        return None

    def _select_expiration(self, min_dte: int = 30, max_dte: int = 45) -> datetime:
        """Select expiration date."""
        # Simple logic: Target ~30-45 days out (standard monthly or weekly)
        # For simulation, just return a date
        target_date = datetime.now() + timedelta(days=40)
        # Adjust to Friday?
        days_ahead = 4 - target_date.weekday()  # 4 is Friday
        if days_ahead < 0:
            days_ahead += 7
        return target_date + timedelta(days=days_ahead)

    def _build_long_call(
        self, symbol: str, current_price: float, confidence: float
    ) -> OptionsOrderRequest:
        """Build Long Call Strategy."""
        expiration = self._select_expiration()

        # Strike Selection: ATM or slightly OTM
        # For Long Call, maybe Delta 0.50 (ATM)
        strike = round(current_price, 0)  # Simplified ATM

        occ_symbol = OptionUtils.generate_occ_symbol(symbol, expiration, "call", strike)

        leg = OptionsLeg(
            symbol=occ_symbol,
            ratio=1,
            side="buy",
            type="call",
            strike=strike,
            expiration=expiration.strftime("%Y-%m-%d"),
            action="buy_to_open",
        )

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Long Call",
            legs=[leg],
            max_loss=500.0,  # Placeholder
            max_profit=float("inf"),
            bpr=500.0,
            rationale=f"Bullish conviction ({confidence:.2f}) in low IV",
            confidence=confidence,
        )

    def _build_long_put(
        self, symbol: str, current_price: float, confidence: float
    ) -> OptionsOrderRequest:
        """Build Long Put Strategy."""
        expiration = self._select_expiration()
        strike = round(current_price, 0)  # Simplified ATM

        occ_symbol = OptionUtils.generate_occ_symbol(symbol, expiration, "put", strike)

        leg = OptionsLeg(
            symbol=occ_symbol,
            ratio=1,
            side="buy",
            type="put",
            strike=strike,
            expiration=expiration.strftime("%Y-%m-%d"),
            action="buy_to_open",
        )

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Long Put",
            legs=[leg],
            max_loss=500.0,
            max_profit=strike * 100,
            bpr=500.0,
            rationale=f"Bearish conviction ({confidence:.2f}) in low IV",
            confidence=confidence,
        )

    def _build_bull_call_spread(
        self, symbol: str, current_price: float, confidence: float
    ) -> OptionsOrderRequest:
        """Build Bull Call Spread (Debit)."""
        expiration = self._select_expiration()

        # Buy ATM Call, Sell OTM Call
        strike_long = round(current_price, 0)
        strike_short = round(current_price * 1.05, 0)  # 5% OTM

        leg1 = OptionsLeg(
            symbol=OptionUtils.generate_occ_symbol(symbol, expiration, "call", strike_long),
            ratio=1,
            side="buy",
            type="call",
            strike=strike_long,
            expiration=expiration.strftime("%Y-%m-%d"),
            action="buy_to_open",
        )

        leg2 = OptionsLeg(
            symbol=OptionUtils.generate_occ_symbol(symbol, expiration, "call", strike_short),
            ratio=1,
            side="sell",
            type="call",
            strike=strike_short,
            expiration=expiration.strftime("%Y-%m-%d"),
            action="sell_to_open",
        )

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Bull Call Spread",
            legs=[leg1, leg2],
            max_loss=200.0,  # Placeholder
            max_profit=300.0,
            bpr=200.0,
            rationale=f"Bullish spread ({confidence:.2f})",
            confidence=confidence,
        )

    def _build_bull_put_spread(
        self, symbol: str, current_price: float, confidence: float
    ) -> OptionsOrderRequest:
        """Build Bull Put Spread (Credit)."""
        expiration = self._select_expiration()

        # Sell ATM Put, Buy OTM Put
        strike_short = round(current_price, 0)
        strike_long = round(current_price * 0.95, 0)  # 5% OTM

        leg1 = OptionsLeg(
            symbol=OptionUtils.generate_occ_symbol(symbol, expiration, "put", strike_short),
            ratio=1,
            side="sell",
            type="put",
            strike=strike_short,
            expiration=expiration.strftime("%Y-%m-%d"),
            action="sell_to_open",
        )

        leg2 = OptionsLeg(
            symbol=OptionUtils.generate_occ_symbol(symbol, expiration, "put", strike_long),
            ratio=1,
            side="buy",
            type="put",
            strike=strike_long,
            expiration=expiration.strftime("%Y-%m-%d"),
            action="buy_to_open",
        )

        max_profit = 150.0  # Net credit received
        max_loss = (strike_short - strike_long) * 100 - max_profit

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Bull Put Spread",
            legs=[leg1, leg2],
            max_loss=max_loss,
            max_profit=max_profit,
            bpr=max_loss,
            rationale=f"Bullish credit spread ({confidence:.2f}) in high IV",
            confidence=confidence,
        )

    def _build_bear_call_spread(
        self, symbol: str, current_price: float, confidence: float
    ) -> OptionsOrderRequest:
        """Build Bear Call Spread (Credit)."""
        expiration = self._select_expiration()

        # Sell ATM Call, Buy OTM Call
        strike_short = round(current_price, 0)
        strike_long = round(current_price * 1.05, 0)  # 5% OTM

        leg1 = OptionsLeg(
            symbol=OptionUtils.generate_occ_symbol(symbol, expiration, "call", strike_short),
            ratio=1,
            side="sell",
            type="call",
            strike=strike_short,
            expiration=expiration.strftime("%Y-%m-%d"),
            action="sell_to_open",
        )

        leg2 = OptionsLeg(
            symbol=OptionUtils.generate_occ_symbol(symbol, expiration, "call", strike_long),
            ratio=1,
            side="buy",
            type="call",
            strike=strike_long,
            expiration=expiration.strftime("%Y-%m-%d"),
            action="buy_to_open",
        )

        max_profit = 150.0  # Net credit received
        max_loss = (strike_long - strike_short) * 100 - max_profit

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Bear Call Spread",
            legs=[leg1, leg2],
            max_loss=max_loss,
            max_profit=max_profit,
            bpr=max_loss,
            rationale=f"Bearish credit spread ({confidence:.2f}) in high IV",
            confidence=confidence,
        )

    def _build_bear_put_spread(
        self, symbol: str, current_price: float, confidence: float
    ) -> OptionsOrderRequest:
        """Build Bear Put Spread (Debit)."""
        expiration = self._select_expiration()

        # Buy ATM Put, Sell OTM Put
        strike_long = round(current_price, 0)
        strike_short = round(current_price * 0.95, 0)  # 5% OTM

        leg1 = OptionsLeg(
            symbol=OptionUtils.generate_occ_symbol(symbol, expiration, "put", strike_long),
            ratio=1,
            side="buy",
            type="put",
            strike=strike_long,
            expiration=expiration.strftime("%Y-%m-%d"),
            action="buy_to_open",
        )

        leg2 = OptionsLeg(
            symbol=OptionUtils.generate_occ_symbol(symbol, expiration, "put", strike_short),
            ratio=1,
            side="sell",
            type="put",
            strike=strike_short,
            expiration=expiration.strftime("%Y-%m-%d"),
            action="sell_to_open",
        )

        max_loss = 200.0  # Net debit paid
        max_profit = (strike_long - strike_short) * 100 - max_loss

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Bear Put Spread",
            legs=[leg1, leg2],
            max_loss=max_loss,
            max_profit=max_profit,
            bpr=max_loss,
            rationale=f"Bearish debit spread ({confidence:.2f})",
            confidence=confidence,
        )

    def _build_iron_condor(
        self, symbol: str, current_price: float, confidence: float
    ) -> OptionsOrderRequest:
        """Build Iron Condor (Credit).

        4-leg neutral strategy: OTM Bull Put Spread + OTM Bear Call Spread
        Used in high IV, range-bound markets.
        """
        expiration = self._select_expiration()

        # Put side (Bull Put Spread) - Below current price
        put_short_strike = round(current_price * 0.95, 0)  # 5% OTM
        put_long_strike = round(current_price * 0.90, 0)  # 10% OTM

        # Call side (Bear Call Spread) - Above current price
        call_short_strike = round(current_price * 1.05, 0)  # 5% OTM
        call_long_strike = round(current_price * 1.10, 0)  # 10% OTM

        legs = [
            # Put spread
            OptionsLeg(
                symbol=OptionUtils.generate_occ_symbol(symbol, expiration, "put", put_short_strike),
                ratio=1,
                side="sell",
                type="put",
                strike=put_short_strike,
                expiration=expiration.strftime("%Y-%m-%d"),
                action="sell_to_open",
            ),
            OptionsLeg(
                symbol=OptionUtils.generate_occ_symbol(symbol, expiration, "put", put_long_strike),
                ratio=1,
                side="buy",
                type="put",
                strike=put_long_strike,
                expiration=expiration.strftime("%Y-%m-%d"),
                action="buy_to_open",
            ),
            # Call spread
            OptionsLeg(
                symbol=OptionUtils.generate_occ_symbol(
                    symbol, expiration, "call", call_short_strike
                ),
                ratio=1,
                side="sell",
                type="call",
                strike=call_short_strike,
                expiration=expiration.strftime("%Y-%m-%d"),
                action="sell_to_open",
            ),
            OptionsLeg(
                symbol=OptionUtils.generate_occ_symbol(
                    symbol, expiration, "call", call_long_strike
                ),
                ratio=1,
                side="buy",
                type="call",
                strike=call_long_strike,
                expiration=expiration.strftime("%Y-%m-%d"),
                action="buy_to_open",
            ),
        ]

        max_profit = 300.0  # Net credit received
        wing_width = (put_short_strike - put_long_strike) * 100
        max_loss = wing_width - max_profit

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Iron Condor",
            legs=legs,
            max_loss=max_loss,
            max_profit=max_profit,
            bpr=max_loss,
            rationale=f"Neutral range-bound ({confidence:.2f}) in high IV",
            confidence=confidence,
        )

    def _build_straddle(
        self, symbol: str, current_price: float, confidence: float
    ) -> OptionsOrderRequest:
        """Build Long Straddle (Debit).

        2-leg volatility strategy: ATM Call + ATM Put (both bought)
        Used when expecting large move but direction uncertain.
        """
        expiration = self._select_expiration()
        strike = round(current_price, 0)  # ATM for both

        legs = [
            OptionsLeg(
                symbol=OptionUtils.generate_occ_symbol(symbol, expiration, "call", strike),
                ratio=1,
                side="buy",
                type="call",
                strike=strike,
                expiration=expiration.strftime("%Y-%m-%d"),
                action="buy_to_open",
            ),
            OptionsLeg(
                symbol=OptionUtils.generate_occ_symbol(symbol, expiration, "put", strike),
                ratio=1,
                side="buy",
                type="put",
                strike=strike,
                expiration=expiration.strftime("%Y-%m-%d"),
                action="buy_to_open",
            ),
        ]

        max_loss = 800.0  # Total premium paid
        max_profit = float("inf")  # Unlimited upside

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Long Straddle",
            legs=legs,
            max_loss=max_loss,
            max_profit=max_profit,
            bpr=max_loss,
            rationale=f"High volatility expected ({confidence:.2f}), no directional bias",
            confidence=confidence,
        )

    def _build_strangle(
        self, symbol: str, current_price: float, confidence: float
    ) -> OptionsOrderRequest:
        """Build Long Strangle (Debit).

        2-leg volatility strategy: OTM Call + OTM Put (both bought)
        Cheaper than straddle but requires larger move to profit.
        """
        expiration = self._select_expiration()

        # OTM strikes
        call_strike = round(current_price * 1.05, 0)  # 5% OTM
        put_strike = round(current_price * 0.95, 0)  # 5% OTM

        legs = [
            OptionsLeg(
                symbol=OptionUtils.generate_occ_symbol(symbol, expiration, "call", call_strike),
                ratio=1,
                side="buy",
                type="call",
                strike=call_strike,
                expiration=expiration.strftime("%Y-%m-%d"),
                action="buy_to_open",
            ),
            OptionsLeg(
                symbol=OptionUtils.generate_occ_symbol(symbol, expiration, "put", put_strike),
                ratio=1,
                side="buy",
                type="put",
                strike=put_strike,
                expiration=expiration.strftime("%Y-%m-%d"),
                action="buy_to_open",
            ),
        ]

        max_loss = 500.0  # Total premium paid (cheaper than straddle)
        max_profit = float("inf")  # Unlimited upside

        return OptionsOrderRequest(
            symbol=symbol,
            strategy_name="Long Strangle",
            legs=legs,
            max_loss=max_loss,
            max_profit=max_profit,
            bpr=max_loss,
            rationale=f"Large move expected ({confidence:.2f}), lower cost than straddle",
            confidence=confidence,
        )
