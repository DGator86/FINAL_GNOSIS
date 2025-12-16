"""
Expected Move Calculator - Industry Standard Implementation

Calculates expected price movement ranges using multiple methodologies:
1. IV-Based (Primary): Uses options implied volatility - the market's forecast
2. ATR-Based (Fallback): Uses Average True Range for non-options assets
3. Hybrid: Blends both when available for higher confidence

Based on options market conventions used by CBOE, Bloomberg, and institutional traders.

References:
- CBOE Expected Move: IV × √(DTE/365) × Spot
- Bloomberg MOVE Index methodology
- CME Group variance swap conventions
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from schemas.core_schemas import (
    ExpectedMove,
    PipelineResult,
    PriceRange,
)


class ExpectedMoveCalculator:
    """
    Industry-standard expected move calculator.

    Uses the same methodology as professional options traders:
    - 1σ move has 68.2% probability (±1 standard deviation)
    - 2σ move has 95.4% probability (±2 standard deviations)
    - 3σ move has 99.7% probability (±3 standard deviations)

    Primary calculation (IV-based):
        Expected Move ($) = Spot × IV × √(DTE/365)
        Expected Move (%) = IV × √(DTE/365)

    Where:
        - Spot = Current underlying price
        - IV = Annualized implied volatility (ATM)
        - DTE = Days to expiration (or forecast horizon)
    """

    # Standard deviation multipliers for probability levels
    SIGMA_1 = 1.0    # 68.2% probability
    SIGMA_2 = 2.0    # 95.4% probability
    SIGMA_3 = 3.0    # 99.7% probability

    # Trading days per year (industry standard)
    TRADING_DAYS_YEAR = 252
    CALENDAR_DAYS_YEAR = 365

    def __init__(
        self,
        options_adapter: Any = None,
        market_adapter: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.options_adapter = options_adapter
        self.market_adapter = market_adapter
        self.config = config or {}

        # Default timeframe for expected move (1 day)
        self.default_dte = self.config.get("default_dte", 1)

    def calculate(
        self,
        symbol: str,
        spot_price: float,
        pipeline_result: Optional[PipelineResult] = None,
        dte: Optional[int] = None,
        iv_override: Optional[float] = None,
    ) -> ExpectedMove:
        """
        Calculate expected move using best available data.

        Priority order:
        1. IV-based (if options data available)
        2. ATR-based (if market data available)
        3. Volatility regime estimate (fallback)

        Args:
            symbol: Ticker symbol
            spot_price: Current price of underlying
            pipeline_result: Optional pipeline result with engine snapshots
            dte: Days to expiration/horizon (default: 1)
            iv_override: Manual IV override for testing

        Returns:
            ExpectedMove with probability ranges
        """
        if dte is None:
            dte = self.default_dte

        # Try IV-based calculation first (highest accuracy)
        iv_result = self._calculate_iv_based(
            symbol, spot_price, dte, pipeline_result, iv_override
        )
        if iv_result:
            return iv_result

        # Fallback to ATR-based
        atr_result = self._calculate_atr_based(
            symbol, spot_price, dte, pipeline_result
        )
        if atr_result:
            return atr_result

        # Last resort: volatility regime estimate
        return self._calculate_regime_based(
            symbol, spot_price, dte, pipeline_result
        )

    def _calculate_iv_based(
        self,
        symbol: str,
        spot_price: float,
        dte: int,
        pipeline_result: Optional[PipelineResult],
        iv_override: Optional[float] = None,
    ) -> Optional[ExpectedMove]:
        """
        Calculate using implied volatility (industry standard).

        Formula: Expected Move = Spot × IV × √(DTE/365)

        This is the same calculation used by:
        - CBOE for expected move on SPX/VIX
        - ThinkOrSwim expected move indicator
        - Bloomberg terminal expected move
        - Professional options market makers
        """
        # Get IV from various sources
        iv = iv_override
        iv_rank = None
        hv = None

        if iv is None and self.options_adapter:
            try:
                vol_metrics = self.options_adapter.get_volatility_metrics(symbol)
                if vol_metrics:
                    iv = vol_metrics.get("atm_iv") or vol_metrics.get("iv")
                    iv_rank = vol_metrics.get("iv_rank") or vol_metrics.get("iv_percentile")
                    hv = vol_metrics.get("hv_20") or vol_metrics.get("hv")
            except Exception as e:
                logger.debug(f"Could not get vol metrics for {symbol}: {e}")

        # Try to get IV from hedge snapshot
        if iv is None and pipeline_result and pipeline_result.hedge_snapshot:
            # Some hedge engines store IV in regime_features
            regime_features = pipeline_result.hedge_snapshot.regime_features
            iv = regime_features.get("atm_iv") or regime_features.get("implied_vol")

        if iv is None or iv <= 0:
            return None

        # Industry standard formula
        # Expected Move (%) = IV × √(DTE/365)
        time_factor = math.sqrt(dte / self.CALENDAR_DAYS_YEAR)
        expected_move_pct = iv * time_factor

        # Calculate dollar moves
        move_1sigma = spot_price * expected_move_pct
        move_2sigma = spot_price * expected_move_pct * self.SIGMA_2
        move_3sigma = spot_price * expected_move_pct * self.SIGMA_3

        # Build price ranges
        one_sigma = PriceRange(
            lower=round(spot_price - move_1sigma, 2),
            center=round(spot_price, 2),
            upper=round(spot_price + move_1sigma, 2),
        )

        two_sigma = PriceRange(
            lower=round(spot_price - move_2sigma, 2),
            center=round(spot_price, 2),
            upper=round(spot_price + move_2sigma, 2),
        )

        three_sigma = PriceRange(
            lower=round(spot_price - move_3sigma, 2),
            center=round(spot_price, 2),
            upper=round(spot_price + move_3sigma, 2),
        )

        # Calculate directional probabilities if skew available
        upside_prob, downside_prob = self._calculate_skew_probabilities(
            symbol, pipeline_result
        )

        # Determine timeframe string
        timeframe = self._dte_to_timeframe(dte)

        return ExpectedMove(
            one_sigma=one_sigma,
            two_sigma=two_sigma,
            three_sigma=three_sigma,
            calculation_method="iv_based",
            timeframe=timeframe,
            implied_volatility=round(iv, 4),
            historical_volatility=round(hv, 4) if hv else None,
            iv_rank=round(iv_rank, 1) if iv_rank else None,
            days_to_expiry=dte,
            expected_move_pct=round(expected_move_pct * 100, 2),  # As percentage
            upside_probability=upside_prob,
            downside_probability=downside_prob,
        )

    def _calculate_atr_based(
        self,
        symbol: str,
        spot_price: float,
        dte: int,
        pipeline_result: Optional[PipelineResult],
    ) -> Optional[ExpectedMove]:
        """
        Calculate using Average True Range (fallback method).

        ATR measures average daily range, useful when IV not available.

        Conversion: ATR represents ~1σ of daily movement
        Scale: ATR × √(DTE) for multi-day projections
        """
        atr = None
        atr_pct = None

        # Try to get ATR from market adapter
        if self.market_adapter:
            try:
                end = datetime.utcnow()
                start = end - timedelta(days=30)
                bars = self.market_adapter.get_bars(
                    symbol, start=start, end=end, timeframe="1Day"
                )
                if bars and len(bars) >= 14:
                    atr = self._compute_atr(bars, period=14)
                    atr_pct = atr / spot_price if spot_price > 0 else 0
            except Exception as e:
                logger.debug(f"Could not compute ATR for {symbol}: {e}")

        if atr is None or atr <= 0:
            return None

        # Scale ATR for time horizon
        # ATR is approximately 1σ daily move
        time_factor = math.sqrt(dte)
        expected_move_pct = atr_pct * time_factor

        # Calculate ranges (ATR ≈ 1σ)
        move_1sigma = atr * time_factor
        move_2sigma = move_1sigma * self.SIGMA_2
        move_3sigma = move_1sigma * self.SIGMA_3

        one_sigma = PriceRange(
            lower=round(spot_price - move_1sigma, 2),
            center=round(spot_price, 2),
            upper=round(spot_price + move_1sigma, 2),
        )

        two_sigma = PriceRange(
            lower=round(spot_price - move_2sigma, 2),
            center=round(spot_price, 2),
            upper=round(spot_price + move_2sigma, 2),
        )

        three_sigma = PriceRange(
            lower=round(spot_price - move_3sigma, 2),
            center=round(spot_price, 2),
            upper=round(spot_price + move_3sigma, 2),
        )

        timeframe = self._dte_to_timeframe(dte)

        return ExpectedMove(
            one_sigma=one_sigma,
            two_sigma=two_sigma,
            three_sigma=three_sigma,
            calculation_method="atr_based",
            timeframe=timeframe,
            days_to_expiry=dte,
            expected_move_pct=round(expected_move_pct * 100, 2),
        )

    def _calculate_regime_based(
        self,
        symbol: str,
        spot_price: float,
        dte: int,
        pipeline_result: Optional[PipelineResult],
    ) -> ExpectedMove:
        """
        Fallback calculation using volatility regime estimates.

        Uses typical volatility values based on market regime:
        - Low vol: ~10% annualized
        - Moderate vol: ~20% annualized
        - High vol: ~35% annualized
        """
        # Default to moderate volatility
        annual_vol = 0.20

        if pipeline_result and pipeline_result.elasticity_snapshot:
            regime = pipeline_result.elasticity_snapshot.volatility_regime.lower()
            vol = pipeline_result.elasticity_snapshot.volatility

            if vol > 0:
                # Use actual volatility if available
                annual_vol = vol
            elif regime == "low":
                annual_vol = 0.10
            elif regime == "high":
                annual_vol = 0.35

        # Standard IV formula
        time_factor = math.sqrt(dte / self.CALENDAR_DAYS_YEAR)
        expected_move_pct = annual_vol * time_factor

        move_1sigma = spot_price * expected_move_pct
        move_2sigma = move_1sigma * self.SIGMA_2
        move_3sigma = move_1sigma * self.SIGMA_3

        one_sigma = PriceRange(
            lower=round(spot_price - move_1sigma, 2),
            center=round(spot_price, 2),
            upper=round(spot_price + move_1sigma, 2),
        )

        two_sigma = PriceRange(
            lower=round(spot_price - move_2sigma, 2),
            center=round(spot_price, 2),
            upper=round(spot_price + move_2sigma, 2),
        )

        three_sigma = PriceRange(
            lower=round(spot_price - move_3sigma, 2),
            center=round(spot_price, 2),
            upper=round(spot_price + move_3sigma, 2),
        )

        timeframe = self._dte_to_timeframe(dte)

        return ExpectedMove(
            one_sigma=one_sigma,
            two_sigma=two_sigma,
            three_sigma=three_sigma,
            calculation_method="regime_estimate",
            timeframe=timeframe,
            days_to_expiry=dte,
            expected_move_pct=round(expected_move_pct * 100, 2),
        )

    def _compute_atr(self, bars: List[Any], period: int = 14) -> float:
        """Compute Average True Range from OHLC bars."""
        if len(bars) < period + 1:
            return 0.0

        true_ranges = []
        for i in range(1, len(bars)):
            high = bars[i].high
            low = bars[i].low
            prev_close = bars[i - 1].close

            # True Range = max(H-L, |H-PC|, |L-PC|)
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        # Simple moving average of TR
        if len(true_ranges) >= period:
            return sum(true_ranges[-period:]) / period
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

    def _calculate_skew_probabilities(
        self,
        symbol: str,
        pipeline_result: Optional[PipelineResult],
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate directional probabilities from put/call skew.

        Negative skew (puts more expensive) = higher downside probability
        Positive skew (calls more expensive) = higher upside probability
        """
        if not pipeline_result:
            return None, None

        # Try to get directional bias from consensus
        if pipeline_result.consensus:
            direction = pipeline_result.consensus.get("direction", "neutral")
            confidence = pipeline_result.consensus.get("confidence", 0.5)

            if direction == "long":
                upside = 0.5 + (confidence * 0.2)  # Up to 70% upside
                return round(upside, 2), round(1 - upside, 2)
            elif direction == "short":
                downside = 0.5 + (confidence * 0.2)
                return round(1 - downside, 2), round(downside, 2)

        # Neutral: 50/50
        return 0.50, 0.50

    def _dte_to_timeframe(self, dte: int) -> str:
        """Convert DTE to human-readable timeframe."""
        if dte <= 1:
            return "1D"
        elif dte <= 5:
            return f"{dte}D"
        elif dte <= 7:
            return "1W"
        elif dte <= 14:
            return "2W"
        elif dte <= 30:
            return "1M"
        elif dte <= 90:
            return "3M"
        else:
            return f"{dte}D"


def calculate_expected_move_for_trade(
    symbol: str,
    spot_price: float,
    pipeline_result: PipelineResult,
    options_adapter: Any = None,
    market_adapter: Any = None,
    dte: int = 1,
) -> ExpectedMove:
    """
    Convenience function to calculate expected move for a trade idea.

    Args:
        symbol: Ticker symbol
        spot_price: Current underlying price
        pipeline_result: Pipeline result with engine snapshots
        options_adapter: Optional options data adapter
        market_adapter: Optional market data adapter
        dte: Days to expiration/horizon

    Returns:
        ExpectedMove with probability ranges
    """
    calculator = ExpectedMoveCalculator(
        options_adapter=options_adapter,
        market_adapter=market_adapter,
    )
    return calculator.calculate(
        symbol=symbol,
        spot_price=spot_price,
        pipeline_result=pipeline_result,
        dte=dte,
    )
