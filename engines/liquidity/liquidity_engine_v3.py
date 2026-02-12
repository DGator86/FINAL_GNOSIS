"""Liquidity Engine v3 - Enhanced market liquidity analysis with 0DTE support."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

from loguru import logger

from engines.inputs.market_data_adapter import MarketDataAdapter
from engines.inputs.options_chain_adapter import OptionsChainAdapter
from schemas.core_schemas import LiquiditySnapshot


class LiquidityEngineV3:
    """
    Liquidity Engine v3 for enhanced market liquidity analysis.

    New V3 Features:
    - 0DTE depth analysis for intraday scalping
    - Gamma squeeze risk detection
    - Enhanced spread analysis for options
    """

    def __init__(
        self,
        market_adapter: MarketDataAdapter,
        options_adapter: OptionsChainAdapter,
        config: Dict[str, Any],
    ):
        """
        Initialize Liquidity Engine V3.

        Args:
            market_adapter: Market data provider
            options_adapter: Options chain data provider
            config: Engine configuration
        """
        self.market_adapter = market_adapter
        self.options_adapter = options_adapter
        self.config = config
        logger.info("LiquidityEngineV3 initialized")

    def run(self, symbol: str, timestamp: datetime) -> LiquiditySnapshot:
        """
        Run liquidity analysis for a symbol.

        Args:
            symbol: Trading symbol
            timestamp: Analysis timestamp

        Returns:
            LiquiditySnapshot with liquidity metrics
        """
        logger.debug(f"Running LiquidityEngineV3 for {symbol} at {timestamp}")

        try:
            # Get current quote
            quote = self.market_adapter.get_quote(symbol)

            # Calculate bid-ask spread
            mid_price = (quote.bid + quote.ask) / 2
            spread_pct = ((quote.ask - quote.bid) / mid_price) * 100 if mid_price > 0 else 0.0

            # Get recent volume data
            bars = self.market_adapter.get_bars(
                symbol, timestamp - timedelta(days=5), timestamp, timeframe="1Day"
            )

            avg_volume = sum(bar.volume for bar in bars) / len(bars) if bars else 0.0

            # Calculate depth (sum of bid and ask sizes)
            depth = quote.bid_size + quote.ask_size

            # V3: Calculate 0DTE depth (intraday liquidity)
            zero_dte_depth = self._calculate_0dte_depth(symbol, timestamp)

            # V3: Detect gamma squeeze risk
            gamma_squeeze_risk = self._detect_gamma_squeeze_risk(symbol, timestamp, avg_volume)

            # Calculate impact cost (simplified)
            impact_cost = spread_pct * 0.5  # Assume crossing half the spread

            # Calculate liquidity score (0-1 scale)
            # Higher volume and tighter spreads = higher score
            volume_score = min(1.0, avg_volume / 10_000_000)  # Normalize to 10M
            spread_score = max(0.0, 1.0 - (spread_pct / 1.0))  # 1% spread = 0 score

            # V3: Boost score for high 0DTE depth
            depth_boost = min(0.2, zero_dte_depth / 1000.0)

            liquidity_score = volume_score * 0.6 + spread_score * 0.3 + depth_boost * 0.1

            # V3: Penalize for gamma squeeze risk
            if gamma_squeeze_risk:
                liquidity_score *= 0.7

            return LiquiditySnapshot(
                timestamp=timestamp,
                symbol=symbol,
                liquidity_score=liquidity_score,
                bid_ask_spread=spread_pct,
                volume=avg_volume,
                depth=depth,
                impact_cost=impact_cost,
            )

        except Exception as e:
            logger.error(f"Error in LiquidityEngineV3 for {symbol}: {e}")
            return LiquiditySnapshot(
                timestamp=timestamp,
                symbol=symbol,
            )

    def _calculate_0dte_depth(self, symbol: str, timestamp: datetime) -> float:
        """
        Calculate 0DTE options depth for intraday scalping.

        Returns:
            Total open interest for 0DTE options
        """
        try:
            chain = self.options_adapter.get_chain(symbol, timestamp)
            if not chain:
                return 0.0

            # Filter for 0DTE (expiring today)
            zero_dte_contracts = [
                c for c in chain if (c.expiration.date() - timestamp.date()).days == 0
            ]

            if not zero_dte_contracts:
                return 0.0

            # Sum open interest
            total_oi = sum(c.open_interest for c in zero_dte_contracts)
            return float(total_oi)

        except Exception as e:
            logger.debug(f"Error calculating 0DTE depth for {symbol}: {e}")
            return 0.0

    def _detect_gamma_squeeze_risk(
        self, symbol: str, timestamp: datetime, avg_volume: float
    ) -> bool:
        """
        Detect gamma squeeze risk.

        High short interest + low float + high volume = squeeze risk

        Returns:
            True if gamma squeeze risk detected
        """
        try:
            # Get current volume
            bars = self.market_adapter.get_bars(
                symbol, timestamp - timedelta(days=1), timestamp, timeframe="1Day"
            )

            if not bars:
                return False

            current_volume = bars[-1].volume

            # Simple heuristic: Volume spike > 3x average
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0.0

            if volume_ratio > 3.0:
                logger.info(
                    f"Gamma squeeze risk detected for {symbol}: volume ratio {volume_ratio:.2f}"
                )
                return True

            return False

        except Exception as e:
            logger.debug(f"Error detecting gamma squeeze for {symbol}: {e}")
            return False
