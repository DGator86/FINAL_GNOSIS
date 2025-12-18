"""Sentiment processors for news, flow, and technical analysis."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from engines.inputs.market_data_adapter import MarketDataAdapter
from engines.inputs.news_adapter import NewsAdapter
from schemas.core_schemas import MTFAnalysis, TimeframeSignal


class NewsSentimentProcessor:
    """Processes news sentiment."""
    
    def __init__(self, news_adapter: NewsAdapter, config: Dict[str, Any]):
        self.news_adapter = news_adapter
        self.config = config
    
    def process(self, symbol: str, timestamp: datetime) -> float:
        """Calculate news sentiment score."""
        try:
            articles = self.news_adapter.get_news(
                symbol,
                timestamp - timedelta(days=7),
                timestamp
            )
            
            if not articles:
                return 0.0
            
            # Weight recent news more heavily
            total_weight = 0.0
            weighted_sentiment = 0.0
            
            for article in articles:
                age_days = (timestamp - article.timestamp).days
                weight = max(0.1, 1.0 - (age_days / 7.0))
                weighted_sentiment += article.sentiment * weight
                total_weight += weight
            
            return weighted_sentiment / total_weight if total_weight > 0 else 0.0
        
        except Exception as e:
            logger.error(f"Error in NewsSentimentProcessor: {e}")
            return 0.0


class FlowSentimentProcessor:
    """Processes order flow sentiment using Unusual Whales when available."""

    def __init__(self, config: Dict[str, Any], flow_adapter: Any | None = None):
        self.config = config
        self.flow_adapter = flow_adapter

    def process(self, symbol: str, timestamp: datetime) -> float:
        """Calculate flow sentiment score from options flow intensity and skew."""

        if not self.flow_adapter:
            return 0.0

        try:
            snapshot = self.flow_adapter.get_flow_snapshot(symbol, timestamp)
            if not snapshot:
                return 0.0

            call_pressure = snapshot.get("call_volume", 0) + snapshot.get("call_premium", 0)
            put_pressure = snapshot.get("put_volume", 0) + snapshot.get("put_premium", 0)
            total = call_pressure + put_pressure
            if total == 0:
                return 0.0

            skew = (call_pressure - put_pressure) / total
            intensity = snapshot.get("sweep_ratio", 0)
            score = (skew * 0.7) + (intensity * 0.3)
            return max(-1.0, min(1.0, score))

        except Exception as exc:
            logger.error(f"Error in FlowSentimentProcessor: {exc}")
            return 0.0


class TechnicalSentimentProcessor:
    """Processes technical analysis sentiment with multi-timeframe support."""

    # Timeframe configurations: (name, lookback_days, bar_timeframe, periods_for_avg)
    TIMEFRAMES = [
        ("1Min", 1, "1Min", 60),      # 60 1-min bars
        ("5Min", 1, "5Min", 60),      # 60 5-min bars
        ("15Min", 2, "15Min", 48),    # 48 15-min bars
        ("30Min", 3, "30Min", 48),    # 48 30-min bars
        ("1Hour", 5, "1Hour", 48),    # 48 hourly bars
        ("4Hour", 14, "4Hour", 42),   # 42 4-hour bars
        ("1Day", 30, "1Day", 20),     # 20 daily bars
    ]

    def __init__(self, market_adapter: MarketDataAdapter, config: Dict[str, Any]):
        self.market_adapter = market_adapter
        self.config = config

    def process(self, symbol: str, timestamp: datetime) -> float:
        """Calculate technical sentiment score (single timeframe, backward compatible)."""
        try:
            bars = self.market_adapter.get_bars(
                symbol,
                timestamp - timedelta(days=20),
                timestamp,
                timeframe="1Day"
            )

            if len(bars) < 2:
                return 0.0

            # Simple momentum: compare recent price to 20-day average
            recent_price = bars[-1].close
            avg_price = sum(bar.close for bar in bars) / len(bars)

            momentum = (recent_price - avg_price) / avg_price if avg_price > 0 else 0.0

            # Normalize to [-1, 1]
            return max(-1.0, min(1.0, momentum * 10))

        except Exception as e:
            logger.error(f"Error in TechnicalSentimentProcessor: {e}")
            return 0.0

    def _get_strategy_for_signal(
        self, tf_name: str, direction: str, strength: float, confidence: float, rsi: float
    ) -> tuple[str, str]:
        """
        Determine the recommended options strategy based on timeframe and signal.

        Returns (strategy_name, strategy_details) tuple.
        """
        # Map timeframes to expiry suggestions
        expiry_map = {
            "1Min": "0DTE",
            "5Min": "0DTE",
            "15Min": "0DTE/1DTE",
            "30Min": "1-3 DTE",
            "1Hour": "3-5 DTE",
            "4Hour": "1-2 weeks",
            "1Day": "2-4 weeks",
        }
        expiry = expiry_map.get(tf_name, "1-2 weeks")

        abs_strength = abs(strength)

        # BULLISH strategies
        if direction == "long":
            if confidence >= 0.7 and abs_strength >= 0.5:
                # High conviction bullish
                strategy = "Long Call"
                details = f"ATM or slightly OTM call, {expiry} expiry"
            elif confidence >= 0.5:
                # Moderate conviction bullish
                strategy = "Bull Call Spread"
                details = f"ATM/OTM call debit spread, {expiry} expiry, defined risk"
            else:
                # Low conviction bullish
                strategy = "Bull Put Spread"
                details = f"OTM put credit spread, {expiry} expiry, collect premium"

            # RSI oversold = extra bullish confirmation
            if rsi < 30:
                strategy = "Long Call"
                details = f"RSI oversold reversal play, ATM call, {expiry} expiry"

        # BEARISH strategies
        elif direction == "short":
            if confidence >= 0.7 and abs_strength >= 0.5:
                # High conviction bearish
                strategy = "Long Put"
                details = f"ATM or slightly OTM put, {expiry} expiry"
            elif confidence >= 0.5:
                # Moderate conviction bearish
                strategy = "Bear Put Spread"
                details = f"ATM/OTM put debit spread, {expiry} expiry, defined risk"
            else:
                # Low conviction bearish
                strategy = "Bear Call Spread"
                details = f"OTM call credit spread, {expiry} expiry, collect premium"

            # RSI overbought = extra bearish confirmation
            if rsi > 70:
                strategy = "Long Put"
                details = f"RSI overbought reversal play, ATM put, {expiry} expiry"

        # NEUTRAL strategies
        else:
            if confidence < 0.3:
                # Very low conviction - stay out
                strategy = "No Trade"
                details = "Insufficient signal clarity, wait for better setup"
            elif abs_strength < 0.1:
                # Truly neutral - range-bound strategies
                strategy = "Iron Condor"
                details = f"Sell OTM strangle, buy further OTM wings, {expiry} expiry"
            else:
                # Slight lean but neutral overall
                strategy = "Iron Butterfly"
                details = f"ATM short straddle with OTM wings, {expiry} expiry"

            # High RSI variance suggests potential move
            if rsi > 65 or rsi < 35:
                strategy = "Straddle"
                details = f"ATM straddle for expected breakout, {expiry} expiry"

        return strategy, details

    def analyze_mtf(self, symbol: str, timestamp: datetime) -> MTFAnalysis:
        """
        Perform multi-timeframe analysis across all configured timeframes.

        Returns MTFAnalysis with signals for each timeframe showing:
        - Direction (long/short/neutral)
        - Strength (-1 to +1)
        - Confidence (0 to 1)
        - Momentum
        - Trend description
        """
        signals: List[TimeframeSignal] = []

        for tf_name, lookback_days, bar_timeframe, periods in self.TIMEFRAMES:
            signal = self._analyze_timeframe(
                symbol, timestamp, tf_name, lookback_days, bar_timeframe, periods
            )
            if signal:
                signals.append(signal)

        # Calculate alignment score (how aligned are all timeframes)
        alignment_score = self._calculate_alignment(signals)

        # Find dominant timeframe (strongest signal)
        dominant_tf = ""
        max_strength = 0.0
        for sig in signals:
            if abs(sig.strength) > max_strength:
                max_strength = abs(sig.strength)
                dominant_tf = sig.timeframe

        # Calculate overall direction (weighted by timeframe importance)
        overall_direction, overall_confidence = self._calculate_overall_direction(signals)

        return MTFAnalysis(
            timestamp=timestamp,
            symbol=symbol,
            signals=signals,
            alignment_score=alignment_score,
            dominant_timeframe=dominant_tf,
            overall_direction=overall_direction,
            overall_confidence=overall_confidence,
        )

    def _analyze_timeframe(
        self,
        symbol: str,
        timestamp: datetime,
        tf_name: str,
        lookback_days: int,
        bar_timeframe: str,
        periods: int,
    ) -> Optional[TimeframeSignal]:
        """Analyze a single timeframe and return its signal."""
        try:
            bars = self.market_adapter.get_bars(
                symbol,
                timestamp - timedelta(days=lookback_days),
                timestamp,
                timeframe=bar_timeframe,
            )

            if len(bars) < 5:
                return TimeframeSignal(
                    timeframe=tf_name,
                    direction="neutral",
                    strength=0.0,
                    confidence=0.0,
                    momentum=0.0,
                    trend="insufficient data",
                    reasoning="Not enough bars for analysis",
                )

            # Use available bars (up to periods)
            analysis_bars = bars[-min(len(bars), periods):]

            # Calculate momentum
            recent_price = analysis_bars[-1].close
            avg_price = sum(b.close for b in analysis_bars) / len(analysis_bars)
            momentum = (recent_price - avg_price) / avg_price if avg_price > 0 else 0.0

            # Calculate short-term vs long-term trend
            if len(analysis_bars) >= 10:
                short_avg = sum(b.close for b in analysis_bars[-5:]) / 5
                long_avg = sum(b.close for b in analysis_bars[-10:]) / 10
                trend_strength = (short_avg - long_avg) / long_avg if long_avg > 0 else 0.0
            else:
                trend_strength = momentum

            # Calculate RSI-like overbought/oversold
            gains = []
            losses = []
            for i in range(1, len(analysis_bars)):
                change = analysis_bars[i].close - analysis_bars[i-1].close
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            avg_gain = sum(gains[-14:]) / min(14, len(gains)) if gains else 0
            avg_loss = sum(losses[-14:]) / min(14, len(losses)) if losses else 0.001
            rs = avg_gain / avg_loss if avg_loss > 0 else 1
            rsi = 100 - (100 / (1 + rs))

            # Determine direction and strength
            strength = max(-1.0, min(1.0, momentum * 10 + trend_strength * 5))

            if strength > 0.3:
                direction = "long"
            elif strength < -0.3:
                direction = "short"
            else:
                direction = "neutral"

            # Trend description
            if trend_strength > 0.02:
                trend = "bullish"
            elif trend_strength < -0.02:
                trend = "bearish"
            else:
                trend = "neutral"

            # Confidence based on consistency and RSI extremes
            confidence = min(1.0, abs(strength) * 0.8 + 0.2)
            if rsi > 70 or rsi < 30:
                confidence *= 0.8  # Less confident at extremes (potential reversal)

            # Build reasoning
            reasons = []
            if momentum > 0.01:
                reasons.append(f"price above {len(analysis_bars)}-period avg by {momentum:.1%}")
            elif momentum < -0.01:
                reasons.append(f"price below {len(analysis_bars)}-period avg by {abs(momentum):.1%}")

            if trend == "bullish":
                reasons.append("short-term trend rising")
            elif trend == "bearish":
                reasons.append("short-term trend falling")

            if rsi > 70:
                reasons.append(f"RSI overbought ({rsi:.0f})")
            elif rsi < 30:
                reasons.append(f"RSI oversold ({rsi:.0f})")

            reasoning = "; ".join(reasons) if reasons else "No strong signals"

            # Get recommended options strategy
            strategy, strategy_details = self._get_strategy_for_signal(
                tf_name, direction, strength, confidence, rsi
            )

            return TimeframeSignal(
                timeframe=tf_name,
                direction=direction,
                strength=round(strength, 3),
                confidence=round(confidence, 3),
                momentum=round(momentum, 4),
                trend=trend,
                reasoning=reasoning,
                strategy=strategy,
                strategy_details=strategy_details,
            )

        except Exception as e:
            logger.debug(f"MTF analysis error for {tf_name}: {e}")
            return TimeframeSignal(
                timeframe=tf_name,
                direction="neutral",
                strength=0.0,
                confidence=0.0,
                momentum=0.0,
                trend="error",
                reasoning=str(e)[:100],
                strategy="No Trade",
                strategy_details="Error in analysis",
            )

    def _calculate_alignment(self, signals: List[TimeframeSignal]) -> float:
        """
        Calculate how aligned timeframes are (0 = conflicting, 1 = all agree).
        """
        if not signals:
            return 0.0

        # Count directions
        long_count = sum(1 for s in signals if s.direction == "long")
        short_count = sum(1 for s in signals if s.direction == "short")
        neutral_count = sum(1 for s in signals if s.direction == "neutral")
        total = len(signals)

        # Perfect alignment = all same direction (excluding neutral)
        if long_count == total or short_count == total:
            return 1.0

        # Mostly aligned
        max_count = max(long_count, short_count)
        if max_count > 0:
            # Alignment weighted by how many agree vs disagree
            opposing = short_count if long_count > short_count else long_count
            alignment = (max_count - opposing) / total
            # Reduce alignment if too many neutrals
            alignment *= (1 - neutral_count / total * 0.3)
            return round(max(0, alignment), 3)

        # All neutral
        return 0.5

    def _calculate_overall_direction(
        self, signals: List[TimeframeSignal]
    ) -> tuple[str, float]:
        """
        Calculate overall direction with higher timeframes weighted more heavily.

        Weights: 1Min=1, 5Min=2, 15Min=3, 30Min=4, 1Hour=5, 4Hour=6, 1Day=7
        """
        if not signals:
            return "neutral", 0.0

        tf_weights = {
            "1Min": 1.0,
            "5Min": 2.0,
            "15Min": 3.0,
            "30Min": 4.0,
            "1Hour": 5.0,
            "4Hour": 6.0,
            "1Day": 7.0,
        }

        weighted_strength = 0.0
        total_weight = 0.0
        weighted_confidence = 0.0

        for sig in signals:
            weight = tf_weights.get(sig.timeframe, 1.0)
            weighted_strength += sig.strength * weight
            weighted_confidence += sig.confidence * weight
            total_weight += weight

        if total_weight == 0:
            return "neutral", 0.0

        avg_strength = weighted_strength / total_weight
        avg_confidence = weighted_confidence / total_weight

        if avg_strength > 0.15:
            direction = "long"
        elif avg_strength < -0.15:
            direction = "short"
        else:
            direction = "neutral"

        return direction, round(avg_confidence, 3)
