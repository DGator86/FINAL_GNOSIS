"""
Gnosis Alpha - Technical Analyzer

Standalone technical analysis for signal generation.
Uses yfinance for data and simple indicators for signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    import numpy as np
    YFINANCE_AVAILABLE = True
except ImportError:
    logger.warning("yfinance not available")
    YFINANCE_AVAILABLE = False


@dataclass
class TechnicalSignals:
    """Technical analysis signals for a symbol."""
    
    # Trend indicators
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    
    # Momentum
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    
    # Price levels
    current_price: Optional[float] = None
    prev_close: Optional[float] = None
    day_change_pct: float = 0.0
    
    # Volume
    volume: Optional[float] = None
    avg_volume: Optional[float] = None
    volume_ratio: float = 1.0
    
    # Volatility
    atr_14: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    
    # Signals
    trend_signal: str = "neutral"  # bullish, bearish, neutral
    momentum_signal: str = "neutral"
    volume_signal: str = "neutral"
    overall_signal: str = "neutral"
    confidence: float = 0.0
    
    # Volume thresholds
    UNUSUAL_VOLUME_THRESHOLD = 2.0   # 2x average = unusual
    HIGH_VOLUME_THRESHOLD = 3.0      # 3x average = very unusual
    
    @property
    def unusual_volume(self) -> bool:
        """Check if current volume is unusual (2x+ average)."""
        return self.volume_ratio >= self.UNUSUAL_VOLUME_THRESHOLD
    
    @property
    def very_high_volume(self) -> bool:
        """Check if current volume is very high (3x+ average)."""
        return self.volume_ratio >= self.HIGH_VOLUME_THRESHOLD
    
    @property
    def volume_description(self) -> str:
        """Human-readable volume description."""
        if self.very_high_volume:
            return f"🔥 Very High ({self.volume_ratio:.1f}x avg)"
        elif self.unusual_volume:
            return f"📈 Unusual ({self.volume_ratio:.1f}x avg)"
        elif self.volume_ratio >= 1.5:
            return f"Above avg ({self.volume_ratio:.1f}x)"
        elif self.volume_ratio <= 0.5:
            return f"Low ({self.volume_ratio:.1f}x avg)"
        return "Normal"
    
    def to_dict(self) -> dict:
        return {
            "sma_20": self.sma_20,
            "sma_50": self.sma_50,
            "sma_200": self.sma_200,
            "rsi_14": self.rsi_14,
            "macd": self.macd,
            "current_price": self.current_price,
            "day_change_pct": self.day_change_pct,
            "volume": self.volume,
            "avg_volume": self.avg_volume,
            "volume_ratio": round(self.volume_ratio, 2),
            "unusual_volume": self.unusual_volume,
            "very_high_volume": self.very_high_volume,
            "volume_description": self.volume_description,
            "trend_signal": self.trend_signal,
            "momentum_signal": self.momentum_signal,
            "volume_signal": self.volume_signal,
            "overall_signal": self.overall_signal,
            "confidence": self.confidence,
        }


class TechnicalAnalyzer:
    """
    Performs technical analysis on stocks using yfinance data.
    
    Calculates common indicators and generates signals.
    """
    
    def __init__(self, lookback_days: int = 100):
        """
        Initialize analyzer.
        
        Args:
            lookback_days: Days of historical data to analyze
        """
        self.lookback_days = lookback_days
        self._cache: Dict[str, Any] = {}
    
    def analyze(self, symbol: str) -> TechnicalSignals:
        """
        Analyze a symbol and return technical signals.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            TechnicalSignals with indicators and signals
        """
        signals = TechnicalSignals()
        
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available for analysis")
            return signals
        
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return signals
            
            # Current price info
            signals.current_price = float(df['Close'].iloc[-1])
            if len(df) > 1:
                signals.prev_close = float(df['Close'].iloc[-2])
                signals.day_change_pct = (
                    (signals.current_price - signals.prev_close) / signals.prev_close * 100
                )
            
            # Volume
            signals.volume = float(df['Volume'].iloc[-1])
            signals.avg_volume = float(df['Volume'].rolling(20).mean().iloc[-1])
            if signals.avg_volume > 0:
                signals.volume_ratio = signals.volume / signals.avg_volume
            
            # Moving averages
            if len(df) >= 20:
                signals.sma_20 = float(df['Close'].rolling(20).mean().iloc[-1])
            if len(df) >= 50:
                signals.sma_50 = float(df['Close'].rolling(50).mean().iloc[-1])
            if len(df) >= 200:
                signals.sma_200 = float(df['Close'].rolling(200).mean().iloc[-1])
            
            # EMA for MACD
            signals.ema_12 = float(df['Close'].ewm(span=12, adjust=False).mean().iloc[-1])
            signals.ema_26 = float(df['Close'].ewm(span=26, adjust=False).mean().iloc[-1])
            
            # MACD
            macd_line = df['Close'].ewm(span=12, adjust=False).mean() - \
                        df['Close'].ewm(span=26, adjust=False).mean()
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            
            signals.macd = float(macd_line.iloc[-1])
            signals.macd_signal = float(signal_line.iloc[-1])
            signals.macd_histogram = signals.macd - signals.macd_signal
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            signals.rsi_14 = float(rsi.iloc[-1])
            
            # ATR
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            tr = high_low.combine(high_close, max).combine(low_close, max)
            signals.atr_14 = float(tr.rolling(14).mean().iloc[-1])
            
            # Bollinger Bands
            sma_20 = df['Close'].rolling(20).mean()
            std_20 = df['Close'].rolling(20).std()
            signals.bollinger_upper = float((sma_20 + 2 * std_20).iloc[-1])
            signals.bollinger_lower = float((sma_20 - 2 * std_20).iloc[-1])
            
            # Generate signals
            signals = self._generate_signals(signals)
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
        
        return signals
    
    def _generate_signals(self, signals: TechnicalSignals) -> TechnicalSignals:
        """Generate buy/sell signals from technical indicators."""
        
        bullish_points = 0
        bearish_points = 0
        total_points = 0
        
        # Trend analysis
        if signals.current_price and signals.sma_20 and signals.sma_50:
            total_points += 3
            
            # Price above MAs = bullish
            if signals.current_price > signals.sma_20:
                bullish_points += 1
            else:
                bearish_points += 1
            
            if signals.current_price > signals.sma_50:
                bullish_points += 1
            else:
                bearish_points += 1
            
            # MA crossover
            if signals.sma_20 > signals.sma_50:
                bullish_points += 1
            else:
                bearish_points += 1
        
        if signals.sma_200:
            total_points += 1
            if signals.current_price > signals.sma_200:
                bullish_points += 1
            else:
                bearish_points += 1
        
        # Set trend signal
        if bullish_points > bearish_points:
            signals.trend_signal = "bullish"
        elif bearish_points > bullish_points:
            signals.trend_signal = "bearish"
        else:
            signals.trend_signal = "neutral"
        
        # Momentum analysis (RSI + MACD)
        momentum_bullish = 0
        momentum_bearish = 0
        momentum_total = 0
        
        if signals.rsi_14:
            momentum_total += 2
            if signals.rsi_14 < 30:
                momentum_bullish += 2  # Oversold = potential buy
            elif signals.rsi_14 > 70:
                momentum_bearish += 2  # Overbought = potential sell
            elif signals.rsi_14 < 50:
                momentum_bearish += 1
            else:
                momentum_bullish += 1
        
        if signals.macd and signals.macd_signal:
            momentum_total += 2
            if signals.macd > signals.macd_signal:
                momentum_bullish += 1
            else:
                momentum_bearish += 1
            
            if signals.macd_histogram and signals.macd_histogram > 0:
                momentum_bullish += 1
            else:
                momentum_bearish += 1
        
        bullish_points += momentum_bullish
        bearish_points += momentum_bearish
        total_points += momentum_total
        
        if momentum_bullish > momentum_bearish:
            signals.momentum_signal = "bullish"
        elif momentum_bearish > momentum_bullish:
            signals.momentum_signal = "bearish"
        else:
            signals.momentum_signal = "neutral"
        
        # Volume analysis
        if signals.volume_ratio:
            total_points += 1
            if signals.volume_ratio > 1.5 and signals.day_change_pct > 0:
                bullish_points += 1
                signals.volume_signal = "bullish"
            elif signals.volume_ratio > 1.5 and signals.day_change_pct < 0:
                bearish_points += 1
                signals.volume_signal = "bearish"
            else:
                signals.volume_signal = "neutral"
        
        # Overall signal
        if total_points > 0:
            bullish_pct = bullish_points / total_points
            bearish_pct = bearish_points / total_points
            
            if bullish_pct > 0.6:
                signals.overall_signal = "bullish"
                signals.confidence = bullish_pct
            elif bearish_pct > 0.6:
                signals.overall_signal = "bearish"
                signals.confidence = bearish_pct
            else:
                signals.overall_signal = "neutral"
                signals.confidence = max(bullish_pct, bearish_pct)
        
        return signals
    
    def get_support_resistance(
        self,
        symbol: str,
        lookback_days: int = 60,
    ) -> Dict[str, List[float]]:
        """
        Find support and resistance levels.
        
        Returns dict with 'support' and 'resistance' price levels.
        """
        levels = {"support": [], "resistance": []}
        
        if not YFINANCE_AVAILABLE:
            return levels
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{lookback_days}d")
            
            if df.empty:
                return levels
            
            # Find local minima (support) and maxima (resistance)
            prices = df['Close'].values
            
            for i in range(2, len(prices) - 2):
                # Local minimum = support
                if prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
                   prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                    levels["support"].append(float(prices[i]))
                
                # Local maximum = resistance
                if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
                   prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                    levels["resistance"].append(float(prices[i]))
            
            # Sort and get most recent/significant
            levels["support"] = sorted(levels["support"], reverse=True)[:3]
            levels["resistance"] = sorted(levels["resistance"])[:3]
            
        except Exception as e:
            logger.error(f"Error finding support/resistance for {symbol}: {e}")
        
        return levels
