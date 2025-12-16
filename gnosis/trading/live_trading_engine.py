"""Live trading engine that connects ML agent signals to Alpaca execution."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from adapters.alpaca_market_adapter import AlpacaMarketDataAdapter
from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
from gnosis.trading.ml_forecasting_agent import AgentSignal, MLForecastingAgent


class GnosisLiveTradingEngine:
    """Run periodic live trading cycles using Alpaca and the ML agent ensemble."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.symbols = config.get("symbols", ["SPY"])
        self.timeframe = config.get("timeframe", "1Hour")
        self.lookback_bars = int(config.get("lookback_bars", 150))
        self.trading_interval = int(config.get("trading_interval", 3600))
        self.max_position_value = float(config.get("max_position_value", 10_000))
        self.paper = bool(config.get("paper", True))

        self.market_data_adapter = AlpacaMarketDataAdapter()
        self.broker = AlpacaBrokerAdapter(paper=self.paper)
        self.agent = MLForecastingAgent("ml_live_agent", config.get("agent_config", {}))

        self.running = False
        self.trading_thread: Optional[threading.Thread] = None
        self.last_signal: Dict[str, Dict[str, Any]] = {}
        self.performance_log: list[Dict[str, Any]] = []

    def start(self) -> None:
        """Start background trading loop."""
        if self.running:
            self.logger.warning("Trading engine already running")
            return

        self.running = True
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        self.logger.info("🚀 GNOSIS Live Trading Engine Started")
        self.logger.info("Symbols: %s", ", ".join(self.symbols))
        self.logger.info("Interval: %ss", self.trading_interval)

    def stop(self) -> None:
        """Stop background trading loop."""
        self.running = False
        if self.trading_thread:
            self.trading_thread.join(timeout=10)
        self.logger.info("GNOSIS Live Trading Engine Stopped")

    def _trading_loop(self) -> None:
        """Run the trading loop until stopped."""
        while self.running:
            try:
                account = self.broker.get_account()
                for symbol in self.symbols:
                    self._trade_symbol(symbol, account)
                time.sleep(self.trading_interval)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("Error in trading loop: %s", exc)
                time.sleep(60)

    def _trade_symbol(self, symbol: str, account) -> None:
        self.logger.info("\n%s", "=" * 60)
        self.logger.info("Analyzing %s", symbol)
        historical = self._get_historical_data(symbol)
        if historical is None or historical.empty:
            self.logger.warning("No historical data for %s", symbol)
            return

        current_price = float(historical["close"].iloc[-1])
        features = self._calculate_features(historical)
        if features is None:
            self.logger.warning("Insufficient data to build features for %s", symbol)
            return

            market_snapshot = {
                "timestamp": datetime.utcnow(),
                "symbol": symbol,
                "close": current_price,
                "price": current_price,
                "volume": float(historical["volume"].iloc[-1]),
                "account_balance": float(account.equity),
                "portfolio_value": float(account.portfolio_value),
                "volatility": features.get("volatility", 0.02),
            }

        signal = self.agent.analyze(market_snapshot, features)
        self.logger.info(
            "Signal: %s (confidence=%.2f)", signal.signal_type.upper(), signal.confidence
        )
        if signal.reasoning:
            self.logger.info("Reasoning: %s", signal.reasoning)

        position = self.broker.get_position(symbol)
        self._execute_signal(symbol, signal, position, current_price)

        self.last_signal[symbol] = {
            "timestamp": datetime.utcnow(),
            "signal": signal,
            "price": current_price,
        }

    def _get_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        end = datetime.utcnow()
        start = end - self._timeframe_to_timedelta(self.lookback_bars)
        bars = self.market_data_adapter.get_bars(symbol, start, end, self.timeframe)
        if not bars:
            return None

        df = pd.DataFrame(
            [
                {
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
                for bar in bars
            ]
        )
        return df

    def _calculate_features(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if len(data) < 60:
            return None

        closes = data["close"]
        volumes = data["volume"]
        returns = closes.pct_change()

        volatility = returns.rolling(window=20).std().iloc[-1]
        rsi = self._compute_rsi(closes, period=14)

        ema_short = closes.ewm(span=12, adjust=False).mean()
        ema_long = closes.ewm(span=26, adjust=False).mean()
        macd_line = ema_short - ema_long

        middle = closes.rolling(window=20).mean()
        std = closes.rolling(window=20).std()
        upper = middle + 2 * std
        lower = middle - 2 * std
        bollinger_width = ((upper - lower) / middle).iloc[-1]

        return {
            "close": float(closes.iloc[-1]),
            "volume": float(volumes.iloc[-1]),
            "volatility": float(volatility if np.isfinite(volatility) else 0.0),
            "rsi": float(rsi if np.isfinite(rsi) else 50.0),
            "macd": float(macd_line.iloc[-1]),
            "bollinger_width": float(bollinger_width if np.isfinite(bollinger_width) else 0.0),
        }

    def _execute_signal(
        self,
        symbol: str,
        signal: AgentSignal,
        current_position,
        current_price: float,
    ) -> None:
        if signal.signal_type == "hold":
            self.logger.info("Action: HOLD")
            return

        target_value = min(signal.position_size * current_price, self.max_position_value)
        quantity = max(int(target_value / current_price), 0)

        if quantity <= 0:
            self.logger.warning("Quantity rounded to zero for %s; skipping", symbol)
            return

        if signal.signal_type == "buy":
            if current_position:
                self.logger.info("Existing position in %s detected; skipping buy", symbol)
                return
            order_id = self.broker.place_order(symbol, quantity, "buy")
            if order_id:
                self.logger.info("✅ Buy order submitted: %s", order_id)
                self._log_trade(symbol, "buy", quantity, current_price, signal)
        elif signal.signal_type == "sell":
            if not current_position:
                self.logger.info("No open position in %s to sell", symbol)
                return
            if self.broker.close_position(symbol):
                self.logger.info("✅ Position in %s closed", symbol)
                self._log_trade(symbol, "sell", float(current_position.quantity), current_price, signal)

    def _log_trade(self, symbol: str, side: str, qty: float, price: float, signal: AgentSignal) -> None:
        self.performance_log.append(
            {
                "timestamp": datetime.utcnow(),
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "signal_confidence": signal.confidence,
                "signal_reasoning": signal.reasoning,
            }
        )

    def _timeframe_to_timedelta(self, periods: int) -> timedelta:
        mapping = {
            "1Min": timedelta(minutes=periods),
            "5Min": timedelta(minutes=5 * periods),
            "15Min": timedelta(minutes=15 * periods),
            "30Min": timedelta(minutes=30 * periods),
            "1Hour": timedelta(hours=periods),
            "4Hour": timedelta(hours=4 * periods),
            "1Day": timedelta(days=periods),
        }
        return mapping.get(self.timeframe, timedelta(days=periods))

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> float:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0

    def get_performance_summary(self) -> Dict[str, Any]:
        positions = self.broker.get_positions()
        return {
            "positions": positions,
            "total_trades": len(self.performance_log),
            "last_signal": self.last_signal,
        }

    def emergency_stop(self) -> None:
        self.logger.warning("🚨 EMERGENCY STOP - Closing positions")
        self.broker.close_all_positions()
        self.stop()

__all__ = ["GnosisLiveTradingEngine"]
