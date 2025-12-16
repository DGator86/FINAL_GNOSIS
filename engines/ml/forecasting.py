"""Adapters for time-series forecasting inspired by Facebook Research Kats."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression

from adapters.market_data_adapter import MarketDataAdapter


class KatsForecasterAdapter:
    """Lightweight wrapper that prefers Kats models but falls back gracefully.

    The adapter mirrors the Kats interface in spirit but avoids strict dependency
    on the library so the system can run even when optional packages are absent.
    """

    def __init__(
        self,
        market_adapter: MarketDataAdapter,
        forecast_horizon: int = 5,
        min_history: int = 30,
    ) -> None:
        self.market_adapter = market_adapter
        self.forecast_horizon = forecast_horizon
        self.min_history = min_history
        self._kats_available = False
        self._kats_classes: Dict[str, Any] = {}
        self._maybe_import_kats()

    def _maybe_import_kats(self) -> None:
        """Attempt to import Kats components if installed."""
        try:
            from kats.consts import TimeSeriesData
            from kats.models.prophet import ProphetModel

            self._kats_available = True
            self._kats_classes = {
                "TimeSeriesData": TimeSeriesData,
                "ProphetModel": ProphetModel,
            }
            logger.info("Kats detected; ProphetModel will be used for forecasts")
        except Exception:
            logger.info("Kats not available; using linear regression fallback")
            self._kats_available = False
            self._kats_classes = {}

    def forecast(
        self,
        symbol: str,
        end_time: datetime,
        target: str = "close",
        horizon: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Produce a short-horizon forecast for the symbol.

        Args:
            symbol: Target symbol.
            end_time: Time of the latest bar to include.
            target: Which OHLCV field to forecast.
            horizon: Optional override for forecast horizon.
        """

        horizon = horizon or self.forecast_horizon
        start = end_time - timedelta(days=max(self.min_history, horizon * 3))
        bars = self.market_adapter.get_bars(symbol, start, end_time, timeframe="1Day")
        if not bars:
            logger.warning(f"No history available for {symbol}; skipping forecast")
            return {
                "model": "unavailable",
                "horizon": horizon,
                "forecast": [],
                "confidence": 0.0,
                "metadata": {"reason": "missing_history"},
            }

        df = pd.DataFrame([bar.model_dump() for bar in bars])
        df.sort_values("timestamp", inplace=True)
        if target not in df.columns:
            logger.warning(f"Target {target} not in OHLCV columns; defaulting to close")
            target = "close"
        series = df[target].astype(float)

        if len(series) < self.min_history:
            logger.warning(
                f"Insufficient history for Kats models ({len(series)}/{self.min_history}); "
                "using fallback",
            )
            use_kats = False
        else:
            use_kats = self._kats_available

        if use_kats:
            forecast_values, confidence = self._forecast_with_kats(df, target, horizon)
            model_name = "kats_prophet"
        else:
            forecast_values, confidence = self._forecast_with_linear_regression(
                series, horizon
            )
            model_name = "linear_regression_fallback"

        return {
            "model": model_name,
            "horizon": horizon,
            "forecast": forecast_values,
            "confidence": confidence,
            "metadata": {
                "history": len(series),
                "target": target,
                "uses_kats": use_kats,
            },
        }

    def _forecast_with_kats(
        self, df: pd.DataFrame, target: str, horizon: int
    ) -> tuple[List[float], float]:
        """Use Kats ProphetModel when available."""
        try:
            kats_ts = self._kats_classes["TimeSeriesData"](
                df[["timestamp", target]].rename(columns={"timestamp": "time"})
            )
            model = self._kats_classes["ProphetModel"](kats_ts)
            model.fit()
            fcst = model.forecast(steps=horizon, freq="D")
            predictions = fcst["fcst"].astype(float).tolist()
            confidence = min(1.0, 0.7 + 0.3 * (len(df) / (self.min_history + 1)))
            return predictions, confidence
        except Exception as exc:  # pragma: no cover - optional dependency path
            logger.error(f"Kats forecast failed: {exc}; falling back to linear model")
            return self._forecast_with_linear_regression(df[target].astype(float), horizon)

    def _forecast_with_linear_regression(
        self, series: pd.Series, horizon: int
    ) -> tuple[List[float], float]:
        """Fallback forecaster using simple linear regression."""
        x = np.arange(len(series)).reshape(-1, 1)
        y = series.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        future_x = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
        preds = model.predict(future_x).flatten()
        confidence = 0.5 if len(series) < self.min_history else 0.65
        return preds.tolist(), confidence
