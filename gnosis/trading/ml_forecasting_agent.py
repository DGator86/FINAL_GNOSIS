"""
Machine learning forecasting agent for the GNOSIS trading system.

This agent aggregates forecasts from multiple model families (LSTM,
Transformer, and XGBoost) and produces trading signals with confidence
scores that reflect model agreement and predictive uncertainty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from models.ensemble.xgboost_model import XGBoostEnsemble
from models.time_series.gnosis_lstm_forecaster import GnosisLSTMForecaster
from models.time_series.transformer_forecaster import TransformerForecaster


@dataclass
class AgentSignal:
    """Structured trading signal produced by an agent."""

    agent_id: str
    timestamp: datetime
    signal_type: str
    confidence: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """Lightweight base agent with risk helpers and logging."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.logger = logging.getLogger(agent_id)
        self.state: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        self.min_confidence: float = config.get("min_confidence", 0.5)

    def validate_features(self, features: Dict[str, Any], required: List[str]) -> bool:
        """Check required features are present and non-null."""

        missing = [name for name in required if name not in features or features[name] is None]
        if missing:
            self.logger.warning("Missing required features: %s", ", ".join(missing))
            return False
        return True

    def calculate_position_size(self, confidence: float, price: float, balance: float) -> float:
        """Risk-based position sizing with confidence weighting."""

        risk_per_trade = float(self.config.get("risk_per_trade", 0.01))
        if price <= 0:
            return 0.0

        raw_size = balance * risk_per_trade / price
        return max(0.0, raw_size * confidence)

    def calculate_stop_loss(self, price: float, signal: str, volatility: float) -> Optional[float]:
        """Simple volatility-aware stop-loss."""

        buffer = max(volatility * 2, 0.005)
        if signal == "buy":
            return price * (1 - buffer)
        if signal == "sell":
            return price * (1 + buffer)
        return None

    def calculate_take_profit(
        self, price: float, signal: str, confidence: float, volatility: float
    ) -> Optional[float]:
        """Confidence-scaled take-profit target."""

        multiplier = max(1.0, confidence * 2)
        buffer = max(volatility * multiplier, 0.01)
        if signal == "buy":
            return price * (1 + buffer)
        if signal == "sell":
            return price * (1 - buffer)
        return None

    def log_signal(self, signal: AgentSignal) -> None:
        self.logger.info(
            "Signal %s | type=%s confidence=%.2f target=%s",
            signal.agent_id,
            signal.signal_type,
            signal.confidence,
            signal.target_price,
        )


class MLForecastingAgent(BaseAgent):
    """Agent that uses ML models for price forecasting."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)

        self.model_type = config.get("model_type", "lstm")
        self.model_path = config.get("model_path")
        self.ensemble_models = config.get("ensemble_models", [])

        self.forecast_horizons = config.get("forecast_horizons", [1, 5, 15])
        self.uncertainty_threshold = config.get("uncertainty_threshold", 0.1)

        self.required_features = ["close", "volume", "volatility", "rsi", "macd", "bollinger_width"]

        self.models = self._load_models()

    def _load_models(self) -> Dict[str, Any]:
        """Load configured ML models while handling missing loaders gracefully."""

        models: Dict[str, Any] = {}

        if self.model_path:
            model = self._initialize_model(self.model_type, self.config.get("model_config", {}))
            self._maybe_load_model(model, self.model_path)
            models["primary"] = model

        for i, model_path in enumerate(self.ensemble_models):
            ensemble_configs = self.config.get("ensemble_configs", [])
            model_config = ensemble_configs[i] if i < len(ensemble_configs) else {}
            model_type = model_config.get("type", "lstm")
            model = self._initialize_model(model_type, model_config)
            self._maybe_load_model(model, model_path)
            models[f"ensemble_{i}"] = model

        self.logger.info("Loaded %d model(s) for forecasting", len(models))
        return models

    def _initialize_model(self, model_type: str, model_config: Dict[str, Any]) -> Any:
        if model_type == "lstm":
            return GnosisLSTMForecaster(model_config)
        if model_type == "transformer":
            return TransformerForecaster(**model_config)
        if model_type == "xgboost":
            return XGBoostEnsemble()
        raise ValueError(f"Unsupported model type: {model_type}")

    def _maybe_load_model(self, model: Any, model_path: str) -> None:
        if hasattr(model, "load_model"):
            try:
                model.load_model(model_path)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("Model load failed from %s: %s", model_path, exc)
        else:
            self.logger.info("Model %s has no load_model method; skipping load", model)

    def analyze(self, market_data: Dict[str, Any], features: Dict[str, Any]) -> AgentSignal:
        """Analyze market conditions using ML forecasts and emit a trading signal."""

        if not self.validate_features(features, self.required_features):
            return self._create_hold_signal("Missing required features")

        feature_vector = self._prepare_features(features)
        forecasts = self._generate_forecasts(feature_vector)
        aggregated_forecast = self._aggregate_forecasts(forecasts)
        confidence = self._calculate_confidence(forecasts, aggregated_forecast)

        if confidence < self.min_confidence:
            return self._create_hold_signal(
                f"Low confidence: {confidence:.2f} < {self.min_confidence}"
            )

        current_price = market_data.get("close", market_data.get("price"))
        signal = self._generate_signal(
            aggregated_forecast, current_price, confidence, market_data
        )

        self.log_signal(signal)
        return signal

    def _prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        values: List[float] = []
        for feature_name in self.required_features:
            value = features.get(feature_name, 0)
            if isinstance(value, (pd.Series, np.ndarray)):
                value = value[-1] if len(value) > 0 else 0
            values.append(float(value))
        return np.array(values).reshape(1, -1)

    def _generate_forecasts(self, feature_vector: np.ndarray) -> Dict[str, Dict[str, Any]]:
        forecasts: Dict[str, Dict[str, Any]] = {}
        for model_name, model in self.models.items():
            try:
                if hasattr(model, "predict"):
                    predictions = model.predict(feature_vector, return_uncertainty=True)
                    forecasts[model_name] = predictions if isinstance(predictions, dict) else {}
                else:
                    self.logger.warning("Model %s does not support predict", model_name)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.error("Model %s prediction failed: %s", model_name, exc)
        return forecasts

    def _aggregate_forecasts(self, forecasts: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        aggregated: Dict[str, float] = {}
        horizon_predictions: Dict[str, List[float]] = {}

        for forecast in forecasts.values():
            if "predictions" not in forecast:
                continue
            for horizon, preds in forecast["predictions"].items():
                if len(preds) == 0:
                    continue
                horizon_predictions.setdefault(horizon, []).append(preds[0])

        for horizon, preds in horizon_predictions.items():
            if preds:
                aggregated[horizon] = float(np.mean(preds))

        return aggregated

    def _calculate_confidence(
        self, forecasts: Dict[str, Dict[str, Any]], aggregated: Dict[str, float]
    ) -> float:
        if not forecasts or not aggregated:
            return 0.0

        agreements: List[float] = []
        for horizon in aggregated.keys():
            horizon_preds: List[float] = []
            for forecast in forecasts.values():
                if "predictions" in forecast and horizon in forecast["predictions"]:
                    preds = forecast["predictions"][horizon]
                    if len(preds) > 0:
                        horizon_preds.append(preds[0])
            if len(horizon_preds) > 1:
                mean_pred = np.mean(horizon_preds)
                if mean_pred != 0:
                    cv = np.std(horizon_preds) / abs(mean_pred)
                    agreements.append(1 / (1 + cv))

        uncertainties: List[float] = []
        for forecast in forecasts.values():
            if "uncertainties" in forecast:
                for uncs in forecast["uncertainties"].values():
                    if len(uncs) > 0:
                        uncertainties.append(uncs[0])

        agreement_score = float(np.mean(agreements)) if agreements else 0.5
        if uncertainties:
            avg_uncertainty = float(np.mean(uncertainties))
            uncertainty_score = 1 / (1 + avg_uncertainty)
        else:
            uncertainty_score = 0.5

        confidence = 0.6 * agreement_score + 0.4 * uncertainty_score
        return float(np.clip(confidence, 0, 1))

    def _generate_signal(
        self,
        forecast: Dict[str, float],
        current_price: float,
        confidence: float,
        market_data: Dict[str, Any],
    ) -> AgentSignal:
        if not forecast:
            return self._create_hold_signal("No valid forecasts")

        horizon_key = min(forecast.keys())
        predicted_price = forecast[horizon_key]
        price_change = (predicted_price - current_price) / current_price

        if price_change > 0.01:
            signal_type = "buy"
        elif price_change < -0.01:
            signal_type = "sell"
        else:
            signal_type = "hold"

        volatility = market_data.get("volatility", 0.02)
        account_balance = market_data.get("account_balance", 100000)

        position_size = self.calculate_position_size(confidence, current_price, account_balance)
        stop_loss = self.calculate_stop_loss(current_price, signal_type, volatility)
        take_profit = self.calculate_take_profit(current_price, signal_type, confidence, volatility)

        return AgentSignal(
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            signal_type=signal_type,
            confidence=confidence,
            target_price=predicted_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            reasoning=f"ML forecast predicts {price_change:.2%} change",
            metadata={
                "forecast": forecast,
                "current_price": current_price,
                "predicted_price": predicted_price,
            },
        )

    def _create_hold_signal(self, reason: str) -> AgentSignal:
        return AgentSignal(
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            signal_type="hold",
            confidence=1.0,
            reasoning=reason,
        )

    def update_state(
        self, market_data: Dict[str, Any], execution_result: Optional[Dict[str, Any]] = None
    ) -> None:
        if execution_result:
            self._update_performance(execution_result)
        self.state["last_update"] = datetime.now().isoformat()
        self.state["last_price"] = market_data.get("close", market_data.get("price"))

    def _update_performance(self, execution_result: Dict[str, Any]) -> None:
        if "profit" not in execution_result:
            return

        metrics = self.performance_metrics
        metrics.setdefault("total_profit", 0)
        metrics.setdefault("trades", 0)
        metrics.setdefault("winning_trades", 0)

        profit = execution_result["profit"]
        metrics["total_profit"] += profit
        metrics["trades"] += 1
        if profit > 0:
            metrics["winning_trades"] += 1
        metrics["win_rate"] = metrics["winning_trades"] / metrics["trades"]
