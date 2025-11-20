"""
Regime Detection Agent for GNOSIS Trading System
Detects market regimes and adjusts strategy accordingly.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .base_agent import BaseAgent, AgentSignal


class RegimeDetectionAgent(BaseAgent):
    """Agent that detects and adapts to market regimes"""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)

        # Regime configuration
        self.n_regimes = config.get("n_regimes", 4)
        self.lookback_period = config.get("lookback_period", 100)
        self.regime_features = config.get(
            "regime_features", ["returns", "volatility", "volume", "trend_strength"]
        )

        # Regime models
        self.regime_model = GaussianMixture(
            n_components=self.n_regimes, covariance_type="full", random_state=42
        )
        self.scaler = StandardScaler()

        # Regime state
        self.current_regime = None
        self.regime_history = []
        self.regime_probabilities = None

        # Strategy parameters per regime
        self.regime_strategies = self._initialize_regime_strategies()

        # Required features
        self.required_features = self.regime_features

    def _initialize_regime_strategies(self) -> Dict[int, Dict[str, Any]]:
        """Initialize strategy parameters for each regime"""
        strategies = {}

        # Regime 0: Bull market (high vol, positive trend)
        strategies[0] = {
            "name": "bull_volatile",
            "position_bias": 0.8,  # Bullish bias
            "max_position": 0.15,
            "stop_loss_multiplier": 1.5,
            "take_profit_multiplier": 2.5,
        }

        # Regime 1: Bear market (high vol, negative trend)
        strategies[1] = {
            "name": "bear_volatile",
            "position_bias": 0.2,  # Bearish bias
            "max_position": 0.08,
            "stop_loss_multiplier": 1.2,
            "take_profit_multiplier": 1.8,
        }

        # Regime 2: Bull market (low vol, positive trend)
        strategies[2] = {
            "name": "bull_stable",
            "position_bias": 0.7,
            "max_position": 0.12,
            "stop_loss_multiplier": 1.0,
            "take_profit_multiplier": 2.0,
        }

        # Regime 3: Range-bound (low vol, weak trend)
        strategies[3] = {
            "name": "range_bound",
            "position_bias": 0.5,
            "max_position": 0.10,
            "stop_loss_multiplier": 0.8,
            "take_profit_multiplier": 1.5,
        }

        return strategies

    def fit_regime_model(self, historical_data: pd.DataFrame):
        """Fit regime detection model on historical data"""

        # Extract regime features
        features = self._extract_regime_features(historical_data)

        if len(features) < self.n_regimes * 10:
            self.logger.warning("Insufficient data for regime fitting")
            return

        # Fit model
        features_scaled = self.scaler.fit_transform(features)
        self.regime_model.fit(features_scaled)

        self.logger.info(f"Fitted regime model with {self.n_regimes} regimes")

    def _extract_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime detection"""
        features = []

        # Returns
        if "close" in data.columns:
            returns = data["close"].pct_change()
            features.append(returns.values)

        # Volatility (rolling std of returns)
        if "close" in data.columns:
            volatility = returns.rolling(20).std()
            features.append(volatility.values)

        # Volume
        if "volume" in data.columns:
            volume_change = data["volume"].pct_change()
            features.append(volume_change.values)

        # Trend strength (moving average slope)
        if "close" in data.columns:
            ma_20 = data["close"].rolling(20).mean()
            ma_50 = data["close"].rolling(50).mean()
            trend = (ma_20 - ma_50) / ma_50
            features.append(trend.values)

        # Stack features
        features_array = np.column_stack(features)

        # Remove NaN rows
        features_array = features_array[~np.isnan(features_array).any(axis=1)]

        return features_array

    def detect_regime(self, market_data: Dict[str, Any], features: Dict[str, Any]) -> int:
        """Detect current market regime"""

        # Prepare feature vector
        feature_vector = []

        for feature_name in self.regime_features:
            value = features.get(feature_name, 0)

            if isinstance(value, (pd.Series, np.ndarray)):
                value = value[-1] if len(value) > 0 else 0

            feature_vector.append(value)

        feature_vector = np.array(feature_vector).reshape(1, -1)

        # Scale features
        feature_scaled = self.scaler.transform(feature_vector)

        # Predict regime
        regime = self.regime_model.predict(feature_scaled)[0]
        self.regime_probabilities = self.regime_model.predict_proba(feature_scaled)[0]

        return regime

    def analyze(self, market_data: Dict[str, Any], features: Dict[str, Any]) -> AgentSignal:
        """
        Analyze market regime and generate signal

        Args:
            market_data: Current market data
            features: Computed features

        Returns:
            Trading signal adapted to current regime
        """

        if not self.validate_features(features):
            return self._create_hold_signal("Missing required features")

        # Detect current regime
        regime = self.detect_regime(market_data, features)
        self.current_regime = regime
        self.regime_history.append(
            {
                "timestamp": datetime.now(),
                "regime": regime,
                "probabilities": self.regime_probabilities.tolist(),
            }
        )

        # Get regime strategy
        strategy = self.regime_strategies[regime]

        # Calculate regime confidence (max probability)
        regime_confidence = float(np.max(self.regime_probabilities))

        if regime_confidence < 0.5:
            return self._create_hold_signal(f"Low regime confidence: {regime_confidence:.2f}")

        # Generate signal based on regime
        signal = self._generate_regime_signal(
            market_data, features, regime, strategy, regime_confidence
        )

        self.log_signal(signal)
        return signal

    def _generate_regime_signal(
        self,
        market_data: Dict[str, Any],
        features: Dict[str, Any],
        regime: int,
        strategy: Dict[str, Any],
        confidence: float,
    ) -> AgentSignal:
        """Generate signal based on detected regime"""

        current_price = market_data.get("close", market_data.get("price"))

        # Get price momentum
        returns = features.get("returns", 0)
        if isinstance(returns, (pd.Series, np.ndarray)):
            returns = returns[-1] if len(returns) > 0 else 0

        # Determine signal type based on regime bias and momentum
        position_bias = strategy["position_bias"]

        if returns > 0.005 and position_bias > 0.5:  # Bullish regime + positive momentum
            signal_type = "buy"
        elif returns < -0.005 and position_bias < 0.5:  # Bearish regime + negative momentum
            signal_type = "sell"
        else:
            signal_type = "hold"

        # Calculate position size based on regime
        volatility = features.get("volatility", 0.02)
        if isinstance(volatility, (pd.Series, np.ndarray)):
            volatility = volatility[-1] if len(volatility) > 0 else 0.02

        account_balance = market_data.get("account_balance", 100000)

        # Regime-adjusted position size
        base_position = self.calculate_position_size(confidence, current_price, account_balance)
        position_size = min(base_position, strategy["max_position"])

        # Regime-adjusted risk parameters
        stop_loss = self.calculate_stop_loss(current_price, signal_type, volatility)
        stop_loss_adjusted = current_price - ((current_price - stop_loss) * strategy["stop_loss_multiplier"])

        take_profit = self.calculate_take_profit(current_price, signal_type, confidence, volatility)
        take_profit_adjusted = current_price + ((take_profit - current_price) * strategy["take_profit_multiplier"])

        # Create signal
        signal = AgentSignal(
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            signal_type=signal_type,
            confidence=confidence,
            target_price=None,
            stop_loss=stop_loss_adjusted,
            take_profit=take_profit_adjusted,
            position_size=position_size,
            reasoning=f"Regime {regime} ({strategy['name']}): bias={position_bias:.2f}",
            metadata={
                "regime": regime,
                "regime_name": strategy["name"],
                "regime_confidence": float(confidence),
                "regime_probabilities": self.regime_probabilities.tolist(),
            },
        )

        return signal

    def _create_hold_signal(self, reason: str) -> AgentSignal:
        """Create hold signal"""
        return AgentSignal(
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            signal_type="hold",
            confidence=1.0,
            reasoning=reason,
        )

    def update_state(self, market_data: Dict[str, Any], execution_result: Optional[Dict[str, Any]] = None):
        """Update agent state"""

        self.state["current_regime"] = self.current_regime
        self.state["regime_probabilities"] = (
            self.regime_probabilities.tolist() if self.regime_probabilities is not None else None
        )
        self.state["last_update"] = datetime.now().isoformat()

        # Update performance metrics per regime
        if execution_result and self.current_regime is not None:
            regime_key = f"regime_{self.current_regime}"

            if regime_key not in self.performance_metrics:
                self.performance_metrics[regime_key] = {"trades": 0, "profit": 0, "wins": 0}

            if "profit" in execution_result:
                profit = execution_result["profit"]
                self.performance_metrics[regime_key]["trades"] += 1
                self.performance_metrics[regime_key]["profit"] += profit

                if profit > 0:
                    self.performance_metrics[regime_key]["wins"] += 1

    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of regime detection"""

        if not self.regime_history:
            return {"no_data": True}

        # Calculate regime distribution
        regime_counts = {}
        for entry in self.regime_history:
            regime = entry["regime"]
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        total = len(self.regime_history)
        regime_distribution = {regime: count / total for regime, count in regime_counts.items()}

        return {
            "current_regime": self.current_regime,
            "current_regime_name": (self.regime_strategies[self.current_regime]["name"] if self.current_regime is not None else None),
            "regime_probabilities": (
                self.regime_probabilities.tolist() if self.regime_probabilities is not None else None
            ),
            "regime_distribution": regime_distribution,
            "total_observations": total,
            "performance_by_regime": self.performance_metrics,
        }
