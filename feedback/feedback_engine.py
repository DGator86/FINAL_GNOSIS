"""Feedback Engine v2 leveraging RL and anomaly detection."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import IsolationForest

try:  # pragma: no cover - optional
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import DummyVecEnv
except Exception:  # pragma: no cover
    PPO = None
    DummyVecEnv = None


class FeedbackEnv:
    """Minimal gym-like environment wrapping ledger metrics."""

    def __init__(self, metrics: Dict[str, float]):
        self.metrics = metrics
        self.action_space = type("space", (), {"shape": (1,), "sample": lambda self: np.array([0.0])})()
        self.observation_space = self.action_space

    def reset(self):
        return np.array(list(self.metrics.values()), dtype=float)

    def step(self, action):
        reward = float(self.metrics.get("sharpe", 0.0) - abs(action[0]))
        obs = np.array(list(self.metrics.values()), dtype=float)
        done = True
        return obs, reward, done, {}


class FeedbackEngine:
    """Adjusts parameters based on ledger metrics using PPO and anomaly filters."""

    def __init__(self) -> None:
        self.model: Optional[Any] = None

    def update_from_feedback(self, ledger_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        from ledger.ledger_metrics import compute_metrics

        metrics = compute_metrics(ledger_df)
        anomalies = self._detect_anomalies(ledger_df)

        update: Dict[str, Any] = {"anomalies": anomalies, "metrics": metrics}
        if PPO and DummyVecEnv:
            env = DummyVecEnv([lambda: FeedbackEnv(metrics)])
            self.model = PPO("MlpPolicy", env, verbose=0)
            self.model.learn(total_timesteps=500)
            action, _ = self.model.predict(env.reset())
            update["risk_per_trade"] = float(config.get("risk_per_trade", 0.01) * (1 + action[0]))
        logger.info("Feedback update computed: %s", update)
        return update

    def _detect_anomalies(self, ledger_df: pd.DataFrame) -> Dict[str, Any]:
        if ledger_df.empty:
            return {}
        try:
            features = []
            for _, row in ledger_df.iterrows():
                payload = json.loads(row.get("payload") or "{}")
                features.append([
                    payload.get("hedge_snapshot", {}).get("elasticity", 0.0),
                    payload.get("liquidity_snapshot", {}).get("liquidity_score", 0.0),
                    payload.get("sentiment_snapshot", {}).get("sentiment_score", 0.0),
                ])
            model = IsolationForest(random_state=42)
            preds = model.fit_predict(features)
            flagged = int(np.sum(preds == -1))
            return {"flagged": flagged, "total": len(preds)}
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Anomaly detection failed: {exc}")
            return {}


# Test: FeedbackEngine()._detect_anomalies(pd.DataFrame()) returns {}
