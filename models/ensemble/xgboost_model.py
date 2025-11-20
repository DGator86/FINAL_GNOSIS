"""
XGBoost-based model for regime classification and feature importance
"""

import logging
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class XGBoostClassifier:
    """
    XGBoost model for regime classification and signal generation

    Features:
    - Multi-class regime classification
    - Feature importance analysis
    - SHAP integration for explainability
    - Hyperparameter optimization
    """

    def __init__(
        self,
        objective: str = "multi:softprob",
        num_class: int = 3,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.params = {
            "objective": objective,
            "num_class": num_class,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "eval_metric": "mlogloss" if "multi" in objective else "logloss",
        }

        self.model = xgb.XGBClassifier(**self.params)
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2, early_stopping_rounds: int = 10) -> Dict:
        self.feature_names = X.columns.tolist()

        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )

        self.is_fitted = True

        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)

        logger.info(f"XGBoost training completed - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        return {"train_accuracy": train_acc, "val_accuracy": val_acc, "feature_importance": self.get_feature_importance()}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict_proba(X)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        importance = self.model.get_booster().get_score(importance_type=importance_type)

        importance_df = (
            pd.DataFrame([{"feature": feat, "importance": imp} for feat, imp in importance.items()])
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        return importance_df

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        import shap

        if not self.is_fitted:
            raise ValueError("Model not fitted")

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)

        return shap_values

    def save(self, path: str):
        model_data = {
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(model_data, path)
        logger.info(f"XGBoost model saved to {path}")

    def load(self, path: str):
        model_data = joblib.load(path)
        self.model = model_data["model"]
        self.params = model_data["params"]
        self.feature_names = model_data["feature_names"]
        self.is_fitted = model_data["is_fitted"]
        logger.info(f"XGBoost model loaded from {path}")


class RegimeClassifier:
    """
    Specialized XGBoost for market regime classification

    Regimes:
    0: Ranging/Low Volatility
    1: Trending/Medium Volatility
    2: Volatile/High Volatility/Crisis
    """

    def __init__(self):
        self.classifier = XGBoostClassifier(
            objective="multi:softprob", num_class=3, max_depth=4, learning_rate=0.05, n_estimators=200
        )

    def create_regime_labels(self, returns: pd.Series, volatility_window: int = 20) -> pd.Series:
        vol = returns.rolling(volatility_window).std()
        trend_strength = np.abs(returns.rolling(volatility_window).mean())

        vol_low_thresh = vol.quantile(0.33)
        vol_high_thresh = vol.quantile(0.67)

        trend_thresh = trend_strength.quantile(0.6)

        regimes = pd.Series(0, index=returns.index)

        regimes[vol > vol_high_thresh] = 2

        trending_mask = (vol > vol_low_thresh) & (vol <= vol_high_thresh) & (trend_strength > trend_thresh)
        regimes[trending_mask] = 1

        return regimes

    def fit(self, features: pd.DataFrame, returns: pd.Series, **kwargs) -> Dict:
        regimes = self.create_regime_labels(returns)

        valid_mask = ~(features.isnull().any(axis=1) | regimes.isnull())
        features_clean = features[valid_mask]
        regimes_clean = regimes[valid_mask]

        return self.classifier.fit(features_clean, regimes_clean, **kwargs)

    def predict_regime(self, features: pd.DataFrame) -> Dict[str, float]:
        probas = self.classifier.predict_proba(features)
        predicted_class = self.classifier.predict(features)[0]

        regime_names = ["ranging", "trending", "volatile"]

        result = {
            "predicted_regime": regime_names[predicted_class],
            "regime_probabilities": {regime_names[i]: float(probas[0][i]) for i in range(len(regime_names))},
        }

        return result


class SignalGenerator:
    """XGBoost-based signal generator for buy/sell/hold decisions"""

    def __init__(self):
        self.classifier = XGBoostClassifier(
            objective="multi:softprob", num_class=3, max_depth=5, learning_rate=0.1, n_estimators=150
        )

    def create_signal_labels(self, returns: pd.Series, forward_window: int = 5, threshold: float = 0.002) -> pd.Series:
        forward_returns = returns.shift(-forward_window)

        signals = pd.Series(1, index=returns.index)

        signals[forward_returns > threshold] = 2
        signals[forward_returns < -threshold] = 0

        return signals

    def fit(self, features: pd.DataFrame, returns: pd.Series, **kwargs) -> Dict:
        signals = self.create_signal_labels(returns)

        valid_mask = ~(features.isnull().any(axis=1) | signals.isnull())
        features_clean = features[valid_mask]
        signals_clean = signals[valid_mask]

        return self.classifier.fit(features_clean, signals_clean, **kwargs)

    def predict_signal(self, features: pd.DataFrame) -> Dict[str, float]:
        probas = self.classifier.predict_proba(features)
        predicted_class = self.classifier.predict(features)[0]

        signal_names = ["sell", "hold", "buy"]

        result = {
            "signal": signal_names[predicted_class],
            "confidence": float(np.max(probas[0])),
            "signal_probabilities": {signal_names[i]: float(probas[0][i]) for i in range(len(signal_names))},
        }

        return result


class XGBoostEnsemble:
    """
    Ensemble of XGBoost models for different tasks
    Integrates with GNOSIS pipeline
    """

    def __init__(self):
        self.regime_classifier = RegimeClassifier()
        self.signal_generator = SignalGenerator()
        self.models = {"regime": self.regime_classifier, "signal": self.signal_generator}

    def fit_all(self, features: pd.DataFrame, returns: pd.Series) -> Dict:
        results = {}

        logger.info("Training regime classifier...")
        results["regime"] = self.regime_classifier.fit(features, returns)

        logger.info("Training signal generator...")
        results["signal"] = self.signal_generator.fit(features, returns)

        return results

    def predict_all(self, features: pd.DataFrame) -> Dict:
        if len(features) > 1:
            features_current = features.tail(1)
        else:
            features_current = features

        results: Dict[str, Dict] = {}

        try:
            regime_pred = self.regime_classifier.predict_regime(features_current)
            results.update(regime_pred)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Regime prediction failed: {exc}")
            results.update(
                {
                    "predicted_regime": "unknown",
                    "regime_probabilities": {"ranging": 0.33, "trending": 0.33, "volatile": 0.34},
                }
            )

        try:
            signal_pred = self.signal_generator.predict_signal(features_current)
            results.update(signal_pred)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Signal prediction failed: {exc}")
            results.update(
                {
                    "signal": "hold",
                    "confidence": 0.5,
                    "signal_probabilities": {"sell": 0.25, "hold": 0.5, "buy": 0.25},
                }
            )

        return results

    def get_feature_importance(self, model_name: str) -> pd.DataFrame:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        return self.models[model_name].classifier.get_feature_importance()

    def save_all(self, base_path: str):
        for name, model in self.models.items():
            model.classifier.save(f"{base_path}_xgb_{name}.pkl")

    def load_all(self, base_path: str):
        for name, model in self.models.items():
            model.classifier.load(f"{base_path}_xgb_{name}.pkl")
