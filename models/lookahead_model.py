"""Simple lookahead model built on top of scikit-learn for adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class LookaheadModel:
    """Trainable model that predicts next-period regime labels."""

    def __init__(self) -> None:
        self.model: Pipeline | None = None

    def train(self, features: pd.DataFrame, target_column: str = "target_regime") -> Tuple[float, float]:
        """Train the model from a feature matrix returned by the feature builder."""

        if target_column not in features.columns:
            raise ValueError(f"Missing target column {target_column}")

        df = features.dropna(subset=[target_column])
        y = df[target_column]
        X = df.drop(columns=[target_column])

        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, shuffle=False)

        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ]
        )
        self.model.fit(train_x, train_y)

        train_score = self.model.score(train_x, train_y)
        test_score = self.model.score(test_x, test_y)
        logger.info("Lookahead model train_score=%.3f test_score=%.3f", train_score, test_score)
        return train_score, test_score

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self.model:
            raise RuntimeError("Model not trained")
        return pd.Series(self.model.predict(features), index=features.index)

    def save(self, path: Path) -> None:
        if not self.model:
            raise RuntimeError("Cannot save an untrained model")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Saved lookahead model to %s", path)

    def load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        self.model = joblib.load(path)
        logger.info("Loaded lookahead model from %s", path)
