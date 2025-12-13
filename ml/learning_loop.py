"""
Complete ML Learning Loop for GNOSIS Trading System

Integrates Massive historical data (market + options) with ML training and backtesting
to create a continuous learning and improvement pipeline.

Pipeline:
1. Fetch historical data (OHLCV + Options) from Massive/Alpaca
2. Build features (technical, options Greeks, GEX, sentiment proxies)
3. Train ML models (LSTM for price prediction, XGBoost for direction)
4. Backtest trained models against historical data
5. Analyze performance and iterate

Usage:
    python -m ml.learning_loop --symbol SPY --start 2020-01-01 --end 2024-12-01
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class LearningConfig:
    """Configuration for the ML learning loop."""

    # Symbol and date range
    symbol: str = "SPY"
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-01"
    timeframe: str = "1Day"

    # Data sources
    data_provider: str = "auto"  # "massive", "alpaca", or "auto"
    use_options_data: bool = True  # Include options features if available

    # Feature configuration
    sequence_length: int = 60  # LSTM lookback window
    target_horizons: List[int] = field(default_factory=lambda: [1, 5, 15, 60])

    # Model configuration
    train_lstm: bool = True
    train_xgboost: bool = True

    # LSTM hyperparameters
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    lstm_learning_rate: float = 0.001
    lstm_patience: int = 10

    # XGBoost hyperparameters
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    # Train/val/test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15  # Remaining goes to test

    # Backtesting configuration
    run_backtest: bool = True
    backtest_initial_capital: float = 100_000.0
    backtest_position_size: float = 0.10
    backtest_stop_loss: float = 0.02
    backtest_take_profit: float = 0.04

    # Output configuration
    output_dir: str = "runs/learning"
    save_models: bool = True
    save_features: bool = True
    save_results: bool = True

    # Tag for this run
    tag: str = ""


@dataclass
class LearningResults:
    """Results from a learning loop iteration."""

    # Config
    config: LearningConfig = None

    # Data stats
    total_samples: int = 0
    num_features: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0

    # Feature info
    options_features_available: bool = False
    feature_columns: List[str] = field(default_factory=list)

    # LSTM results
    lstm_trained: bool = False
    lstm_train_loss: float = 0.0
    lstm_val_loss: float = 0.0
    lstm_test_mae: float = 0.0
    lstm_test_direction_accuracy: float = 0.0

    # XGBoost results
    xgb_trained: bool = False
    xgb_train_accuracy: float = 0.0
    xgb_val_accuracy: float = 0.0
    xgb_test_accuracy: float = 0.0
    xgb_feature_importance: Dict[str, float] = field(default_factory=dict)

    # Backtest results
    backtest_run: bool = False
    backtest_total_return: float = 0.0
    backtest_sharpe_ratio: float = 0.0
    backtest_max_drawdown: float = 0.0
    backtest_win_rate: float = 0.0
    backtest_total_trades: int = 0

    # Paths
    lstm_model_path: str = ""
    xgb_model_path: str = ""
    features_path: str = ""
    results_path: str = ""

    # Timestamps
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0


class MLLearningLoop:
    """Complete ML learning loop for Gnosis trading system."""

    def __init__(self, config: LearningConfig):
        self.config = config
        self.results = LearningResults(config=config)

        # Initialize adapters lazily
        self._market_adapter = None
        self._options_adapter = None
        self._ml_integration = None

        # Data storage
        self.features_df: Optional[pd.DataFrame] = None
        self.targets_df: Optional[pd.DataFrame] = None
        self.splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

        # Models
        self.lstm_model = None
        self.xgb_model = None
        self.feature_scaler = None

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _init_adapters(self):
        """Initialize data adapters based on configuration."""
        provider = self.config.data_provider

        if provider == "auto":
            # Try Massive first, fall back to Alpaca
            if os.getenv("MASSIVE_API_ENABLED", "").lower() == "true":
                provider = "massive"
            elif os.getenv("ALPACA_API_KEY"):
                provider = "alpaca"
            else:
                raise ValueError(
                    "No data provider available. Set MASSIVE_API_ENABLED=true or "
                    "ALPACA_API_KEY/ALPACA_SECRET_KEY environment variables."
                )

        if provider == "massive":
            self._init_massive_adapters()
        elif provider == "alpaca":
            self._init_alpaca_adapters()
        else:
            raise ValueError(f"Unknown data provider: {provider}")

    def _init_massive_adapters(self):
        """Initialize Massive data adapters."""
        try:
            from engines.inputs.massive_market_adapter import MassiveMarketDataAdapter
            from engines.inputs.massive_options_adapter import MassiveOptionsAdapter
            from engines.ml.massive_options_integration import MassiveOptionsMLIntegration

            self._market_adapter = MassiveMarketDataAdapter()
            self._options_adapter = MassiveOptionsAdapter() if self.config.use_options_data else None
            self._ml_integration = MassiveOptionsMLIntegration(
                market_adapter=self._market_adapter,
                options_adapter=self._options_adapter,
            )
            logger.info("Initialized Massive data adapters")
        except Exception as e:
            logger.error(f"Failed to initialize Massive adapters: {e}")
            raise

    def _init_alpaca_adapters(self):
        """Initialize Alpaca data adapters."""
        try:
            from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
            self._market_adapter = AlpacaMarketDataAdapter()
            self._options_adapter = None  # Alpaca doesn't have comprehensive options data
            self._ml_integration = None
            logger.info("Initialized Alpaca data adapters (options data not available)")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca adapters: {e}")
            raise

    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch historical data and build features/targets."""
        logger.info(f"Fetching data for {self.config.symbol}")

        start = datetime.strptime(self.config.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(self.config.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        if self._ml_integration:
            # Use Massive ML integration for comprehensive features
            features_df, targets_df = self._ml_integration.get_training_data(
                symbol=self.config.symbol,
                start_date=start,
                end_date=end,
                timeframe=self.config.timeframe.lower().replace("day", "day").replace("hour", "hour"),
                include_options=self.config.use_options_data and self._options_adapter is not None,
                target_horizons=self.config.target_horizons,
            )
            self.results.options_features_available = self._options_adapter is not None
        else:
            # Fall back to basic feature building
            features_df, targets_df = self._build_basic_features(start, end)
            self.results.options_features_available = False

        self.features_df = features_df
        self.targets_df = targets_df

        self.results.total_samples = len(features_df)
        self.results.num_features = features_df.shape[1] if not features_df.empty else 0
        self.results.feature_columns = list(features_df.columns) if not features_df.empty else []

        logger.info(f"Fetched {self.results.total_samples} samples with {self.results.num_features} features")

        return features_df, targets_df

    def _build_basic_features(self, start: datetime, end: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build basic features from market data when Massive integration unavailable."""
        if not self._market_adapter:
            return pd.DataFrame(), pd.DataFrame()

        # Fetch OHLCV data
        bars = self._market_adapter.get_bars(
            symbol=self.config.symbol,
            start=start,
            end=end,
            timeframe=self.config.timeframe,
        )

        if not bars:
            return pd.DataFrame(), pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
            }
            for bar in bars
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        # Build technical features
        features_df = self._compute_technical_features(df)

        # Build targets
        targets_df = self._compute_targets(df)

        return features_df, targets_df

    def _compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical analysis features."""
        features = pd.DataFrame(index=df.index)

        # Price features
        features['return_1'] = df['close'].pct_change()
        features['return_5'] = df['close'].pct_change(5)
        features['return_10'] = df['close'].pct_change(10)
        features['return_20'] = df['close'].pct_change(20)

        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean() / df['close'] - 1
            features[f'ema_{period}'] = df['close'].ewm(span=period).mean() / df['close'] - 1

        # Volatility
        features['volatility_10'] = df['close'].pct_change().rolling(10).std()
        features['volatility_20'] = df['close'].pct_change().rolling(20).std()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        features['macd'] = (ema12 - ema26) / df['close']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()

        # Bollinger Bands
        bb_mean = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        features['bb_upper'] = (df['close'] - (bb_mean + 2 * bb_std)) / df['close']
        features['bb_lower'] = (df['close'] - (bb_mean - 2 * bb_std)) / df['close']

        # Volume features
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']

        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean() / df['close']

        return features.dropna()

    def _compute_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute target variables (forward returns)."""
        targets = pd.DataFrame(index=df.index)

        for horizon in self.config.target_horizons:
            # Forward returns
            targets[f'return_{horizon}'] = df['close'].shift(-horizon) / df['close'] - 1
            # Direction (1 = up, 0 = down)
            targets[f'direction_{horizon}'] = (targets[f'return_{horizon}'] > 0).astype(int)

        return targets.dropna()

    def create_splits(self):
        """Create train/val/test splits."""
        if self.features_df is None or self.features_df.empty:
            raise ValueError("No features available. Run fetch_data() first.")

        if self._ml_integration:
            self.splits = self._ml_integration.create_train_val_test_split(
                self.features_df,
                self.targets_df,
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio,
            )
        else:
            n = len(self.features_df)
            train_end = int(n * self.config.train_ratio)
            val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

            self.splits = {
                "train": (self.features_df.iloc[:train_end], self.targets_df.iloc[:train_end]),
                "val": (self.features_df.iloc[train_end:val_end], self.targets_df.iloc[train_end:val_end]),
                "test": (self.features_df.iloc[val_end:], self.targets_df.iloc[val_end:]),
            }

        self.results.train_samples = len(self.splits["train"][0])
        self.results.val_samples = len(self.splits["val"][0])
        self.results.test_samples = len(self.splits["test"][0])

        logger.info(
            f"Split data: train={self.results.train_samples}, "
            f"val={self.results.val_samples}, test={self.results.test_samples}"
        )

    def train_lstm(self):
        """Train LSTM model for price prediction."""
        if not self.config.train_lstm:
            logger.info("LSTM training disabled")
            return

        logger.info("Training LSTM model...")

        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("PyTorch not available. Skipping LSTM training.")
            return

        # Prepare sequences
        train_X, train_y = self._prepare_lstm_sequences(self.splits["train"])
        val_X, val_y = self._prepare_lstm_sequences(self.splits["val"])

        if len(train_X) == 0:
            logger.warning("Insufficient data for LSTM training")
            return

        # Scale features
        self.feature_scaler = StandardScaler()
        train_X_flat = train_X.reshape(-1, train_X.shape[-1])
        self.feature_scaler.fit(train_X_flat)

        train_X_scaled = self._scale_sequences(train_X, self.feature_scaler)
        val_X_scaled = self._scale_sequences(val_X, self.feature_scaler)

        # Convert to tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_dataset = TensorDataset(
            torch.FloatTensor(train_X_scaled),
            torch.FloatTensor(train_y),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_X_scaled),
            torch.FloatTensor(val_y),
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.lstm_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.lstm_batch_size)

        # Build model
        input_dim = train_X.shape[2]
        self.lstm_model = self._build_lstm_model(input_dim).to(device)

        # Training loop
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=self.config.lstm_learning_rate)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.lstm_epochs):
            # Train
            self.lstm_model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = self.lstm_model(X_batch)
                loss = criterion(output.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validate
            self.lstm_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    output = self.lstm_model(X_batch)
                    loss = criterion(output.squeeze(), y_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.lstm_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Evaluate on test set
        test_X, test_y = self._prepare_lstm_sequences(self.splits["test"])
        if len(test_X) > 0:
            test_X_scaled = self._scale_sequences(test_X, self.feature_scaler)
            test_dataset = TensorDataset(torch.FloatTensor(test_X_scaled), torch.FloatTensor(test_y))
            test_loader = DataLoader(test_dataset, batch_size=self.config.lstm_batch_size)

            self.lstm_model.eval()
            preds, actuals = [], []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    output = self.lstm_model(X_batch)
                    preds.extend(output.squeeze().cpu().numpy())
                    actuals.extend(y_batch.numpy())

            preds = np.array(preds)
            actuals = np.array(actuals)

            # Metrics
            mae = np.mean(np.abs(preds - actuals))
            direction_acc = np.mean((preds > 0) == (actuals > 0))

            self.results.lstm_test_mae = float(mae)
            self.results.lstm_test_direction_accuracy = float(direction_acc)

        self.results.lstm_trained = True
        self.results.lstm_train_loss = float(train_loss)
        self.results.lstm_val_loss = float(best_val_loss)

        logger.info(
            f"LSTM training complete. Test MAE: {self.results.lstm_test_mae:.6f}, "
            f"Direction accuracy: {self.results.lstm_test_direction_accuracy:.2%}"
        )

    def _build_lstm_model(self, input_dim: int):
        """Build LSTM model architecture."""
        import torch.nn as nn

        class LSTMPredictor(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                    bidirectional=True,
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Use last timestep
                out = self.fc(lstm_out[:, -1, :])
                return out

        return LSTMPredictor(
            input_dim=input_dim,
            hidden_dim=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_num_layers,
            dropout=self.config.lstm_dropout,
        )

    def _prepare_lstm_sequences(
        self,
        split: Tuple[pd.DataFrame, pd.DataFrame],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training."""
        features_df, targets_df = split

        if len(features_df) < self.config.sequence_length + 1:
            return np.array([]), np.array([])

        features = features_df.fillna(0).values
        targets = targets_df['return_1'].fillna(0).values if 'return_1' in targets_df.columns else targets_df.iloc[:, 0].fillna(0).values

        X, y = [], []
        for i in range(len(features) - self.config.sequence_length):
            X.append(features[i:i + self.config.sequence_length])
            y.append(targets[i + self.config.sequence_length])

        return np.array(X), np.array(y)

    def _scale_sequences(self, X: np.ndarray, scaler) -> np.ndarray:
        """Scale feature sequences."""
        original_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.transform(X_flat)
        return X_scaled.reshape(original_shape)

    def train_xgboost(self):
        """Train XGBoost model for direction classification."""
        if not self.config.train_xgboost:
            logger.info("XGBoost training disabled")
            return

        logger.info("Training XGBoost model...")

        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("XGBoost not available. Skipping XGBoost training.")
            return

        train_X, train_targets = self.splits["train"]
        val_X, val_targets = self.splits["val"]
        test_X, test_targets = self.splits["test"]

        # Get direction targets
        target_col = 'direction_1' if 'direction_1' in train_targets.columns else train_targets.columns[1]
        train_y = train_targets[target_col].fillna(0).astype(int)
        val_y = val_targets[target_col].fillna(0).astype(int)
        test_y = test_targets[target_col].fillna(0).astype(int)

        # Fill NaN in features
        train_X = train_X.fillna(0)
        val_X = val_X.fillna(0)
        test_X = test_X.fillna(0)

        # Train model
        self.xgb_model = XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            use_label_encoder=False,
            eval_metric='logloss',
            early_stopping_rounds=20,
        )

        self.xgb_model.fit(
            train_X, train_y,
            eval_set=[(val_X, val_y)],
            verbose=False,
        )

        # Evaluate
        train_acc = self.xgb_model.score(train_X, train_y)
        val_acc = self.xgb_model.score(val_X, val_y)
        test_acc = self.xgb_model.score(test_X, test_y)

        self.results.xgb_trained = True
        self.results.xgb_train_accuracy = float(train_acc)
        self.results.xgb_val_accuracy = float(val_acc)
        self.results.xgb_test_accuracy = float(test_acc)

        # Feature importance
        importance = self.xgb_model.feature_importances_
        feature_names = train_X.columns.tolist()
        importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20])
        self.results.xgb_feature_importance = importance_dict

        logger.info(
            f"XGBoost training complete. Train: {train_acc:.2%}, Val: {val_acc:.2%}, Test: {test_acc:.2%}"
        )

    def run_backtest(self):
        """Run backtest using trained models."""
        if not self.config.run_backtest:
            logger.info("Backtesting disabled")
            return

        if not self.xgb_model and not self.lstm_model:
            logger.warning("No trained models available for backtesting")
            return

        logger.info("Running backtest...")

        try:
            from backtesting.ml_backtest_engine import MLBacktestConfig, MLBacktestEngine
        except ImportError:
            logger.warning("ML backtest engine not available")
            return

        # Use XGBoost predictions for signals if available
        test_X, test_targets = self.splits["test"]

        if self.xgb_model:
            # Get predictions
            predictions = self.xgb_model.predict_proba(test_X.fillna(0))
            long_prob = predictions[:, 1]  # Probability of up direction
        else:
            long_prob = np.full(len(test_X), 0.5)

        # Simple backtest simulation
        capital = self.config.backtest_initial_capital
        position = 0
        trades = []
        equity_curve = [capital]

        target_col = 'return_1' if 'return_1' in test_targets.columns else test_targets.columns[0]
        returns = test_targets[target_col].fillna(0).values

        for i in range(len(returns)):
            signal = long_prob[i] - 0.5  # Convert to -0.5 to +0.5

            if position == 0 and abs(signal) > 0.2:
                # Open position
                position = 1 if signal > 0 else -1
                entry_price = 100  # Normalized
                entry_capital = capital * self.config.backtest_position_size

            elif position != 0:
                # Check P&L
                pnl = returns[i] * position * entry_capital

                # Apply stop loss / take profit
                if abs(returns[i]) > self.config.backtest_stop_loss or abs(returns[i]) > self.config.backtest_take_profit:
                    capital += pnl
                    trades.append(pnl)
                    position = 0
                elif (position > 0 and signal < -0.1) or (position < 0 and signal > 0.1):
                    capital += pnl
                    trades.append(pnl)
                    position = 0

            equity_curve.append(capital)

        # Calculate metrics
        if trades:
            self.results.backtest_run = True
            self.results.backtest_total_return = (capital - self.config.backtest_initial_capital) / self.config.backtest_initial_capital
            self.results.backtest_total_trades = len(trades)
            self.results.backtest_win_rate = sum(1 for t in trades if t > 0) / len(trades)

            # Sharpe ratio
            if len(equity_curve) > 1:
                equity_returns = np.diff(equity_curve) / equity_curve[:-1]
                if np.std(equity_returns) > 0:
                    self.results.backtest_sharpe_ratio = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(252)

            # Max drawdown
            peak = equity_curve[0]
            max_dd = 0
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak
                if dd > max_dd:
                    max_dd = dd
            self.results.backtest_max_drawdown = max_dd

        logger.info(
            f"Backtest complete. Return: {self.results.backtest_total_return:.2%}, "
            f"Sharpe: {self.results.backtest_sharpe_ratio:.2f}, "
            f"Win rate: {self.results.backtest_win_rate:.2%}"
        )

    def save_models(self):
        """Save trained models to disk."""
        if not self.config.save_models:
            return

        tag = self.config.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        models_dir = self.output_dir / "models" / tag
        models_dir.mkdir(parents=True, exist_ok=True)

        # Save LSTM
        if self.lstm_model:
            import torch
            lstm_path = models_dir / "lstm_model.pt"
            torch.save(self.lstm_model.state_dict(), lstm_path)
            self.results.lstm_model_path = str(lstm_path)

            # Save scaler
            if self.feature_scaler:
                scaler_path = models_dir / "feature_scaler.pkl"
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.feature_scaler, f)

            logger.info(f"LSTM model saved to {lstm_path}")

        # Save XGBoost
        if self.xgb_model:
            xgb_path = models_dir / "xgb_model.json"
            self.xgb_model.save_model(str(xgb_path))
            self.results.xgb_model_path = str(xgb_path)
            logger.info(f"XGBoost model saved to {xgb_path}")

    def save_features(self):
        """Save feature data to disk."""
        if not self.config.save_features or self.features_df is None:
            return

        tag = self.config.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        features_dir = self.output_dir / "features" / tag
        features_dir.mkdir(parents=True, exist_ok=True)

        features_path = features_dir / "features.parquet"
        self.features_df.to_parquet(features_path)
        self.results.features_path = str(features_path)

        targets_path = features_dir / "targets.parquet"
        self.targets_df.to_parquet(targets_path)

        logger.info(f"Features saved to {features_dir}")

    def save_results(self):
        """Save learning results to disk."""
        if not self.config.save_results:
            return

        tag = self.config.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.output_dir / f"{tag}_results.json"

        # Convert results to dict
        results_dict = asdict(self.results)
        # Remove config (too large) but keep key params
        results_dict['config'] = {
            'symbol': self.config.symbol,
            'start_date': self.config.start_date,
            'end_date': self.config.end_date,
            'timeframe': self.config.timeframe,
            'use_options_data': self.config.use_options_data,
        }

        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

        self.results.results_path = str(results_path)
        logger.info(f"Results saved to {results_path}")

    def run(self) -> LearningResults:
        """Run the complete learning loop."""
        self.results.started_at = datetime.now().isoformat()
        start_time = datetime.now()

        try:
            logger.info("="*60)
            logger.info("GNOSIS ML LEARNING LOOP")
            logger.info("="*60)
            logger.info(f"Symbol: {self.config.symbol}")
            logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
            logger.info(f"Data provider: {self.config.data_provider}")
            logger.info("="*60)

            # Initialize adapters
            self._init_adapters()

            # Fetch data
            self.fetch_data()

            if self.results.total_samples < 100:
                raise ValueError(f"Insufficient data: {self.results.total_samples} samples")

            # Create splits
            self.create_splits()

            # Train models
            self.train_lstm()
            self.train_xgboost()

            # Run backtest
            self.run_backtest()

            # Save outputs
            self.save_models()
            self.save_features()
            self.save_results()

        except Exception as e:
            logger.error(f"Learning loop failed: {e}")
            raise

        finally:
            self.results.completed_at = datetime.now().isoformat()
            self.results.duration_seconds = (datetime.now() - start_time).total_seconds()

        # Print summary
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print results summary."""
        print("\n" + "="*60)
        print("LEARNING LOOP RESULTS")
        print("="*60)
        print(f"Symbol: {self.config.symbol}")
        print(f"Period: {self.config.start_date} to {self.config.end_date}")
        print(f"Duration: {self.results.duration_seconds:.1f}s")
        print("-"*60)
        print("DATA")
        print(f"  Total samples: {self.results.total_samples}")
        print(f"  Features: {self.results.num_features}")
        print(f"  Options data: {'Yes' if self.results.options_features_available else 'No'}")
        print("-"*60)

        if self.results.lstm_trained:
            print("LSTM MODEL")
            print(f"  Train loss: {self.results.lstm_train_loss:.6f}")
            print(f"  Val loss: {self.results.lstm_val_loss:.6f}")
            print(f"  Test MAE: {self.results.lstm_test_mae:.6f}")
            print(f"  Direction accuracy: {self.results.lstm_test_direction_accuracy:.2%}")
            print("-"*60)

        if self.results.xgb_trained:
            print("XGBOOST MODEL")
            print(f"  Train accuracy: {self.results.xgb_train_accuracy:.2%}")
            print(f"  Val accuracy: {self.results.xgb_val_accuracy:.2%}")
            print(f"  Test accuracy: {self.results.xgb_test_accuracy:.2%}")
            print("  Top features:")
            for i, (name, imp) in enumerate(list(self.results.xgb_feature_importance.items())[:5]):
                print(f"    {i+1}. {name}: {imp:.4f}")
            print("-"*60)

        if self.results.backtest_run:
            print("BACKTEST")
            print(f"  Total return: {self.results.backtest_total_return:.2%}")
            print(f"  Sharpe ratio: {self.results.backtest_sharpe_ratio:.2f}")
            print(f"  Max drawdown: {self.results.backtest_max_drawdown:.2%}")
            print(f"  Win rate: {self.results.backtest_win_rate:.2%}")
            print(f"  Total trades: {self.results.backtest_total_trades}")

        print("="*60)


def run_learning_loop(
    symbol: str = "SPY",
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-01",
    **kwargs
) -> LearningResults:
    """Convenience function to run the learning loop."""
    config = LearningConfig(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )
    loop = MLLearningLoop(config)
    return loop.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GNOSIS ML Learning Loop")
    parser.add_argument("--symbol", type=str, default="SPY", help="Trading symbol")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-12-01", help="End date")
    parser.add_argument("--timeframe", type=str, default="1Day", help="Data timeframe")
    parser.add_argument("--provider", type=str, default="auto", help="Data provider")
    parser.add_argument("--no-options", action="store_true", help="Disable options data")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM training")
    parser.add_argument("--no-xgb", action="store_true", help="Skip XGBoost training")
    parser.add_argument("--no-backtest", action="store_true", help="Skip backtesting")
    parser.add_argument("--tag", type=str, default="", help="Run tag")

    args = parser.parse_args()

    results = run_learning_loop(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        timeframe=args.timeframe,
        data_provider=args.provider,
        use_options_data=not args.no_options,
        train_lstm=not args.no_lstm,
        train_xgboost=not args.no_xgb,
        run_backtest=not args.no_backtest,
        tag=args.tag,
    )
