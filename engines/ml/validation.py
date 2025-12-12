"""
Model Validation System for GNOSIS ML Models
Comprehensive validation including backtesting, performance metrics, and risk assessment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""

    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    information_ratio: float

    def to_dict(self) -> Dict[str, float]:
        """Convert the metrics to a dictionary."""
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r2": self.r2,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
            "profit_factor": self.profit_factor,
            "calmar_ratio": self.calmar_ratio,
            "sortino_ratio": self.sortino_ratio,
            "information_ratio": self.information_ratio,
        }


class BacktestEngine:
    """Backtesting engine for trading models."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
    ) -> None:
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def run_backtest(
        self,
        predictions: np.ndarray,
        actual_prices: np.ndarray,
        signals: Optional[np.ndarray] = None,
        position_sizing: str = "fixed",
    ) -> Dict[str, Any]:
        """
        Run a backtest on predictions.

        Args:
            predictions: Model predictions.
            actual_prices: Actual price data.
            signals: Trading signals (optional, generated from predictions if None).
            position_sizing: Position sizing method ("fixed", "volatility", "kelly").
        """
        if signals is None:
            signals = self._generate_signals(predictions)

        returns = np.diff(actual_prices) / actual_prices[:-1]
        positions = self._apply_position_sizing(signals, returns, position_sizing)
        portfolio_returns = positions[:-1] * returns
        portfolio_returns = self._apply_transaction_costs(portfolio_returns, positions)

        cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
        portfolio_value = self.initial_capital * (1 + cumulative_returns)

        return {
            "portfolio_returns": portfolio_returns,
            "cumulative_returns": cumulative_returns,
            "portfolio_value": portfolio_value,
            "positions": positions,
            "signals": signals,
            "total_return": cumulative_returns[-1] if len(cumulative_returns) > 0 else 0.0,
            "annualized_return": self._annualized_return(portfolio_returns),
            "volatility": float(np.std(portfolio_returns) * np.sqrt(252)),
            "max_drawdown": self._max_drawdown(cumulative_returns),
            "sharpe_ratio": self._sharpe_ratio(portfolio_returns),
            "calmar_ratio": self._calmar_ratio(portfolio_returns, cumulative_returns),
        }

    def _generate_signals(self, predictions: np.ndarray) -> np.ndarray:
        """Generate trading signals from predictions."""
        signals = np.zeros_like(predictions, dtype=float)
        signals[predictions > 0] = 1.0
        signals[predictions < 0] = -1.0
        return signals

    def _apply_position_sizing(
        self, signals: np.ndarray, returns: np.ndarray, method: str
    ) -> np.ndarray:
        """Apply position sizing strategy."""
        if method == "fixed":
            size = np.full_like(signals, 0.1, dtype=float)
        elif method == "volatility":
            size = np.full_like(signals, 0.1, dtype=float)
            if len(returns) > 0:
                vol_window = min(20, len(returns))
                rolling_vol = pd.Series(returns).rolling(vol_window).std()
                vol_target = 0.15  # 15% annual volatility target
                vol_position_sizes = (
                    vol_target / (rolling_vol * np.sqrt(252))
                ).fillna(0.1)
                size[1:] = vol_position_sizes.to_numpy()[: len(signals) - 1]
        elif method == "kelly":
            win_rate = float(np.mean(returns > 0)) if len(returns) > 0 else 0.0
            avg_win = float(np.mean(returns[returns > 0])) if np.any(returns > 0) else 0.0
            avg_loss = (
                float(np.mean(returns[returns < 0])) if np.any(returns < 0) else 0.0
            )
            kelly_fraction = 0.1
            if avg_loss != 0:
                kelly_fraction = win_rate - (1 - win_rate) * (avg_win / abs(avg_loss))
                kelly_fraction = float(np.clip(kelly_fraction, 0, 0.25))
            size = np.full_like(signals, kelly_fraction, dtype=float)
        else:
            size = np.full_like(signals, 0.1, dtype=float)

        return signals * size

    def _apply_transaction_costs(
        self, returns: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """Apply transaction costs and slippage."""
        position_changes = np.abs(np.diff(positions))
        costs = position_changes * (self.transaction_cost + self.slippage)
        return returns - costs

    def _max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(cumulative_returns) == 0:
            return 0.0

        cumulative = 1 + cumulative_returns
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return float(abs(np.min(drawdown)))

    def _sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate the annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = np.mean(returns) - risk_free_rate / 252
        return float(excess_returns / np.std(returns) * np.sqrt(252))

    def _annualized_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0
        return float(np.mean(returns) * 252)

    def _calmar_ratio(self, returns: np.ndarray, cumulative_returns: np.ndarray) -> float:
        """Calculate Calmar ratio."""
        annual_return = self._annualized_return(returns)
        max_dd = self._max_drawdown(cumulative_returns)
        if max_dd == 0:
            return 0.0
        return float(annual_return / max_dd)


class ModelValidator:
    """Comprehensive model validation system."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backtest_engine = BacktestEngine(
            initial_capital=config.get("initial_capital", 100_000),
            transaction_cost=config.get("transaction_cost", 0.001),
            slippage=config.get("slippage", 0.0005),
        )

    def validate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        prices: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Comprehensive model validation."""
        validation_results: Dict[str, Any] = {}
        try:
            predictions = model.predict(X_test)
            if isinstance(predictions, dict):
                for horizon, preds in predictions.get("predictions", {}).items():
                    if len(preds) == 0:
                        continue
                    horizon_results = self._validate_predictions(
                        np.asarray(preds), y_test[: len(preds)]
                    )
                    validation_results[horizon] = horizon_results
                    if prices is not None and len(prices) >= len(preds):
                        backtest_results = self.backtest_engine.run_backtest(
                            np.asarray(preds), prices[: len(preds)]
                        )
                        validation_results[horizon]["backtest"] = backtest_results
            else:
                validation_results["single"] = self._validate_predictions(
                    np.asarray(predictions), y_test
                )
                if prices is not None:
                    backtest_results = self.backtest_engine.run_backtest(
                        np.asarray(predictions), prices
                    )
                    validation_results["single"]["backtest"] = backtest_results
        except Exception as exc:  # noqa: BLE001
            validation_results["error"] = str(exc)

        return validation_results

    def _validate_predictions(
        self, predictions: np.ndarray, actual: np.ndarray
    ) -> Dict[str, Any]:
        """Validate predictions against actual values."""
        min_len = min(len(predictions), len(actual))
        pred = predictions[:min_len]
        act = actual[:min_len]

        if len(pred) == 0:
            return {"error": "No predictions to validate"}

        mse = mean_squared_error(act, pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(act, pred)
        mape = float(np.mean(np.abs((act - pred) / np.where(act != 0, act, 1))) * 100)
        r2 = r2_score(act, pred)

        actual_direction = np.sign(np.diff(act))
        pred_direction = np.sign(np.diff(pred))
        hit_rate = float(
            np.mean(actual_direction == pred_direction) if len(actual_direction) > 0 else 0.0
        )

        residuals = act - pred
        _, jb_pvalue = stats.jarque_bera(residuals)

        lb_pvalue: Optional[float] = None
        if acorr_ljungbox is not None:
            lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
            lb_pvalue = float(lb_result["lb_pvalue"].iloc[-1])

        bp_pvalue: Optional[float] = None
        if het_breuschpagan is not None:
            _, bp_pvalue, _, _ = het_breuschpagan(residuals, np.column_stack([pred]))
            bp_pvalue = float(bp_pvalue)

        return {
            "basic_metrics": {
                "mse": float(mse),
                "rmse": rmse,
                "mae": float(mae),
                "mape": mape,
                "r2": float(r2),
                "hit_rate": hit_rate,
            },
            "statistical_tests": {
                "jarque_bera_pvalue": float(jb_pvalue),
                "ljung_box_pvalue": lb_pvalue,
                "breusch_pagan_pvalue": bp_pvalue,
            },
            "residual_stats": {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "skewness": float(stats.skew(residuals)),
                "kurtosis": float(stats.kurtosis(residuals)),
            },
        }

    def cross_validate_model(
        self,
        model_class: Any,
        model_config: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        time_series_split: bool = True,
    ) -> Dict[str, Any]:
        """Perform cross-validation."""
        fold_results: List[Dict[str, Any]] = []

        if time_series_split:
            fold_size = len(X) // (cv_folds + 1)
            for i in range(cv_folds):
                train_end = fold_size * (i + 2)
                test_start = fold_size * (i + 1)
                test_end = fold_size * (i + 2)

                X_train = X[:train_end]
                y_train = y[:train_end]
                X_test = X[test_start:test_end]
                y_test = y[test_start:test_end]

                try:
                    model = model_class(model_config)
                    model.train(X_train, y_train)
                    fold_result = self.validate_model(model, X_test, y_test)
                    fold_results.append(fold_result)
                except Exception as exc:  # noqa: BLE001
                    fold_results.append({"error": str(exc)})
        else:
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                try:
                    model = model_class(model_config)
                    model.train(X_train, y_train)
                    fold_result = self.validate_model(model, X_test, y_test)
                    fold_results.append(fold_result)
                except Exception as exc:  # noqa: BLE001
                    fold_results.append({"error": str(exc)})

        return self._aggregate_cv_results(fold_results)

    def _aggregate_cv_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate cross-validation results."""
        valid_results = [result for result in fold_results if "error" not in result]
        if not valid_results:
            return {"error": "All folds failed"}

        metrics_by_horizon: Dict[str, List[Dict[str, float]]] = {}
        for result in valid_results:
            for horizon, horizon_data in result.items():
                if horizon not in metrics_by_horizon:
                    metrics_by_horizon[horizon] = []
                if "basic_metrics" in horizon_data:
                    metrics_by_horizon[horizon].append(horizon_data["basic_metrics"])

        aggregated: Dict[str, Any] = {}
        for horizon, metrics_list in metrics_by_horizon.items():
            if not metrics_list:
                continue

            aggregated[horizon] = {
                "mean_metrics": {},
                "std_metrics": {},
                "fold_count": len(metrics_list),
            }

            for metric_name in metrics_list[0].keys():
                values = [m[metric_name] for m in metrics_list if not np.isnan(m[metric_name])]
                if values:
                    aggregated[horizon]["mean_metrics"][metric_name] = float(np.mean(values))
                    aggregated[horizon]["std_metrics"][metric_name] = float(np.std(values))

        return aggregated

    def generate_validation_report(
        self, validation_results: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive validation report."""
        report_lines: List[str] = []
        report_lines.append("=" * 60)
        report_lines.append("GNOSIS MODEL VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {pd.Timestamp.now()}")
        report_lines.append("")

        for horizon, results in validation_results.items():
            if "error" in results:
                report_lines.append(f"Horizon {horizon}: ERROR - {results['error']}")
                continue

            report_lines.append(f"HORIZON: {horizon}")
            report_lines.append("-" * 40)

            if "basic_metrics" in results:
                metrics = results["basic_metrics"]
                report_lines.append("Performance Metrics:")
                report_lines.append(f"  MSE: {metrics['mse']:.6f}")
                report_lines.append(f"  RMSE: {metrics['rmse']:.6f}")
                report_lines.append(f"  MAE: {metrics['mae']:.6f}")
                report_lines.append(f"  MAPE: {metrics['mape']:.2f}%")
                report_lines.append(f"  R²: {metrics['r2']:.4f}")
                report_lines.append(f"  Hit Rate: {metrics['hit_rate'] * 100:.2f}%")
                report_lines.append("")

            if "backtest" in results:
                bt = results["backtest"]
                report_lines.append("Backtest Results:")
                report_lines.append(f"  Total Return: {bt['total_return']:.2%}")
                report_lines.append(f"  Annualized Return: {bt['annualized_return']:.2%}")
                report_lines.append(f"  Volatility: {bt['volatility']:.2%}")
                report_lines.append(f"  Sharpe Ratio: {bt['sharpe_ratio']:.3f}")
                report_lines.append(f"  Max Drawdown: {bt['max_drawdown']:.2%}")
                report_lines.append(f"  Calmar Ratio: {bt['calmar_ratio']:.3f}")
                report_lines.append("")

            if "statistical_tests" in results:
                tests = results["statistical_tests"]
                report_lines.append("Statistical Tests:")
                jb_result = (
                    "PASS" if tests["jarque_bera_pvalue"] is None or tests["jarque_bera_pvalue"] > 0.05 else "FAIL"
                )
                report_lines.append(
                    f"  Normality (JB): {jb_result} (p={tests['jarque_bera_pvalue']:.4f})"
                )

                if tests.get("ljung_box_pvalue") is not None:
                    lb_result = "PASS" if tests["ljung_box_pvalue"] > 0.05 else "FAIL"
                    report_lines.append(
                        f"  No Autocorr (LB): {lb_result} (p={tests['ljung_box_pvalue']:.4f})"
                    )

                if tests.get("breusch_pagan_pvalue") is not None:
                    bp_result = "PASS" if tests["breusch_pagan_pvalue"] > 0.05 else "FAIL"
                    report_lines.append(
                        f"  Homoscedasticity (BP): {bp_result} (p={tests['breusch_pagan_pvalue']:.4f})"
                    )
                report_lines.append("")

        report_text = "\n".join(report_lines)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(report_text)

        return report_text
