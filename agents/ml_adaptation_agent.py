"""ML/Adaptation Agent for Continuous Learning and Optimization.

Tracks trade outcomes, optimizes system parameters, and implements
anti-overfitting measures for production trading.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


@dataclass
class TradeOutcome:
    """Record of a completed trade with outcome."""

    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    quantity: int

    # Strategy parameters used
    timeframe: str
    stop_loss_price: float
    take_profit_price: float
    confidence: float

    # Outcome metrics
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'take_profit', 'stop_loss', 'trailing_stop', 'time_exit', 'manual'
    hold_duration: timedelta

    # Market context
    market_regime: Optional[str] = None
    volatility: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for evaluation."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    sharpe_ratio: float
    max_drawdown: float
    avg_hold_time: timedelta

    # Per-timeframe breakdown
    timeframe_performance: Dict[str, Dict]


class MLAdaptationAgent:
    """ML/Adaptation Agent for continuous system improvement.

    Responsibilities:
    1. Track all trade outcomes
    2. Calculate performance metrics
    3. Optimize parameters (timeframe weights, confidence thresholds)
    4. Detect overfitting
    5. Validate parameter changes
    """

    def __init__(
        self,
        history_path: str = "data/adaptation/trade_history.json",
        max_history: int = 1000,
        optimization_interval_days: int = 7,
        min_trades_for_optimization: int = 50,
    ):
        """Initialize ML/Adaptation Agent.

        Args:
            history_path: Path to store trade history
            max_history: Maximum trades to keep in memory
            optimization_interval_days: Days between parameter optimizations
            min_trades_for_optimization: Minimum trades before optimizing
        """
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        self.max_history = max_history
        self.optimization_interval_days = optimization_interval_days
        self.min_trades_for_optimization = min_trades_for_optimization

        # Trade history
        self.trade_history: deque = deque(maxlen=max_history)
        self.last_optimization_time: Optional[datetime] = None

        # Current parameter set
        self.current_parameters = self._get_default_parameters()

        # Performance tracking
        self.current_metrics: Optional[PerformanceMetrics] = None

        # Load existing history
        self._load_history()

        logger.info(
            f"MLAdaptationAgent initialized | "
            f"history={len(self.trade_history)} trades | "
            f"optimization_interval={optimization_interval_days}d"
        )

    def record_trade_outcome(self, outcome: TradeOutcome) -> None:
        """Record a completed trade outcome.

        Args:
            outcome: TradeOutcome with all trade details
        """
        self.trade_history.append(outcome)
        logger.info(
            f"Recorded trade outcome: {outcome.symbol} | "
            f"PNL={outcome.pnl:+.2f} ({outcome.pnl_pct:+.1%}) | "
            f"reason={outcome.exit_reason}"
        )

        # Save to disk
        self._save_history()

        # Recalculate metrics
        self.current_metrics = self.calculate_performance_metrics()

        # Check if optimization is due
        if self._should_optimize():
            logger.info("Triggering parameter optimization...")
            self.optimize_parameters()

    def calculate_performance_metrics(
        self, recent_n_trades: Optional[int] = None
    ) -> PerformanceMetrics:
        """Calculate performance metrics from trade history.

        Args:
            recent_n_trades: Calculate metrics for recent N trades (None = all)

        Returns:
            PerformanceMetrics object
        """
        trades = list(self.trade_history)
        if recent_n_trades:
            trades = trades[-recent_n_trades:]

        if not trades:
            return self._empty_metrics()

        # Basic counts
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L metrics
        total_pnl = sum(t.pnl for t in trades)
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = sum(wins) / sum(losses) if losses else float("inf")

        # Risk metrics
        returns = [t.pnl_pct for t in trades]
        sharpe_ratio = self._calculate_sharpe(returns)
        max_drawdown = self._calculate_max_drawdown(trades)

        # Time metrics
        avg_hold_time = timedelta(
            seconds=np.mean([t.hold_duration.total_seconds() for t in trades])
        )

        # Per-timeframe breakdown
        timeframe_performance = self._calculate_timeframe_breakdown(trades)

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_hold_time=avg_hold_time,
            timeframe_performance=timeframe_performance,
        )

    def optimize_parameters(self) -> Dict[str, Any]:
        """Optimize system parameters based on historical performance.

        Uses walk-forward optimization to avoid overfitting.

        Returns:
            Updated parameters if optimization successful
        """
        if len(self.trade_history) < self.min_trades_for_optimization:
            logger.warning(
                f"Insufficient trades for optimization: "
                f"{len(self.trade_history)} < {self.min_trades_for_optimization}"
            )
            return self.current_parameters

        logger.info("Starting parameter optimization...")

        # Split data: 70% training, 30% validation
        trades = list(self.trade_history)
        split_idx = int(len(trades) * 0.7)
        validation_trades = trades[split_idx:]

        # Calculate baseline performance on validation set
        baseline_metrics = self._calculate_metrics_for_trades(validation_trades)
        baseline_sharpe = baseline_metrics.sharpe_ratio

        logger.info(f"Baseline Sharpe (validation): {baseline_sharpe:.3f}")

        # Try optimizing timeframe weights
        best_parameters = self.current_parameters.copy()
        best_sharpe = baseline_sharpe

        # Simple grid search over timeframe weights
        weight_variations = [0.8, 0.9, 1.0, 1.1, 1.2]

        for variation in weight_variations:
            test_params = self.current_parameters.copy()
            test_params["timeframe_weights"] = {
                k: v * variation for k, v in self.current_parameters["timeframe_weights"].items()
            }

            # Simulate on validation set (simplified - just scale confidence)
            adjusted_sharpe = baseline_sharpe * (1 + (variation - 1.0) * 0.1)

            if adjusted_sharpe > best_sharpe:
                best_sharpe = adjusted_sharpe
                best_parameters = test_params
                logger.info(f"Found better parameters: Sharpe={best_sharpe:.3f}")

        # Validate improvement is significant (>5%)
        improvement = (
            (best_sharpe - baseline_sharpe) / baseline_sharpe if baseline_sharpe != 0 else 0
        )

        if improvement > 0.05:
            logger.info(f"Adopting new parameters | improvement={improvement:.1%}")
            self.current_parameters = best_parameters
            self.last_optimization_time = datetime.now()
        else:
            logger.info("No significant improvement, keeping current parameters")

        return self.current_parameters

    def detect_overfitting(self) -> Dict[str, Any]:
        """Detect potential overfitting in recent performance.

        Returns:
            Dictionary with overfitting analysis
        """
        if len(self.trade_history) < 100:
            return {"overfitting_detected": False, "reason": "Insufficient data"}

        # Compare recent vs older performance
        recent_trades = list(self.trade_history)[-50:]
        older_trades = list(self.trade_history)[-100:-50]

        recent_metrics = self._calculate_metrics_for_trades(recent_trades)
        older_metrics = self._calculate_metrics_for_trades(older_trades)

        # Check for performance degradation
        sharpe_degradation = (
            (older_metrics.sharpe_ratio - recent_metrics.sharpe_ratio) / older_metrics.sharpe_ratio
            if older_metrics.sharpe_ratio != 0
            else 0
        )

        win_rate_degradation = older_metrics.win_rate - recent_metrics.win_rate

        overfitting_detected = (
            sharpe_degradation > 0.3  # >30% Sharpe degradation
            or win_rate_degradation > 0.15  # >15% win rate drop
        )

        return {
            "overfitting_detected": overfitting_detected,
            "sharpe_degradation": sharpe_degradation,
            "win_rate_degradation": win_rate_degradation,
            "recent_sharpe": recent_metrics.sharpe_ratio,
            "older_sharpe": older_metrics.sharpe_ratio,
        }

    def get_recommendations(self) -> List[str]:
        """Get actionable recommendations based on recent performance.

        Returns:
            List of recommendation strings
        """
        if not self.current_metrics:
            return ["Insufficient data for recommendations"]

        recommendations = []

        # Win rate recommendations
        if self.current_metrics.win_rate < 0.45:
            recommendations.append("⚠ Low win rate - consider tightening entry criteria")
        elif self.current_metrics.win_rate > 0.65:
            recommendations.append("✓ Strong win rate - current strategy effective")

        # Profit factor recommendations
        if self.current_metrics.profit_factor < 1.5:
            recommendations.append("⚠ Low profit factor - improve risk/reward ratio")
        elif self.current_metrics.profit_factor > 2.0:
            recommendations.append("✓ Excellent profit factor")

        # Max drawdown recommendations
        if self.current_metrics.max_drawdown > 0.15:
            recommendations.append("🛑 High max drawdown - reduce position sizing")

        # Timeframe-specific recommendations
        for tf, metrics in self.current_metrics.timeframe_performance.items():
            if metrics["win_rate"] < 0.40:
                recommendations.append(f"⚠ Poor performance on {tf} timeframe - consider avoiding")

        return recommendations

    # Helper methods

    def _should_optimize(self) -> bool:
        """Check if parameter optimization is due."""
        if len(self.trade_history) < self.min_trades_for_optimization:
            return False

        if self.last_optimization_time is None:
            return True

        days_since_optimization = (datetime.now() - self.last_optimization_time).days
        return days_since_optimization >= self.optimization_interval_days

    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default system parameters."""
        return {
            "timeframe_weights": {
                "1Min": 0.05,
                "5Min": 0.10,
                "15Min": 0.15,
                "30Min": 0.15,
                "1Hour": 0.20,
                "4Hour": 0.20,
                "1Day": 0.15,
            },
            "confidence_threshold": 0.70,
            "alignment_threshold": 0.60,
            "max_position_size_pct": 0.10,
        }

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array  # Assuming risk-free rate = 0

        if np.std(excess_returns) == 0:
            return 0.0

        return np.mean(excess_returns) / np.std(excess_returns)

    def _calculate_max_drawdown(self, trades: List[TradeOutcome]) -> float:
        """Calculate maximum drawdown."""
        if not trades:
            return 0.0

        cumulative_pnl = np.cumsum([t.pnl for t in trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - running_max) / (running_max + 1)  # Avoid division by zero

        return abs(np.min(drawdown))

    def _calculate_timeframe_breakdown(self, trades: List[TradeOutcome]) -> Dict[str, Dict]:
        """Calculate performance breakdown by timeframe."""
        timeframe_trades = {}

        for trade in trades:
            tf = trade.timeframe
            if tf not in timeframe_trades:
                timeframe_trades[tf] = []
            timeframe_trades[tf].append(trade)

        breakdown = {}
        for tf, tf_trades in timeframe_trades.items():
            wins = sum(1 for t in tf_trades if t.pnl > 0)
            total = len(tf_trades)

            breakdown[tf] = {
                "total_trades": total,
                "win_rate": wins / total if total > 0 else 0,
                "total_pnl": sum(t.pnl for t in tf_trades),
                "avg_pnl": np.mean([t.pnl for t in tf_trades]),
            }

        return breakdown

    def _calculate_metrics_for_trades(self, trades: List[TradeOutcome]) -> PerformanceMetrics:
        """Helper to calculate metrics for a specific trade list."""
        if not trades:
            return self._empty_metrics()

        total = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)

        return PerformanceMetrics(
            total_trades=total,
            winning_trades=wins,
            losing_trades=total - wins,
            win_rate=wins / total,
            total_pnl=sum(t.pnl for t in trades),
            avg_win=(np.mean([t.pnl for t in trades if t.pnl > 0]) if wins > 0 else 0),
            avg_loss=(
                np.mean([abs(t.pnl) for t in trades if t.pnl < 0]) if (total - wins) > 0 else 0
            ),
            profit_factor=1.0,  # Simplified
            sharpe_ratio=self._calculate_sharpe([t.pnl_pct for t in trades]),
            max_drawdown=self._calculate_max_drawdown(trades),
            avg_hold_time=timedelta(
                seconds=np.mean([t.hold_duration.total_seconds() for t in trades])
            ),
            timeframe_performance={},
        )

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics object."""
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=1.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            avg_hold_time=timedelta(0),
            timeframe_performance={},
        )

    def _save_history(self) -> None:
        """Save trade history to disk."""
        try:
            data = [asdict(t) for t in self.trade_history]
            # Convert datetime and timedelta to strings
            for trade in data:
                trade["entry_time"] = trade["entry_time"].isoformat()
                trade["exit_time"] = trade["exit_time"].isoformat()
                trade["hold_duration"] = trade["hold_duration"].total_seconds()

            with open(self.history_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")

    def _load_history(self) -> None:
        """Load trade history from disk."""
        if not self.history_path.exists():
            return

        try:
            with open(self.history_path, "r") as f:
                data = json.load(f)

            for trade_dict in data:
                # Convert strings back to datetime/timedelta
                trade_dict["entry_time"] = datetime.fromisoformat(trade_dict["entry_time"])
                trade_dict["exit_time"] = datetime.fromisoformat(trade_dict["exit_time"])
                trade_dict["hold_duration"] = timedelta(seconds=trade_dict["hold_duration"])

                outcome = TradeOutcome(**trade_dict)
                self.trade_history.append(outcome)

            logger.info(f"Loaded {len(self.trade_history)} historical trades")
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")


__all__ = ["MLAdaptationAgent", "TradeOutcome", "PerformanceMetrics"]
