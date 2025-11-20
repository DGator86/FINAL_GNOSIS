"""Risk Management Agent for GNOSIS Trading System."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from agents.base_agent import AgentSignal, BaseAgent


class RiskManagementAgent(BaseAgent):
    """Agent that enforces risk limits and portfolio constraints."""

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)

        # Risk limits
        self.max_portfolio_risk = config.get("max_portfolio_risk", 0.02)  # 2% max risk
        self.max_position_size = config.get("max_position_size", 0.10)  # 10% per position
        self.max_drawdown = config.get("max_drawdown", 0.15)  # 15% max drawdown
        self.max_leverage = config.get("max_leverage", 1.0)  # No leverage by default
        self.max_correlation = config.get("max_correlation", 0.7)  # Max correlation between positions

        # Daily limits
        self.max_daily_loss = config.get("max_daily_loss", 0.05)  # 5% daily loss limit
        self.max_daily_trades = config.get("max_daily_trades", 10)

        # Portfolio state
        self.portfolio_value = config.get("initial_capital", 100000)
        self.peak_value = self.portfolio_value
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.trade_history = deque(maxlen=1000)

        # Risk metrics
        self.var_95 = 0.0  # Value at Risk (95%)
        self.expected_shortfall = 0.0
        self.sharpe_ratio = 0.0

    def analyze(self, market_data: Dict[str, Any], features: Dict[str, Any]) -> AgentSignal:
        """Analyze portfolio risk and generate risk-adjusted signal."""

        self._update_portfolio_state(market_data)
        self._calculate_risk_metrics()
        risk_check = self._check_risk_limits()

        if not risk_check["pass"]:
            return self._create_risk_signal(
                "hold",
                f"Risk limit exceeded: {risk_check['reason']}",
                risk_check,
            )

        if not self._can_open_position():
            return self._create_risk_signal(
                "hold",
                "Cannot open new position: daily limits reached",
                risk_check,
            )

        optimal_size = self._calculate_optimal_position_size(market_data, features)

        signal = self._generate_risk_signal("buy", "Risk check passed", risk_check)
        signal.position_size = optimal_size

        return signal

    def _update_portfolio_state(self, market_data: Dict[str, Any]) -> None:
        """Update portfolio state from latest market data."""

        current_price = market_data.get("close", market_data.get("price", 0.0))
        timestamp = market_data.get("timestamp", datetime.now())

        total_position_value = 0.0
        for position in self.positions.values():
            position_value = position["quantity"] * current_price
            total_position_value += abs(position_value)

        cash = self.portfolio_value - total_position_value
        self.portfolio_value = cash + total_position_value

        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        if hasattr(self, "last_update_date"):
            if timestamp.date() != self.last_update_date:
                self.daily_pnl = 0.0
                self.daily_trades = 0

        self.last_update_date = timestamp.date()

    def _calculate_risk_metrics(self) -> None:
        """Calculate portfolio risk metrics."""

        if len(self.trade_history) < 30:
            return

        returns = np.array([t["return"] for t in self.trade_history if "return" in t])

        if len(returns) == 0:
            return

        self.var_95 = float(np.percentile(returns, 5))
        tail_losses = returns[returns <= self.var_95]
        if len(tail_losses) > 0:
            self.expected_shortfall = float(np.mean(tail_losses))

        if np.std(returns) > 0:
            self.sharpe_ratio = float((np.mean(returns) / np.std(returns)) * np.sqrt(252))

    def _check_risk_limits(self) -> Dict[str, Any]:
        """Check all risk limits and return diagnostic info."""

        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        if current_drawdown > self.max_drawdown:
            return {
                "pass": False,
                "reason": f"Max drawdown exceeded: {current_drawdown:.2%} > {self.max_drawdown:.2%}",
                "metric": "drawdown",
                "value": current_drawdown,
            }

        daily_loss_pct = abs(self.daily_pnl) / self.portfolio_value
        if self.daily_pnl < 0 and daily_loss_pct > self.max_daily_loss:
            return {
                "pass": False,
                "reason": f"Daily loss limit exceeded: {daily_loss_pct:.2%} > {self.max_daily_loss:.2%}",
                "metric": "daily_loss",
                "value": daily_loss_pct,
            }

        if self.daily_trades >= self.max_daily_trades:
            return {
                "pass": False,
                "reason": f"Daily trade limit reached: {self.daily_trades} >= {self.max_daily_trades}",
                "metric": "daily_trades",
                "value": self.daily_trades,
            }

        total_exposure = sum(abs(pos["quantity"] * pos["current_price"]) for pos in self.positions.values())

        if total_exposure > self.portfolio_value * self.max_leverage:
            return {
                "pass": False,
                "reason": "Leverage limit exceeded",
                "metric": "leverage",
                "value": total_exposure / self.portfolio_value,
            }

        return {
            "pass": True,
            "drawdown": current_drawdown,
            "daily_loss": daily_loss_pct,
            "daily_trades": self.daily_trades,
            "var_95": self.var_95,
            "sharpe_ratio": self.sharpe_ratio,
        }

    def _can_open_position(self) -> bool:
        """Return True if daily limits allow new positions."""

        if self.daily_trades >= self.max_daily_trades:
            return False

        daily_loss_pct = abs(self.daily_pnl) / self.portfolio_value
        if self.daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss:
            return False

        return True

    def _calculate_optimal_position_size(self, market_data: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Calculate optimal position size using a conservative Kelly criterion."""

        del market_data, features  # presently unused inputs

        if len(self.trade_history) < 10:
            return self.max_position_size * 0.5

        returns = [t["return"] for t in self.trade_history if "return" in t]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]

        if not wins or not losses:
            return self.max_position_size * 0.5

        win_rate = len(wins) / len(returns)
        avg_win = float(np.mean(wins))
        avg_loss = abs(float(np.mean(losses)))

        if avg_loss > 0:
            kelly_fraction = win_rate - ((1 - win_rate) * (avg_win / avg_loss))
            kelly_fraction = float(np.clip(kelly_fraction, 0, 1))
        else:
            kelly_fraction = 0.5

        fractional_kelly = kelly_fraction * 0.25
        optimal_size = min(fractional_kelly, self.max_position_size)

        return float(optimal_size)

    def _generate_risk_signal(self, signal_type: str, reason: str, risk_check: Dict[str, Any]) -> AgentSignal:
        """Generate risk management signal."""

        return AgentSignal(
            agent_id=self.agent_id,
            timestamp=datetime.now(),
            signal_type=signal_type,
            confidence=1.0 if risk_check["pass"] else 0.0,
            reasoning=reason,
            metadata={
                "risk_check": risk_check,
                "portfolio_value": self.portfolio_value,
                "drawdown": (self.peak_value - self.portfolio_value) / self.peak_value,
                "daily_pnl": self.daily_pnl,
                "var_95": self.var_95,
                "sharpe_ratio": self.sharpe_ratio,
            },
        )

    def _create_risk_signal(self, signal_type: str, reason: str, risk_check: Dict[str, Any]) -> AgentSignal:
        """Create a risk signal (alias for :meth:`_generate_risk_signal`)."""

        return self._generate_risk_signal(signal_type, reason, risk_check)

    def update_position(self, symbol: str, quantity: float, price: float) -> None:
        """Update or close a position and maintain average price."""

        if symbol not in self.positions:
            self.positions[symbol] = {"quantity": 0.0, "avg_price": 0.0, "current_price": price}

        position = self.positions[symbol]

        total_cost = position["quantity"] * position["avg_price"]
        new_cost = quantity * price
        new_quantity = position["quantity"] + quantity

        if new_quantity != 0:
            position["avg_price"] = (total_cost + new_cost) / new_quantity

        position["quantity"] = new_quantity
        position["current_price"] = price

        if abs(position["quantity"]) < 1e-6:
            del self.positions[symbol]

    def record_trade(self, trade: Dict[str, Any]) -> None:
        """Record trade for risk calculations and daily limits."""

        self.trade_history.append(trade)
        self.daily_trades += 1

        if "pnl" in trade:
            self.daily_pnl += trade["pnl"]

    def update_state(self, market_data: Dict[str, Any], execution_result: Optional[Dict[str, Any]] = None) -> None:
        """Update agent state based on market data and executed trades."""

        self._update_portfolio_state(market_data)

        if execution_result:
            if "symbol" in execution_result:
                self.record_trade(execution_result)

                if "quantity" in execution_result and "price" in execution_result:
                    self.update_position(
                        execution_result["symbol"],
                        execution_result["quantity"],
                        execution_result["price"],
                    )

        self.state["portfolio_value"] = self.portfolio_value
        self.state["positions"] = self.positions
        self.state["daily_pnl"] = self.daily_pnl
        self.state["daily_trades"] = self.daily_trades
        self.state["risk_metrics"] = {
            "var_95": self.var_95,
            "expected_shortfall": self.expected_shortfall,
            "sharpe_ratio": self.sharpe_ratio,
            "drawdown": (self.peak_value - self.portfolio_value) / self.peak_value,
        }

    def get_risk_summary(self) -> Dict[str, Any]:
        """Return a summary of current risk metrics and limits."""

        risk_check = self._check_risk_limits()

        return {
            "portfolio_value": self.portfolio_value,
            "peak_value": self.peak_value,
            "current_drawdown": (self.peak_value - self.portfolio_value) / self.peak_value,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "var_95": self.var_95,
            "expected_shortfall": self.expected_shortfall,
            "sharpe_ratio": self.sharpe_ratio,
            "risk_limits": {
                "max_drawdown": self.max_drawdown,
                "max_daily_loss": self.max_daily_loss,
                "max_daily_trades": self.max_daily_trades,
                "max_leverage": self.max_leverage,
            },
            "risk_check": risk_check,
            "position_count": len(self.positions),
            "positions": self.positions,
        }
