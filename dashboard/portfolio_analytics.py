"""
Portfolio Analytics UI

Visual analytics interface for:
- Performance attribution
- Risk analysis
- Greeks visualization
- Historical analysis
- Sector allocation

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class TimeFrame(str, Enum):
    """Analytics time frames."""
    DAY = "1D"
    WEEK = "1W"
    MONTH = "1M"
    QUARTER = "3M"
    YEAR = "1Y"
    ALL = "ALL"


class ChartType(str, Enum):
    """Chart visualization types."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"
    HEATMAP = "heatmap"
    SCATTER = "scatter"


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    total_return: float
    total_return_pct: float
    daily_return_avg: float
    daily_return_std: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Drawdown
    max_drawdown: float
    max_drawdown_duration_days: int
    current_drawdown: float
    
    # Win/Loss
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Time stats
    best_day: float
    worst_day: float
    positive_days: int
    negative_days: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "daily_return_avg": self.daily_return_avg,
            "daily_return_std": self.daily_return_std,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "current_drawdown": self.current_drawdown,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "best_day": self.best_day,
            "worst_day": self.worst_day,
            "positive_days": self.positive_days,
            "negative_days": self.negative_days,
        }


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""
    # Greeks
    portfolio_delta: float
    portfolio_gamma: float
    portfolio_theta: float
    portfolio_vega: float
    
    # Beta
    portfolio_beta: float
    beta_weighted_delta: float
    
    # Value at Risk
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # Conditional VaR
    
    # Exposure
    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    
    # Concentration
    largest_position_pct: float
    top_5_concentration: float
    hhi_index: float  # Herfindahl-Hirschman Index
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "greeks": {
                "delta": self.portfolio_delta,
                "gamma": self.portfolio_gamma,
                "theta": self.portfolio_theta,
                "vega": self.portfolio_vega,
            },
            "beta": self.portfolio_beta,
            "beta_weighted_delta": self.beta_weighted_delta,
            "var": {
                "var_95": self.var_95,
                "var_99": self.var_99,
                "cvar_95": self.cvar_95,
            },
            "exposure": {
                "gross": self.gross_exposure,
                "net": self.net_exposure,
                "long": self.long_exposure,
                "short": self.short_exposure,
            },
            "concentration": {
                "largest_position_pct": self.largest_position_pct,
                "top_5_concentration": self.top_5_concentration,
                "hhi_index": self.hhi_index,
            },
        }


@dataclass
class SectorAllocation:
    """Sector allocation data."""
    sector: str
    market_value: float
    percentage: float
    pnl: float
    pnl_pct: float
    positions: int


@dataclass
class ChartData:
    """Data for chart visualization."""
    chart_type: ChartType
    title: str
    labels: List[str]
    datasets: List[Dict[str, Any]]
    options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Chart.js compatible format."""
        return {
            "type": self.chart_type.value,
            "data": {
                "labels": self.labels,
                "datasets": self.datasets,
            },
            "options": self.options,
        }


class PortfolioAnalytics:
    """
    Portfolio analytics and visualization.
    
    Features:
    - Performance metrics calculation
    - Risk analysis
    - Attribution analysis
    - Chart generation
    - Historical analysis
    """
    
    def __init__(self):
        """Initialize analytics."""
        # Historical data storage
        self._portfolio_history: List[Dict[str, Any]] = []
        self._trade_history: List[Dict[str, Any]] = []
        self._positions: List[Dict[str, Any]] = []
        
        # Cached metrics
        self._last_performance: Optional[PerformanceMetrics] = None
        self._last_risk: Optional[RiskMetrics] = None
        
        logger.info("PortfolioAnalytics initialized")
    
    def update_portfolio_snapshot(
        self,
        timestamp: datetime,
        total_value: float,
        cash: float,
        positions: List[Dict[str, Any]],
    ) -> None:
        """Add portfolio snapshot."""
        self._portfolio_history.append({
            "timestamp": timestamp,
            "total_value": total_value,
            "cash": cash,
            "positions_count": len(positions),
        })
        
        self._positions = positions
        
        # Keep last 365 days
        cutoff = datetime.now() - timedelta(days=365)
        self._portfolio_history = [
            h for h in self._portfolio_history if h["timestamp"] > cutoff
        ]
    
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """Add trade to history."""
        trade["timestamp"] = trade.get("timestamp", datetime.now())
        self._trade_history.append(trade)
    
    def calculate_performance(
        self,
        timeframe: TimeFrame = TimeFrame.MONTH,
    ) -> PerformanceMetrics:
        """Calculate performance metrics."""
        # Filter history by timeframe
        history = self._filter_by_timeframe(timeframe)
        
        if len(history) < 2:
            return self._empty_performance()
        
        # Calculate returns
        values = [h["total_value"] for h in history]
        daily_returns = [
            (values[i] - values[i-1]) / values[i-1] if values[i-1] > 0 else 0
            for i in range(1, len(values))
        ]
        
        if not daily_returns:
            return self._empty_performance()
        
        # Basic stats
        total_return = values[-1] - values[0]
        total_return_pct = total_return / values[0] * 100 if values[0] > 0 else 0
        
        avg_return = statistics.mean(daily_returns) if daily_returns else 0
        std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe = (avg_return * 252) / (std_return * math.sqrt(252)) if std_return > 0 else 0
        
        # Sortino ratio (downside deviation)
        negative_returns = [r for r in daily_returns if r < 0]
        downside_std = statistics.stdev(negative_returns) if len(negative_returns) > 1 else std_return
        sortino = (avg_return * 252) / (downside_std * math.sqrt(252)) if downside_std > 0 else 0
        
        # Drawdown calculation
        peak = values[0]
        max_dd = 0
        dd_duration = 0
        current_dd_start = None
        max_dd_duration = 0
        
        for i, val in enumerate(values):
            if val > peak:
                peak = val
                if current_dd_start is not None:
                    dd_duration = i - current_dd_start
                    max_dd_duration = max(max_dd_duration, dd_duration)
                current_dd_start = None
            else:
                dd = (peak - val) / peak
                if dd > max_dd:
                    max_dd = dd
                    if current_dd_start is None:
                        current_dd_start = i
        
        current_dd = (peak - values[-1]) / peak if peak > 0 else 0
        
        # Calmar ratio
        calmar = (total_return_pct / 100 * (252 / len(values))) / max_dd if max_dd > 0 else 0
        
        # Win/Loss analysis
        positive_returns = [r for r in daily_returns if r > 0]
        negative_returns = [r for r in daily_returns if r < 0]
        
        win_rate = len(positive_returns) / len(daily_returns) * 100 if daily_returns else 0
        avg_win = statistics.mean(positive_returns) * 100 if positive_returns else 0
        avg_loss = statistics.mean(negative_returns) * 100 if negative_returns else 0
        
        gross_profit = sum(positive_returns)
        gross_loss = abs(sum(negative_returns))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            daily_return_avg=avg_return * 100,
            daily_return_std=std_return * 100,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd * 100,
            max_drawdown_duration_days=max_dd_duration,
            current_drawdown=current_dd * 100,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            best_day=max(daily_returns) * 100 if daily_returns else 0,
            worst_day=min(daily_returns) * 100 if daily_returns else 0,
            positive_days=len(positive_returns),
            negative_days=len(negative_returns),
        )
    
    def _empty_performance(self) -> PerformanceMetrics:
        """Return empty performance metrics."""
        return PerformanceMetrics(
            total_return=0, total_return_pct=0, daily_return_avg=0, daily_return_std=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            max_drawdown=0, max_drawdown_duration_days=0, current_drawdown=0,
            win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
            best_day=0, worst_day=0, positive_days=0, negative_days=0,
        )
    
    def calculate_risk(self, portfolio_value: float = 100000) -> RiskMetrics:
        """Calculate risk metrics."""
        # Aggregate Greeks from positions
        total_delta = sum(p.get("delta", 0) for p in self._positions)
        total_gamma = sum(p.get("gamma", 0) for p in self._positions)
        total_theta = sum(p.get("theta", 0) for p in self._positions)
        total_vega = sum(p.get("vega", 0) for p in self._positions)
        
        # Beta
        weighted_beta = sum(
            p.get("market_value", 0) * p.get("beta", 1) 
            for p in self._positions
        )
        total_mv = sum(p.get("market_value", 0) for p in self._positions)
        portfolio_beta = weighted_beta / total_mv if total_mv > 0 else 1
        
        # Exposure
        long_mv = sum(p.get("market_value", 0) for p in self._positions if p.get("market_value", 0) > 0)
        short_mv = abs(sum(p.get("market_value", 0) for p in self._positions if p.get("market_value", 0) < 0))
        gross_exposure = long_mv + short_mv
        net_exposure = long_mv - short_mv
        
        # VaR calculation (simplified parametric VaR)
        history = self._filter_by_timeframe(TimeFrame.MONTH)
        if len(history) > 1:
            values = [h["total_value"] for h in history]
            returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values)) if values[i-1] > 0]
            
            if returns:
                mean_ret = statistics.mean(returns)
                std_ret = statistics.stdev(returns) if len(returns) > 1 else 0
                
                var_95 = portfolio_value * (mean_ret - 1.645 * std_ret)
                var_99 = portfolio_value * (mean_ret - 2.326 * std_ret)
                
                # CVaR (expected shortfall)
                sorted_returns = sorted(returns)
                cutoff_idx = int(len(sorted_returns) * 0.05)
                worst_returns = sorted_returns[:max(1, cutoff_idx)]
                cvar_95 = portfolio_value * statistics.mean(worst_returns) if worst_returns else var_95
            else:
                var_95 = var_99 = cvar_95 = 0
        else:
            var_95 = var_99 = cvar_95 = 0
        
        # Concentration
        position_values = [abs(p.get("market_value", 0)) for p in self._positions]
        if position_values and portfolio_value > 0:
            largest_pct = max(position_values) / portfolio_value * 100
            sorted_values = sorted(position_values, reverse=True)
            top_5 = sum(sorted_values[:5]) / portfolio_value * 100 if len(sorted_values) >= 5 else 100
            
            # HHI
            shares = [v / portfolio_value for v in position_values]
            hhi = sum(s ** 2 for s in shares) * 10000
        else:
            largest_pct = 0
            top_5 = 0
            hhi = 0
        
        return RiskMetrics(
            portfolio_delta=total_delta,
            portfolio_gamma=total_gamma,
            portfolio_theta=total_theta,
            portfolio_vega=total_vega,
            portfolio_beta=portfolio_beta,
            beta_weighted_delta=total_delta * portfolio_beta,
            var_95=abs(var_95),
            var_99=abs(var_99),
            cvar_95=abs(cvar_95),
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            long_exposure=long_mv,
            short_exposure=short_mv,
            largest_position_pct=largest_pct,
            top_5_concentration=top_5,
            hhi_index=hhi,
        )
    
    def get_sector_allocation(self) -> List[SectorAllocation]:
        """Get sector allocation breakdown."""
        sector_data: Dict[str, Dict[str, Any]] = {}
        
        for pos in self._positions:
            sector = pos.get("sector", "Other")
            mv = pos.get("market_value", 0)
            pnl = pos.get("unrealized_pnl", 0)
            
            if sector not in sector_data:
                sector_data[sector] = {
                    "market_value": 0,
                    "pnl": 0,
                    "positions": 0,
                    "cost_basis": 0,
                }
            
            sector_data[sector]["market_value"] += mv
            sector_data[sector]["pnl"] += pnl
            sector_data[sector]["positions"] += 1
            sector_data[sector]["cost_basis"] += pos.get("cost_basis", mv)
        
        total_mv = sum(d["market_value"] for d in sector_data.values())
        
        allocations = []
        for sector, data in sector_data.items():
            pct = data["market_value"] / total_mv * 100 if total_mv > 0 else 0
            pnl_pct = data["pnl"] / data["cost_basis"] * 100 if data["cost_basis"] > 0 else 0
            
            allocations.append(SectorAllocation(
                sector=sector,
                market_value=data["market_value"],
                percentage=pct,
                pnl=data["pnl"],
                pnl_pct=pnl_pct,
                positions=data["positions"],
            ))
        
        return sorted(allocations, key=lambda x: x.market_value, reverse=True)
    
    def generate_pnl_chart(self, timeframe: TimeFrame = TimeFrame.MONTH) -> ChartData:
        """Generate P&L chart data."""
        history = self._filter_by_timeframe(timeframe)
        
        if not history:
            return ChartData(
                chart_type=ChartType.LINE,
                title="Portfolio P&L",
                labels=[],
                datasets=[],
            )
        
        labels = [h["timestamp"].strftime("%Y-%m-%d") for h in history]
        values = [h["total_value"] for h in history]
        
        # Calculate cumulative P&L
        initial_value = values[0]
        pnl = [v - initial_value for v in values]
        
        return ChartData(
            chart_type=ChartType.LINE,
            title="Portfolio P&L",
            labels=labels,
            datasets=[{
                "label": "P&L",
                "data": pnl,
                "borderColor": "#10b981",
                "backgroundColor": "rgba(16, 185, 129, 0.1)",
                "fill": True,
            }],
            options={
                "responsive": True,
                "plugins": {
                    "legend": {"display": True},
                    "title": {"display": True, "text": "Cumulative P&L"},
                },
            },
        )
    
    def generate_sector_pie_chart(self) -> ChartData:
        """Generate sector allocation pie chart."""
        allocations = self.get_sector_allocation()
        
        colors = [
            "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
            "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
        ]
        
        return ChartData(
            chart_type=ChartType.PIE,
            title="Sector Allocation",
            labels=[a.sector for a in allocations],
            datasets=[{
                "data": [a.percentage for a in allocations],
                "backgroundColor": colors[:len(allocations)],
            }],
            options={
                "responsive": True,
                "plugins": {
                    "legend": {"position": "right"},
                    "title": {"display": True, "text": "Sector Allocation"},
                },
            },
        )
    
    def generate_greeks_bar_chart(self) -> ChartData:
        """Generate Greeks exposure bar chart."""
        risk = self.calculate_risk()
        
        return ChartData(
            chart_type=ChartType.BAR,
            title="Greeks Exposure",
            labels=["Delta", "Gamma", "Theta", "Vega"],
            datasets=[{
                "label": "Exposure",
                "data": [
                    risk.portfolio_delta,
                    risk.portfolio_gamma * 100,  # Scale gamma
                    risk.portfolio_theta,
                    risk.portfolio_vega,
                ],
                "backgroundColor": [
                    "#3b82f6",
                    "#10b981",
                    "#ef4444",
                    "#f59e0b",
                ],
            }],
            options={
                "responsive": True,
                "plugins": {
                    "title": {"display": True, "text": "Portfolio Greeks"},
                },
                "scales": {
                    "y": {"beginAtZero": True},
                },
            },
        )
    
    def generate_drawdown_chart(self, timeframe: TimeFrame = TimeFrame.MONTH) -> ChartData:
        """Generate drawdown chart."""
        history = self._filter_by_timeframe(timeframe)
        
        if not history:
            return ChartData(
                chart_type=ChartType.AREA,
                title="Drawdown",
                labels=[],
                datasets=[],
            )
        
        values = [h["total_value"] for h in history]
        labels = [h["timestamp"].strftime("%Y-%m-%d") for h in history]
        
        # Calculate drawdown series
        peak = values[0]
        drawdowns = []
        
        for val in values:
            if val > peak:
                peak = val
            dd = (val - peak) / peak * 100 if peak > 0 else 0
            drawdowns.append(dd)
        
        return ChartData(
            chart_type=ChartType.AREA,
            title="Portfolio Drawdown",
            labels=labels,
            datasets=[{
                "label": "Drawdown %",
                "data": drawdowns,
                "borderColor": "#ef4444",
                "backgroundColor": "rgba(239, 68, 68, 0.3)",
                "fill": True,
            }],
            options={
                "responsive": True,
                "plugins": {
                    "title": {"display": True, "text": "Drawdown Analysis"},
                },
                "scales": {
                    "y": {"max": 0},
                },
            },
        )
    
    def _filter_by_timeframe(self, timeframe: TimeFrame) -> List[Dict[str, Any]]:
        """Filter history by timeframe."""
        now = datetime.now()
        
        days_map = {
            TimeFrame.DAY: 1,
            TimeFrame.WEEK: 7,
            TimeFrame.MONTH: 30,
            TimeFrame.QUARTER: 90,
            TimeFrame.YEAR: 365,
            TimeFrame.ALL: 9999,
        }
        
        cutoff = now - timedelta(days=days_map.get(timeframe, 30))
        
        return [h for h in self._portfolio_history if h["timestamp"] > cutoff]
    
    def generate_full_report(self) -> Dict[str, Any]:
        """Generate full analytics report."""
        performance = self.calculate_performance()
        risk = self.calculate_risk()
        sectors = self.get_sector_allocation()
        
        return {
            "generated_at": datetime.now().isoformat(),
            "performance": performance.to_dict(),
            "risk": risk.to_dict(),
            "sector_allocation": [
                {
                    "sector": s.sector,
                    "market_value": s.market_value,
                    "percentage": s.percentage,
                    "pnl": s.pnl,
                    "pnl_pct": s.pnl_pct,
                    "positions": s.positions,
                }
                for s in sectors
            ],
            "charts": {
                "pnl": self.generate_pnl_chart().to_dict(),
                "sector": self.generate_sector_pie_chart().to_dict(),
                "greeks": self.generate_greeks_bar_chart().to_dict(),
                "drawdown": self.generate_drawdown_chart().to_dict(),
            },
            "summary": {
                "total_positions": len(self._positions),
                "history_days": len(self._portfolio_history),
                "total_trades": len(self._trade_history),
            },
        }
    
    def generate_html_report(self) -> str:
        """Generate HTML analytics report."""
        report = self.generate_full_report()
        perf = report["performance"]
        risk = report["risk"]
        
        return f'''<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Analytics Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: system-ui; background: #0d1117; color: #c9d1d9; padding: 2rem; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1, h2 {{ color: #58a6ff; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }}
        .card {{ background: #161b22; border-radius: 8px; padding: 1rem; }}
        .metric {{ margin-bottom: 0.5rem; }}
        .metric-label {{ color: #8b949e; font-size: 0.875rem; }}
        .metric-value {{ font-size: 1.5rem; font-weight: bold; }}
        .positive {{ color: #238636; }}
        .negative {{ color: #da3633; }}
        .chart-container {{ height: 300px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Portfolio Analytics Report</h1>
        <p>Generated: {report["generated_at"]}</p>
        
        <h2>Performance Summary</h2>
        <div class="grid">
            <div class="card">
                <div class="metric">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {'positive' if perf['total_return_pct'] >= 0 else 'negative'}">
                        {perf['total_return_pct']:.2f}%
                    </div>
                </div>
            </div>
            <div class="card">
                <div class="metric">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">{perf['sharpe_ratio']:.2f}</div>
                </div>
            </div>
            <div class="card">
                <div class="metric">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">-{perf['max_drawdown']:.2f}%</div>
                </div>
            </div>
            <div class="card">
                <div class="metric">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{perf['win_rate']:.1f}%</div>
                </div>
            </div>
        </div>
        
        <h2>Risk Metrics</h2>
        <div class="grid">
            <div class="card">
                <div class="metric">
                    <div class="metric-label">Portfolio Beta</div>
                    <div class="metric-value">{risk['beta']:.2f}</div>
                </div>
            </div>
            <div class="card">
                <div class="metric">
                    <div class="metric-label">VaR (95%)</div>
                    <div class="metric-value">${risk['var']['var_95']:,.0f}</div>
                </div>
            </div>
            <div class="card">
                <div class="metric">
                    <div class="metric-label">Net Exposure</div>
                    <div class="metric-value">${risk['exposure']['net']:,.0f}</div>
                </div>
            </div>
            <div class="card">
                <div class="metric">
                    <div class="metric-label">Concentration (Top 5)</div>
                    <div class="metric-value">{risk['concentration']['top_5_concentration']:.1f}%</div>
                </div>
            </div>
        </div>
        
        <h2>Greeks Exposure</h2>
        <div class="grid">
            <div class="card">
                <div class="metric">
                    <div class="metric-label">Delta</div>
                    <div class="metric-value">{risk['greeks']['delta']:.1f}</div>
                </div>
            </div>
            <div class="card">
                <div class="metric">
                    <div class="metric-label">Gamma</div>
                    <div class="metric-value">{risk['greeks']['gamma']:.3f}</div>
                </div>
            </div>
            <div class="card">
                <div class="metric">
                    <div class="metric-label">Theta</div>
                    <div class="metric-value">${risk['greeks']['theta']:.2f}/day</div>
                </div>
            </div>
            <div class="card">
                <div class="metric">
                    <div class="metric-label">Vega</div>
                    <div class="metric-value">{risk['greeks']['vega']:.1f}</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>'''


# Singleton instance
portfolio_analytics = PortfolioAnalytics()


# Convenience functions
def calculate_performance(timeframe: TimeFrame = TimeFrame.MONTH) -> PerformanceMetrics:
    """Calculate performance metrics."""
    return portfolio_analytics.calculate_performance(timeframe)


def calculate_risk(portfolio_value: float = 100000) -> RiskMetrics:
    """Calculate risk metrics."""
    return portfolio_analytics.calculate_risk(portfolio_value)


def generate_report() -> Dict[str, Any]:
    """Generate full analytics report."""
    return portfolio_analytics.generate_full_report()
