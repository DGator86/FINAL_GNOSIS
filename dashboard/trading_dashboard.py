"""
Trading Dashboard - Web UI for Portfolio Monitoring

Real-time dashboard for:
- Portfolio overview
- Position tracking
- Greeks exposure
- P&L monitoring
- Alert management
- Order execution

Author: Super Gnosis Elite Trading System
Version: 1.0.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import asyncio

from loguru import logger


class DashboardTheme(str, Enum):
    """Dashboard color themes."""
    DARK = "dark"
    LIGHT = "light"
    TRADING = "trading"


class WidgetType(str, Enum):
    """Dashboard widget types."""
    PORTFOLIO_SUMMARY = "portfolio_summary"
    POSITIONS_TABLE = "positions_table"
    GREEKS_DISPLAY = "greeks_display"
    PNL_CHART = "pnl_chart"
    ALERTS_PANEL = "alerts_panel"
    ORDER_ENTRY = "order_entry"
    WATCHLIST = "watchlist"
    MARKET_OVERVIEW = "market_overview"
    FLOW_SCANNER = "flow_scanner"
    NEWS_FEED = "news_feed"


@dataclass
class WidgetConfig:
    """Widget configuration."""
    widget_type: WidgetType
    title: str
    position: Dict[str, int]  # x, y, width, height
    refresh_interval: int = 5  # seconds
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardLayout:
    """Dashboard layout configuration."""
    name: str
    theme: DashboardTheme = DashboardTheme.DARK
    widgets: List[WidgetConfig] = field(default_factory=list)
    grid_columns: int = 12
    grid_rows: int = 8


class DashboardDataProvider:
    """Provides data for dashboard widgets."""
    
    def __init__(self):
        """Initialize data provider."""
        self._portfolio_data: Dict[str, Any] = {}
        self._positions: List[Dict[str, Any]] = []
        self._greeks: Dict[str, float] = {}
        self._alerts: List[Dict[str, Any]] = []
        self._pnl_history: List[Dict[str, Any]] = []
        
    def update_portfolio(self, data: Dict[str, Any]) -> None:
        """Update portfolio data."""
        self._portfolio_data = data
        self._portfolio_data["last_updated"] = datetime.now().isoformat()
    
    def update_positions(self, positions: List[Dict[str, Any]]) -> None:
        """Update positions data."""
        self._positions = positions
    
    def update_greeks(self, greeks: Dict[str, float]) -> None:
        """Update Greeks exposure."""
        self._greeks = greeks
    
    def add_alert(self, alert: Dict[str, Any]) -> None:
        """Add new alert."""
        alert["timestamp"] = datetime.now().isoformat()
        self._alerts.insert(0, alert)
        self._alerts = self._alerts[:100]  # Keep last 100
    
    def add_pnl_point(self, pnl: float, timestamp: Optional[datetime] = None) -> None:
        """Add P&L data point."""
        self._pnl_history.append({
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "pnl": pnl,
        })
        # Keep last 1000 points
        self._pnl_history = self._pnl_history[-1000:]
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary for widget."""
        return {
            "total_value": self._portfolio_data.get("total_value", 0),
            "cash": self._portfolio_data.get("cash", 0),
            "buying_power": self._portfolio_data.get("buying_power", 0),
            "day_pnl": self._portfolio_data.get("day_pnl", 0),
            "day_pnl_pct": self._portfolio_data.get("day_pnl_pct", 0),
            "total_pnl": self._portfolio_data.get("total_pnl", 0),
            "positions_count": len(self._positions),
            "last_updated": self._portfolio_data.get("last_updated"),
        }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions for widget."""
        return self._positions
    
    def get_greeks(self) -> Dict[str, Any]:
        """Get Greeks for widget."""
        return {
            "delta": self._greeks.get("delta", 0),
            "gamma": self._greeks.get("gamma", 0),
            "theta": self._greeks.get("theta", 0),
            "vega": self._greeks.get("vega", 0),
            "rho": self._greeks.get("rho", 0),
            "beta_weighted_delta": self._greeks.get("beta_weighted_delta", 0),
        }
    
    def get_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return self._alerts[:limit]
    
    def get_pnl_chart_data(self) -> List[Dict[str, Any]]:
        """Get P&L chart data."""
        return self._pnl_history


class TradingDashboard:
    """
    Main trading dashboard class.
    
    Provides web-based UI for portfolio monitoring
    and trade management.
    """
    
    def __init__(self, layout: Optional[DashboardLayout] = None):
        """Initialize dashboard."""
        self.layout = layout or self._default_layout()
        self.data_provider = DashboardDataProvider()
        self._connected_clients: List[Any] = []
        
        logger.info("TradingDashboard initialized")
    
    def _default_layout(self) -> DashboardLayout:
        """Create default dashboard layout."""
        return DashboardLayout(
            name="Default Trading Dashboard",
            theme=DashboardTheme.DARK,
            widgets=[
                WidgetConfig(
                    widget_type=WidgetType.PORTFOLIO_SUMMARY,
                    title="Portfolio Summary",
                    position={"x": 0, "y": 0, "width": 4, "height": 2},
                    refresh_interval=5,
                ),
                WidgetConfig(
                    widget_type=WidgetType.GREEKS_DISPLAY,
                    title="Greeks Exposure",
                    position={"x": 4, "y": 0, "width": 4, "height": 2},
                    refresh_interval=5,
                ),
                WidgetConfig(
                    widget_type=WidgetType.ALERTS_PANEL,
                    title="Alerts",
                    position={"x": 8, "y": 0, "width": 4, "height": 2},
                    refresh_interval=2,
                ),
                WidgetConfig(
                    widget_type=WidgetType.POSITIONS_TABLE,
                    title="Positions",
                    position={"x": 0, "y": 2, "width": 8, "height": 3},
                    refresh_interval=5,
                ),
                WidgetConfig(
                    widget_type=WidgetType.PNL_CHART,
                    title="P&L Chart",
                    position={"x": 8, "y": 2, "width": 4, "height": 3},
                    refresh_interval=10,
                ),
                WidgetConfig(
                    widget_type=WidgetType.ORDER_ENTRY,
                    title="Order Entry",
                    position={"x": 0, "y": 5, "width": 4, "height": 3},
                    refresh_interval=0,
                ),
                WidgetConfig(
                    widget_type=WidgetType.WATCHLIST,
                    title="Watchlist",
                    position={"x": 4, "y": 5, "width": 4, "height": 3},
                    refresh_interval=5,
                ),
                WidgetConfig(
                    widget_type=WidgetType.FLOW_SCANNER,
                    title="Options Flow",
                    position={"x": 8, "y": 5, "width": 4, "height": 3},
                    refresh_interval=3,
                ),
            ],
        )
    
    def get_widget_data(self, widget_type: WidgetType) -> Dict[str, Any]:
        """Get data for specific widget."""
        if widget_type == WidgetType.PORTFOLIO_SUMMARY:
            return self.data_provider.get_portfolio_summary()
        elif widget_type == WidgetType.POSITIONS_TABLE:
            return {"positions": self.data_provider.get_positions()}
        elif widget_type == WidgetType.GREEKS_DISPLAY:
            return self.data_provider.get_greeks()
        elif widget_type == WidgetType.ALERTS_PANEL:
            return {"alerts": self.data_provider.get_alerts()}
        elif widget_type == WidgetType.PNL_CHART:
            return {"data": self.data_provider.get_pnl_chart_data()}
        else:
            return {}
    
    def get_full_dashboard_data(self) -> Dict[str, Any]:
        """Get all dashboard data."""
        return {
            "layout": {
                "name": self.layout.name,
                "theme": self.layout.theme.value,
                "grid_columns": self.layout.grid_columns,
                "grid_rows": self.layout.grid_rows,
            },
            "widgets": [
                {
                    "type": w.widget_type.value,
                    "title": w.title,
                    "position": w.position,
                    "refresh_interval": w.refresh_interval,
                    "data": self.get_widget_data(w.widget_type),
                }
                for w in self.layout.widgets
            ],
            "timestamp": datetime.now().isoformat(),
        }
    
    def generate_html(self) -> str:
        """Generate dashboard HTML."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNOSIS Trading Dashboard</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent-green: #238636;
            --accent-red: #da3633;
            --accent-blue: #58a6ff;
            --border-color: #30363d;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }}
        
        .header {{
            background: var(--bg-secondary);
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .header h1 {{
            font-size: 1.5rem;
            font-weight: 600;
        }}
        
        .header .status {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        
        .status-indicator {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--accent-green);
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            grid-auto-rows: minmax(100px, auto);
            gap: 1rem;
            padding: 1rem;
        }}
        
        .widget {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .widget-header {{
            background: var(--bg-tertiary);
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-color);
            font-weight: 600;
            font-size: 0.875rem;
        }}
        
        .widget-content {{
            padding: 1rem;
        }}
        
        /* Portfolio Summary */
        .portfolio-summary {{
            grid-column: span 4;
            grid-row: span 2;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }}
        
        .summary-item {{
            text-align: center;
        }}
        
        .summary-value {{
            font-size: 1.5rem;
            font-weight: 700;
        }}
        
        .summary-label {{
            color: var(--text-secondary);
            font-size: 0.75rem;
            margin-top: 0.25rem;
        }}
        
        .positive {{ color: var(--accent-green); }}
        .negative {{ color: var(--accent-red); }}
        
        /* Greeks Display */
        .greeks-display {{
            grid-column: span 4;
            grid-row: span 2;
        }}
        
        .greeks-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
        }}
        
        .greek-item {{
            background: var(--bg-tertiary);
            padding: 0.75rem;
            border-radius: 4px;
            text-align: center;
        }}
        
        .greek-name {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
        }}
        
        .greek-value {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-top: 0.25rem;
        }}
        
        /* Positions Table */
        .positions-table {{
            grid-column: span 8;
            grid-row: span 3;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background: var(--bg-tertiary);
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
        }}
        
        /* Alerts Panel */
        .alerts-panel {{
            grid-column: span 4;
            grid-row: span 2;
        }}
        
        .alert-item {{
            padding: 0.5rem;
            margin-bottom: 0.5rem;
            background: var(--bg-tertiary);
            border-radius: 4px;
            border-left: 3px solid var(--accent-blue);
            font-size: 0.875rem;
        }}
        
        .alert-item.warning {{ border-left-color: #f0883e; }}
        .alert-item.critical {{ border-left-color: var(--accent-red); }}
        
        .alert-time {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        
        /* P&L Chart */
        .pnl-chart {{
            grid-column: span 4;
            grid-row: span 3;
        }}
        
        .chart-container {{
            height: 200px;
            display: flex;
            align-items: flex-end;
            gap: 2px;
        }}
        
        .chart-bar {{
            flex: 1;
            background: var(--accent-green);
            min-height: 1px;
            border-radius: 2px 2px 0 0;
        }}
        
        .chart-bar.negative {{
            background: var(--accent-red);
        }}
        
        /* Order Entry */
        .order-entry {{
            grid-column: span 4;
            grid-row: span 3;
        }}
        
        .order-form {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}
        
        .form-group {{
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}
        
        .form-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
        
        .form-input {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 0.5rem;
            color: var(--text-primary);
            font-size: 0.875rem;
        }}
        
        .form-input:focus {{
            outline: none;
            border-color: var(--accent-blue);
        }}
        
        .btn {{
            padding: 0.75rem 1rem;
            border: none;
            border-radius: 4px;
            font-weight: 600;
            cursor: pointer;
            font-size: 0.875rem;
        }}
        
        .btn-buy {{
            background: var(--accent-green);
            color: white;
        }}
        
        .btn-sell {{
            background: var(--accent-red);
            color: white;
        }}
        
        .btn-row {{
            display: flex;
            gap: 0.5rem;
        }}
        
        .btn-row .btn {{
            flex: 1;
        }}
        
        /* Watchlist */
        .watchlist {{
            grid-column: span 4;
            grid-row: span 3;
        }}
        
        .watchlist-item {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .watchlist-symbol {{
            font-weight: 600;
        }}
        
        .watchlist-price {{
            font-family: monospace;
        }}
        
        /* Flow Scanner */
        .flow-scanner {{
            grid-column: span 4;
            grid-row: span 3;
        }}
        
        .flow-item {{
            padding: 0.5rem;
            margin-bottom: 0.5rem;
            background: var(--bg-tertiary);
            border-radius: 4px;
            font-size: 0.875rem;
        }}
        
        .flow-bullish {{ border-left: 3px solid var(--accent-green); }}
        .flow-bearish {{ border-left: 3px solid var(--accent-red); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 GNOSIS Trading Dashboard</h1>
        <div class="status">
            <div class="status-indicator"></div>
            <span>Live</span>
            <span id="last-update">--:--:--</span>
        </div>
    </div>
    
    <div class="dashboard">
        <!-- Portfolio Summary -->
        <div class="widget portfolio-summary">
            <div class="widget-header">Portfolio Summary</div>
            <div class="widget-content">
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="summary-value" id="total-value">$0</div>
                        <div class="summary-label">Total Value</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value" id="buying-power">$0</div>
                        <div class="summary-label">Buying Power</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value positive" id="day-pnl">$0</div>
                        <div class="summary-label">Day P&L</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-value" id="positions-count">0</div>
                        <div class="summary-label">Positions</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Greeks Display -->
        <div class="widget greeks-display">
            <div class="widget-header">Greeks Exposure</div>
            <div class="widget-content">
                <div class="greeks-grid">
                    <div class="greek-item">
                        <div class="greek-name">Delta</div>
                        <div class="greek-value" id="delta">0</div>
                    </div>
                    <div class="greek-item">
                        <div class="greek-name">Gamma</div>
                        <div class="greek-value" id="gamma">0</div>
                    </div>
                    <div class="greek-item">
                        <div class="greek-name">Theta</div>
                        <div class="greek-value" id="theta">0</div>
                    </div>
                    <div class="greek-item">
                        <div class="greek-name">Vega</div>
                        <div class="greek-value" id="vega">0</div>
                    </div>
                    <div class="greek-item">
                        <div class="greek-name">Rho</div>
                        <div class="greek-value" id="rho">0</div>
                    </div>
                    <div class="greek-item">
                        <div class="greek-name">Beta Δ</div>
                        <div class="greek-value" id="beta-delta">0</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Alerts Panel -->
        <div class="widget alerts-panel">
            <div class="widget-header">Alerts</div>
            <div class="widget-content" id="alerts-container">
                <div class="alert-item">No alerts</div>
            </div>
        </div>
        
        <!-- Positions Table -->
        <div class="widget positions-table">
            <div class="widget-header">Positions</div>
            <div class="widget-content">
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Qty</th>
                            <th>Avg Cost</th>
                            <th>Current</th>
                            <th>P&L</th>
                            <th>P&L %</th>
                        </tr>
                    </thead>
                    <tbody id="positions-body">
                        <tr><td colspan="6" style="text-align: center;">No positions</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- P&L Chart -->
        <div class="widget pnl-chart">
            <div class="widget-header">P&L Chart</div>
            <div class="widget-content">
                <div class="chart-container" id="pnl-chart">
                    <!-- Chart bars will be inserted here -->
                </div>
            </div>
        </div>
        
        <!-- Order Entry -->
        <div class="widget order-entry">
            <div class="widget-header">Quick Order</div>
            <div class="widget-content">
                <form class="order-form" onsubmit="return false;">
                    <div class="form-group">
                        <label class="form-label">Symbol</label>
                        <input type="text" class="form-input" id="order-symbol" placeholder="SPY">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Quantity</label>
                        <input type="number" class="form-input" id="order-qty" placeholder="100">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Order Type</label>
                        <select class="form-input" id="order-type">
                            <option value="market">Market</option>
                            <option value="limit">Limit</option>
                            <option value="stop">Stop</option>
                        </select>
                    </div>
                    <div class="btn-row">
                        <button class="btn btn-buy" onclick="submitOrder('buy')">Buy</button>
                        <button class="btn btn-sell" onclick="submitOrder('sell')">Sell</button>
                    </div>
                </form>
            </div>
        </div>
        
        <!-- Watchlist -->
        <div class="widget watchlist">
            <div class="widget-header">Watchlist</div>
            <div class="widget-content" id="watchlist-container">
                <div class="watchlist-item">
                    <span class="watchlist-symbol">SPY</span>
                    <span class="watchlist-price">$500.00</span>
                </div>
                <div class="watchlist-item">
                    <span class="watchlist-symbol">QQQ</span>
                    <span class="watchlist-price">$430.00</span>
                </div>
                <div class="watchlist-item">
                    <span class="watchlist-symbol">IWM</span>
                    <span class="watchlist-price">$220.00</span>
                </div>
            </div>
        </div>
        
        <!-- Flow Scanner -->
        <div class="widget flow-scanner">
            <div class="widget-header">Options Flow</div>
            <div class="widget-content" id="flow-container">
                <div class="flow-item flow-bullish">
                    <strong>AAPL</strong> - 1000 calls bought @ $180 strike
                </div>
                <div class="flow-item flow-bearish">
                    <strong>TSLA</strong> - 500 puts bought @ $250 strike
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        let ws;
        
        function connectWebSocket() {{
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${{protocol}}//${{window.location.host}}/ws/dashboard`);
            
            ws.onmessage = function(event) {{
                const data = JSON.parse(event.data);
                updateDashboard(data);
            }};
            
            ws.onclose = function() {{
                setTimeout(connectWebSocket, 3000);
            }};
        }}
        
        function updateDashboard(data) {{
            if (data.portfolio) {{
                document.getElementById('total-value').textContent = formatMoney(data.portfolio.total_value);
                document.getElementById('buying-power').textContent = formatMoney(data.portfolio.buying_power);
                document.getElementById('day-pnl').textContent = formatMoney(data.portfolio.day_pnl);
                document.getElementById('positions-count').textContent = data.portfolio.positions_count;
            }}
            
            if (data.greeks) {{
                document.getElementById('delta').textContent = data.greeks.delta.toFixed(2);
                document.getElementById('gamma').textContent = data.greeks.gamma.toFixed(3);
                document.getElementById('theta').textContent = data.greeks.theta.toFixed(2);
                document.getElementById('vega').textContent = data.greeks.vega.toFixed(2);
            }}
            
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
        }}
        
        function formatMoney(value) {{
            return new Intl.NumberFormat('en-US', {{
                style: 'currency',
                currency: 'USD'
            }}).format(value || 0);
        }}
        
        function submitOrder(side) {{
            const symbol = document.getElementById('order-symbol').value;
            const qty = document.getElementById('order-qty').value;
            const orderType = document.getElementById('order-type').value;
            
            console.log(`Order: ${{side}} ${{qty}} ${{symbol}} @ ${{orderType}}`);
            alert(`Order submitted: ${{side.toUpperCase()}} ${{qty}} ${{symbol}}`);
        }}
        
        // Initialize
        // connectWebSocket();
        
        // Simulate updates for demo
        setInterval(() => {{
            const mockData = {{
                portfolio: {{
                    total_value: 100000 + Math.random() * 1000,
                    buying_power: 50000,
                    day_pnl: (Math.random() - 0.5) * 2000,
                    positions_count: 5
                }},
                greeks: {{
                    delta: (Math.random() - 0.5) * 200,
                    gamma: Math.random() * 10,
                    theta: -Math.random() * 100,
                    vega: Math.random() * 500
                }}
            }};
            updateDashboard(mockData);
        }}, 5000);
    </script>
</body>
</html>'''
    
    def get_api_routes(self) -> List[Dict[str, Any]]:
        """Get API routes for dashboard."""
        return [
            {"path": "/dashboard", "method": "GET", "handler": "render_dashboard"},
            {"path": "/dashboard/data", "method": "GET", "handler": "get_dashboard_data"},
            {"path": "/ws/dashboard", "method": "WS", "handler": "websocket_dashboard"},
        ]


# Convenience function to create dashboard
def create_dashboard(layout: Optional[DashboardLayout] = None) -> TradingDashboard:
    """Create trading dashboard."""
    return TradingDashboard(layout)
