#!/usr/bin/env python3
"""
Magnificent 7 Web Dashboard API

FastAPI endpoint for the Magnificent 7 Portfolio Dashboard.
Provides JSON API and serves HTML dashboard.

Author: GNOSIS Trading System
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware


# =============================================================================
# Configuration
# =============================================================================

MAGNIFICENT_7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

COMPANY_INFO = {
    "AAPL": {"name": "Apple", "sector": "Technology", "color": "#A2AAAD"},
    "MSFT": {"name": "Microsoft", "sector": "Technology", "color": "#00A4EF"},
    "GOOGL": {"name": "Alphabet", "sector": "Technology", "color": "#4285F4"},
    "AMZN": {"name": "Amazon", "sector": "Consumer", "color": "#FF9900"},
    "NVDA": {"name": "NVIDIA", "sector": "Technology", "color": "#76B900"},
    "META": {"name": "Meta", "sector": "Technology", "color": "#0081FB"},
    "TSLA": {"name": "Tesla", "sector": "Automotive", "color": "#CC0000"},
}


def load_env():
    """Load environment variables from .env file."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ[key] = value
    except FileNotFoundError:
        pass


load_env()


# =============================================================================
# API Clients
# =============================================================================

def alpaca_request(endpoint: str) -> Optional[Dict[str, Any]]:
    """Make Alpaca API request."""
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        return None
    
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    
    try:
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Alpaca API error: {e}")
        return None


def alpaca_data_request(endpoint: str) -> Optional[Dict[str, Any]]:
    """Make Alpaca Data API request."""
    base_url = "https://data.alpaca.markets"
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        return None
    
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    
    try:
        url = f"{base_url}/{endpoint.lstrip('/')}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Alpaca Data API error: {e}")
        return None


def unusual_whales_request(endpoint: str) -> Optional[Dict[str, Any]]:
    """Make Unusual Whales API request."""
    base_url = "https://api.unusualwhales.com"
    api_token = os.getenv("UNUSUAL_WHALES_API_TOKEN")
    
    if not api_token:
        return None
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json",
    }
    
    try:
        url = f"{base_url}{endpoint}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return None


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Magnificent 7 Dashboard API",
    description="Real-time portfolio dashboard for Magnificent 7 stocks",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/account")
async def get_account():
    """Get Alpaca account info."""
    account = alpaca_request("account")
    if not account:
        raise HTTPException(status_code=500, detail="Failed to fetch account")
    return account


@app.get("/api/positions")
async def get_positions():
    """Get Magnificent 7 positions."""
    positions = alpaca_request("positions")
    if positions is None:
        raise HTTPException(status_code=500, detail="Failed to fetch positions")
    
    # Filter for Magnificent 7
    mag7_positions = [p for p in positions if p.get("symbol") in MAGNIFICENT_7]
    return mag7_positions


@app.get("/api/quotes")
async def get_quotes():
    """Get Magnificent 7 quotes."""
    symbols_str = ",".join(MAGNIFICENT_7)
    quotes = alpaca_data_request(f"v2/stocks/quotes/latest?symbols={symbols_str}")
    
    if not quotes:
        raise HTTPException(status_code=500, detail="Failed to fetch quotes")
    
    return quotes.get("quotes", {})


@app.get("/api/flow/{symbol}")
async def get_flow(symbol: str):
    """Get options flow for a symbol."""
    if symbol.upper() not in MAGNIFICENT_7:
        raise HTTPException(status_code=400, detail="Symbol not in Magnificent 7")
    
    flow = unusual_whales_request(f"/api/stock/{symbol.upper()}/flow-alerts")
    return flow.get("data", []) if flow else []


@app.get("/api/greeks/{symbol}")
async def get_greeks(symbol: str):
    """Get Greeks for a symbol."""
    if symbol.upper() not in MAGNIFICENT_7:
        raise HTTPException(status_code=400, detail="Symbol not in Magnificent 7")
    
    greeks = unusual_whales_request(f"/api/stock/{symbol.upper()}/option-contracts")
    return greeks.get("data", []) if greeks else []


@app.get("/api/summary")
async def get_summary():
    """Get full dashboard summary."""
    # Account
    account = alpaca_request("account") or {}
    
    # Positions
    positions = alpaca_request("positions") or []
    mag7_positions = [p for p in positions if p.get("symbol") in MAGNIFICENT_7]
    
    # Quotes
    symbols_str = ",".join(MAGNIFICENT_7)
    quotes_data = alpaca_data_request(f"v2/stocks/quotes/latest?symbols={symbols_str}") or {}
    quotes = quotes_data.get("quotes", {})
    
    # Calculate totals
    total_value = sum(float(p.get("market_value", 0)) for p in mag7_positions)
    total_pnl = sum(float(p.get("unrealized_pl", 0)) for p in mag7_positions)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "account": {
            "status": account.get("status"),
            "equity": float(account.get("equity", 0)),
            "cash": float(account.get("cash", 0)),
            "buying_power": float(account.get("buying_power", 0)),
        },
        "magnificent_7": {
            "symbols": MAGNIFICENT_7,
            "positions": mag7_positions,
            "total_value": total_value,
            "total_pnl": total_pnl,
            "quotes": quotes,
        },
        "strategy": {
            "min_confidence": float(os.getenv("MIN_CONFIDENCE", "0.70")),
            "min_reward_risk": float(os.getenv("MIN_REWARD_RISK", "2.5")),
            "max_positions": int(os.getenv("MAX_POSITIONS", "5")),
            "backtest_return": 0.67,
            "backtest_sharpe": 1.03,
            "backtest_max_dd": 0.34,
            "backtest_win_rate": 52.9,
        },
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the HTML dashboard."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Magnificent 7 Portfolio Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }
        .card { background: rgba(22, 27, 34, 0.8); backdrop-filter: blur(10px); }
        .glow { box-shadow: 0 0 15px rgba(88, 166, 255, 0.3); }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .loading { animation: pulse 1.5s infinite; }
    </style>
</head>
<body class="min-h-screen text-gray-100 p-4">
    <div class="max-w-7xl mx-auto">
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                Magnificent 7 Portfolio Dashboard
            </h1>
            <p class="text-gray-400 mt-2">GNOSIS Trading System | Paper Trading Mode</p>
            <p id="timestamp" class="text-sm text-gray-500 mt-1"></p>
        </div>

        <!-- Account Summary -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div class="card rounded-xl p-4 border border-gray-700">
                <div class="text-gray-400 text-sm">Account Status</div>
                <div id="status" class="text-2xl font-bold text-green-400 loading">--</div>
            </div>
            <div class="card rounded-xl p-4 border border-gray-700">
                <div class="text-gray-400 text-sm">Portfolio Value</div>
                <div id="equity" class="text-2xl font-bold loading">--</div>
            </div>
            <div class="card rounded-xl p-4 border border-gray-700">
                <div class="text-gray-400 text-sm">Buying Power</div>
                <div id="buying_power" class="text-2xl font-bold loading">--</div>
            </div>
            <div class="card rounded-xl p-4 border border-gray-700">
                <div class="text-gray-400 text-sm">Magnificent 7 P&L</div>
                <div id="mag7_pnl" class="text-2xl font-bold loading">--</div>
            </div>
        </div>

        <!-- Magnificent 7 Grid -->
        <h2 class="text-xl font-semibold mb-4 text-blue-400">Magnificent 7 Stocks</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <div id="AAPL-card" class="card rounded-xl p-4 border border-gray-700 glow">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-lg font-bold">🍎 AAPL</span>
                    <span class="text-sm text-gray-400">Apple</span>
                </div>
                <div id="AAPL-price" class="text-2xl font-bold loading">--</div>
                <div id="AAPL-position" class="text-sm text-gray-400 mt-2">No position</div>
            </div>
            <div id="MSFT-card" class="card rounded-xl p-4 border border-gray-700 glow">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-lg font-bold">💻 MSFT</span>
                    <span class="text-sm text-gray-400">Microsoft</span>
                </div>
                <div id="MSFT-price" class="text-2xl font-bold loading">--</div>
                <div id="MSFT-position" class="text-sm text-gray-400 mt-2">No position</div>
            </div>
            <div id="GOOGL-card" class="card rounded-xl p-4 border border-gray-700 glow">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-lg font-bold">🔍 GOOGL</span>
                    <span class="text-sm text-gray-400">Alphabet</span>
                </div>
                <div id="GOOGL-price" class="text-2xl font-bold loading">--</div>
                <div id="GOOGL-position" class="text-sm text-gray-400 mt-2">No position</div>
            </div>
            <div id="AMZN-card" class="card rounded-xl p-4 border border-gray-700 glow">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-lg font-bold">📦 AMZN</span>
                    <span class="text-sm text-gray-400">Amazon</span>
                </div>
                <div id="AMZN-price" class="text-2xl font-bold loading">--</div>
                <div id="AMZN-position" class="text-sm text-gray-400 mt-2">No position</div>
            </div>
            <div id="NVDA-card" class="card rounded-xl p-4 border border-gray-700 glow">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-lg font-bold">🎮 NVDA</span>
                    <span class="text-sm text-gray-400">NVIDIA</span>
                </div>
                <div id="NVDA-price" class="text-2xl font-bold loading">--</div>
                <div id="NVDA-position" class="text-sm text-gray-400 mt-2">No position</div>
            </div>
            <div id="META-card" class="card rounded-xl p-4 border border-gray-700 glow">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-lg font-bold">👥 META</span>
                    <span class="text-sm text-gray-400">Meta</span>
                </div>
                <div id="META-price" class="text-2xl font-bold loading">--</div>
                <div id="META-position" class="text-sm text-gray-400 mt-2">No position</div>
            </div>
            <div id="TSLA-card" class="card rounded-xl p-4 border border-gray-700 glow">
                <div class="flex items-center justify-between mb-2">
                    <span class="text-lg font-bold">🚗 TSLA</span>
                    <span class="text-sm text-gray-400">Tesla</span>
                </div>
                <div id="TSLA-price" class="text-2xl font-bold loading">--</div>
                <div id="TSLA-position" class="text-sm text-gray-400 mt-2">No position</div>
            </div>
            <!-- Strategy Card -->
            <div class="card rounded-xl p-4 border border-blue-700 bg-blue-900/20">
                <div class="text-blue-400 font-semibold mb-2">Strategy Performance</div>
                <div class="grid grid-cols-2 gap-2 text-sm">
                    <div>Return: <span class="text-green-400">+0.67%</span></div>
                    <div>Sharpe: <span class="text-green-400">1.03</span></div>
                    <div>Max DD: <span class="text-yellow-400">-0.34%</span></div>
                    <div>Win Rate: <span class="text-green-400">52.9%</span></div>
                </div>
            </div>
        </div>

        <!-- Positions Table -->
        <h2 class="text-xl font-semibold mb-4 text-blue-400">Open Positions</h2>
        <div class="card rounded-xl border border-gray-700 overflow-hidden mb-8">
            <table class="w-full">
                <thead class="bg-gray-800">
                    <tr>
                        <th class="px-4 py-3 text-left text-sm text-gray-400">Symbol</th>
                        <th class="px-4 py-3 text-right text-sm text-gray-400">Qty</th>
                        <th class="px-4 py-3 text-right text-sm text-gray-400">Avg Entry</th>
                        <th class="px-4 py-3 text-right text-sm text-gray-400">Current</th>
                        <th class="px-4 py-3 text-right text-sm text-gray-400">Market Value</th>
                        <th class="px-4 py-3 text-right text-sm text-gray-400">P&L</th>
                        <th class="px-4 py-3 text-right text-sm text-gray-400">P&L %</th>
                    </tr>
                </thead>
                <tbody id="positions-table">
                    <tr><td colspan="7" class="px-4 py-8 text-center text-gray-500 loading">Loading...</td></tr>
                </tbody>
            </table>
        </div>

        <!-- Footer -->
        <div class="text-center text-gray-500 text-sm">
            <p>Auto-refresh: 30 seconds | GNOSIS Trading System</p>
            <p class="mt-1">Commands: <code class="bg-gray-800 px-2 py-1 rounded">./start.sh local</code> to start trading</p>
        </div>
    </div>

    <script>
        const MAGNIFICENT_7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'];
        
        function formatMoney(value) {
            const v = parseFloat(value) || 0;
            if (Math.abs(v) >= 1000000) return '$' + (v/1000000).toFixed(2) + 'M';
            if (Math.abs(v) >= 1000) return '$' + (v/1000).toFixed(2) + 'K';
            return '$' + v.toFixed(2);
        }
        
        function formatPercent(value) {
            const v = parseFloat(value) * 100;
            return (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
        }
        
        function colorClass(value) {
            const v = parseFloat(value);
            return v >= 0 ? 'text-green-400' : 'text-red-400';
        }
        
        async function fetchData() {
            try {
                const response = await fetch('/api/summary');
                const data = await response.json();
                
                // Update timestamp
                document.getElementById('timestamp').textContent = 
                    'Last updated: ' + new Date(data.timestamp).toLocaleString();
                
                // Update account
                document.getElementById('status').textContent = data.account.status || 'N/A';
                document.getElementById('status').className = 'text-2xl font-bold ' + 
                    (data.account.status === 'ACTIVE' ? 'text-green-400' : 'text-yellow-400');
                document.getElementById('equity').textContent = formatMoney(data.account.equity);
                document.getElementById('equity').className = 'text-2xl font-bold text-white';
                document.getElementById('buying_power').textContent = formatMoney(data.account.buying_power);
                document.getElementById('buying_power').className = 'text-2xl font-bold text-blue-400';
                
                const mag7_pnl = data.magnificent_7.total_pnl;
                document.getElementById('mag7_pnl').textContent = 
                    (mag7_pnl >= 0 ? '+' : '') + formatMoney(mag7_pnl);
                document.getElementById('mag7_pnl').className = 
                    'text-2xl font-bold ' + colorClass(mag7_pnl);
                
                // Update quotes
                const quotes = data.magnificent_7.quotes;
                MAGNIFICENT_7.forEach(symbol => {
                    const quote = quotes[symbol];
                    const priceEl = document.getElementById(symbol + '-price');
                    if (quote) {
                        const mid = (parseFloat(quote.bp) + parseFloat(quote.ap)) / 2;
                        priceEl.textContent = '$' + mid.toFixed(2);
                        priceEl.className = 'text-2xl font-bold text-white';
                    }
                });
                
                // Update positions
                const positions = data.magnificent_7.positions;
                const posMap = {};
                positions.forEach(p => posMap[p.symbol] = p);
                
                MAGNIFICENT_7.forEach(symbol => {
                    const pos = posMap[symbol];
                    const posEl = document.getElementById(symbol + '-position');
                    if (pos) {
                        const pnl = parseFloat(pos.unrealized_pl);
                        const pnlPct = parseFloat(pos.unrealized_plpc) * 100;
                        posEl.textContent = `${pos.qty} shares | P&L: ${(pnl >= 0 ? '+' : '')}$${pnl.toFixed(2)} (${pnlPct.toFixed(2)}%)`;
                        posEl.className = 'text-sm ' + colorClass(pnl);
                    } else {
                        posEl.textContent = 'No position';
                        posEl.className = 'text-sm text-gray-400';
                    }
                });
                
                // Update positions table
                const tableBody = document.getElementById('positions-table');
                if (positions.length === 0) {
                    tableBody.innerHTML = '<tr><td colspan="7" class="px-4 py-8 text-center text-gray-500">No Magnificent 7 positions</td></tr>';
                } else {
                    tableBody.innerHTML = positions.map(p => {
                        const pnl = parseFloat(p.unrealized_pl);
                        const pnlPct = parseFloat(p.unrealized_plpc);
                        return `
                            <tr class="border-t border-gray-700">
                                <td class="px-4 py-3 font-semibold">${p.symbol}</td>
                                <td class="px-4 py-3 text-right">${p.qty}</td>
                                <td class="px-4 py-3 text-right">$${parseFloat(p.avg_entry_price).toFixed(2)}</td>
                                <td class="px-4 py-3 text-right">$${parseFloat(p.current_price).toFixed(2)}</td>
                                <td class="px-4 py-3 text-right">${formatMoney(p.market_value)}</td>
                                <td class="px-4 py-3 text-right ${colorClass(pnl)}">${(pnl >= 0 ? '+' : '')}${formatMoney(pnl)}</td>
                                <td class="px-4 py-3 text-right ${colorClass(pnlPct)}">${formatPercent(pnlPct)}</td>
                            </tr>
                        `;
                    }).join('');
                }
                
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }
        
        // Initial fetch
        fetchData();
        
        // Auto-refresh every 30 seconds
        setInterval(fetchData, 30000);
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
