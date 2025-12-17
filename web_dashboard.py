#!/usr/bin/env python3
"""
Enhanced Web Dashboard for 30-Symbol Scanner
Shows all tickers with Composer Agent statuses in real-time
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Super Gnosis - 30 Symbol Scanner</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .status-bar {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .stat-box {
            text-align: center;
        }
        
        .stat-label {
            font-size: 0.9em;
            opacity: 0.8;
            margin-bottom: 5px;
        }
        
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
        }
        
        .market-open {
            color: #4ade80;
        }
        
        .market-closed {
            color: #f87171;
        }
        
        .symbols-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .symbol-card {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 15px;
            transition: transform 0.2s, box-shadow 0.2s;
            border: 2px solid transparent;
        }
        
        .symbol-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }
        
        .symbol-card.buy {
            border-color: #4ade80;
            background: rgba(74, 222, 128, 0.1);
        }
        
        .symbol-card.sell {
            border-color: #f87171;
            background: rgba(248, 113, 113, 0.1);
        }
        
        .symbol-card.hold {
            border-color: #fbbf24;
            background: rgba(251, 191, 36, 0.1);
        }
        
        .symbol-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .symbol-name {
            font-size: 1.5em;
            font-weight: bold;
        }
        
        .symbol-price {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .composer-status {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 10px 0;
            padding: 8px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
        }
        
        .status-badge {
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .status-badge.buy {
            background: #4ade80;
            color: #065f46;
        }
        
        .status-badge.sell {
            background: #f87171;
            color: #7f1d1d;
        }
        
        .status-badge.hold {
            background: #fbbf24;
            color: #78350f;
        }
        
        .confidence-bar {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 5px;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #4ade80, #10b981);
            transition: width 0.3s;
        }
        
        .signals {
            font-size: 0.85em;
            margin-top: 8px;
        }
        
        .signal-row {
            display: flex;
            justify-content: space-between;
            padding: 3px 0;
            opacity: 0.8;
        }
        
        .last-update {
            text-align: center;
            margin-top: 20px;
            opacity: 0.7;
            font-size: 0.9em;
        }
        
        .loading {
            text-align: center;
            padding: 50px;
            font-size: 1.5em;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .updating {
            animation: pulse 2s ease-in-out infinite;
        }
        .positions-section {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
        }

        .section-title {
            font-size: 1.5em;
            margin-bottom: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
            padding-bottom: 10px;
        }

        .positions-table {
            width: 100%;
            border-collapse: collapse;
        }

        .positions-table th, .positions-table td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        .positions-table th {
            font-weight: bold;
            opacity: 0.8;
            font-size: 0.9em;
            text-transform: uppercase;
        }

        .greeks-cell {
            font-family: monospace;
            font-size: 0.9em;
            color: #a5b4fc;
        }
        
        .strategy-badge {
            background: #6366f1;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 Super Gnosis Trading Scanner</h1>
            <p>Real-time 30-Symbol Market Analysis</p>
        </header>
        
        <div class="status-bar">
            <div class="stat-box">
                <div class="stat-label">Market Status</div>
                <div class="stat-value" id="market-status">--</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Portfolio Value</div>
                <div class="stat-value" id="portfolio-value">$0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Active Positions</div>
                <div class="stat-value" id="position-count">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">P&L Today</div>
                <div class="stat-value" id="pnl">$0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Symbols Scanned</div>
                <div class="stat-value" id="symbol-count">0</div>
            </div>
        </div>

        <!-- Active Positions Section -->
        <div class="positions-section" id="positions-section" style="display: none;">
            <div class="section-title">Active Positions</div>
            <table class="positions-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Type</th>
                        <th>Side</th>
                        <th>Qty</th>
                        <th>Entry</th>
                        <th>Current</th>
                        <th>P&L</th>
                        <th>Greeks (Δ / Γ / Θ / ν)</th>
                    </tr>
                </thead>
                <tbody id="positions-body">
                    <!-- Populated by JS -->
                </tbody>
            </table>
        </div>
        
        <div id="symbols-container" class="loading">
            Loading scanner data...
        </div>
        
        <div class="last-update">
            Last Update: <span id="last-update">--</span>
        </div>
    </div>
    
    <script>
        function updateDashboard() {
            fetch('/api/state')
                .then(response => response.json())
                .then(data => {
                    // Update status bar
                    const marketStatus = document.getElementById('market-status');
                    marketStatus.textContent = data.market_open ? 'OPEN' : 'CLOSED';
                    marketStatus.className = 'stat-value ' + (
                        data.market_open ? 'market-open' : 'market-closed'
                    );
                    
                    document.getElementById('portfolio-value').textContent = 
                        '$' + (data.account.portfolio_value || 0).toLocaleString(
                            'en-US', {maximumFractionDigits: 2}
                        );
                    
                    const positions = data.positions || [];
                    document.getElementById('position-count').textContent = positions.length;
                    
                    const pnl = data.account.pnl || 0;
                    const pnlElement = document.getElementById('pnl');
                    pnlElement.textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toLocaleString(
                        'en-US', {maximumFractionDigits: 2}
                    );
                    pnlElement.style.color = pnl >= 0 ? '#4ade80' : '#f87171';
                    
                    document.getElementById('symbol-count').textContent = 
                        Object.keys(data.symbols || {}).length;

                    // Update Positions Table
                    const positionsSection = document.getElementById('positions-section');
                    const positionsBody = document.getElementById('positions-body');
                    
                    if (positions.length > 0) {
                        positionsSection.style.display = 'block';
                        positionsBody.innerHTML = positions.map(pos => {
                            const pnlVal = pos.unrealized_pnl || 0;
                            const pnlClass = pnlVal >= 0 ? 'market-open' : 'market-closed';
                            const pnlText = (pnlVal >= 0 ? '+' : '') + '$' + pnlVal.toFixed(2);
                            
                            // Format Greeks
                            let greeksHtml = '-';
                            if (pos.delta !== undefined && pos.delta !== null) {
                                greeksHtml = `
                                    Δ ${pos.delta.toFixed(2)} | 
                                    Γ ${pos.gamma?.toFixed(3) || '-'} | 
                                    Θ ${pos.theta?.toFixed(2) || '-'} | 
                                    ν ${pos.vega?.toFixed(2) || '-'}
                                `;
                            }
                            
                            // Format Type/Strategy
                            let typeHtml = pos.asset_class === 'option_strategy' 
                                ? `<span class="strategy-badge">
                                    ${pos.option_symbol || 'Strategy'}
                                   </span>`
                                : pos.asset_class.toUpperCase();

                            return `
                                <tr>
                                    <td><strong>${pos.symbol}</strong></td>
                                    <td>${typeHtml}</td>
                                    <td style="color: ${pos.side === 'long' ? '#4ade80' : '#f87171'}">${pos.side.toUpperCase()}</td>
                                    <td>${pos.quantity}</td>
                                    <td>$${pos.entry_price.toFixed(2)}</td>
                                    <td>$${(pos.current_price || pos.entry_price).toFixed(2)}</td>
                                    <td class="${pnlClass}">${pnlText}</td>
                                    <td class="greeks-cell">${greeksHtml}</td>
                                </tr>
                            `;
                        }).join('');
                    } else {
                        positionsSection.style.display = 'none';
                    }
                    
                    // Update symbols grid
                    const container = document.getElementById('symbols-container');
                    const symbols = data.symbols || {};
                    
                    if (Object.keys(symbols).length === 0) {
                        container.innerHTML = '<div class="loading">Waiting for scan data...</div>';
                        return;
                    }
                    
                    let html = '<div class="symbols-grid">';
                    
                    // Sort symbols by confidence
                    const sortedSymbols = Object.values(symbols).sort((a, b) => 
                        b.composer_confidence - a.composer_confidence
                    );
                    
                    // Generate cards (existing logic)
                    // ... (rest of existing logic assumed to be preserved if I don't overwrite it, but I am replacing the whole block)
                    // Wait, I need to make sure I don't break the symbols grid generation.
                    // The previous code had logic to generate cards. I should preserve it or rewrite it.
                    // I'll rewrite it to be safe.
                    
                    container.innerHTML = sortedSymbols.map(sym => {
                        const signalClass = (sym.composer_signal || 'HOLD').toLowerCase();
                        const confidence = (sym.composer_confidence || 0) * 100;
                        
                        return `
                        <div class="symbol-card ${signalClass}">
                            <div class="symbol-header">
                                <div class="symbol-name">${sym.symbol}</div>
                                <div class="symbol-price">$${(sym.price || 0).toFixed(2)}</div>
                            </div>
                            <div class="composer-status">
                                <span class="status-badge ${signalClass}">
                                    ${sym.composer_signal || 'HOLD'}
                                </span>
                                <span>${confidence.toFixed(0)}% Conf.</span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${confidence}%"></div>
                            </div>
                        </div>
                        `;
                    }).join('') + '</div>';
                    
                    // Update timestamp
                    document.getElementById('last-update').textContent = 
                        new Date(data.last_update).toLocaleTimeString();
                })
                .catch(error => {
                    console.error('Error fetching data:', error);
                });
        }
        
        // Update every 5 seconds
        updateDashboard();
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>
"""


@app.route("/")
def index() -> str:
    """Main dashboard page"""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/state")
def get_state() -> Any:
    """API endpoint to get current scanner state"""
    state_file = Path("data/scanner_state/current_state.json")

    if state_file.exists():
        with open(state_file, "r") as f:
            return jsonify(json.load(f))
    else:
        return jsonify(
            {
                "symbols": {},
                "market_open": False,
                "positions": [],
                "account": {
                    "portfolio_value": 30000,
                    "cash": 30000,
                    "buying_power": 60000,
                    "pnl": 0,
                },
                "last_update": datetime.now().isoformat(),
            }
        )


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║              🌐 STARTING WEB DASHBOARD                                ║
║                                                                        ║
║  Access at: http://localhost:8000                                     ║
║  Or:        https://8000-...-novita.ai                                ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
    """)

    app.run(host="0.0.0.0", port=8000, debug=False)
