#!/usr/bin/env python3
"""
Enhanced Web Dashboard for 30-Symbol Scanner
Shows all tickers with Composer Agent statuses in real-time
"""

from flask import Flask, render_template_string, jsonify
import json
from pathlib import Path
from datetime import datetime

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
                    marketStatus.className = 'stat-value ' + (data.market_open ? 'market-open' : 'market-closed');
                    
                    document.getElementById('portfolio-value').textContent = 
                        '$' + (data.account.portfolio_value || 0).toLocaleString('en-US', {maximumFractionDigits: 2});
                    
                    document.getElementById('position-count').textContent = 
                        (data.positions || []).length;
                    
                    const pnl = data.account.pnl || 0;
                    const pnlElement = document.getElementById('pnl');
                    pnlElement.textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toLocaleString('en-US', {maximumFractionDigits: 2});
                    pnlElement.style.color = pnl >= 0 ? '#4ade80' : '#f87171';
                    
                    document.getElementById('symbol-count').textContent = 
                        Object.keys(data.symbols || {}).length;
                    
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
                    
                    sortedSymbols.forEach(symbol => {
                        const status = symbol.composer_status.toLowerCase();
                        const confidence = symbol.composer_confidence * 100;
                        
                        html += `
                            <div class="symbol-card ${status}">
                                <div class="symbol-header">
                                    <div class="symbol-name">${symbol.symbol}</div>
                                    <div class="symbol-price">$${symbol.price.toFixed(2)}</div>
                                </div>
                                
                                <div class="composer-status">
                                    <span style="font-size: 0.9em;">Composer Agent</span>
                                    <span class="status-badge ${status}">${symbol.composer_status}</span>
                                </div>
                                
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${confidence}%"></div>
                                </div>
                                <div style="text-align: center; font-size: 0.8em; margin-top: 3px;">
                                    Confidence: ${confidence.toFixed(1)}%
                                </div>
                                
                                <div class="signals">
                                    <div class="signal-row">
                                        <span>🔄 Flow Signal:</span>
                                        <span>${symbol.signals.flow.signal}</span>
                                    </div>
                                    <div class="signal-row">
                                        <span>📈 Momentum:</span>
                                        <span>${symbol.signals.momentum.signal}</span>
                                    </div>
                                    ${symbol.flow_data.total_flows ? `
                                    <div class="signal-row">
                                        <span>📊 Options Flow:</span>
                                        <span>${symbol.flow_data.bullish_count}↑ ${symbol.flow_data.bearish_count}↓</span>
                                    </div>
                                    ` : ''}
                                </div>
                            </div>
                        `;
                    });
                    
                    html += '</div>';
                    container.innerHTML = html;
                    
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


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/state')
def get_state():
    """API endpoint to get current scanner state"""
    state_file = Path("data/scanner_state/current_state.json")
    
    if state_file.exists():
        with open(state_file, 'r') as f:
            return jsonify(json.load(f))
    else:
        return jsonify({
            'symbols': {},
            'market_open': False,
            'positions': [],
            'account': {
                'portfolio_value': 30000,
                'cash': 30000,
                'buying_power': 60000,
                'pnl': 0
            },
            'last_update': datetime.now().isoformat()
        })


if __name__ == '__main__':
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
    
    app.run(host='0.0.0.0', port=8000, debug=False)