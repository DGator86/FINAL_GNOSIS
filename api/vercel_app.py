#!/usr/bin/env python3
"""
GNOSIS Vercel Adapter - Minimal API version
Serves static data without heavy ML/trading dependencies
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from datetime import datetime, timezone
import os

app = FastAPI(title="GNOSIS Lite", version="3.0.0")

# Mock data for demonstration
MOCK_STOCKS = [
    {"rank": 1, "symbol": "AAPL", "name": "Apple Inc.", "score": 85, "trend": "bullish"},
    {"rank": 2, "symbol": "MSFT", "name": "Microsoft Corp.", "score": 82, "trend": "bullish"},
    {"rank": 3, "symbol": "GOOGL", "name": "Alphabet Inc.", "score": 80, "trend": "neutral"},
    {"rank": 4, "symbol": "AMZN", "name": "Amazon.com Inc.", "score": 78, "trend": "bullish"},
    {"rank": 5, "symbol": "NVDA", "name": "NVIDIA Corp.", "score": 90, "trend": "bullish"},
]

@app.get("/")
async def home():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GNOSIS Lite - Vercel</title>
        <style>
            body { font-family: system-ui; max-width: 800px; margin: 40px auto; padding: 20px; background: #0a0a0f; color: white; }
            .card { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; margin: 20px 0; }
            h1 { color: #818cf8; }
            .stock { padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); }
            .bullish { color: #22c55e; }
            .neutral { color: #fbbf24; }
        </style>
    </head>
    <body>
        <h1>🧠 GNOSIS Lite</h1>
        <p>Lightweight version for Vercel serverless deployment</p>

        <div class="card">
            <h2>⚠️ Limited Functionality</h2>
            <p>This is a minimal version. For full features, deploy on Railway, Render, or Fly.io</p>
            <p><a href="/api/status" style="color: #818cf8;">API Status</a> |
               <a href="/api/rankings" style="color: #818cf8;">Rankings</a></p>
        </div>

        <div class="card">
            <h3>Top 5 Stocks (Mock Data)</h3>
            <div id="stocks"></div>
        </div>

        <script>
            fetch('/api/rankings').then(r => r.json()).then(data => {
                document.getElementById('stocks').innerHTML = data.rankings.map(s =>
                    `<div class="stock">
                        <strong>${s.rank}. ${s.symbol}</strong> - ${s.name}
                        <span class="${s.trend}">${s.trend}</span> (Score: ${s.score})
                    </div>`
                ).join('');
            });
        </script>
    </body>
    </html>
    """)

@app.get("/api/status")
async def status():
    return {
        "status": "running",
        "mode": "vercel-lite",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "3.0.0-lite",
        "note": "Limited functionality - deploy on Railway/Render for full features"
    }

@app.get("/api/rankings")
async def rankings():
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rankings": MOCK_STOCKS,
        "note": "Mock data - real engine requires Railway/Render deployment"
    }

@app.get("/api/top10")
async def top10():
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "top_10": MOCK_STOCKS[:10]
    }

# Vercel serverless handler
handler = app
