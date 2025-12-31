#!/bin/bash

# Simple script to launch the ticker viewer
# This starts the SaaS dashboard and automatically opens the ticker viewer page

echo "🚀 Starting Super Gnosis Ticker Viewer..."
echo ""
echo "📊 The ticker viewer will be available at:"
echo "   http://localhost:8000/tickers"
echo ""
echo "   (Main dashboard: http://localhost:8000)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"

python -m uvicorn saas.app:app --host 0.0.0.0 --port 8000
