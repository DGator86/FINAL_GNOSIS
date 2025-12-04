#!/bin/bash
# Cleanup any existing trading processes and start fresh

echo "🧹 Cleaning up existing processes..."

# Kill any existing python trading processes
pkill -9 -f "start_.*trading" 2>/dev/null || true
pkill -9 -f "python3.*start_" 2>/dev/null || true

# Wait a moment for connections to close
sleep 2

echo "✅ Cleanup complete"
echo ""
echo "🚀 Starting trading system..."
echo ""

# Check which script to run (default to simple trading)
SCRIPT=${1:-start_trading_now.py}

python3 "$SCRIPT"
