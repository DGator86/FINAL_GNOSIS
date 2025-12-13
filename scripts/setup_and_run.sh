#!/bin/bash
# =============================================================================
# GNOSIS ML BACKTESTING - SETUP AND RUN SCRIPT
# =============================================================================
# This script sets up the environment and runs the ML learning loop.
#
# Usage:
#   ./scripts/setup_and_run.sh                    # Run with defaults (SPY, 5 years)
#   ./scripts/setup_and_run.sh AAPL 2022-01-01   # Custom symbol and start date
# =============================================================================

set -e

SYMBOL="${1:-SPY}"
START_DATE="${2:-2019-01-01}"
END_DATE="${3:-2024-12-01}"

echo "============================================================"
echo "GNOSIS ML BACKTESTING SYSTEM"
echo "============================================================"
echo "Symbol:     $SYMBOL"
echo "Start Date: $START_DATE"
echo "End Date:   $END_DATE"
echo "============================================================"

# Check for .env file
if [ ! -f .env ]; then
    echo ""
    echo "ERROR: .env file not found!"
    echo ""
    echo "Create a .env file with your API credentials:"
    echo ""
    echo "  # Alpaca (required)"
    echo "  ALPACA_API_KEY=your_key"
    echo "  ALPACA_SECRET_KEY=your_secret"
    echo ""
    echo "  # Massive (optional, for options data)"
    echo "  MASSIVE_API_KEY=your_key"
    echo "  MASSIVE_API_ENABLED=true"
    echo ""
    exit 1
fi

# Load environment variables
echo "Loading environment variables..."
set -a
source .env
set +a

# Check required credentials
if [ -z "$ALPACA_API_KEY" ] && [ -z "$MASSIVE_API_KEY" ]; then
    echo "ERROR: No API credentials found. Set ALPACA_API_KEY or MASSIVE_API_KEY in .env"
    exit 1
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip install -q alpaca-py pandas numpy loguru pydantic torch xgboost scikit-learn 2>/dev/null || true

# Determine data provider
if [ "$MASSIVE_API_ENABLED" = "true" ] && [ -n "$MASSIVE_API_KEY" ]; then
    pip install -q massive 2>/dev/null || true
    PROVIDER="massive"
    echo "Using Massive API (with options data)"
else
    PROVIDER="alpaca"
    echo "Using Alpaca API"
fi

echo ""
echo "Starting ML Learning Loop..."
echo "============================================================"

# Run the learning loop
python -m ml.learning_loop \
    --symbol "$SYMBOL" \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --provider "$PROVIDER" \
    --tag "${SYMBOL}_$(date +%Y%m%d)"

echo ""
echo "============================================================"
echo "COMPLETE!"
echo ""
echo "Results saved to: runs/learning/"
echo "============================================================"
