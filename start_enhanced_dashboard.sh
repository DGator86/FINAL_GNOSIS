#!/bin/bash
# ============================================================================
# Super Gnosis DHPE v4 - Enhanced Dashboard Launcher
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${PURPLE}"
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║     🚀 SUPER GNOSIS DHPE v4 - PREMIUM DASHBOARD                      ║"
echo "║                                                                      ║"
echo "║     Enhanced Trading Intelligence Dashboard                          ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for required packages
echo -e "${CYAN}Checking dependencies...${NC}"

pip install -q streamlit plotly pandas numpy python-dotenv 2>/dev/null || true

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Default port
PORT=${1:-8501}

echo -e "${GREEN}"
echo "┌──────────────────────────────────────────────────────────────────────┐"
echo "│  Dashboard starting on port $PORT                                      │"
echo "│                                                                      │"
echo "│  Local URL:   http://localhost:$PORT                                   │"
echo "│  Network URL: http://0.0.0.0:$PORT                                     │"
echo "│                                                                      │"
echo "│  Press Ctrl+C to stop the server                                     │"
echo "└──────────────────────────────────────────────────────────────────────┘"
echo -e "${NC}"

# Launch Streamlit
streamlit run dashboard_enhanced.py \
    --server.port "$PORT" \
    --server.address "0.0.0.0" \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.primaryColor "#667eea" \
    --theme.backgroundColor "#0d1117" \
    --theme.secondaryBackgroundColor "#161b22" \
    --theme.textColor "#f0f6fc"
