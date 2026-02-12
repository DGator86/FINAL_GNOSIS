#!/bin/bash
# =============================================================================
# GNOSIS - Start Everything (One Command)
# =============================================================================
# Usage: ./scripts/startup_all.sh
# =============================================================================

set -e
cd /home/root/webapp

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "============================================================"
echo "  GNOSIS TRADING SYSTEM - FULL STARTUP"
echo "  $(date)"
echo "============================================================"
echo -e "${NC}"

# Create logs directory
mkdir -p logs

# Kill any existing processes first (clean slate)
echo -e "${YELLOW}Cleaning up old processes...${NC}"
pkill -f "train_daily.py" 2>/dev/null || true
pkill -f "run_continuous_learning.py" 2>/dev/null || true
pkill -f "run_trading_daemon.py" 2>/dev/null || true
pkill -f "gnosis_service.py" 2>/dev/null || true
pkill -f "dashboard.py" 2>/dev/null || true
pkill -f "master_control.py" 2>/dev/null || true
pkill -f "active_risk_monitor.py" 2>/dev/null || true
sleep 2

echo -e "${GREEN}✓ Cleanup complete${NC}"

# 1. Start Dashboard (Streamlit)
echo -e "${BLUE}[1/4] Starting Dashboard...${NC}"
nohup python3 -m streamlit run dashboard.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    > logs/dashboard.log 2>&1 &
echo -e "${GREEN}✓ Dashboard started on port 8501${NC}"

# 2. Start Risk Monitor
echo -e "${BLUE}[2/4] Starting Risk Monitor...${NC}"
nohup python3 scripts/active_risk_monitor.py \
    > logs/risk_monitor.log 2>&1 &
echo -e "${GREEN}✓ Risk Monitor started${NC}"

# 3. Start Master Control
echo -e "${BLUE}[3/4] Starting Master Control...${NC}"
nohup python3 master_control.py \
    > logs/master_control.log 2>&1 &
echo -e "${GREEN}✓ Master Control started${NC}"

# 4. Start GNOSIS Service
echo -e "${BLUE}[4/4] Starting GNOSIS Service...${NC}"
nohup python3 scripts/gnosis_service.py \
    > logs/gnosis_service.log 2>&1 &
echo -e "${GREEN}✓ GNOSIS Service started${NC}"

sleep 3

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  ALL SERVICES STARTED SUCCESSFULLY!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Dashboard:     http://$(hostname -I | awk '{print $1}'):8501"
echo ""
echo "Log files:"
echo "  - logs/dashboard.log"
echo "  - logs/risk_monitor.log"
echo "  - logs/master_control.log"
echo "  - logs/gnosis_service.log"
echo ""
echo "Commands:"
echo "  tail -f logs/master_control.log    # Watch main logs"
echo "  ./scripts/run_daily_training.sh    # Run training"
echo "  ps aux | grep python               # Check processes"
echo "  pkill -f gnosis                     # Stop all"
echo ""

# Show running processes
echo "Running Processes:"
ps aux | grep python | grep -v grep | awk '{print "  PID " $2 ": " $11 " " $12}'
