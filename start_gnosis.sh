#!/bin/bash
# Gnosis Trading System Startup Script
# Starts all required services for the Gnosis framework

set -e

GNOSIS_ROOT="/home/user/FINAL_GNOSIS"
LOG_DIR="$GNOSIS_ROOT/logs"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "========================================="
echo "  Gnosis Trading System Startup"
echo "========================================="

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to check if a service is running on a port
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# 1. Start PostgreSQL
echo -e "\n${YELLOW}[1/3]${NC} Starting PostgreSQL..."
if ps aux | grep -v grep | grep postgres > /dev/null; then
    echo -e "${GREEN}✓${NC} PostgreSQL already running"
else
    service postgresql start
    sleep 2
    echo -e "${GREEN}✓${NC} PostgreSQL started"
fi

# 2. Start Backend API
echo -e "\n${YELLOW}[2/3]${NC} Starting Backend API..."
if check_port 8000; then
    echo -e "${YELLOW}!${NC} Backend already running on port 8000, stopping it first..."
    pkill -f "uvicorn web_api:app" || true
    sleep 2
fi

cd "$GNOSIS_ROOT"
nohup python3 -m uvicorn web_api:app --host 0.0.0.0 --port 8000 >> "$LOG_DIR/backend.log" 2>&1 &
sleep 5

if check_port 8000; then
    echo -e "${GREEN}✓${NC} Backend API started on http://localhost:8000"
else
    echo -e "${RED}✗${NC} Backend API failed to start. Check $LOG_DIR/backend.log"
    exit 1
fi

# 3. Start Frontend
echo -e "\n${YELLOW}[3/3]${NC} Starting Frontend..."
if check_port 3000; then
    echo -e "${YELLOW}!${NC} Frontend already running on port 3000, stopping it first..."
    pkill -f "next dev" || true
    sleep 2
fi

cd "$GNOSIS_ROOT/saas_frontend"
nohup npm run dev >> "$LOG_DIR/frontend.log" 2>&1 &
sleep 5

if check_port 3000; then
    echo -e "${GREEN}✓${NC} Frontend started on http://localhost:3000"
else
    echo -e "${RED}✗${NC} Frontend failed to start. Check $LOG_DIR/frontend.log"
    exit 1
fi

# Summary
echo -e "\n========================================="
echo -e "${GREEN}✓ Gnosis Trading System is running!${NC}"
echo "========================================="
echo ""
echo "Services:"
echo "  • Backend API:  http://localhost:8000"
echo "  • API Docs:     http://localhost:8000/docs"
echo "  • Frontend:     http://localhost:3000"
echo "  • PostgreSQL:   localhost:5432"
echo ""
echo "Logs:"
echo "  • Backend:  $LOG_DIR/backend.log"
echo "  • Frontend: $LOG_DIR/frontend.log"
echo ""
echo "To stop all services: ./stop_gnosis.sh"
echo "To check status: ./status_gnosis.sh"
echo "========================================="
