#!/bin/bash
# Gnosis Trading System Stop Script
# Stops all Gnosis services

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "Stopping Gnosis Trading System..."

# Stop Backend
if pgrep -f "uvicorn web_api:app" > /dev/null; then
    pkill -f "uvicorn web_api:app"
    echo -e "${GREEN}✓${NC} Backend stopped"
else
    echo "Backend not running"
fi

# Stop Frontend
if pgrep -f "next dev" > /dev/null; then
    pkill -f "next dev"
    echo -e "${GREEN}✓${NC} Frontend stopped"
else
    echo "Frontend not running"
fi

# Optionally stop PostgreSQL (commented out by default)
# service postgresql stop
# echo -e "${GREEN}✓${NC} PostgreSQL stopped"

echo -e "\n${GREEN}Gnosis services stopped${NC}"
