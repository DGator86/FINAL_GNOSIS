#!/bin/bash
# Gnosis Trading System Status Script
# Check status of all services

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================="
echo "  Gnosis Trading System Status"
echo "========================================="

# Check Backend
if lsof -i :8000 > /dev/null 2>&1; then
    echo -e "Backend API (8000):  ${GREEN}✓ Running${NC}"
    curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null | head -5 || echo "  (API responding)"
else
    echo -e "Backend API (8000):  ${RED}✗ Not Running${NC}"
fi

# Check Frontend
if lsof -i :3000 > /dev/null 2>&1; then
    echo -e "Frontend (3000):     ${GREEN}✓ Running${NC}"
else
    echo -e "Frontend (3000):     ${RED}✗ Not Running${NC}"
fi

# Check PostgreSQL
if ps aux | grep -v grep | grep postgres > /dev/null; then
    echo -e "PostgreSQL (5432):   ${GREEN}✓ Running${NC}"
else
    echo -e "PostgreSQL (5432):   ${RED}✗ Not Running${NC}"
fi

echo "========================================="

# Show recent processes
echo -e "\nProcesses:"
ps aux | grep -E "uvicorn|next dev|postgres" | grep -v grep | awk '{printf "  PID: %-6s CPU: %-5s MEM: %-5s %s\n", $2, $3"%", $4"%", $11}'
