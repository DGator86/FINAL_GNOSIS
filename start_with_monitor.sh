#!/bin/bash
# Start Gnosis with process monitoring
# This starts all services AND a background monitor to keep them alive

GNOSIS_ROOT="/home/user/FINAL_GNOSIS"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Starting Gnosis with process monitoring..."

# Start services
"$GNOSIS_ROOT/start_gnosis.sh"

# Check if monitor is already running
if pgrep -f "monitor_gnosis.sh" > /dev/null; then
    echo -e "\n${YELLOW}Process monitor already running${NC}"
else
    # Start monitor in background
    nohup "$GNOSIS_ROOT/monitor_gnosis.sh" > /dev/null 2>&1 &
    MONITOR_PID=$!
    echo -e "\n${GREEN}✓ Process monitor started (PID: $MONITOR_PID)${NC}"
    echo "Monitor logs: $GNOSIS_ROOT/logs/monitor.log"
    echo ""
    echo "The monitor will automatically restart services if they crash."
    echo "To stop monitoring: pkill -f monitor_gnosis.sh"
fi
