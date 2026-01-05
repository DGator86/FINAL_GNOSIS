#!/bin/bash
# Install Gnosis systemd services
# Run this script to enable auto-start on boot

set -e

GNOSIS_ROOT="/home/user/FINAL_GNOSIS"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "  Installing Gnosis Systemd Services"
echo "========================================="

# Check if systemd is available
if ! command -v systemctl &> /dev/null; then
    echo -e "${YELLOW}Warning: systemd not available in this environment${NC}"
    echo "Systemd services cannot be installed, but you can still use:"
    echo "  - ./start_gnosis.sh to start services"
    echo "  - ./stop_gnosis.sh to stop services"
    echo "  - Process monitoring with ./monitor_gnosis.sh"
    exit 0
fi

# Create logs directory
mkdir -p "$GNOSIS_ROOT/logs"

# Copy service files to systemd directory
echo -e "\n${YELLOW}Installing service files...${NC}"
cp "$GNOSIS_ROOT/gnosis-backend.service" /etc/systemd/system/
cp "$GNOSIS_ROOT/gnosis-frontend.service" /etc/systemd/system/

# Reload systemd
echo -e "${YELLOW}Reloading systemd...${NC}"
systemctl daemon-reload

# Enable services
echo -e "${YELLOW}Enabling services...${NC}"
systemctl enable gnosis-backend.service
systemctl enable gnosis-frontend.service

echo -e "\n${GREEN}✓ Systemd services installed!${NC}"
echo ""
echo "Services will now start automatically on boot."
echo ""
echo "Manual control commands:"
echo "  Start:   systemctl start gnosis-backend gnosis-frontend"
echo "  Stop:    systemctl stop gnosis-backend gnosis-frontend"
echo "  Status:  systemctl status gnosis-backend gnosis-frontend"
echo "  Restart: systemctl restart gnosis-backend gnosis-frontend"
echo "  Logs:    journalctl -u gnosis-backend -f"
echo "           journalctl -u gnosis-frontend -f"
echo ""
echo "To start services now, run:"
echo "  systemctl start gnosis-backend gnosis-frontend"
echo "========================================="
