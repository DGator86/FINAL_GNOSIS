#!/bin/bash
# Install GNOSIS Trading System as a systemd service

set -e

echo "🚀 Installing GNOSIS Trading Service..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run as root (use sudo)"
    exit 1
fi

# Create logs directory
mkdir -p /root/FINAL_GNOSIS/logs
echo "✅ Created logs directory"

# Copy service file to systemd
cp /root/FINAL_GNOSIS/deployment/gnosis-trading.service /etc/systemd/system/
echo "✅ Copied service file to /etc/systemd/system/"

# Reload systemd
systemctl daemon-reload
echo "✅ Reloaded systemd daemon"

# Enable service (auto-start on boot)
systemctl enable gnosis-trading.service
echo "✅ Enabled service for auto-start on boot"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ GNOSIS Trading Service installed successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 MANAGEMENT COMMANDS:"
echo ""
echo "  Start service:    sudo systemctl start gnosis-trading"
echo "  Stop service:     sudo systemctl stop gnosis-trading"
echo "  Restart service:  sudo systemctl restart gnosis-trading"
echo "  View status:      sudo systemctl status gnosis-trading"
echo "  View logs:        sudo journalctl -u gnosis-trading -f"
echo "  View app logs:    tail -f /root/FINAL_GNOSIS/logs/trading-service.log"
echo ""
echo "⚠️  IMPORTANT: Service will NOT start automatically. Use:"
echo "     sudo systemctl start gnosis-trading"
echo ""
