#!/bin/bash
# Quick Cloud Deployment Script for Super Gnosis Trading System

set -e

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                        ║"
echo "║         🚀 SUPER GNOSIS - CLOUD DEPLOYMENT SETUP 🚀                   ║"
echo "║                                                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your API credentials:"
    echo "   - ALPACA_API_KEY"
    echo "   - ALPACA_SECRET_KEY"
    echo "   - UNUSUAL_WHALES_API_TOKEN"
    echo ""
    echo "After editing .env, run this script again."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "📦 Docker not found. Installing..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "✅ Docker installed!"
    echo ""
    echo "⚠️  You need to log out and back in for Docker permissions."
    echo "    Then run this script again."
    exit 0
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "📦 Installing docker-compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ docker-compose installed!"
fi

echo "🔍 Checking .env configuration..."
if grep -q "your_alpaca_api_key_here" .env || grep -q "your_alpaca_secret_key_here" .env; then
    echo "❌ .env file still contains placeholder values!"
    echo ""
    echo "Please edit .env and add your real API credentials:"
    echo "   nano .env"
    exit 1
fi

echo "✅ .env file configured"
echo ""

# Choose deployment mode
echo "Which mode do you want to run?"
echo ""
echo "1) start_trading_now.py       - Simple paper trading (5 symbols)"
echo "2) start_scanner_trading.py   - Multi-timeframe scanner + trading"
echo "3) start_full_trading_system.py - Complete system with all engines"
echo "4) start_dynamic_trading.py   - Dynamic universe (25+ symbols)"
echo ""
read -p "Enter choice [1-4] (default: 2): " mode_choice
mode_choice=${mode_choice:-2}

case $mode_choice in
    1) COMMAND="python start_trading_now.py" ;;
    2) COMMAND="python start_scanner_trading.py" ;;
    3) COMMAND="python start_full_trading_system.py" ;;
    4) COMMAND="python start_dynamic_trading.py" ;;
    *) echo "Invalid choice. Using default (scanner trading)"; COMMAND="python start_scanner_trading.py" ;;
esac

echo ""
echo "🏗️  Building Docker container..."
docker-compose build

echo ""
echo "🚀 Starting trading system..."
echo "   Command: $COMMAND"

# Update docker-compose with chosen command
cat > docker-compose.override.yml << EOF
version: '3.8'

services:
  trading-bot:
    command: $COMMAND
EOF

docker-compose up -d

echo ""
echo "✅ Trading system is now running in the background!"
echo ""
echo "📊 Useful commands:"
echo "   View logs:    docker-compose logs -f"
echo "   Stop:         docker-compose down"
echo "   Restart:      docker-compose restart"
echo "   Status:       docker-compose ps"
echo ""
echo "🌐 If running dashboard mode, access at: http://$(hostname -I | awk '{print $1}'):8080"
echo ""
echo "💰 Check your trades at: https://app.alpaca.markets/paper/dashboard"
echo ""
