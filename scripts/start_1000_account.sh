#!/bin/bash
# ============================================================================
# GNOSIS $1000 Starter Account - Launch Script
# ============================================================================
# This script launches GNOSIS configured for a $1000 account
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
GNOSIS_DIR="/home/user/FINAL_GNOSIS"
ENV_FILE="$GNOSIS_DIR/.env.starter"
CONFIG_FILE="$GNOSIS_DIR/config/config_starter.yaml"
LOG_DIR="/var/log/gnosis"

echo -e "${GREEN}=== GNOSIS $1000 Starter Account ===${NC}"
echo ""

# Check if running in correct directory
if [ ! -d "$GNOSIS_DIR" ]; then
    echo -e "${RED}Error: GNOSIS directory not found at $GNOSIS_DIR${NC}"
    exit 1
fi

cd "$GNOSIS_DIR"

# Check if .env.starter exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Warning: $ENV_FILE not found${NC}"
    echo "Creating from template..."

    if [ -f ".env.starter.template" ]; then
        cp .env.starter.template .env.starter
        echo -e "${YELLOW}Please edit .env.starter with your Alpaca API keys${NC}"
        echo "Then run this script again."
        exit 1
    else
        echo -e "${RED}Error: .env.starter.template not found${NC}"
        exit 1
    fi
fi

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Source environment
source .env.starter

# Check critical environment variables
if [ -z "$ALPACA_API_KEY" ] || [ "$ALPACA_API_KEY" = "your_paper_api_key_here" ]; then
    echo -e "${RED}Error: ALPACA_API_KEY not set in .env.starter${NC}"
    echo "Please edit .env.starter and add your Alpaca API key"
    exit 1
fi

if [ -z "$ALPACA_SECRET_KEY" ] || [ "$ALPACA_SECRET_KEY" = "your_paper_secret_key_here" ]; then
    echo -e "${RED}Error: ALPACA_SECRET_KEY not set in .env.starter${NC}"
    echo "Please edit .env.starter and add your Alpaca secret key"
    exit 1
fi

# Check if using paper trading
if [[ "$ALPACA_BASE_URL" == *"paper"* ]]; then
    echo -e "${GREEN}✓ Using PAPER TRADING (recommended for testing)${NC}"
else
    echo -e "${YELLOW}⚠ WARNING: Using LIVE TRADING${NC}"
    echo -e "${YELLOW}Are you sure? Paper trading is recommended first.${NC}"
    read -p "Continue with LIVE trading? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Aborted. Switch to paper trading in .env.starter"
        exit 1
    fi
fi

# Check account balance
echo ""
echo "Checking Alpaca account..."
python3 -c "
from alpaca_trade_api import REST
import os
api = REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url=os.getenv('ALPACA_BASE_URL')
)
account = api.get_account()
print(f'Account Equity: \${float(account.equity):,.2f}')
print(f'Cash Available: \${float(account.cash):,.2f}')
print(f'Buying Power: \${float(account.buying_power):,.2f}')

if float(account.equity) < 900:
    print('⚠ WARNING: Account balance is less than \$900')
    print('   Consider adding funds or adjusting position sizes')
" 2>/dev/null || {
    echo -e "${RED}Error: Could not connect to Alpaca API${NC}"
    echo "Please check your API keys in .env.starter"
    exit 1
}

echo -e "${GREEN}✓ Alpaca connection successful${NC}"

# Show current configuration
echo ""
echo -e "${GREEN}=== Configuration ===${NC}"
echo "Capital: \$$DEFAULT_CAPITAL"
echo "Max Positions: $MAX_POSITIONS"
echo "Max Position Size: ${MAX_POSITION_SIZE}% (\$$(echo "$DEFAULT_CAPITAL * $MAX_POSITION_SIZE" | bc))"
echo "Max Daily Loss: \$$MAX_DAILY_LOSS"
echo "Min Confidence: $MIN_CONFIDENCE"
echo "Symbols: $TRADING_SYMBOLS"
echo "Trade Type: $TRADE_TYPE"
echo "Trading Enabled: $ENABLE_TRADING"

# Menu
echo ""
echo -e "${GREEN}=== What would you like to do? ===${NC}"
echo "1) Test single analysis (SPY)"
echo "2) Run paper trading (recommended first)"
echo "3) Start automated trading"
echo "4) Launch dashboard"
echo "5) View logs"
echo "6) Check account status"
echo "7) Exit"
echo ""
read -p "Select option (1-7): " option

case $option in
    1)
        echo ""
        echo -e "${GREEN}Running single analysis on SPY...${NC}"
        python main.py run-once --symbol SPY \
            --config "$CONFIG_FILE" 2>&1 | tee "$LOG_DIR/test_run.log"
        ;;

    2)
        echo ""
        echo -e "${GREEN}Starting paper trading loop (1 hour)...${NC}"
        echo "Trading: OFF (just monitoring)"
        ENABLE_TRADING=false python main.py multi-symbol-loop \
            --symbols SPY QQQ \
            --duration 3600 \
            --config "$CONFIG_FILE" 2>&1 | tee "$LOG_DIR/paper_trading.log"
        ;;

    3)
        if [ "$ENABLE_TRADING" != "true" ]; then
            echo -e "${YELLOW}Warning: ENABLE_TRADING=false in .env.starter${NC}"
            read -p "Enable trading and continue? (yes/no): " enable
            if [ "$enable" = "yes" ]; then
                export ENABLE_TRADING=true
            else
                echo "Aborted"
                exit 0
            fi
        fi

        echo ""
        echo -e "${GREEN}Starting automated trading daemon...${NC}"
        echo "Press Ctrl+C to stop"
        echo ""

        python run_trading_daemon.py \
            --config "$CONFIG_FILE" 2>&1 | tee "$LOG_DIR/trading_daemon.log"
        ;;

    4)
        echo ""
        echo -e "${GREEN}Launching dashboard...${NC}"
        echo "Open browser to: http://localhost:8501"
        streamlit run dashboard.py
        ;;

    5)
        echo ""
        echo -e "${GREEN}Recent logs:${NC}"
        if [ -f "$LOG_DIR/trading_daemon.log" ]; then
            tail -50 "$LOG_DIR/trading_daemon.log"
        else
            echo "No logs found yet"
        fi
        ;;

    6)
        echo ""
        echo -e "${GREEN}Account Status:${NC}"
        python3 -c "
from alpaca_trade_api import REST
import os
api = REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url=os.getenv('ALPACA_BASE_URL')
)
account = api.get_account()
positions = api.list_positions()

print(f'Equity: \${float(account.equity):,.2f}')
print(f'Cash: \${float(account.cash):,.2f}')
print(f'P&L Today: \${float(account.equity) - float(account.last_equity):,.2f}')
print(f'Positions: {len(positions)}')

for pos in positions:
    pnl = float(pos.unrealized_pl)
    pnl_pct = float(pos.unrealized_plpc) * 100
    print(f'  {pos.symbol}: {pos.qty} shares @ \${float(pos.avg_entry_price):.2f}')
    print(f'    P&L: \${pnl:.2f} ({pnl_pct:+.2f}%)')
"
        ;;

    7)
        echo "Goodbye!"
        exit 0
        ;;

    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
