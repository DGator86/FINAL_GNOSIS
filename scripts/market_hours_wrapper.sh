#!/bin/bash
#
# Market Hours Wrapper Script for GNOSIS
# Only executes trading during US market hours (Mon-Fri, 9:30 AM - 4:00 PM EST)
#

set -e  # Exit on error

# Configuration
GNOSIS_DIR="/home/user/FINAL_GNOSIS"
VENV_PATH="$GNOSIS_DIR/venv/bin/activate"
LOG_DIR="/var/log/gnosis"
PYTHON_SCRIPT="main.py"
SCRIPT_ARGS="multi-symbol-loop --top 5 --duration 300"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Get current time in EST (adjust TZ as needed)
export TZ="America/New_York"
CURRENT_HOUR=$(date +%H)
CURRENT_MINUTE=$(date +%M)
CURRENT_DAY=$(date +%u)  # 1-7 (Monday-Sunday)

# Convert to minutes since midnight for easier comparison
CURRENT_TIME_MINUTES=$((10#$CURRENT_HOUR * 60 + 10#$CURRENT_MINUTE))

# Market hours: 9:30 AM - 4:00 PM EST
MARKET_OPEN_MINUTES=$((9 * 60 + 30))   # 9:30 AM = 570 minutes
MARKET_CLOSE_MINUTES=$((16 * 60))       # 4:00 PM = 960 minutes

# Check if today is a weekday (1=Monday, 5=Friday)
if [ "$CURRENT_DAY" -gt 5 ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Market closed: Weekend (day $CURRENT_DAY)" >> "$LOG_DIR/market_hours.log"
    exit 0
fi

# Check if within market hours
if [ "$CURRENT_TIME_MINUTES" -lt "$MARKET_OPEN_MINUTES" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Market not yet open (opens at 9:30 AM EST)" >> "$LOG_DIR/market_hours.log"
    exit 0
fi

if [ "$CURRENT_TIME_MINUTES" -gt "$MARKET_CLOSE_MINUTES" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Market closed (closed at 4:00 PM EST)" >> "$LOG_DIR/market_hours.log"
    exit 0
fi

# Market is open - execute GNOSIS
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Market open - executing GNOSIS" >> "$LOG_DIR/market_hours.log"

cd "$GNOSIS_DIR" || exit 1

# Activate virtual environment
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
else
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: Virtual environment not found at $VENV_PATH" >> "$LOG_DIR/error.log"
    exit 1
fi

# Execute the trading script
python "$PYTHON_SCRIPT" $SCRIPT_ARGS >> "$LOG_DIR/trading.log" 2>> "$LOG_DIR/error.log"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: GNOSIS exited with code $EXIT_CODE" >> "$LOG_DIR/error.log"
    exit $EXIT_CODE
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] GNOSIS execution completed successfully" >> "$LOG_DIR/market_hours.log"
