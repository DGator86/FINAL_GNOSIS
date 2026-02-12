#!/bin/bash

# GNOSIS Daily Training Wrapper
# Can be scheduled via cron

# 1. Setup paths
PROJECT_DIR="/home/root/webapp"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

DATE=$(date +"%Y-%m-%d")
LOG_FILE="$LOG_DIR/training_$DATE.log"

echo "========================================" >> "$LOG_FILE"
echo "Starting Training Cycle: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# 2. Navigate
cd "$PROJECT_DIR" || exit 1

# 3. Execute
# Using python3 -u for unbuffered output to logs
python3 -u train_daily.py >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Training completed at $(date)" >> "$LOG_FILE"
else
    echo "FAILURE: Training failed with code $EXIT_CODE at $(date)" >> "$LOG_FILE"
fi

echo "" >> "$LOG_FILE"
