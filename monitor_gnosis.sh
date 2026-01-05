#!/bin/bash
# Gnosis Process Monitor
# Monitors services and automatically restarts them if they crash
# Run in background: nohup ./monitor_gnosis.sh &

GNOSIS_ROOT="/home/user/FINAL_GNOSIS"
LOG_DIR="$GNOSIS_ROOT/logs"
MONITOR_LOG="$LOG_DIR/monitor.log"
CHECK_INTERVAL=30  # Check every 30 seconds

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MONITOR_LOG"
}

check_and_restart_backend() {
    if ! lsof -i :8000 > /dev/null 2>&1; then
        log "⚠️  Backend not running on port 8000, restarting..."
        cd "$GNOSIS_ROOT"
        nohup python3 -m uvicorn web_api:app --host 0.0.0.0 --port 8000 >> "$LOG_DIR/backend.log" 2>&1 &
        sleep 5
        if lsof -i :8000 > /dev/null 2>&1; then
            log "✓ Backend restarted successfully"
        else
            log "✗ Backend restart failed - check logs"
        fi
    fi
}

check_and_restart_frontend() {
    if ! lsof -i :3000 > /dev/null 2>&1; then
        log "⚠️  Frontend not running on port 3000, restarting..."
        cd "$GNOSIS_ROOT/saas_frontend"
        nohup npm run dev >> "$LOG_DIR/frontend.log" 2>&1 &
        sleep 5
        if lsof -i :3000 > /dev/null 2>&1; then
            log "✓ Frontend restarted successfully"
        else
            log "✗ Frontend restart failed - check logs"
        fi
    fi
}

check_and_restart_postgres() {
    if ! ps aux | grep -v grep | grep postgres > /dev/null; then
        log "⚠️  PostgreSQL not running, restarting..."
        service postgresql start
        sleep 3
        if ps aux | grep -v grep | grep postgres > /dev/null; then
            log "✓ PostgreSQL restarted successfully"
        else
            log "✗ PostgreSQL restart failed"
        fi
    fi
}

log "🔍 Gnosis Process Monitor started (PID: $$)"
log "Monitoring interval: ${CHECK_INTERVAL}s"
log "Logs: $MONITOR_LOG"

while true; do
    check_and_restart_postgres
    check_and_restart_backend
    check_and_restart_frontend
    sleep $CHECK_INTERVAL
done
