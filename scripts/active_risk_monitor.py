#!/usr/bin/env python3
"""
ACTIVE RISK MONITOR - Continuous Position Management

This runs continuously and:
1. Monitors all positions every 30 seconds
2. Closes positions that breach stop-loss thresholds
3. Updates trailing stops for profitable positions
4. Manages options positions manually (since Alpaca doesn't support option stops)
5. Logs all activity

Author: GNOSIS Risk Management System
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import signal

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

# Load environment
def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value

load_env()

ALPACA_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").replace("/v2", "")

# Risk Parameters
STOCK_STOP_LOSS_PCT = -12.0      # Close stocks down more than 12%
OPTION_STOP_LOSS_PCT = -50.0     # Close options down more than 50%
PROFIT_LOCK_THRESHOLD = 20.0     # Start trailing at 20% profit
TRAILING_STOP_PCT = 30.0         # Trail 30% from peak (for big winners)
MAX_DAILY_LOSS = 5000            # Stop trading if daily loss exceeds this
CHECK_INTERVAL = 30              # Check every 30 seconds

# Tracking
position_peaks = {}  # Track high-water marks for trailing stops
daily_pnl = 0
running = True

headers = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"risk_monitor_{datetime.now().strftime('%Y%m%d')}.log"


def log(message: str, level: str = "INFO"):
    """Log message to file and console."""
    timestamp = datetime.now().isoformat()
    log_line = f"[{timestamp}] [{level}] {message}"
    print(log_line)
    with open(LOG_FILE, "a") as f:
        f.write(log_line + "\n")


def get_account():
    """Get account info."""
    try:
        response = httpx.get(f"{ALPACA_BASE_URL}/v2/account", headers=headers, timeout=10.0)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        log(f"Error getting account: {e}", "ERROR")
    return None


def get_positions():
    """Get all positions."""
    try:
        response = httpx.get(f"{ALPACA_BASE_URL}/v2/positions", headers=headers, timeout=10.0)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        log(f"Error getting positions: {e}", "ERROR")
    return []


def close_position(symbol: str, reason: str):
    """Close a position immediately."""
    global daily_pnl
    
    try:
        response = httpx.delete(
            f"{ALPACA_BASE_URL}/v2/positions/{symbol}",
            headers=headers,
            timeout=30.0
        )
        if response.status_code in [200, 204]:
            result = response.json() if response.text else {}
            log(f"🔴 CLOSED {symbol} - Reason: {reason}", "TRADE")
            return True
        else:
            log(f"Failed to close {symbol}: {response.text}", "ERROR")
    except Exception as e:
        log(f"Error closing {symbol}: {e}", "ERROR")
    return False


def check_position(position: dict) -> tuple:
    """
    Check if position should be closed.
    Returns: (should_close, reason)
    """
    global position_peaks
    
    symbol = position["symbol"]
    qty = float(position["qty"])
    current_price = float(position.get("current_price", 0))
    avg_entry = float(position.get("avg_entry_price", 0))
    unrealized_pl = float(position.get("unrealized_pl", 0))
    unrealized_plpc = float(position.get("unrealized_plpc", 0)) * 100
    
    is_option = len(symbol) > 10
    
    # Determine stop loss threshold
    stop_threshold = OPTION_STOP_LOSS_PCT if is_option else STOCK_STOP_LOSS_PCT
    
    # Check hard stop loss
    if unrealized_plpc <= stop_threshold:
        return True, f"STOP LOSS ({unrealized_plpc:.1f}% <= {stop_threshold}%)"
    
    # Track peak for trailing stop
    if symbol not in position_peaks:
        position_peaks[symbol] = unrealized_plpc
    else:
        if unrealized_plpc > position_peaks[symbol]:
            position_peaks[symbol] = unrealized_plpc
    
    peak_pnl = position_peaks[symbol]
    
    # Trailing stop for profitable positions
    if peak_pnl >= PROFIT_LOCK_THRESHOLD:
        # Calculate drawdown from peak
        drawdown = peak_pnl - unrealized_plpc
        max_drawdown = peak_pnl * (TRAILING_STOP_PCT / 100)
        
        if drawdown >= max_drawdown:
            return True, f"TRAILING STOP (Peak: {peak_pnl:.1f}%, Current: {unrealized_plpc:.1f}%, Drawdown: {drawdown:.1f}%)"
    
    return False, None


def run_check_cycle():
    """Run one check cycle."""
    global daily_pnl
    
    positions = get_positions()
    if not positions:
        return
    
    log(f"📊 Checking {len(positions)} positions...")
    
    closed_count = 0
    total_unrealized = 0
    
    for p in positions:
        symbol = p["symbol"]
        unrealized_pl = float(p.get("unrealized_pl", 0))
        unrealized_plpc = float(p.get("unrealized_plpc", 0)) * 100
        total_unrealized += unrealized_pl
        
        should_close, reason = check_position(p)
        
        if should_close:
            log(f"⚠️ {symbol}: {reason} | P&L: ${unrealized_pl:+.2f} ({unrealized_plpc:+.1f}%)", "ALERT")
            if close_position(symbol, reason):
                closed_count += 1
                daily_pnl += unrealized_pl
    
    if closed_count > 0:
        log(f"Closed {closed_count} positions. Daily realized P&L: ${daily_pnl:+.2f}")
    
    # Check daily loss limit
    account = get_account()
    if account:
        day_pl = float(account.get("equity", 0)) - float(account.get("last_equity", 0))
        if day_pl <= -MAX_DAILY_LOSS:
            log(f"🚨 DAILY LOSS LIMIT REACHED: ${day_pl:.2f}", "CRITICAL")
            # Could add logic to close all positions here
    
    log(f"Total unrealized P&L: ${total_unrealized:+.2f}")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global running
    log("Received shutdown signal. Stopping...", "INFO")
    running = False


def main():
    global running
    
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    log("="*60)
    log("🛡️ GNOSIS ACTIVE RISK MONITOR STARTED")
    log("="*60)
    log(f"Stock Stop Loss: {STOCK_STOP_LOSS_PCT}%")
    log(f"Option Stop Loss: {OPTION_STOP_LOSS_PCT}%")
    log(f"Profit Lock Threshold: {PROFIT_LOCK_THRESHOLD}%")
    log(f"Trailing Stop: {TRAILING_STOP_PCT}% from peak")
    log(f"Max Daily Loss: ${MAX_DAILY_LOSS}")
    log(f"Check Interval: {CHECK_INTERVAL}s")
    log("="*60)
    
    while running:
        try:
            run_check_cycle()
        except Exception as e:
            log(f"Error in check cycle: {e}", "ERROR")
        
        # Wait for next cycle
        for _ in range(CHECK_INTERVAL):
            if not running:
                break
            time.sleep(1)
    
    log("Risk monitor stopped.")


if __name__ == "__main__":
    main()
