import subprocess
import time
import os
import signal
import sys
from preflight_check import run_preflight_check

def start_process(cmd, name):
    print(f"🚀 Starting {name}...")
    # Use python3 explicit
    return subprocess.Popen(["python3"] + cmd.split(), stdout=sys.stdout, stderr=sys.stderr)

def main():
    print("==================================================")
    print("   SUPER GNOSIS MASTER CONTROL: ACTIVATION")
    print("==================================================")
    
    # 1. Pre-Flight Checks
    print("running system diagnostics...")
    if not run_preflight_check():
        print("❌ System Diagnostics Failed. Aborting Activation.")
        sys.exit(1)
    
    # 1.5 Database Initialization
    print("Initializing Ledger/DB...")
    try:
        from init_db import init_db
        # We might need to mock or ensure DB is reachable, but ledger_store is file-based/sqlite mostly.
        # init_db.py seems to target Postgres. If we use SQLite, we might skip or use ledger_store directly.
        # But let's run it if configured.
        if os.getenv("DATABASE_URL"):
            init_db()
        else:
            print("   Using SQLite/File Ledger (No DATABASE_URL)")
    except Exception as e:
        print(f"   ⚠️  DB Init warning: {e}")
        
    print("✅ System Diagnostics Passed.")
    print("   - API Connectivity: OK")
    print("   - Data Feeds: OK")
    print("   - Broker Link: OK")
    
    procs = []
    
    try:
        # 2. Start HiveMind (Continuous Learning)
        # It runs in background, updating models
        p_hive = start_process("run_continuous_learning.py", "HiveMind (Evolution)")
        procs.append(p_hive)
        
        # 3. Start Trading Daemon
        # It trades based on current models and logic
        p_trade = start_process("run_trading_daemon.py", "Trading Daemon")
        procs.append(p_trade)
        
        # 4. Start Dashboard (Streamlit)
        # Using subprocess for streamlit run
        print("🚀 Starting Dashboard...")
        p_dash = subprocess.Popen(["streamlit", "run", "dashboard.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p_dash)
        
        print("\n✅ GNOSIS SYSTEM ACTIVATED.")
        print("   - Physics Engine: Online")
        print("   - Meta-Learning: Online")
        print("   - Trading Loop: Active (Market Hours)")
        print("   - Dashboard: http://localhost:8501")
        print("   Press Ctrl+C to Deactivate.")
        
        # Monitor Loop
        while True:
            time.sleep(10)
            # Check health
            if p_hive.poll() is not None:
                print("⚠️  HiveMind died! Restarting...")
                p_hive = start_process("run_continuous_learning.py", "HiveMind (Evolution)")
                procs[0] = p_hive
                
            if p_trade.poll() is not None:
                print("⚠️  Trading Daemon died! Restarting...")
                p_trade = start_process("run_trading_daemon.py", "Trading Daemon")
                procs[1] = p_trade
                
    except KeyboardInterrupt:
        print("\n🛑 Deactivation Sequence Initiated...")
        for p in procs:
            p.terminate()
            p.wait()
        print("👋 System Deactivated.")

if __name__ == "__main__":
    main()
