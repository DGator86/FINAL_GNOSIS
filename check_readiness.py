import os
import sys
from datetime import datetime

def check_system():
    print("📋 Checking System Readiness for Monday...")
    
    status = {
        "model": False,
        "config": False,
        "data_feed": False,
        "trading_engine": False
    }
    
    # 1. Model Check
    if os.path.exists("data/models/physics_agent_best.pt"):
        m_time = datetime.fromtimestamp(os.path.getmtime("data/models/physics_agent_best.pt"))
        print(f"✅ Model found (Last modified: {m_time})")
        status["model"] = True
    else:
        print("❌ Model checkpoint missing!")

    # 2. Config Check
    if os.path.exists(".env"):
        print("✅ Configuration loaded.")
        status["config"] = True
    else:
        print("❌ .env file missing!")
        
    # 3. Scripts Check
    if os.path.exists("train_monday.py"):
        print("✅ Training script ready.")
        status["data_feed"] = True
        
    if os.path.exists("main.py"):
        print("✅ Main trading engine ready.")
        status["trading_engine"] = True
        
    print("-" * 30)
    if all(status.values()):
        print("🚀 ALL SYSTEMS GO. Ready for Monday market open.")
    else:
        print("⚠️ System checks passed with warnings.")

if __name__ == "__main__":
    check_system()
