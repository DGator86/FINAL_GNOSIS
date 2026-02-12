#!/usr/bin/env python3
"""
GNOSIS Service Runner
Handles graceful startup, shutdown, and logging for production deployment.
"""

import os
import sys
import signal
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "gnosis_saas.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GNOSIS")

# Load environment
def load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value
        logger.info("Environment loaded from .env")

load_env()

# Import after env is loaded
import uvicorn
from saas.gnosis_saas import app

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    port = int(os.getenv("GNOSIS_PORT", "8888"))
    host = os.getenv("GNOSIS_HOST", "0.0.0.0")
    
    logger.info("="*60)
    logger.info("  GNOSIS SaaS - Starting Production Server")
    logger.info("="*60)
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  PID: {os.getpid()}")
    logger.info(f"  Started: {datetime.now().isoformat()}")
    logger.info("="*60)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )
