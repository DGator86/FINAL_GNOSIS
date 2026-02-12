#!/usr/bin/env python3
"""
GNOSIS Trading System - Launch Script

This script initializes and starts all trading components:
1. Loads configuration from .env
2. Validates API connections
3. Starts the paper trading engine
4. Enables real-time monitoring

Usage:
    python scripts/launch.py [--dry-run] [--train-first] [--symbols SPY,QQQ]
"""

import os
import sys
import asyncio
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/gnosis_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger('GNOSIS')


class GnosisLauncher:
    """Main launcher for the GNOSIS trading system."""
    
    def __init__(self, dry_run: bool = False, symbols: list = None):
        self.dry_run = dry_run
        self.symbols = symbols or os.getenv('TRADING_SYMBOLS', 'SPY,QQQ,AAPL,NVDA,MSFT').split(',')
        self.config = self._load_config()
        
    def _load_config(self) -> dict:
        """Load configuration from environment."""
        return {
            'alpaca': {
                'api_key': os.getenv('ALPACA_API_KEY'),
                'secret_key': os.getenv('ALPACA_SECRET_KEY'),
                'base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2'),
            },
            'unusual_whales': {
                'token': os.getenv('UNUSUAL_WHALES_API_TOKEN'),
            },
            'trading': {
                'max_positions': int(os.getenv('MAX_POSITIONS', 10)),
                'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', 5000)),
                'scan_interval': int(os.getenv('SCAN_INTERVAL', 60)),
            },
            'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
        }
    
    async def validate_connections(self) -> bool:
        """Validate all API connections."""
        logger.info("=" * 60)
        logger.info("GNOSIS Trading System - Connection Validation")
        logger.info("=" * 60)
        
        all_valid = True
        
        # 1. Validate Alpaca Connection
        logger.info("\n[1/3] Checking Alpaca Paper Trading API...")
        alpaca_valid = await self._check_alpaca()
        if alpaca_valid:
            logger.info("  ✅ Alpaca connection successful")
        else:
            logger.error("  ❌ Alpaca connection failed")
            all_valid = False
        
        # 2. Validate Unusual Whales Connection
        logger.info("\n[2/3] Checking Unusual Whales API...")
        uw_valid = await self._check_unusual_whales()
        if uw_valid:
            logger.info("  ✅ Unusual Whales connection successful")
        else:
            logger.warning("  ⚠️ Unusual Whales connection failed (optional)")
        
        # 3. Check Redis
        logger.info("\n[3/3] Checking Redis connection...")
        redis_valid = await self._check_redis()
        if redis_valid:
            logger.info("  ✅ Redis connection successful")
        else:
            logger.warning("  ⚠️ Redis not available (will use in-memory cache)")
        
        return all_valid
    
    async def _check_alpaca(self) -> bool:
        """Check Alpaca API connection."""
        try:
            import aiohttp
            
            headers = {
                'APCA-API-KEY-ID': self.config['alpaca']['api_key'],
                'APCA-API-SECRET-KEY': self.config['alpaca']['secret_key'],
            }
            
            async with aiohttp.ClientSession() as session:
                # Check account
                url = f"{self.config['alpaca']['base_url']}/account"
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        account = await resp.json()
                        logger.info(f"  Account Status: {account.get('status')}")
                        logger.info(f"  Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
                        logger.info(f"  Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")
                        logger.info(f"  Cash: ${float(account.get('cash', 0)):,.2f}")
                        return True
                    else:
                        error = await resp.text()
                        logger.error(f"  Error: {error}")
                        return False
                        
        except Exception as e:
            logger.error(f"  Exception: {e}")
            return False
    
    async def _check_unusual_whales(self) -> bool:
        """Check Unusual Whales API connection."""
        try:
            import aiohttp
            
            token = self.config['unusual_whales']['token']
            if not token:
                return False
            
            headers = {
                'Authorization': f'Bearer {token}',
                'Accept': 'application/json',
            }
            
            async with aiohttp.ClientSession() as session:
                url = "https://api.unusualwhales.com/api/market/overview"
                async with session.get(url, headers=headers) as resp:
                    return resp.status == 200
                    
        except Exception as e:
            logger.warning(f"  Unusual Whales check failed: {e}")
            return False
    
    async def _check_redis(self) -> bool:
        """Check Redis connection."""
        try:
            import redis.asyncio as redis
            
            client = redis.from_url(self.config['redis_url'])
            await client.ping()
            await client.close()
            return True
            
        except Exception as e:
            logger.warning(f"  Redis check failed: {e}")
            return False
    
    async def start_trading(self):
        """Start the paper trading engine."""
        logger.info("\n" + "=" * 60)
        logger.info("Starting GNOSIS Paper Trading Engine")
        logger.info("=" * 60)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'PAPER TRADING'}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Max Positions: {self.config['trading']['max_positions']}")
        logger.info(f"Max Daily Loss: ${self.config['trading']['max_daily_loss']:,.2f}")
        logger.info("=" * 60 + "\n")
        
        try:
            from trade.paper_trading_engine import PaperTradingEngine
            
            # Create engine configuration
            engine_config = {
                'scan_interval': self.config['trading']['scan_interval'],
                'position_check_interval': 30,
                'max_daily_loss': self.config['trading']['max_daily_loss'],
                'max_positions': self.config['trading']['max_positions'],
            }
            
            # Initialize the paper trading engine
            engine = PaperTradingEngine(
                symbols=self.symbols,
                config=engine_config,
                dry_run=self.dry_run
            )
            
            # Start the engine
            logger.info("🚀 Engine starting...")
            await engine.start()
            
        except KeyboardInterrupt:
            logger.info("\n⛔ Shutdown signal received")
            await engine.stop()
            logger.info("Engine stopped gracefully")
            
        except Exception as e:
            logger.error(f"Engine error: {e}")
            raise
    
    async def run(self):
        """Main entry point."""
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        logger.info("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗ ███╗   ██╗ ██████╗ ███████╗██╗███████╗             ║
║  ██╔════╝ ████╗  ██║██╔═══██╗██╔════╝██║██╔════╝             ║
║  ██║  ███╗██╔██╗ ██║██║   ██║███████╗██║███████╗             ║
║  ██║   ██║██║╚██╗██║██║   ██║╚════██║██║╚════██║             ║
║  ╚██████╔╝██║ ╚████║╚██████╔╝███████║██║███████║             ║
║   ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚═╝╚══════╝             ║
║                                                               ║
║           Super Gnosis Elite Trading System                   ║
║                    Paper Trading Mode                         ║
╚═══════════════════════════════════════════════════════════════╝
        """)
        
        # Validate connections
        if not await self.validate_connections():
            logger.error("\n❌ Critical connection validation failed!")
            logger.error("Please check your API credentials in .env file")
            sys.exit(1)
        
        logger.info("\n✅ All critical connections validated!")
        
        if self.dry_run:
            logger.info("\n🔍 DRY RUN MODE - No actual trades will be placed")
        
        # Start trading
        await self.start_trading()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='GNOSIS Trading System Launcher')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Run without placing actual trades')
    parser.add_argument('--symbols', type=str, 
                        help='Comma-separated list of symbols to trade')
    parser.add_argument('--train-first', action='store_true',
                        help='Train ML models before starting')
    
    args = parser.parse_args()
    
    symbols = args.symbols.split(',') if args.symbols else None
    
    # Train models first if requested
    if args.train_first:
        print("\n🧠 Training ML models first...")
        from scripts.quick_train import quick_train
        await quick_train()
    
    # Launch trading system
    launcher = GnosisLauncher(dry_run=args.dry_run, symbols=symbols)
    await launcher.run()


if __name__ == '__main__':
    asyncio.run(main())
