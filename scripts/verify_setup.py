#!/usr/bin/env python3
"""
GNOSIS Trading System - Setup Verification

Verifies all components are properly configured and ready to trade.

Usage:
    python scripts/verify_setup.py
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}  {text}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")


def print_ok(text):
    print(f"  {GREEN}✓{RESET} {text}")


def print_fail(text):
    print(f"  {RED}✗{RESET} {text}")


def print_warn(text):
    print(f"  {YELLOW}!{RESET} {text}")


async def verify_alpaca():
    """Verify Alpaca connection."""
    print_header("ALPACA CONNECTION")
    
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    base_url = os.getenv('ALPACA_BASE_URL')
    
    if not api_key or not secret_key:
        print_fail("API credentials not configured")
        return False
    
    print_ok(f"API Key: {api_key[:8]}...")
    print_ok(f"Base URL: {base_url}")
    
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            headers = {
                'APCA-API-KEY-ID': api_key,
                'APCA-API-SECRET-KEY': secret_key
            }
            
            async with session.get(f"{base_url}/account", headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print_ok(f"Account Status: {data.get('status', 'N/A')}")
                    print_ok(f"Buying Power: ${float(data.get('buying_power', 0)):,.2f}")
                    print_ok(f"Portfolio Value: ${float(data.get('portfolio_value', 0)):,.2f}")
                    print_ok(f"Cash: ${float(data.get('cash', 0)):,.2f}")
                    return True
                else:
                    error = await resp.text()
                    print_fail(f"API Error ({resp.status}): {error}")
                    return False
                    
    except Exception as e:
        print_fail(f"Connection failed: {e}")
        return False


async def verify_unusual_whales():
    """Verify Unusual Whales connection."""
    print_header("UNUSUAL WHALES API")
    
    api_token = os.getenv('UNUSUAL_WHALES_API_TOKEN')
    
    if not api_token:
        print_warn("API token not configured (optional)")
        return True
    
    print_ok(f"API Token: {api_token[:8]}...")
    
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            headers = {'Authorization': f'Bearer {api_token}'}
            
            async with session.get(
                "https://api.unusualwhales.com/api/market/flow",
                headers=headers
            ) as resp:
                if resp.status == 200:
                    print_ok("Connection successful")
                    return True
                elif resp.status == 401:
                    print_warn("Authentication failed - check token")
                    return True  # Non-fatal
                else:
                    print_warn(f"API returned status {resp.status}")
                    return True  # Non-fatal
                    
    except Exception as e:
        print_warn(f"Connection test failed: {e}")
        return True  # Non-fatal


def verify_environment():
    """Verify environment configuration."""
    print_header("ENVIRONMENT")
    
    checks = [
        ('ALPACA_API_KEY', True),
        ('ALPACA_SECRET_KEY', True),
        ('ALPACA_BASE_URL', True),
        ('TRADING_SYMBOLS', False),
        ('MAX_DAILY_LOSS', False),
        ('MAX_POSITIONS', False),
        ('REDIS_HOST', False),
    ]
    
    all_ok = True
    for key, required in checks:
        value = os.getenv(key)
        if value:
            display = value[:20] + '...' if len(value) > 20 else value
            print_ok(f"{key}: {display}")
        elif required:
            print_fail(f"{key}: NOT SET (required)")
            all_ok = False
        else:
            print_warn(f"{key}: Not set (optional)")
    
    return all_ok


def verify_modules():
    """Verify required modules can be imported."""
    print_header("MODULES")
    
    # Core modules required for trading
    modules = [
        ('trade.paper_trading_engine', 'PaperTradingEngine'),
        ('integration.trading_hub', 'TradingHub'),
        ('ml.training.rl_trainer', 'RLTrainer'),
        ('ml.training.transformer_trainer', 'TransformerTrainer'),
        ('ml.training.orchestrator', 'TrainingOrchestrator'),
        ('middleware.rate_limiter', 'RateLimiter'),
        ('scanner.options_flow_scanner', 'OptionsFlowScanner'),
        ('trade.greeks_hedger', 'GreeksHedger'),
    ]
    
    all_ok = True
    for module_name, attr_name in modules:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            if hasattr(module, attr_name):
                print_ok(f"{module_name}")
            else:
                print_fail(f"{module_name} - missing {attr_name}")
                all_ok = False
        except ImportError as e:
            print_fail(f"{module_name} - {e}")
            all_ok = False
    
    return all_ok


def verify_models():
    """Check for trained models."""
    print_header("ML MODELS")
    
    models_dir = Path(os.getenv('ML_MODELS_DIR', 'models/trained'))
    
    if not models_dir.exists():
        print_warn(f"Models directory not found: {models_dir}")
        print_warn("Run 'python scripts/quick_train.py' to train models")
        return True  # Non-fatal
    
    rl_models = list(models_dir.glob('rl_agent_*.pkl'))
    transformer_models = list(models_dir.glob('transformer_*.pkl'))
    
    if rl_models:
        print_ok(f"RL models found: {len(rl_models)}")
        for m in rl_models[-3:]:  # Show last 3
            print(f"       - {m.name}")
    else:
        print_warn("No RL models found")
    
    if transformer_models:
        print_ok(f"Transformer models found: {len(transformer_models)}")
        for m in transformer_models[-3:]:
            print(f"       - {m.name}")
    else:
        print_warn("No Transformer models found")
    
    if not rl_models and not transformer_models:
        print_warn("No trained models - system will use rule-based signals")
    
    return True


async def main():
    """Run all verifications."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}  GNOSIS TRADING SYSTEM - SETUP VERIFICATION{RESET}")
    print(f"{BLUE}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    results = []
    
    # Run verifications
    results.append(('Environment', verify_environment()))
    results.append(('Modules', verify_modules()))
    results.append(('ML Models', verify_models()))
    results.append(('Alpaca', await verify_alpaca()))
    results.append(('Unusual Whales', await verify_unusual_whales()))
    
    # Summary
    print_header("SUMMARY")
    
    all_ok = True
    for name, ok in results:
        if ok:
            print_ok(f"{name}")
        else:
            print_fail(f"{name}")
            all_ok = False
    
    print("")
    if all_ok:
        print(f"{GREEN}{'='*60}{RESET}")
        print(f"{GREEN}  ALL CHECKS PASSED - READY TO TRADE!{RESET}")
        print(f"{GREEN}{'='*60}{RESET}")
        print("")
        print("Start trading with:")
        print("  ./start.sh local       # Start locally")
        print("  ./start.sh docker      # Start with Docker")
        print("  ./start.sh dry-run     # Test without orders")
        print("")
    else:
        print(f"{RED}{'='*60}{RESET}")
        print(f"{RED}  SOME CHECKS FAILED - Please fix issues above{RESET}")
        print(f"{RED}{'='*60}{RESET}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
