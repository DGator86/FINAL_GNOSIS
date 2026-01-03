import os
import sys
from pathlib import Path
# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.active_risk_monitor import get_positions, load_env

load_env()
positions = get_positions()

print(f"\n📊 Checking {len(positions)} Open Positions:\n")
print(f"{'SYMBOL':<10} {'TYPE':<8} {'P&L %':<10} {'P&L $':<10} {'VALUE':<10}")
print("-" * 55)

for p in positions:
    symbol = p['symbol']
    # Simple heuristic for type
    asset_type = "OPTION" if len(symbol) > 6 else "STOCK" 
    pnl_pct = float(p.get('unrealized_plpc', 0)) * 100
    pnl_usd = float(p.get('unrealized_pl', 0))
    value = float(p.get('market_value', 0))
    
    status = "✅" if pnl_pct >= 0 else "🔻"
    print(f"{status} {symbol:<7} {asset_type:<8} {pnl_pct:>7.2f}% {pnl_usd:>9.2f} {value:>9.2f}")

print("\n")
