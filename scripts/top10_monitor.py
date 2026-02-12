#!/usr/bin/env python3
"""
Top 10 Watchlist Monitor

Real-time monitoring dashboard for the dynamically ranked top 10 stocks.

Features:
- Live price updates
- Ranking changes detection
- Alert on significant moves
- Options flow integration
- Position tracking

Author: GNOSIS Trading System
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from universe.dynamic_universe import DynamicUniverseManager, DYNAMIC_UNIVERSE


# =============================================================================
# Configuration
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def load_env():
    """Load environment variables."""
    env_path = Path(__file__).parent.parent / ".env"
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ[key] = value
    except FileNotFoundError:
        pass


load_env()


# =============================================================================
# API Clients
# =============================================================================

def get_alpaca_account():
    """Get Alpaca account info."""
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        return None
    
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    
    try:
        url = f"{base_url.rstrip('/')}/account"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return None


def get_alpaca_positions():
    """Get Alpaca positions."""
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not api_secret:
        return []
    
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    
    try:
        url = f"{base_url.rstrip('/')}/positions"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []


def get_unusual_whales_flow(symbol: str):
    """Get options flow from Unusual Whales."""
    api_token = os.getenv("UNUSUAL_WHALES_API_TOKEN")
    
    if not api_token:
        return None
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json",
    }
    
    try:
        url = f"https://api.unusualwhales.com/api/stock/{symbol}/flow-alerts"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("data", [])[:3]  # Top 3 alerts
    except Exception:
        return None


# =============================================================================
# Monitor Class
# =============================================================================

class Top10Monitor:
    """
    Real-time monitor for the top 10 ranked stocks.
    """
    
    def __init__(self):
        self.universe_manager = DynamicUniverseManager()
        self.previous_top_10: List[str] = []
        self.alerts: List[Dict[str, Any]] = []
        
    def clear_screen(self):
        """Clear terminal screen."""
        os.system("clear" if os.name != "nt" else "cls")
    
    def format_money(self, value) -> str:
        """Format value as money."""
        try:
            v = float(value)
            if abs(v) >= 1_000_000:
                return f"${v/1_000_000:,.2f}M"
            elif abs(v) >= 1_000:
                return f"${v/1_000:,.2f}K"
            return f"${v:,.2f}"
        except:
            return str(value)
    
    def format_change(self, value) -> str:
        """Format change with color."""
        try:
            v = float(value)
            if v > 0:
                return f"{Colors.GREEN}+{v:.2f}%{Colors.END}"
            elif v < 0:
                return f"{Colors.RED}{v:.2f}%{Colors.END}"
            return f"{v:.2f}%"
        except:
            return str(value)
    
    def detect_ranking_changes(self, new_top_10: List[str]) -> List[str]:
        """Detect changes in top 10 ranking."""
        changes = []
        
        for symbol in new_top_10:
            if symbol not in self.previous_top_10:
                changes.append(f"{Colors.GREEN}+ {symbol} entered Top 10{Colors.END}")
        
        for symbol in self.previous_top_10:
            if symbol not in new_top_10:
                changes.append(f"{Colors.RED}- {symbol} dropped from Top 10{Colors.END}")
        
        return changes
    
    def print_header(self):
        """Print dashboard header."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}" + "="*100 + Colors.END)
        print(f"{Colors.BOLD}{Colors.CYAN}            TOP 10 WATCHLIST MONITOR - DYNAMIC UNIVERSE{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}                    GNOSIS Trading System{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}" + "="*100 + Colors.END)
        print(f"  {Colors.YELLOW}{now}{Colors.END}  |  Universe: 25 stocks  |  Watching: Top 10")
        print()
    
    def print_account_summary(self):
        """Print account summary."""
        account = get_alpaca_account()
        
        if account:
            equity = float(account.get("equity", 0))
            cash = float(account.get("cash", 0))
            buying_power = float(account.get("buying_power", 0))
            
            print(f"{Colors.BOLD}{Colors.BLUE}ACCOUNT STATUS{Colors.END}")
            print("-"*50)
            print(f"  Status: {Colors.GREEN}{account.get('status', 'N/A')}{Colors.END}")
            print(f"  Portfolio: {Colors.BOLD}{self.format_money(equity)}{Colors.END}")
            print(f"  Cash: {self.format_money(cash)}")
            print(f"  Buying Power: {self.format_money(buying_power)}")
            print()
    
    def print_top_10_watchlist(self):
        """Print the top 10 watchlist."""
        # Get fresh rankings
        rankings = self.universe_manager.rank_stocks()
        top_10 = self.universe_manager.get_top_10()
        new_top_10_symbols = [r.symbol for r in top_10]
        
        # Detect changes
        changes = self.detect_ranking_changes(new_top_10_symbols)
        if changes:
            print(f"{Colors.BOLD}{Colors.YELLOW}RANKING CHANGES{Colors.END}")
            for change in changes:
                print(f"  {change}")
            print()
        
        self.previous_top_10 = new_top_10_symbols
        
        # Get positions
        positions = get_alpaca_positions()
        pos_map = {p["symbol"]: p for p in positions}
        
        print(f"{Colors.BOLD}{Colors.BLUE}TOP 10 WATCHLIST{Colors.END}")
        print("-"*100)
        print(f"  {'Rank':<6} {'Symbol':<8} {'Company':<18} {'Price':>10} {'Score':>8} "
              f"{'Trend':<10} {'Position':>12} {'P&L':>12}")
        print("-"*100)
        
        for r in top_10:
            info = self.universe_manager.universe.get(r.symbol)
            price_data = self.universe_manager.price_cache.get(r.symbol, {})
            price = price_data.get("mid", 0)
            
            # Position info
            pos = pos_map.get(r.symbol)
            if pos:
                qty = int(float(pos.get("qty", 0)))
                pnl = float(pos.get("unrealized_pl", 0))
                pos_str = f"{qty} shares"
                pnl_str = f"{Colors.GREEN}+{self.format_money(pnl)}{Colors.END}" if pnl >= 0 else f"{Colors.RED}{self.format_money(pnl)}{Colors.END}"
            else:
                pos_str = "-"
                pnl_str = "-"
            
            # Trend color
            trend = r.signals.get("trend", "neutral")
            if trend == "bullish":
                trend_str = f"{Colors.GREEN}{trend}{Colors.END}"
            elif trend == "bearish":
                trend_str = f"{Colors.RED}{trend}{Colors.END}"
            else:
                trend_str = trend
            
            # Rank marker
            if r.rank <= 3:
                rank_color = Colors.GREEN + Colors.BOLD
            elif r.rank <= 7:
                rank_color = Colors.YELLOW
            else:
                rank_color = ""
            
            print(f"  {rank_color}#{r.rank:<5}{Colors.END} {r.symbol:<8} {info.name[:16]:<18} "
                  f"${price:>9.2f} {r.score:>7.1f} {trend_str:<10} {pos_str:>12} {pnl_str:>12}")
        
        print("-"*100)
        
        # Summary stats
        mag7_count = sum(1 for r in top_10 if self.universe_manager.universe[r.symbol].category == "magnificent_7")
        bullish_count = sum(1 for r in top_10 if r.signals.get("trend") == "bullish")
        
        print(f"\n  {Colors.CYAN}Summary:{Colors.END} Magnificent 7: {mag7_count}/7 in Top 10 | "
              f"Bullish: {bullish_count}/10 | Avg Score: {sum(r.score for r in top_10)/10:.1f}")
        print()
    
    def print_detailed_scores(self):
        """Print detailed score breakdown."""
        top_10 = self.universe_manager.get_top_10()
        
        print(f"{Colors.BOLD}{Colors.BLUE}SCORE BREAKDOWN{Colors.END}")
        print("-"*90)
        print(f"  {'Symbol':<8} {'Momentum':>10} {'Volume':>10} {'Volatility':>12} {'Flow':>10} {'Technical':>12}")
        print("-"*90)
        
        for r in top_10[:5]:  # Top 5 only for brevity
            mom_color = Colors.GREEN if r.momentum_score > 60 else (Colors.RED if r.momentum_score < 40 else "")
            vol_color = Colors.GREEN if r.volume_score > 70 else ""
            tech_color = Colors.GREEN if r.technical_score > 65 else ""
            
            print(f"  {r.symbol:<8} {mom_color}{r.momentum_score:>9.1f}{Colors.END} "
                  f"{vol_color}{r.volume_score:>9.1f}{Colors.END} {r.volatility_score:>11.1f} "
                  f"{r.flow_score:>9.1f} {tech_color}{r.technical_score:>11.1f}{Colors.END}")
        
        print()
    
    def print_options_flow(self):
        """Print options flow for top stocks."""
        top_10 = self.universe_manager.get_top_10()
        
        print(f"{Colors.BOLD}{Colors.BLUE}OPTIONS FLOW - TOP 3{Colors.END}")
        print("-"*80)
        
        for r in top_10[:3]:
            flow = get_unusual_whales_flow(r.symbol)
            
            if flow:
                print(f"\n  {Colors.BOLD}{r.symbol}{Colors.END}")
                for alert in flow[:2]:
                    opt_type = alert.get("put_call", "N/A")
                    strike = float(alert.get("strike", 0))
                    expiry = str(alert.get("expiry", "N/A"))[:10]
                    volume = int(alert.get("volume", 0))
                    
                    type_color = Colors.GREEN if opt_type == "CALL" else Colors.RED
                    print(f"    {type_color}{opt_type}{Colors.END} ${strike:.0f} exp {expiry} | Vol: {volume:,}")
            else:
                print(f"\n  {Colors.BOLD}{r.symbol}{Colors.END}: No flow data")
        
        print()
    
    def print_universe_overview(self):
        """Print overview of entire universe."""
        rankings = self.universe_manager.rank_stocks()
        
        print(f"{Colors.BOLD}{Colors.BLUE}UNIVERSE OVERVIEW (25 stocks){Colors.END}")
        print("-"*60)
        
        # Category breakdown
        categories = {}
        for r in rankings:
            cat = self.universe_manager.universe[r.symbol].category
            if cat not in categories:
                categories[cat] = {"count": 0, "in_top10": 0, "avg_score": 0}
            categories[cat]["count"] += 1
            categories[cat]["avg_score"] += r.score
            if r.rank <= 10:
                categories[cat]["in_top10"] += 1
        
        for cat, data in categories.items():
            data["avg_score"] /= data["count"]
        
        print(f"\n  {'Category':<20} {'Count':>8} {'In Top 10':>12} {'Avg Score':>12}")
        print("  " + "-"*52)
        
        for cat, data in sorted(categories.items(), key=lambda x: -x[1]["avg_score"]):
            print(f"  {cat:<20} {data['count']:>8} {data['in_top10']:>12} {data['avg_score']:>11.1f}")
        
        print()
    
    def print_footer(self):
        """Print dashboard footer."""
        symbols = ",".join(self.universe_manager.top_10)
        
        print(f"{Colors.CYAN}" + "="*100 + Colors.END)
        print(f"  Top 10 Symbols: {Colors.BOLD}{symbols}{Colors.END}")
        print(f"  Commands: {Colors.BOLD}Ctrl+C{Colors.END} to exit")
        print(f"  Start Trading: {Colors.BOLD}./start.sh local{Colors.END}")
        print(f"{Colors.CYAN}" + "="*100 + Colors.END)
    
    def update_env_file(self):
        """Update .env with current top 10."""
        symbols = ",".join(self.universe_manager.top_10)
        env_path = Path(__file__).parent.parent / ".env"
        
        try:
            with open(env_path, "r") as f:
                lines = f.readlines()
            
            # Update TRADING_SYMBOLS line
            new_lines = []
            found = False
            for line in lines:
                if line.startswith("TRADING_SYMBOLS="):
                    new_lines.append(f"TRADING_SYMBOLS={symbols}\n")
                    found = True
                else:
                    new_lines.append(line)
            
            if not found:
                new_lines.append(f"TRADING_SYMBOLS={symbols}\n")
            
            with open(env_path, "w") as f:
                f.writelines(new_lines)
            
            print(f"\n  {Colors.GREEN}Updated .env with Top 10 symbols{Colors.END}")
        except Exception as e:
            print(f"\n  {Colors.RED}Error updating .env: {e}{Colors.END}")
    
    def run(self, refresh_interval: int = 60, auto_update_env: bool = True):
        """Run the monitor with auto-refresh."""
        try:
            while True:
                self.clear_screen()
                self.print_header()
                self.print_account_summary()
                self.print_top_10_watchlist()
                self.print_detailed_scores()
                self.print_options_flow()
                self.print_universe_overview()
                self.print_footer()
                
                if auto_update_env:
                    self.update_env_file()
                
                # Wait for refresh
                for remaining in range(refresh_interval, 0, -1):
                    print(f"\r  Refreshing in {remaining}s... (Press Ctrl+C to exit)", end="", flush=True)
                    time.sleep(1)
        
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Monitor stopped.{Colors.END}\n")
    
    def run_once(self):
        """Run once without auto-refresh."""
        self.clear_screen()
        self.print_header()
        self.print_account_summary()
        self.print_top_10_watchlist()
        self.print_detailed_scores()
        self.print_options_flow()
        self.print_universe_overview()
        self.print_footer()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Top 10 Watchlist Monitor")
    parser.add_argument("--refresh", "-r", type=int, default=60,
                        help="Refresh interval in seconds (default: 60)")
    parser.add_argument("--once", "-1", action="store_true",
                        help="Run once without auto-refresh")
    parser.add_argument("--no-update", action="store_true",
                        help="Don't auto-update .env file")
    
    args = parser.parse_args()
    
    monitor = Top10Monitor()
    
    if args.once:
        monitor.run_once()
    else:
        monitor.run(
            refresh_interval=args.refresh,
            auto_update_env=not args.no_update,
        )


if __name__ == "__main__":
    main()
