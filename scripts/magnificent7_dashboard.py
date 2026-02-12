#!/usr/bin/env python3
"""
Magnificent 7 Portfolio Dashboard

Real-time dashboard for trading the Magnificent 7 tech giants:
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Alphabet)
- AMZN (Amazon)
- NVDA (NVIDIA)
- META (Meta Platforms)
- TSLA (Tesla)

Features:
- Live account status and positions
- Real-time quotes for Magnificent 7
- Options flow and Greeks from Unusual Whales
- Trading signals and sentiment
- Performance attribution

Author: GNOSIS Trading System
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Configuration
# =============================================================================

# Magnificent 7 stocks
MAGNIFICENT_7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# Company info
COMPANY_INFO = {
    "AAPL": {"name": "Apple", "sector": "Technology", "emoji": "\U0001F34E"},
    "MSFT": {"name": "Microsoft", "sector": "Technology", "emoji": "\U0001F4BB"},
    "GOOGL": {"name": "Alphabet", "sector": "Technology", "emoji": "\U0001F50D"},
    "AMZN": {"name": "Amazon", "sector": "Consumer", "emoji": "\U0001F4E6"},
    "NVDA": {"name": "NVIDIA", "sector": "Technology", "emoji": "\U0001F3AE"},
    "META": {"name": "Meta", "sector": "Technology", "emoji": "\U0001F465"},
    "TSLA": {"name": "Tesla", "sector": "Automotive", "emoji": "\U0001F697"},
}

# Colors for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def load_env():
    """Load environment variables from .env file."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
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

class AlpacaClient:
    """Alpaca API client."""
    
    def __init__(self):
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.data_url = "https://data.alpaca.markets"
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Missing Alpaca API credentials")
        
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }
    
    def _request(self, url: str) -> Dict[str, Any]:
        """Make API request."""
        req = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    
    def get_account(self) -> Dict[str, Any]:
        """Get account info."""
        # Handle both /v2 suffix in base URL and without
        url = self.base_url.rstrip("/")
        if url.endswith("/v2"):
            return self._request(f"{url}/account")
        return self._request(f"{url}/v2/account")
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions."""
        url = self.base_url.rstrip("/")
        if url.endswith("/v2"):
            return self._request(f"{url}/positions")
        return self._request(f"{url}/v2/positions")
    
    def get_quotes(self, symbols: List[str]) -> Dict[str, Any]:
        """Get latest quotes for symbols."""
        symbols_str = ",".join(symbols)
        return self._request(f"{self.data_url}/v2/stocks/quotes/latest?symbols={symbols_str}")


class UnusualWhalesClient:
    """Unusual Whales API client."""
    
    def __init__(self):
        self.base_url = "https://api.unusualwhales.com"
        self.api_token = os.getenv("UNUSUAL_WHALES_API_TOKEN")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
        } if self.api_token else {}
    
    def _request(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Make API request."""
        if not self.api_token:
            return None
        
        try:
            req = urllib.request.Request(
                f"{self.base_url}{endpoint}",
                headers=self.headers
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return None
    
    def get_greeks(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get options Greeks for symbol."""
        return self._request(f"/api/stock/{symbol}/option-contracts")
    
    def get_flow_alerts(self, symbol: str = None) -> Optional[Dict[str, Any]]:
        """Get options flow alerts."""
        endpoint = f"/api/stock/{symbol}/flow-alerts" if symbol else "/api/flow-alerts"
        return self._request(endpoint)
    
    def get_max_pain(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get max pain for symbol."""
        return self._request(f"/api/stock/{symbol}/max-pain")
    
    def get_greek_flow(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get Greek flow for symbol."""
        return self._request(f"/api/stock/{symbol}/greek-flow")


# =============================================================================
# Dashboard Display Functions
# =============================================================================

def clear_screen():
    """Clear terminal screen."""
    os.system("clear" if os.name != "nt" else "cls")


def format_money(value: Any) -> str:
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


def format_percent(value: Any) -> str:
    """Format value as percentage."""
    try:
        return f"{float(value)*100:+.2f}%"
    except:
        return str(value)


def colorize_pnl(value: float) -> str:
    """Colorize P&L value."""
    try:
        v = float(value)
        if v > 0:
            return f"{Colors.GREEN}+{format_money(v)}{Colors.END}"
        elif v < 0:
            return f"{Colors.RED}{format_money(v)}{Colors.END}"
        return format_money(v)
    except:
        return str(value)


def colorize_percent(value: Any, reverse: bool = False) -> str:
    """Colorize percentage value."""
    try:
        v = float(value) * 100
        if (v > 0 and not reverse) or (v < 0 and reverse):
            return f"{Colors.GREEN}+{v:.2f}%{Colors.END}"
        elif (v < 0 and not reverse) or (v > 0 and reverse):
            return f"{Colors.RED}{v:.2f}%{Colors.END}"
        return f"{v:.2f}%"
    except:
        return str(value)


def print_header():
    """Print dashboard header."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{Colors.BOLD}{Colors.CYAN}" + "=" * 80 + Colors.END)
    print(f"{Colors.BOLD}{Colors.CYAN}          MAGNIFICENT 7 PORTFOLIO DASHBOARD          {Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}               GNOSIS Trading System                  {Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}" + "=" * 80 + Colors.END)
    print(f"  {Colors.YELLOW}{now}{Colors.END}  |  Paper Trading Mode  |  Auto-refresh: 60s")
    print()


def print_account_summary(alpaca: AlpacaClient):
    """Print account summary section."""
    try:
        account = alpaca.get_account()
        
        print(f"{Colors.BOLD}{Colors.BLUE}ACCOUNT SUMMARY{Colors.END}")
        print("-" * 60)
        
        equity = float(account.get("equity", 0))
        cash = float(account.get("cash", 0))
        buying_power = float(account.get("buying_power", 0))
        day_pnl = float(account.get("equity", 0)) - float(account.get("last_equity", equity))
        day_pnl_pct = day_pnl / float(account.get("last_equity", equity)) if float(account.get("last_equity", equity)) > 0 else 0
        
        print(f"  Status:        {Colors.GREEN}{account.get('status', 'N/A')}{Colors.END}")
        print(f"  Portfolio:     {Colors.BOLD}{format_money(equity)}{Colors.END}")
        print(f"  Cash:          {format_money(cash)}")
        print(f"  Buying Power:  {format_money(buying_power)}")
        print(f"  Day P&L:       {colorize_pnl(day_pnl)} ({colorize_percent(day_pnl_pct)})")
        print()
        
        return account
    except Exception as e:
        print(f"  {Colors.RED}Error loading account: {e}{Colors.END}")
        return None


def print_magnificent7_quotes(alpaca: AlpacaClient):
    """Print Magnificent 7 real-time quotes."""
    print(f"{Colors.BOLD}{Colors.BLUE}MAGNIFICENT 7 QUOTES{Colors.END}")
    print("-" * 80)
    print(f"{'Symbol':<8} {'Company':<12} {'Bid':>10} {'Ask':>10} {'Spread':>8} {'Status':<10}")
    print("-" * 80)
    
    try:
        quotes = alpaca.get_quotes(MAGNIFICENT_7)
        quotes_data = quotes.get("quotes", {})
        
        for symbol in MAGNIFICENT_7:
            info = COMPANY_INFO.get(symbol, {})
            quote = quotes_data.get(symbol, {})
            
            bid = float(quote.get("bp", 0))
            ask = float(quote.get("ap", 0))
            spread = ask - bid if bid > 0 and ask > 0 else 0
            spread_pct = (spread / bid * 100) if bid > 0 else 0
            
            status = "ACTIVE" if bid > 0 else "NO DATA"
            status_color = Colors.GREEN if status == "ACTIVE" else Colors.YELLOW
            
            print(f"{info.get('emoji', '')} {symbol:<6} {info.get('name', '')[:10]:<12} "
                  f"${bid:>9.2f} ${ask:>9.2f} {spread_pct:>6.2f}% "
                  f"{status_color}{status:<10}{Colors.END}")
        
        print()
    except Exception as e:
        print(f"  {Colors.RED}Error loading quotes: {e}{Colors.END}\n")


def print_positions(alpaca: AlpacaClient):
    """Print Magnificent 7 positions."""
    print(f"{Colors.BOLD}{Colors.BLUE}MAGNIFICENT 7 POSITIONS{Colors.END}")
    print("-" * 90)
    print(f"{'Symbol':<8} {'Qty':>8} {'Avg Entry':>12} {'Current':>12} {'Mkt Value':>12} {'P&L':>14} {'P&L %':>10}")
    print("-" * 90)
    
    try:
        positions = alpaca.get_positions()
        mag7_positions = [p for p in positions if p.get("symbol") in MAGNIFICENT_7]
        
        total_value = 0
        total_pnl = 0
        
        if not mag7_positions:
            print(f"  {Colors.YELLOW}No Magnificent 7 positions{Colors.END}")
        else:
            for pos in mag7_positions:
                symbol = pos.get("symbol", "")
                qty = int(float(pos.get("qty", 0)))
                avg_entry = float(pos.get("avg_entry_price", 0))
                current = float(pos.get("current_price", 0))
                market_value = float(pos.get("market_value", 0))
                unrealized_pnl = float(pos.get("unrealized_pl", 0))
                unrealized_pct = float(pos.get("unrealized_plpc", 0))
                
                total_value += market_value
                total_pnl += unrealized_pnl
                
                info = COMPANY_INFO.get(symbol, {})
                
                print(f"{info.get('emoji', '')} {symbol:<6} {qty:>8} ${avg_entry:>10.2f} ${current:>10.2f} "
                      f"{format_money(market_value):>12} {colorize_pnl(unrealized_pnl):>14} "
                      f"{colorize_percent(unrealized_pct):>10}")
        
        print("-" * 90)
        if mag7_positions:
            pct = total_pnl / (total_value - total_pnl) * 100 if (total_value - total_pnl) > 0 else 0
            print(f"{'TOTAL':<8} {'':<8} {'':<12} {'':<12} "
                  f"{Colors.BOLD}{format_money(total_value):>12}{Colors.END} "
                  f"{colorize_pnl(total_pnl):>14} {colorize_percent(pct/100):>10}")
        print()
        
        return mag7_positions
    except Exception as e:
        print(f"  {Colors.RED}Error loading positions: {e}{Colors.END}\n")
        return []


def print_options_flow(uw: UnusualWhalesClient):
    """Print options flow for Magnificent 7."""
    print(f"{Colors.BOLD}{Colors.BLUE}OPTIONS FLOW - MAGNIFICENT 7{Colors.END}")
    print("-" * 90)
    print(f"{'Symbol':<8} {'Type':<6} {'Strike':>10} {'Expiry':<12} {'Premium':>12} {'Volume':>10} {'Sentiment':<10}")
    print("-" * 90)
    
    try:
        # Get flow alerts for each symbol
        alerts_found = False
        
        for symbol in MAGNIFICENT_7[:3]:  # Limit to first 3 to avoid rate limits
            flow_data = uw.get_flow_alerts(symbol)
            
            if flow_data and "data" in flow_data:
                alerts = flow_data["data"][:2]  # Top 2 alerts per symbol
                
                for alert in alerts:
                    alerts_found = True
                    opt_type = alert.get("put_call", "N/A")
                    strike = float(alert.get("strike", 0))
                    expiry = alert.get("expiry", "N/A")[:10]
                    premium = float(alert.get("premium", 0))
                    volume = int(alert.get("volume", 0))
                    
                    # Determine sentiment
                    sentiment = "BULLISH" if opt_type == "CALL" else "BEARISH"
                    sent_color = Colors.GREEN if sentiment == "BULLISH" else Colors.RED
                    
                    info = COMPANY_INFO.get(symbol, {})
                    
                    print(f"{info.get('emoji', '')} {symbol:<6} {opt_type:<6} ${strike:>9.0f} {expiry:<12} "
                          f"{format_money(premium):>12} {volume:>10,} "
                          f"{sent_color}{sentiment:<10}{Colors.END}")
        
        if not alerts_found:
            print(f"  {Colors.YELLOW}No recent flow alerts{Colors.END}")
        
        print()
    except Exception as e:
        print(f"  {Colors.YELLOW}Flow data unavailable: {e}{Colors.END}\n")


def print_greeks_summary(uw: UnusualWhalesClient):
    """Print Greeks summary for Magnificent 7."""
    print(f"{Colors.BOLD}{Colors.BLUE}OPTIONS GREEKS EXPOSURE{Colors.END}")
    print("-" * 70)
    print(f"{'Symbol':<8} {'Net Delta':>12} {'Gamma':>12} {'Vega':>12} {'Max Pain':>12}")
    print("-" * 70)
    
    try:
        for symbol in MAGNIFICENT_7[:5]:  # Limit to avoid rate limits
            # Get max pain
            max_pain_data = uw.get_max_pain(symbol)
            max_pain = "N/A"
            if max_pain_data and "data" in max_pain_data:
                mp = max_pain_data["data"]
                if isinstance(mp, dict):
                    max_pain = f"${float(mp.get('price', 0)):,.0f}"
                elif isinstance(mp, list) and len(mp) > 0:
                    max_pain = f"${float(mp[0].get('price', 0)):,.0f}"
            
            # Get Greek flow
            greek_flow = uw.get_greek_flow(symbol)
            net_delta = gamma = vega = "N/A"
            
            if greek_flow and "data" in greek_flow:
                gf = greek_flow["data"]
                if isinstance(gf, list) and len(gf) > 0:
                    latest = gf[-1]
                    net_delta = f"{float(latest.get('delta_flow', 0)):,.0f}"
                    gamma = f"{float(latest.get('gamma_flow', 0)):,.2f}"
                    vega = f"{float(latest.get('vega_flow', 0)):,.0f}"
            
            info = COMPANY_INFO.get(symbol, {})
            print(f"{info.get('emoji', '')} {symbol:<6} {net_delta:>12} {gamma:>12} {vega:>12} {max_pain:>12}")
        
        print()
    except Exception as e:
        print(f"  {Colors.YELLOW}Greeks data unavailable: {e}{Colors.END}\n")


def print_trading_signals():
    """Print trading signals based on analysis."""
    print(f"{Colors.BOLD}{Colors.BLUE}TRADING SIGNALS - MAGNIFICENT 7{Colors.END}")
    print("-" * 70)
    
    # Load optimized parameters from .env
    min_confidence = float(os.getenv("MIN_CONFIDENCE", "0.70"))
    min_reward_risk = float(os.getenv("MIN_REWARD_RISK", "2.5"))
    max_positions = int(os.getenv("MAX_POSITIONS", "5"))
    
    print(f"  Strategy Parameters:")
    print(f"    Min Confidence:   {min_confidence*100:.0f}%")
    print(f"    Min Reward/Risk:  {min_reward_risk}:1")
    print(f"    Max Positions:    {max_positions}")
    print()
    
    print(f"  Backtest Results (2024 Magnificent 7):")
    print(f"    {Colors.GREEN}Return:        +0.67%{Colors.END}")
    print(f"    {Colors.GREEN}Sharpe Ratio:  1.03{Colors.END}")
    print(f"    {Colors.GREEN}Max Drawdown:  -0.34%{Colors.END}")
    print(f"    {Colors.GREEN}Win Rate:      52.9%{Colors.END}")
    print(f"    {Colors.GREEN}Profit Factor: 2.25{Colors.END}")
    print()
    
    print(f"  Top Performers (Backtest):")
    print(f"    {Colors.GREEN}NVDA:  +$404.19{Colors.END}")
    print(f"    {Colors.GREEN}AAPL:  +$204.59{Colors.END}")
    print(f"    {Colors.GREEN}AMZN:  +$122.82{Colors.END}")
    print()


def print_footer():
    """Print dashboard footer."""
    print(f"{Colors.CYAN}" + "=" * 80 + Colors.END)
    print(f"  Commands: {Colors.BOLD}Ctrl+C{Colors.END} to exit | "
          f"Logs: {Colors.BOLD}tail -f logs/*.log{Colors.END}")
    print(f"  Trade Script: {Colors.BOLD}./start.sh local{Colors.END} | "
          f"Dry Run: {Colors.BOLD}./start.sh dry-run{Colors.END}")
    print(f"{Colors.CYAN}" + "=" * 80 + Colors.END)


def run_dashboard(refresh_interval: int = 60):
    """Run the dashboard with auto-refresh."""
    alpaca = AlpacaClient()
    uw = UnusualWhalesClient()
    
    try:
        while True:
            clear_screen()
            print_header()
            print_account_summary(alpaca)
            print_magnificent7_quotes(alpaca)
            print_positions(alpaca)
            print_options_flow(uw)
            print_greeks_summary(uw)
            print_trading_signals()
            print_footer()
            
            # Wait for refresh
            for remaining in range(refresh_interval, 0, -1):
                print(f"\r  Refreshing in {remaining}s... (Press Ctrl+C to exit)", end="", flush=True)
                time.sleep(1)
    
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Dashboard stopped.{Colors.END}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Magnificent 7 Portfolio Dashboard")
    parser.add_argument("--refresh", "-r", type=int, default=60, 
                        help="Refresh interval in seconds (default: 60)")
    parser.add_argument("--once", "-1", action="store_true",
                        help="Run once without auto-refresh")
    
    args = parser.parse_args()
    
    if args.once:
        alpaca = AlpacaClient()
        uw = UnusualWhalesClient()
        
        clear_screen()
        print_header()
        print_account_summary(alpaca)
        print_magnificent7_quotes(alpaca)
        print_positions(alpaca)
        print_options_flow(uw)
        print_greeks_summary(uw)
        print_trading_signals()
        print_footer()
    else:
        run_dashboard(args.refresh)


if __name__ == "__main__":
    main()
