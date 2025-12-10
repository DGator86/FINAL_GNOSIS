#!/usr/bin/env python3
import json
import os
import urllib.error
import urllib.request
from datetime import datetime


def load_env_from_file(dotenv_path: str = ".env") -> None:
    """
    Minimal .env loader so the dashboard works even when python-dotenv isn't installed.

    Only lines containing KEY=VALUE pairs are loaded; comments and blank lines are ignored.
    """

    try:
        with open(dotenv_path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ[key] = value
    except FileNotFoundError:
        pass


# Load .env from repo root without relying on external packages
load_env_from_file(dotenv_path=".env")

BASE_URL = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")
KEY = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
SECRET = (
    os.getenv("ALPACA_API_SECRET")
    or os.getenv("ALPACA_SECRET_KEY")
    or os.getenv("APCA_API_SECRET_KEY")
)

if not KEY or not SECRET:
    print(
        "ERROR: Missing Alpaca API keys in .env (ALPACA_API_KEY/APCA_API_KEY_ID or ALPACA_API_SECRET/ALPACA_SECRET_KEY/APCA_API_SECRET_KEY)"
    )
    raise SystemExit(1)

HEADERS = {
    "APCA-API-KEY-ID": KEY,
    "APCA-API-SECRET-KEY": SECRET,
}

def fetch_account_and_positions():
    def get_json(path: str):
        req = urllib.request.Request(f"{BASE_URL}{path}", headers=HEADERS)
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        acct = get_json("/v2/account")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print("Error fetching account:", exc)
        raise SystemExit(1)

    try:
        positions = get_json("/v2/positions")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print("Error getting positions:", exc)
        positions = []

    return acct, positions

def format_money(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)

def main():
    os.system("clear")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"GNOSIS TERMINAL DASHBOARD  |  {now}")
    print("-" * 70)

    acct, positions = fetch_account_and_positions()

    # Account summary
    print("ACCOUNT")
    print(f"  Status      : {acct.get('status')}")
    print(f"  Equity      : {format_money(acct.get('equity', 0))}")
    print(f"  Cash        : {format_money(acct.get('cash', 0))}")
    print(f"  Buying Power: {format_money(acct.get('buying_power', 0))}")
    print(f"  Portfolio P&L (today): {format_money(acct.get('portfolio_value', 0))}")
    print("-" * 70)

    # Positions table
    if not positions:
        print("No open positions.")
        return

    headers = [
        "Symbol",
        "Side",
        "Qty",
        "Avg Price",
        "Current Price",
        "Unrealized P&L",
        "Unreal P&L %"
    ]
    print("{:<8} {:<6} {:>6} {:>12} {:>14} {:>16} {:>12}".format(*headers))
    print("-" * 70)

    for p in positions:
        symbol = p.get("symbol")
        side = p.get("side")
        qty = p.get("qty")
        avg_price = p.get("avg_entry_price")
        current_price = p.get("current_price", p.get("asset_current_price", ""))
        unreal = p.get("unrealized_pl")
        unreal_pct = p.get("unrealized_plpc")
        if unreal_pct is not None:
            try:
                unreal_pct_str = f"{float(unreal_pct)*100:,.2f}%"
            except Exception:
                unreal_pct_str = str(unreal_pct)
        else:
            unreal_pct_str = ""

        print(
            "{:<8} {:<6} {:>6} {:>12} {:>14} {:>16} {:>12}".format(
                symbol or "",
                side or "",
                qty or "",
                format_money(avg_price or 0),
                format_money(current_price or 0),
                format_money(unreal or 0),
                unreal_pct_str,
            )
        )

    print("-" * 70)
    print("Tip: In another pane run `tail -f logs/dynamic_trading_*.log` to see trade logic.")
    print("-" * 70)

if __name__ == "__main__":
    main()
