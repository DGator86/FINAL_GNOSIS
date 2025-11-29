import os
import sys
from dotenv import load_dotenv
from alpaca_trade_api import REST

# Add project root to path
sys.path.append(os.getcwd())


def show_positions():
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not secret_key:
        print("Error: ALPACA_API_KEY or ALPACA_SECRET_KEY not found in environment.")
        return

    try:
        api = REST(api_key, secret_key, base_url)
        account = api.get_account()
        positions = api.list_positions()

        print(f"\nAccount Status: {account.status}")
        print(f"Equity: ${float(account.equity):,.2f}")
        print(f"Buying Power: ${float(account.buying_power):,.2f}")
        print("-" * 80)

        if not positions:
            print("No open positions.")
        else:
            print(
                f"{'Symbol':<10} {'Qty':<10} {'Entry':<10} {'Current':<10} {'P/L $':<10} {'P/L %':<10}"
            )
            print("-" * 80)

            for p in positions:
                pl_dollar = float(p.unrealized_pl)
                pl_pct = float(p.unrealized_plpc) * 100

                print(
                    f"{p.symbol:<10} {p.qty:<10} {float(p.avg_entry_price):<10.2f} {float(p.current_price):<10.2f} {pl_dollar:<10.2f} {pl_pct:<10.2f}%"
                )

        print("-" * 80)

    except Exception as e:
        print(f"Error fetching positions: {e}")


if __name__ == "__main__":
    show_positions()
