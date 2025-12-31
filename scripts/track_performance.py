#!/usr/bin/env python3
"""
GNOSIS $1000 Account - Performance Tracker
Tracks daily performance and calculates key metrics
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from alpaca_trade_api import REST
    from dotenv import load_dotenv
except ImportError:
    print("Error: Required packages not installed")
    print("Run: pip install alpaca-trade-api python-dotenv")
    sys.exit(1)

# Load environment
env_file = Path(__file__).parent.parent / ".env.starter"
if env_file.exists():
    load_dotenv(env_file)
else:
    print(f"Warning: {env_file} not found, using default .env")
    load_dotenv()

# Initialize Alpaca
api = REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
)

# Data file
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
TRACKING_FILE = DATA_DIR / "performance_tracking.json"


def load_tracking_data():
    """Load historical tracking data"""
    if TRACKING_FILE.exists():
        with open(TRACKING_FILE, 'r') as f:
            return json.load(f)
    return {"daily": [], "trades": [], "metadata": {"start_date": str(datetime.now().date()), "initial_capital": 1000}}


def save_tracking_data(data):
    """Save tracking data"""
    with open(TRACKING_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def get_account_info():
    """Get current account information"""
    account = api.get_account()
    return {
        "equity": float(account.equity),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "portfolio_value": float(account.portfolio_value),
        "last_equity": float(account.last_equity),
    }


def get_positions():
    """Get current positions"""
    positions = api.list_positions()
    return [
        {
            "symbol": p.symbol,
            "qty": int(p.qty),
            "side": p.side,
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc),
        }
        for p in positions
    ]


def get_closed_trades_today():
    """Get trades closed today"""
    today = datetime.now().date()
    orders = api.list_orders(status='filled', limit=100)

    trades = []
    for order in orders:
        filled_at = datetime.fromisoformat(order.filled_at.replace('Z', '+00:00'))
        if filled_at.date() == today and order.side == 'sell':
            # Try to find corresponding buy
            buy_orders = [o for o in orders if o.symbol == order.symbol and o.side == 'buy' and o.filled_at < order.filled_at]
            if buy_orders:
                buy_order = buy_orders[-1]  # Most recent buy
                buy_price = float(buy_order.filled_avg_price)
                sell_price = float(order.filled_avg_price)
                qty = int(order.filled_qty)
                pnl = (sell_price - buy_price) * qty

                trades.append({
                    "symbol": order.symbol,
                    "qty": qty,
                    "entry_price": buy_price,
                    "exit_price": sell_price,
                    "pnl": pnl,
                    "pnl_pct": (sell_price / buy_price - 1) * 100,
                    "entry_time": buy_order.filled_at,
                    "exit_time": order.filled_at,
                    "hold_time_hours": (filled_at - datetime.fromisoformat(buy_order.filled_at.replace('Z', '+00:00'))).total_seconds() / 3600,
                })

    return trades


def calculate_metrics(data):
    """Calculate performance metrics"""
    if not data["daily"]:
        return {}

    # Get all equity values
    equities = [day["equity"] for day in data["daily"]]
    initial = data["metadata"]["initial_capital"]

    # Calculate returns
    current_equity = equities[-1]
    total_return = (current_equity / initial - 1) * 100

    # Calculate drawdown
    peak = initial
    max_drawdown = 0
    for equity in equities:
        if equity > peak:
            peak = equity
        drawdown = (equity / peak - 1) * 100
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    # Trade statistics
    trades = data["trades"]
    if trades:
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] < 0]

        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
        profit_factor = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) if losses and sum(t["pnl"] for t in losses) != 0 else 0

        expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * abs(avg_loss))
    else:
        win_rate = avg_win = avg_loss = profit_factor = expectancy = 0

    # Daily stats
    days_trading = len(data["daily"])
    if days_trading > 1:
        daily_returns = [(equities[i] / equities[i-1] - 1) * 100 for i in range(1, len(equities))]
        avg_daily_return = sum(daily_returns) / len(daily_returns)
        winning_days = len([r for r in daily_returns if r > 0])
        daily_win_rate = winning_days / len(daily_returns) * 100
    else:
        avg_daily_return = daily_win_rate = 0

    return {
        "total_return_pct": total_return,
        "total_return_dollars": current_equity - initial,
        "max_drawdown_pct": max_drawdown,
        "days_trading": days_trading,
        "total_trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_daily_return_pct": avg_daily_return,
        "daily_win_rate": daily_win_rate,
    }


def print_report():
    """Print performance report"""
    # Load data
    data = load_tracking_data()

    # Get current account info
    account = get_account_info()
    positions = get_positions()
    today_trades = get_closed_trades_today()

    # Update tracking data
    today = str(datetime.now().date())
    today_entry = {
        "date": today,
        "equity": account["equity"],
        "cash": account["cash"],
        "positions_count": len(positions),
        "daily_pnl": account["equity"] - account["last_equity"],
        "daily_pnl_pct": (account["equity"] / account["last_equity"] - 1) * 100 if account["last_equity"] > 0 else 0,
    }

    # Add or update today's entry
    if data["daily"] and data["daily"][-1]["date"] == today:
        data["daily"][-1] = today_entry
    else:
        data["daily"].append(today_entry)

    # Add today's trades
    for trade in today_trades:
        if trade not in data["trades"]:
            data["trades"].append(trade)

    # Save updated data
    save_tracking_data(data)

    # Calculate metrics
    metrics = calculate_metrics(data)

    # Print report
    print("=" * 70)
    print(f"GNOSIS $1000 ACCOUNT - PERFORMANCE REPORT")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\n📊 ACCOUNT STATUS")
    print("-" * 70)
    print(f"Current Equity:     ${account['equity']:>10,.2f}")
    print(f"Cash Available:     ${account['cash']:>10,.2f}")
    print(f"Buying Power:       ${account['buying_power']:>10,.2f}")

    print("\n📈 PERFORMANCE")
    print("-" * 70)
    initial = data["metadata"]["initial_capital"]
    print(f"Initial Capital:    ${initial:>10,.2f}")
    print(f"Current Capital:    ${account['equity']:>10,.2f}")
    if metrics:
        print(f"Total Return:       ${metrics['total_return_dollars']:>10,.2f}  ({metrics['total_return_pct']:>+6.2f}%)")
        print(f"Max Drawdown:       {metrics['max_drawdown_pct']:>16.2f}%")
        print(f"Days Trading:       {metrics['days_trading']:>17}")

        # Daily performance
        if account['last_equity'] > 0:
            daily_pnl = account['equity'] - account['last_equity']
            daily_pnl_pct = (account['equity'] / account['last_equity'] - 1) * 100
            print(f"\nToday's P&L:        ${daily_pnl:>10,.2f}  ({daily_pnl_pct:>+6.2f}%)")

    print("\n🎯 TRADE STATISTICS")
    print("-" * 70)
    if metrics and metrics["total_trades"] > 0:
        print(f"Total Trades:       {metrics['total_trades']:>17}")
        print(f"Win Rate:           {metrics['win_rate']:>16.1f}%")
        print(f"Average Win:        ${metrics['avg_win']:>10,.2f}")
        print(f"Average Loss:       ${abs(metrics['avg_loss']):>10,.2f}")
        print(f"Profit Factor:      {metrics['profit_factor']:>17.2f}")
        print(f"Expectancy:         ${metrics['expectancy']:>10,.2f}")
        print(f"Avg Daily Return:   {metrics['avg_daily_return_pct']:>16.2f}%")
        print(f"Daily Win Rate:     {metrics['daily_win_rate']:>16.1f}%")
    else:
        print("No trades yet")

    print("\n💼 CURRENT POSITIONS")
    print("-" * 70)
    if positions:
        for pos in positions:
            pnl_str = f"${pos['unrealized_pl']:>8,.2f} ({pos['unrealized_plpc']*100:>+6.2f}%)"
            print(f"{pos['symbol']:<6} {pos['qty']:>4} @ ${pos['avg_entry_price']:>7.2f}  →  ${pos['current_price']:>7.2f}  {pnl_str}")
    else:
        print("No open positions")

    print("\n📝 TODAY'S TRADES")
    print("-" * 70)
    if today_trades:
        for trade in today_trades:
            pnl_str = f"${trade['pnl']:>8,.2f} ({trade['pnl_pct']:>+6.2f}%)"
            hold_hours = trade['hold_time_hours']
            print(f"{trade['symbol']:<6} {trade['qty']:>4} @ ${trade['entry_price']:>7.2f}  →  ${trade['exit_price']:>7.2f}  {pnl_str}  [{hold_hours:.1f}h]")
    else:
        print("No trades today")

    # Performance targets
    print("\n🎯 MONTHLY TARGETS")
    print("-" * 70)
    days_in_month = 30
    current_return_pct = metrics.get('total_return_pct', 0)
    days_so_far = metrics.get('days_trading', 1)

    target_5pct = initial * 0.05  # 5% target
    target_10pct = initial * 0.10  # 10% stretch

    if days_so_far > 0:
        projected_monthly = (current_return_pct / days_so_far) * days_in_month
        print(f"5% Target:          ${target_5pct:>10,.2f}  ({5.0:>6.2f}%)")
        print(f"10% Target:         ${target_10pct:>10,.2f}  ({10.0:>6.2f}%)")
        print(f"Projected (30d):    ${initial * projected_monthly / 100:>10,.2f}  ({projected_monthly:>6.2f}%)")

    print("\n" + "=" * 70)

    # Warning checks
    warnings = []
    if metrics:
        if metrics['total_return_pct'] < -10:
            warnings.append("⚠️  Account down >10% - Review strategy")
        if metrics['max_drawdown_pct'] < -15:
            warnings.append("🚨 Max drawdown exceeded 15% - STOP TRADING")
        if metrics['win_rate'] < 45 and metrics['total_trades'] >= 10:
            warnings.append("⚠️  Win rate below 45% - Increase min confidence")
        if metrics['profit_factor'] < 1.0 and metrics['total_trades'] >= 5:
            warnings.append("⚠️  Profit factor < 1.0 - Losing money overall")

    if account['equity'] < 850:
        warnings.append("🚨 CIRCUIT BREAKER: Account below $850 - STOP TRADING")

    if warnings:
        print("\n⚠️  WARNINGS")
        print("-" * 70)
        for warning in warnings:
            print(warning)
        print("=" * 70)

    return data, metrics


def export_csv():
    """Export data to CSV"""
    data = load_tracking_data()

    # Export daily performance
    csv_file = DATA_DIR / "daily_performance.csv"
    with open(csv_file, 'w') as f:
        f.write("Date,Equity,Cash,Positions,Daily P&L,Daily P&L %\n")
        for day in data["daily"]:
            f.write(f"{day['date']},{day['equity']:.2f},{day['cash']:.2f},{day['positions_count']},{day['daily_pnl']:.2f},{day['daily_pnl_pct']:.2f}\n")

    print(f"✓ Exported daily performance to: {csv_file}")

    # Export trades
    if data["trades"]:
        trades_file = DATA_DIR / "trades.csv"
        with open(trades_file, 'w') as f:
            f.write("Symbol,Qty,Entry Price,Exit Price,P&L,P&L %,Entry Time,Exit Time,Hold Hours\n")
            for trade in data["trades"]:
                f.write(f"{trade['symbol']},{trade['qty']},{trade['entry_price']:.2f},{trade['exit_price']:.2f}," +
                       f"{trade['pnl']:.2f},{trade['pnl_pct']:.2f},{trade['entry_time']},{trade['exit_time']},{trade['hold_time_hours']:.2f}\n")

        print(f"✓ Exported trades to: {trades_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GNOSIS $1000 Account Performance Tracker")
    parser.add_argument("--export", action="store_true", help="Export data to CSV")
    args = parser.parse_args()

    try:
        data, metrics = print_report()

        if args.export:
            print()
            export_csv()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
