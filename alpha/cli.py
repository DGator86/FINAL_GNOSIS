#!/usr/bin/env python3
"""
Gnosis Alpha - Command Line Interface

Simple CLI for generating and viewing trading signals.
Supports both stock and options trading.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha.alpha_config import AlphaConfig
from alpha.signal_generator import AlphaSignalGenerator, SignalDirection
from alpha.pdt_tracker import PDTTracker
from alpha.options_signal import (
    OptionsSignalGenerator,
    OptionStrategy,
    OptionSignalDirection,
)
from alpha.zero_dte import (
    ZeroDTEGenerator,
    ZeroDTESignal,
    ZeroDTEStrategy,
    ZeroDTERisk,
    print_0dte_disclaimer,
)


def cmd_scan(args):
    """Scan universe and show signals."""
    config = AlphaConfig.from_env()
    
    if args.symbols:
        config.universe = [s.strip().upper() for s in args.symbols.split(",")]
    
    generator = AlphaSignalGenerator(config=config)
    
    # Set minimum confidence
    min_conf = args.min_confidence or config.min_confidence
    
    # Scan for signals
    signals = generator.scan_universe(min_confidence=min_conf)
    
    # Filter by direction if specified
    if args.direction:
        direction = SignalDirection(args.direction.upper())
        signals = [s for s in signals if s.direction == direction]
    
    # Print signals
    generator.print_signals(signals)
    
    # Save if requested
    if args.output:
        generator.save_signals(signals, args.output)
        print(f"\nSaved to: {args.output}")


def cmd_signal(args):
    """Generate signal for a specific symbol."""
    config = AlphaConfig.from_env()
    generator = AlphaSignalGenerator(config=config)
    
    symbol = args.symbol.upper()
    signal = generator.generate_signal(symbol)
    
    if args.json:
        print(json.dumps(signal.to_dict(), indent=2))
    else:
        print(signal.to_robinhood_format())
        print()
        print(f"Reasoning: {signal.reasoning}")
        if signal.risk_factors:
            print(f"Risk Factors: {', '.join(signal.risk_factors)}")


def cmd_status(args):
    """Show account and PDT status."""
    from alpha.alpha_trader import AlphaTrader
    
    config = AlphaConfig.from_env()
    trader = AlphaTrader(config=config, paper=True)
    
    status = trader.get_status()
    
    print("\n" + "="*50)
    print("  GNOSIS ALPHA - Account Status")
    print("="*50 + "\n")
    
    # Account info
    account = status.get("account", {})
    if "error" in account:
        print(f"Account Error: {account['error']}")
    else:
        print(f"Account: {account.get('account_number', 'N/A')}")
        print(f"Status: {account.get('status', 'N/A')}")
        print(f"Equity: ${account.get('equity', 0):,.2f}")
        print(f"Cash: ${account.get('cash', 0):,.2f}")
        print(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
    
    print()
    
    # PDT status
    pdt = status.get("pdt_status", {})
    print("PDT Status:")
    print(f"  Restricted: {pdt.get('is_pdt_restricted', 'N/A')}")
    print(f"  Day Trades Used: {pdt.get('day_trades_used', 0)}/{pdt.get('max_day_trades', 3)}")
    print(f"  Remaining: {pdt.get('day_trades_remaining', 'N/A')}")
    
    print()
    
    # Positions
    positions = status.get("positions", [])
    if positions:
        print(f"Open Positions ({len(positions)}):")
        for pos in positions:
            pnl = pos.get("unrealized_pnl", 0)
            pnl_pct = pos.get("unrealized_pnl_pct", 0) * 100
            emoji = "🟢" if pnl >= 0 else "🔴"
            print(f"  {emoji} {pos['symbol']}: {pos['quantity']} shares @ ${pos['entry_price']:.2f}")
            print(f"     P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%)")
    else:
        print("No open positions")
    
    print()
    
    # Open orders
    orders = status.get("open_orders", [])
    if orders:
        print(f"Open Orders ({len(orders)}):")
        for order in orders:
            print(f"  {order['side'].upper()} {order['quantity']} {order['symbol']} ({order['status']})")
    else:
        print("No open orders")
    
    print()
    print("="*50)


def cmd_trade(args):
    """Execute a trade based on signal."""
    from alpha.alpha_trader import AlphaTrader
    
    config = AlphaConfig.from_env()
    generator = AlphaSignalGenerator(config=config)
    trader = AlphaTrader(config=config, paper=True)
    
    symbol = args.symbol.upper()
    
    # Generate signal
    signal = generator.generate_signal(symbol)
    
    print(f"\nSignal for {symbol}:")
    print(signal.to_robinhood_format())
    print()
    
    if signal.direction == SignalDirection.HOLD:
        print("Signal is HOLD - no trade executed")
        return
    
    if not args.confirm:
        response = input(f"Execute {signal.direction.value} order? (y/N): ")
        if response.lower() != 'y':
            print("Trade cancelled")
            return
    
    # Execute trade
    order = trader.execute_signal(signal, force=args.force)
    
    if order:
        print(f"\nOrder executed:")
        print(f"  ID: {order.order_id}")
        print(f"  Side: {order.side}")
        print(f"  Quantity: {order.quantity}")
        print(f"  Status: {order.status}")
    else:
        print("\nOrder failed - check logs for details")


def cmd_close(args):
    """Close a position."""
    from alpha.alpha_trader import AlphaTrader
    
    config = AlphaConfig.from_env()
    trader = AlphaTrader(config=config, paper=True)
    
    if args.all:
        orders = trader.close_all_positions()
        print(f"Closed {len(orders)} positions")
    else:
        symbol = args.symbol.upper()
        order = trader.close_position(symbol)
        if order:
            print(f"Position closed: {symbol}")
            print(f"Order ID: {order.order_id}")
        else:
            print(f"Failed to close {symbol}")


# ============================================================
# OPTIONS COMMANDS
# ============================================================

def cmd_options_scan(args):
    """Scan for options opportunities."""
    config = AlphaConfig.from_env()
    
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = config.universe[:10]  # Default to top 10
    
    generator = OptionsSignalGenerator(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
        min_confidence=args.min_confidence or 0.6,
    )
    
    # Filter strategies
    strategies = None
    if args.strategy:
        strategy_map = {
            "call": OptionStrategy.LONG_CALL,
            "put": OptionStrategy.LONG_PUT,
            "covered": OptionStrategy.COVERED_CALL,
            "csp": OptionStrategy.CASH_SECURED_PUT,
        }
        if args.strategy.lower() in strategy_map:
            strategies = [strategy_map[args.strategy.lower()]]
    
    print(f"\n📊 Scanning {len(symbols)} symbols for options opportunities...\n")
    
    signals = generator.scan_for_options(symbols, strategies=strategies)
    
    if not signals:
        print("No options signals found matching criteria.")
        return
    
    # Print signals
    print("="*60)
    print("  GNOSIS ALPHA - Options Signals")
    print("="*60 + "\n")
    
    # Group by direction
    bullish = [s for s in signals if s.direction == OptionSignalDirection.BULLISH]
    bearish = [s for s in signals if s.direction == OptionSignalDirection.BEARISH]
    
    if bullish:
        print("🟢 BULLISH OPTIONS:")
        print("-"*40)
        for signal in bullish[:5]:
            print(signal.to_robinhood_format())
            print()
    
    if bearish:
        print("🔴 BEARISH OPTIONS:")
        print("-"*40)
        for signal in bearish[:5]:
            print(signal.to_robinhood_format())
            print()
    
    print("="*60)
    print(f"Found {len(signals)} options signals")


def cmd_options_signal(args):
    """Get options signal for a specific symbol."""
    config = AlphaConfig.from_env()
    
    symbol = args.symbol.upper()
    
    # Parse strategy
    strategy = None
    if args.strategy:
        strategy_map = {
            "call": OptionStrategy.LONG_CALL,
            "put": OptionStrategy.LONG_PUT,
            "covered": OptionStrategy.COVERED_CALL,
            "csp": OptionStrategy.CASH_SECURED_PUT,
        }
        strategy = strategy_map.get(args.strategy.lower())
    
    generator = OptionsSignalGenerator(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
    )
    
    signal = generator.generate_signal(symbol, strategy=strategy)
    
    if not signal:
        print(f"No options signal generated for {symbol}")
        return
    
    if args.json:
        print(json.dumps(signal.to_dict(), indent=2))
    else:
        print()
        print(signal.to_robinhood_format())
        print()
        print(f"Reasoning: {signal.reasoning}")
        if signal.risk_factors:
            print(f"⚠️ Risks: {', '.join(signal.risk_factors)}")


def cmd_options_chain(args):
    """Show options chain for a symbol."""
    config = AlphaConfig.from_env()
    
    symbol = args.symbol.upper()
    
    from alpha.options_signal import OptionsChainFetcher
    
    fetcher = OptionsChainFetcher(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
    )
    
    contracts = fetcher.get_chain(
        symbol,
        min_dte=args.min_dte or 7,
        max_dte=args.max_dte or 45,
    )
    
    if not contracts:
        print(f"No options chain available for {symbol}")
        return
    
    print(f"\n📊 Options Chain for {symbol}")
    print("="*70)
    
    # Group by expiration
    by_exp = {}
    for c in contracts:
        key = c.expiration.isoformat()
        if key not in by_exp:
            by_exp[key] = {"calls": [], "puts": []}
        if c.option_type.value == "call":
            by_exp[key]["calls"].append(c)
        else:
            by_exp[key]["puts"].append(c)
    
    for exp, chains in sorted(by_exp.items()):
        print(f"\n📅 Expiration: {exp}")
        print("-"*70)
        
        # Header
        print(f"{'CALLS':<30} {'Strike':^10} {'PUTS':>30}")
        print(f"{'Bid':>8} {'Ask':>8} {'Vol':>8} {'':^10} {'Bid':>8} {'Ask':>8} {'Vol':>8}")
        print("-"*70)
        
        # Get unique strikes
        all_strikes = sorted(set([c.strike for c in chains["calls"]] + [c.strike for c in chains["puts"]]))
        
        for strike in all_strikes:
            call = next((c for c in chains["calls"] if c.strike == strike), None)
            put = next((c for c in chains["puts"] if c.strike == strike), None)
            
            call_str = f"{call.bid or 0:>8.2f} {call.ask or 0:>8.2f} {call.volume:>8}" if call else " "*24
            put_str = f"{put.bid or 0:>8.2f} {put.ask or 0:>8.2f} {put.volume:>8}" if put else " "*24
            
            print(f"{call_str} {strike:^10.2f} {put_str}")
    
    print()


def cmd_options_trade(args):
    """Execute an options trade."""
    from alpha.options_trader import OptionsTrader
    
    config = AlphaConfig.from_env()
    
    symbol = args.symbol.upper()
    
    # Parse strategy
    strategy_map = {
        "call": OptionStrategy.LONG_CALL,
        "put": OptionStrategy.LONG_PUT,
    }
    strategy = strategy_map.get(args.strategy.lower(), OptionStrategy.LONG_CALL)
    
    # Generate signal
    generator = OptionsSignalGenerator(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
    )
    
    signal = generator.generate_signal(symbol, strategy=strategy)
    
    if not signal:
        print(f"No options signal for {symbol}")
        return
    
    print("\n📊 Options Signal:")
    print(signal.to_robinhood_format())
    print()
    
    if not args.confirm:
        response = input("Execute this trade? (y/N): ")
        if response.lower() != 'y':
            print("Trade cancelled")
            return
    
    # Execute
    trader = OptionsTrader(config=config, paper=True)
    
    contracts = args.contracts or signal.suggested_contracts
    order = trader.execute_signal(signal, contracts=contracts)
    
    if order:
        print(f"\n✅ Order Executed:")
        print(f"  Order ID: {order.order_id}")
        print(f"  Contract: {order.contract_symbol}")
        print(f"  Side: {order.side}")
        print(f"  Quantity: {order.quantity}")
        print(f"  Status: {order.status}")
    else:
        print("\n❌ Order failed")


def cmd_options_positions(args):
    """Show current options positions."""
    from alpha.options_trader import OptionsTrader
    
    config = AlphaConfig.from_env()
    trader = OptionsTrader(config=config, paper=True)
    
    positions = trader.get_options_positions()
    
    print("\n" + "="*60)
    print("  GNOSIS ALPHA - Options Positions")
    print("="*60 + "\n")
    
    if not positions:
        print("No open options positions")
    else:
        for pos in positions:
            pnl_emoji = "🟢" if pos.unrealized_pnl >= 0 else "🔴"
            print(f"{pnl_emoji} {pos.symbol}")
            print(f"   Contract: {pos.contract_symbol}")
            print(f"   Quantity: {pos.quantity} ({pos.side})")
            print(f"   Entry: ${pos.avg_entry_price:.2f}")
            if pos.current_price:
                print(f"   Current: ${pos.current_price:.2f}")
            print(f"   P&L: ${pos.unrealized_pnl:,.2f}")
            print()
    
    # Show buying power
    buying_power = trader.get_account_buying_power()
    print(f"💰 Buying Power: ${buying_power:,.2f}")
    print("="*60)


# ============================================================
# 0DTE OPTIONS COMMANDS (HIGH RISK)
# ============================================================

def cmd_0dte_scan(args):
    """Scan for 0DTE opportunities."""
    config = AlphaConfig.from_env()
    
    # Show disclaimer first
    if not args.no_disclaimer:
        print_0dte_disclaimer()
        if not args.confirm:
            response = input("\nI understand the risks. Continue? (y/N): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return
    
    # Default 0DTE symbols (daily expirations)
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]
    
    generator = ZeroDTEGenerator(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
        max_position_dollars=args.max_dollars or 200,
    )
    
    print(f"\n⚡ Scanning {len(symbols)} symbols for 0DTE opportunities...\n")
    
    signals = generator.scan_0dte(symbols=symbols, max_dollars=args.max_dollars or 200)
    
    if not signals:
        print("No 0DTE signals found.")
        print("\nPossible reasons:")
        print("  • Market is closed")
        print("  • Less than 15 minutes to market close")
        print("  • No 0DTE expirations available today")
        return
    
    # Print signals
    print("="*70)
    print("  ⚡ GNOSIS ALPHA - 0DTE OPTIONS (HIGH RISK) ⚡")
    print("="*70 + "\n")
    
    for signal in signals:
        print(signal.to_robinhood_format())
        print()
        print("-"*70)
        print()
    
    print(f"\nFound {len(signals)} 0DTE opportunities")
    print("⚠️ Remember: 0DTE = High risk of 100% loss")


def cmd_0dte_signal(args):
    """Get 0DTE signal for a specific symbol."""
    config = AlphaConfig.from_env()
    
    symbol = args.symbol.upper()
    
    # Show disclaimer first
    if not args.no_disclaimer:
        print_0dte_disclaimer()
    
    generator = ZeroDTEGenerator(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
        max_position_dollars=args.max_dollars or 200,
    )
    
    # Check if 0DTE is available
    if not generator.is_0dte_available(symbol):
        print(f"\n❌ 0DTE not available for {symbol}")
        print("\n0DTE is available for:")
        print("  • SPY, QQQ, IWM (daily expirations)")
        print("  • AAPL, TSLA, NVDA, META, GOOGL, AMZN, MSFT")
        return
    
    # Parse strategy if provided
    strategy = None
    if args.strategy:
        strategy_map = {
            "scalp_call": ZeroDTEStrategy.SCALP_CALL,
            "scalp_put": ZeroDTEStrategy.SCALP_PUT,
            "momentum_call": ZeroDTEStrategy.MOMENTUM_CALL,
            "momentum_put": ZeroDTEStrategy.MOMENTUM_PUT,
            "lotto_call": ZeroDTEStrategy.LOTTO_CALL,
            "lotto_put": ZeroDTEStrategy.LOTTO_PUT,
        }
        strategy = strategy_map.get(args.strategy.lower())
    
    signal = generator.generate_signal(
        symbol,
        strategy=strategy,
        max_dollars=args.max_dollars or 200,
    )
    
    if not signal:
        print(f"\n❌ No 0DTE signal for {symbol}")
        print("\nPossible reasons:")
        print("  • Market is closed")
        print("  • No 0DTE options expiring today")
        print("  • Less than 15 minutes to close")
        return
    
    if args.json:
        import json
        print(json.dumps(signal.to_dict(), indent=2))
    else:
        print()
        print(signal.to_robinhood_format())
        print()
        print(f"📊 Reasoning: {signal.reasoning}")


def cmd_0dte_trade(args):
    """Execute a 0DTE trade."""
    from alpha.options_trader import OptionsTrader
    
    config = AlphaConfig.from_env()
    
    symbol = args.symbol.upper()
    
    # Show disclaimer first
    print_0dte_disclaimer()
    
    if not args.confirm:
        response = input("\nI understand the EXTREME risks of 0DTE. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Trade cancelled. Good choice!")
            return
    
    generator = ZeroDTEGenerator(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
        max_position_dollars=args.max_dollars or 200,
    )
    
    # Generate signal
    signal = generator.generate_signal(
        symbol,
        max_dollars=args.max_dollars or 200,
    )
    
    if not signal:
        print(f"\n❌ No 0DTE signal for {symbol}")
        return
    
    print("\n" + signal.to_robinhood_format())
    print()
    
    if not args.confirm:
        print(f"\n⚠️ You are about to risk ${signal.max_loss:.2f}")
        response = input("Execute this 0DTE trade? (y/N): ")
        if response.lower() != 'y':
            print("Trade cancelled.")
            return
    
    # Execute trade
    trader = OptionsTrader(config=config, paper=True)
    
    # Build a signal-compatible object for the trader
    from alpha.options_signal import OptionsSignal, OptionSignalDirection
    
    options_signal = OptionsSignal(
        symbol=signal.symbol,
        strategy=signal.strategy.value,  # Will be converted
        direction=signal.direction,
        confidence=signal.confidence,
        contracts=[signal.contract] if signal.contract else [],
        underlying_price=signal.underlying_price,
        entry_price=signal.entry_price,
        max_loss=signal.max_loss,
        break_even=signal.break_even,
        suggested_contracts=signal.suggested_contracts,
        risk_factors=signal.warnings,
    )
    
    order = trader.execute_signal(options_signal, contracts=signal.suggested_contracts)
    
    if order:
        print(f"\n✅ 0DTE Order Executed:")
        print(f"  Order ID: {order.order_id}")
        print(f"  Contract: {order.contract_symbol}")
        print(f"  Quantity: {order.quantity}")
        print(f"  Status: {order.status}")
        print("\n⚠️ Monitor closely - expires TODAY!")
    else:
        print("\n❌ Order failed")


def main():
    parser = argparse.ArgumentParser(
        description="Gnosis Alpha - Short-term Trading Signals (Stocks & Options)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stock Commands:
  alpha scan                      # Scan for stock signals
  alpha scan -s AAPL,TSLA        # Scan specific symbols
  alpha scan -d BUY              # Show only BUY signals
  alpha signal AAPL              # Get stock signal for AAPL
  alpha trade AAPL               # Execute stock trade
  alpha status                   # Show account status
  alpha close AAPL               # Close stock position

Options Commands:
  alpha options scan              # Scan for options opportunities
  alpha options scan -s call      # Scan for calls only
  alpha options signal AAPL       # Get options signal for AAPL
  alpha options signal AAPL -s put  # Get put signal
  alpha options chain AAPL        # Show options chain
  alpha options trade AAPL call   # Execute call trade
  alpha options positions         # Show options positions

⚡ 0DTE Commands (HIGH RISK):
  alpha 0dte scan                 # Scan for 0DTE opportunities
  alpha 0dte signal SPY           # Get 0DTE signal for SPY
  alpha 0dte signal TSLA -s lotto_call  # Lotto ticket play
  alpha 0dte trade TSLA           # Execute 0DTE trade (RISKY!)
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # ============ STOCK COMMANDS ============
    
    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan for stock signals")
    scan_parser.add_argument("-s", "--symbols", help="Comma-separated symbols")
    scan_parser.add_argument("-d", "--direction", choices=["BUY", "SELL", "HOLD"])
    scan_parser.add_argument("-c", "--min-confidence", type=float)
    scan_parser.add_argument("-o", "--output", help="Output file")
    scan_parser.set_defaults(func=cmd_scan)
    
    # Signal command
    signal_parser = subparsers.add_parser("signal", help="Get stock signal for symbol")
    signal_parser.add_argument("symbol", help="Stock symbol")
    signal_parser.add_argument("--json", action="store_true", help="Output as JSON")
    signal_parser.set_defaults(func=cmd_signal)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show account status")
    status_parser.set_defaults(func=cmd_status)
    
    # Trade command
    trade_parser = subparsers.add_parser("trade", help="Execute stock trade")
    trade_parser.add_argument("symbol", help="Stock symbol")
    trade_parser.add_argument("-y", "--confirm", action="store_true", help="Skip confirmation")
    trade_parser.add_argument("-f", "--force", action="store_true", help="Force low confidence")
    trade_parser.set_defaults(func=cmd_trade)
    
    # Close command
    close_parser = subparsers.add_parser("close", help="Close stock position")
    close_parser.add_argument("symbol", nargs="?", help="Stock symbol")
    close_parser.add_argument("--all", action="store_true", help="Close all positions")
    close_parser.set_defaults(func=cmd_close)
    
    # ============ OPTIONS COMMANDS ============
    
    options_parser = subparsers.add_parser("options", help="Options trading commands")
    options_subparsers = options_parser.add_subparsers(dest="options_command", help="Options commands")
    
    # Options scan
    opt_scan_parser = options_subparsers.add_parser("scan", help="Scan for options opportunities")
    opt_scan_parser.add_argument("-s", "--symbols", help="Comma-separated symbols")
    opt_scan_parser.add_argument("-t", "--strategy", choices=["call", "put", "covered", "csp"],
                                  help="Filter by strategy")
    opt_scan_parser.add_argument("-c", "--min-confidence", type=float, default=0.6)
    opt_scan_parser.set_defaults(func=cmd_options_scan)
    
    # Options signal
    opt_signal_parser = options_subparsers.add_parser("signal", help="Get options signal")
    opt_signal_parser.add_argument("symbol", help="Stock symbol")
    opt_signal_parser.add_argument("-s", "--strategy", choices=["call", "put", "covered", "csp"],
                                    help="Specific strategy")
    opt_signal_parser.add_argument("--json", action="store_true", help="Output as JSON")
    opt_signal_parser.set_defaults(func=cmd_options_signal)
    
    # Options chain
    opt_chain_parser = options_subparsers.add_parser("chain", help="Show options chain")
    opt_chain_parser.add_argument("symbol", help="Stock symbol")
    opt_chain_parser.add_argument("--min-dte", type=int, default=7, help="Min days to expiration")
    opt_chain_parser.add_argument("--max-dte", type=int, default=45, help="Max days to expiration")
    opt_chain_parser.set_defaults(func=cmd_options_chain)
    
    # Options trade
    opt_trade_parser = options_subparsers.add_parser("trade", help="Execute options trade")
    opt_trade_parser.add_argument("symbol", help="Stock symbol")
    opt_trade_parser.add_argument("strategy", choices=["call", "put"], help="Strategy type")
    opt_trade_parser.add_argument("-n", "--contracts", type=int, help="Number of contracts")
    opt_trade_parser.add_argument("-y", "--confirm", action="store_true", help="Skip confirmation")
    opt_trade_parser.set_defaults(func=cmd_options_trade)
    
    # Options positions
    opt_pos_parser = options_subparsers.add_parser("positions", help="Show options positions")
    opt_pos_parser.set_defaults(func=cmd_options_positions)
    
    # ============ 0DTE COMMANDS (HIGH RISK) ============
    
    zero_dte_parser = subparsers.add_parser("0dte", help="⚡ 0DTE options (HIGH RISK)")
    zero_dte_subparsers = zero_dte_parser.add_subparsers(dest="zero_dte_command", help="0DTE commands")
    
    # 0DTE scan
    zero_scan_parser = zero_dte_subparsers.add_parser("scan", help="Scan for 0DTE opportunities")
    zero_scan_parser.add_argument("-s", "--symbols", help="Comma-separated symbols (default: SPY,QQQ,AAPL,TSLA,NVDA)")
    zero_scan_parser.add_argument("-m", "--max-dollars", type=float, default=200,
                                   help="Max position size in dollars (default: $200)")
    zero_scan_parser.add_argument("-y", "--confirm", action="store_true", help="Skip risk confirmation")
    zero_scan_parser.add_argument("--no-disclaimer", action="store_true", help="Hide disclaimer")
    zero_scan_parser.set_defaults(func=cmd_0dte_scan)
    
    # 0DTE signal
    zero_signal_parser = zero_dte_subparsers.add_parser("signal", help="Get 0DTE signal for symbol")
    zero_signal_parser.add_argument("symbol", help="Stock symbol (e.g., SPY, TSLA)")
    zero_signal_parser.add_argument("-s", "--strategy",
                                     choices=["scalp_call", "scalp_put", "momentum_call", 
                                              "momentum_put", "lotto_call", "lotto_put"],
                                     help="Specific 0DTE strategy")
    zero_signal_parser.add_argument("-m", "--max-dollars", type=float, default=200,
                                     help="Max position size in dollars")
    zero_signal_parser.add_argument("--json", action="store_true", help="Output as JSON")
    zero_signal_parser.add_argument("--no-disclaimer", action="store_true", help="Hide disclaimer")
    zero_signal_parser.set_defaults(func=cmd_0dte_signal)
    
    # 0DTE trade
    zero_trade_parser = zero_dte_subparsers.add_parser("trade", help="Execute 0DTE trade (RISKY!)")
    zero_trade_parser.add_argument("symbol", help="Stock symbol")
    zero_trade_parser.add_argument("-m", "--max-dollars", type=float, default=200,
                                    help="Max position size in dollars")
    zero_trade_parser.add_argument("-y", "--confirm", action="store_true", help="Skip confirmation")
    zero_trade_parser.set_defaults(func=cmd_0dte_trade)
    
    # ============ ML COMMANDS ============
    
    ml_parser = subparsers.add_parser("ml", help="🤖 Machine Learning commands")
    ml_subparsers = ml_parser.add_subparsers(dest="ml_command", help="ML commands")
    
    # ML train
    ml_train_parser = ml_subparsers.add_parser("train", help="Train ML model")
    ml_train_parser.add_argument("-s", "--symbols", help="Comma-separated symbols")
    ml_train_parser.add_argument("-d", "--days", type=int, default=180, 
                                  help="Days of historical data (default: 180)")
    ml_train_parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    ml_train_parser.set_defaults(func=cmd_ml_train)
    
    # ML predict
    ml_predict_parser = ml_subparsers.add_parser("predict", help="Get ML prediction")
    ml_predict_parser.add_argument("symbol", help="Stock symbol")
    ml_predict_parser.add_argument("--json", action="store_true", help="Output as JSON")
    ml_predict_parser.set_defaults(func=cmd_ml_predict)
    
    # ML backtest
    ml_backtest_parser = ml_subparsers.add_parser("backtest", help="Run ML backtest")
    ml_backtest_parser.add_argument("-s", "--symbols", help="Comma-separated symbols")
    ml_backtest_parser.add_argument("-d", "--days", type=int, default=90,
                                     help="Days to backtest (default: 90)")
    ml_backtest_parser.set_defaults(func=cmd_ml_backtest)
    
    # ML features
    ml_features_parser = ml_subparsers.add_parser("features", help="Show ML features for symbol")
    ml_features_parser.add_argument("symbol", help="Stock symbol")
    ml_features_parser.set_defaults(func=cmd_ml_features)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Handle options subcommands
    if args.command == "options":
        if args.options_command is None:
            options_parser.print_help()
            return
    
    # Handle 0DTE subcommands
    if args.command == "0dte":
        if args.zero_dte_command is None:
            zero_dte_parser.print_help()
            return
    
    # Handle ML subcommands
    if args.command == "ml":
        if args.ml_command is None:
            ml_parser.print_help()
            return
    
    args.func(args)


# ============================================================
# ML COMMANDS
# ============================================================

def cmd_ml_train(args):
    """Train ML model."""
    from alpha.ml.trainer import AlphaTrainer, TrainingConfig
    from alpha.ml.models import ModelType
    
    print("\n🤖 Training Alpha ML Model...")
    
    config = TrainingConfig(
        train_start_days_ago=args.days,
        tune_hyperparameters=args.tune,
    )
    
    if args.symbols:
        config.symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    trainer = AlphaTrainer(config)
    result = trainer.train()
    
    result.print_summary()


def cmd_ml_predict(args):
    """Get ML prediction for symbol."""
    from alpha.ml.features import AlphaFeatureEngine
    from alpha.ml.models import AlphaModel
    from pathlib import Path
    import json
    
    symbol = args.symbol.upper()
    
    # Load latest model
    model_path = Path("alpha/models/alpha_directional_latest.pkl")
    if not model_path.exists():
        print("❌ No trained model found. Run 'alpha ml train' first.")
        return
    
    model = AlphaModel.load(model_path)
    engine = AlphaFeatureEngine()
    
    # Extract features
    features = engine.extract(symbol)
    
    if not features.has_sufficient_data:
        print(f"❌ Insufficient data for {symbol}")
        return
    
    # Predict
    prediction = model.predict(features.to_array())
    
    if args.json:
        print(json.dumps(prediction.to_dict(), indent=2))
    else:
        direction_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}[prediction.direction]
        
        print(f"\n🤖 ML Prediction for {symbol}")
        print("="*40)
        print(f"{direction_emoji} Direction: {prediction.direction}")
        print(f"📊 Confidence: {prediction.confidence:.1%}")
        print(f"📈 Expected 5-day Return: {prediction.expected_return:+.2%}")
        print(f"\n📉 Probabilities:")
        for direction, prob in prediction.probabilities.items():
            print(f"   {direction}: {prob:.1%}")
        print(f"\n⏰ Model Version: {prediction.model_version}")
        print("="*40)


def cmd_ml_backtest(args):
    """Run ML backtest."""
    from alpha.ml.features import AlphaFeatureEngine
    from alpha.ml.models import DirectionalClassifier, ModelConfig
    from alpha.ml.backtest import AlphaBacktester, BacktestConfig
    from datetime import datetime, timedelta
    
    print("\n📊 Running ML Backtest...")
    
    # Parse symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    
    # Setup dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Create components
    model = DirectionalClassifier(ModelConfig())
    engine = AlphaFeatureEngine()
    
    backtest_config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.10,
        max_positions=5,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        max_hold_days=7,
        min_confidence=0.6,
    )
    
    backtester = AlphaBacktester(backtest_config)
    
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("Running walk-forward backtest...")
    
    result = backtester.run(
        model=model,
        feature_engine=engine,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )
    
    result.print_summary()


def cmd_ml_features(args):
    """Show ML features for a symbol."""
    from alpha.ml.features import AlphaFeatureEngine
    
    symbol = args.symbol.upper()
    engine = AlphaFeatureEngine()
    
    print(f"\n📊 ML Features for {symbol}")
    print("="*60)
    
    features = engine.extract(symbol)
    
    if not features.has_sufficient_data:
        print(f"❌ Insufficient data for {symbol}")
        return
    
    # Group features by category
    categories = {
        "Price/Trend": ["price_vs_sma", "sma", "price_pct"],
        "Momentum": ["rsi", "macd", "stoch"],
        "Volatility": ["atr", "bb_", "volatility"],
        "Volume": ["volume", "obv"],
        "Pattern": ["higher", "trend_strength", "gap", "range", "support", "resistance"],
        "Market": ["relative", "beta", "correlation", "outperformance"],
    }
    
    for category, prefixes in categories.items():
        print(f"\n{category}:")
        for name, value in sorted(features.features.items()):
            if any(name.startswith(p) or p in name for p in prefixes):
                print(f"  {name}: {value:.4f}")
    
    print("\n" + "="*60)
    print(f"Total features: {len(features.features)}")
    print(f"Missing: {len(features.missing_features)}")


if __name__ == "__main__":
    main()
