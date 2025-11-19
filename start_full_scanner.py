#!/usr/bin/env python3
"""
Full 30-Symbol Scanner with Trading
Monitors all configured symbols simultaneously
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import yaml
import json
from collections import defaultdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment
from dotenv import load_dotenv
load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Try to load Unusual Whales
try:
    from engines.inputs.unusual_whales_adapter import UnusualWhalesAdapter
    UW_AVAILABLE = True
except:
    UW_AVAILABLE = False
    print("⚠️  Unusual Whales not available")

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║          🚀 FULL 30-SYMBOL SCANNER WITH TRADING 🚀                    ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
""")

# Load watchlist
with open("config/watchlist.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Build symbol list
symbols = []
watchlist = config['watchlist']
for category in ['indices', 'mega_tech', 'financials', 'high_volume', 'strategic']:
    if category in watchlist:
        for item in watchlist[category]:
            if item['symbol'] != 'VIX':  # VIX not tradeable
                symbols.append(item['symbol'])

print(f"📊 Loaded {len(symbols)} symbols for scanning")
print(f"   {', '.join(symbols[:10])}...")

# Initialize clients
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

trading_client = TradingClient(api_key, secret_key, paper=True)
data_client = StockHistoricalDataClient(api_key, secret_key)

if UW_AVAILABLE:
    uw_client = UnusualWhalesAdapter()
    print("✅ Unusual Whales connected")
else:
    uw_client = None

# Global state for dashboard
scanner_state = {
    'symbols': {},
    'last_update': None,
    'market_open': False,
    'positions': [],
    'account': {}
}

# Create state directory
state_dir = Path("data/scanner_state")
state_dir.mkdir(parents=True, exist_ok=True)


def save_state():
    """Save scanner state to file for dashboard"""
    state_file = state_dir / "current_state.json"
    scanner_state['last_update'] = datetime.now().isoformat()
    
    with open(state_file, 'w') as f:
        json.dump(scanner_state, f, indent=2, default=str)


async def scan_symbol(symbol: str):
    """Scan a single symbol and return analysis"""
    try:
        # Get latest quote
        request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        quotes = data_client.get_stock_latest_quote(request)
        
        if symbol not in quotes:
            return None
        
        quote = quotes[symbol]
        price = quote.ask_price
        
        # Get options flow from Unusual Whales
        flow_signal = "neutral"
        flow_confidence = 0.5
        flow_data = {}
        
        if uw_client:
            try:
                flow = uw_client.get_ticker_flow(symbol, limit=10)
                if flow and 'data' in flow:
                    flows = flow['data'][:10] if isinstance(flow['data'], list) else []
                    
                    # Count bullish/bearish flows
                    bullish = sum(1 for f in flows if isinstance(f, dict) and 
                                f.get('type', '').lower() in ['call', 'sweep'])
                    bearish = sum(1 for f in flows if isinstance(f, dict) and 
                                f.get('type', '').lower() in ['put'])
                    
                    if bullish > bearish + 2:
                        flow_signal = "bullish"
                        flow_confidence = min(0.9, 0.5 + (bullish - bearish) * 0.1)
                    elif bearish > bullish + 2:
                        flow_signal = "bearish"
                        flow_confidence = min(0.9, 0.5 + (bearish - bullish) * 0.1)
                    
                    flow_data = {
                        'bullish_count': bullish,
                        'bearish_count': bearish,
                        'total_flows': len(flows)
                    }
            except:
                pass
        
        # Get recent bars for momentum
        bars_request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            limit=20
        )
        
        bars = data_client.get_stock_bars(bars_request)
        momentum_signal = "neutral"
        momentum_confidence = 0.5
        
        if symbol in bars and len(bars[symbol]) >= 10:
            df = bars[symbol].df
            returns = df['close'].pct_change().tail(10).mean()
            
            if returns > 0.001:
                momentum_signal = "bullish"
                momentum_confidence = min(0.9, 0.5 + abs(returns) * 100)
            elif returns < -0.001:
                momentum_signal = "bearish"
                momentum_confidence = min(0.9, 0.5 + abs(returns) * 100)
        
        # Combine signals - Simple composer logic
        signals = {
            'flow': {'signal': flow_signal, 'confidence': flow_confidence},
            'momentum': {'signal': momentum_signal, 'confidence': momentum_confidence}
        }
        
        # Composer decision
        bullish_votes = sum(1 for s in signals.values() if s['signal'] == 'bullish')
        bearish_votes = sum(1 for s in signals.values() if s['signal'] == 'bearish')
        
        if bullish_votes >= 2:
            composer_status = "BUY"
            composer_confidence = sum(s['confidence'] for s in signals.values() if s['signal'] == 'bullish') / bullish_votes
        elif bearish_votes >= 2:
            composer_status = "SELL"
            composer_confidence = sum(s['confidence'] for s in signals.values() if s['signal'] == 'bearish') / bearish_votes
        else:
            composer_status = "HOLD"
            composer_confidence = 0.5
        
        return {
            'symbol': symbol,
            'price': float(price),
            'bid': float(quote.bid_price),
            'ask': float(quote.ask_price),
            'timestamp': datetime.now().isoformat(),
            'composer_status': composer_status,
            'composer_confidence': composer_confidence,
            'signals': signals,
            'flow_data': flow_data
        }
        
    except Exception as e:
        print(f"   ⚠️ Error scanning {symbol}: {e}")
        return None


async def scan_all_symbols():
    """Scan all symbols in parallel"""
    print(f"\n🔍 Scanning {len(symbols)} symbols...")
    
    # Create scan tasks
    tasks = [scan_symbol(symbol) for symbol in symbols]
    
    # Execute in parallel with limited concurrency
    results = []
    batch_size = 10
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
    
    # Filter valid results
    valid_results = [r for r in results if r is not None]
    
    # Update global state
    for result in valid_results:
        scanner_state['symbols'][result['symbol']] = result
    
    # Count signals
    buy_signals = sum(1 for r in valid_results if r['composer_status'] == 'BUY')
    sell_signals = sum(1 for r in valid_results if r['composer_status'] == 'SELL')
    hold_signals = sum(1 for r in valid_results if r['composer_status'] == 'HOLD')
    
    print(f"   ✅ Scanned {len(valid_results)} symbols")
    print(f"   📊 Signals: {buy_signals} BUY, {sell_signals} SELL, {hold_signals} HOLD")
    
    return valid_results


async def execute_trades(scan_results):
    """Execute trades based on scan results"""
    
    try:
        # Get current positions
        positions = trading_client.get_all_positions()
        position_symbols = [p.symbol for p in positions]
        
        # Limit to 5 positions max
        if len(positions) >= 5:
            print("   ⚠️ Max positions reached (5)")
            return
        
        # Get account info
        account = trading_client.get_account()
        portfolio_value = float(account.portfolio_value)
        
        # Find top BUY signals
        buy_signals = [r for r in scan_results if r['composer_status'] == 'BUY' 
                      and r['composer_confidence'] > 0.7
                      and r['symbol'] not in position_symbols]
        
        # Sort by confidence
        buy_signals.sort(key=lambda x: x['composer_confidence'], reverse=True)
        
        # Take top 3 opportunities
        for signal in buy_signals[:3]:
            if len(positions) >= 5:
                break
            
            symbol = signal['symbol']
            price = signal['price']
            confidence = signal['composer_confidence']
            
            # Position size: 2% of portfolio
            position_value = portfolio_value * 0.02
            shares = int(position_value / price)
            
            if shares > 0:
                print(f"\n   🎯 PLACING TRADE:")
                print(f"      {symbol}: {shares} shares @ ${price:.2f}")
                print(f"      Confidence: {confidence:.2%}")
                
                # Place order
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=shares,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                
                order = trading_client.submit_order(order_request)
                print(f"      ✅ Order ID: {order.id}")
                
                positions.append(order)  # Update local count
    
    except Exception as e:
        print(f"   ❌ Trading error: {e}")


async def update_account_state():
    """Update account and positions in state"""
    try:
        account = trading_client.get_account()
        positions = trading_client.get_all_positions()
        
        scanner_state['account'] = {
            'portfolio_value': float(account.portfolio_value),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'pnl': float(account.portfolio_value) - 30000
        }
        
        scanner_state['positions'] = [
            {
                'symbol': p.symbol,
                'qty': float(p.qty),
                'avg_price': float(p.avg_entry_price),
                'current_price': float(p.current_price),
                'pnl': float(p.unrealized_pl) if hasattr(p, 'unrealized_pl') else 0
            }
            for p in positions
        ]
    except Exception as e:
        print(f"   ⚠️ Error updating account: {e}")


async def main_loop():
    """Main trading loop"""
    
    print("\n🚀 Starting trading loop...")
    print("   Checking every 60 seconds")
    print("   Dashboard state saved to: data/scanner_state/current_state.json")
    
    scan_count = 0
    
    while True:
        try:
            # Check market status
            clock = trading_client.get_clock()
            scanner_state['market_open'] = clock.is_open
            
            if not clock.is_open:
                print(f"\n⏰ Market closed. Next open: {clock.next_open}")
                await asyncio.sleep(60)
                continue
            
            scan_count += 1
            print(f"\n{'='*60}")
            print(f"📊 SCAN #{scan_count} - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*60}")
            
            # Scan all symbols
            results = await scan_all_symbols()
            
            # Update account state
            await update_account_state()
            
            # Save state for dashboard
            save_state()
            
            # Execute trades
            await execute_trades(results)
            
            # Show summary
            print(f"\n   Portfolio: ${scanner_state['account'].get('portfolio_value', 0):,.2f}")
            print(f"   Positions: {len(scanner_state['positions'])}")
            print(f"   P&L: ${scanner_state['account'].get('pnl', 0):+,.2f}")
            
            # Wait before next scan
            await asyncio.sleep(60)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Stopping scanner...")
            break
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            await asyncio.sleep(30)
    
    print("\n✅ Scanner stopped")


if __name__ == "__main__":
    try:
        # Check connection
        account = trading_client.get_account()
        print(f"✅ Connected to Alpaca (Balance: ${float(account.cash):,.2f})")
        
        # Run main loop
        asyncio.run(main_loop())
        
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()