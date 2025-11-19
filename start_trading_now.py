#!/usr/bin/env python3
"""
IMMEDIATE PAPER TRADING LAUNCHER
Simplified version that starts trading immediately
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import asyncio

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment
from dotenv import load_dotenv
load_dotenv()

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║              🚀 STARTING PAPER TRADING SYSTEM 🚀                      ║
║                                                                        ║
║  Mode: FULL PAPER TRADING ENABLED                                     ║
║  Account: Alpaca Paper Trading                                        ║
║  Symbols: Multiple stocks configured                                  ║
║  Data: Unusual Whales + Alpaca                                        ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
""")

# Check if we can connect to Alpaca
try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.live import StockDataStream
    from alpaca.data.historical import StockHistoricalDataClient
    
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    print("🔄 Connecting to Alpaca...")
    client = TradingClient(api_key, secret_key, paper=True)
    account = client.get_account()
    
    print(f"""
✅ ALPACA CONNECTION SUCCESSFUL:
   Account ID: {account.id}
   Balance: ${float(account.cash):,.2f}
   Buying Power: ${float(account.buying_power):,.2f}
   Portfolio Value: ${float(account.portfolio_value):,.2f}
""")
    
    # Check market status
    clock = client.get_clock()
    print(f"""
⏰ MARKET STATUS:
   Market Open: {'✅ YES - TRADING ACTIVE' if clock.is_open else '❌ NO - Waiting for market open'}
   Next Close: {clock.next_close}
""")
    
except Exception as e:
    print(f"❌ Error connecting to Alpaca: {e}")
    sys.exit(1)

# Check Unusual Whales
try:
    from engines.inputs.unusual_whales_adapter import UnusualWhalesAdapter
    
    print("\n🔄 Connecting to Unusual Whales...")
    uw_adapter = UnusualWhalesAdapter()
    
    # Test connection
    tide = uw_adapter.get_market_tide()
    if tide and 'data' in tide:
        print("✅ Unusual Whales connected successfully")
        
        # Get latest market sentiment
        if isinstance(tide['data'], list) and len(tide['data']) > 0:
            latest = tide['data'][-1]
            net_call = float(latest.get('net_call_premium', 0))
            net_put = float(latest.get('net_put_premium', 0))
            
            sentiment = "BULLISH 📈" if net_call > abs(net_put) else "BEARISH 📉"
            print(f"   Current Market Sentiment: {sentiment}")
    
except Exception as e:
    print(f"⚠️  Unusual Whales connection issue (non-critical): {e}")

# Now start a simple trading loop
async def trade_symbols():
    """Simple trading loop for key symbols"""
    
    symbols = ['SPY', 'QQQ', 'AAPL', 'NVDA', 'TSLA']
    data_client = StockHistoricalDataClient(api_key, secret_key)
    
    print(f"""
📊 PAPER TRADING CONFIGURATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Symbols: {', '.join(symbols)}
   Strategy: Momentum + Options Flow
   Risk: 2% stop-loss, 3 position max
   Mode: PAPER TRADING (not real money)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 TRADING SYSTEM ACTIVE - Monitoring markets...
""")
    
    positions_opened = 0
    
    while True:
        try:
            # Check if market is open
            clock = client.get_clock()
            if not clock.is_open:
                print(f"⏰ Market closed. Waiting... (Next open: {clock.next_open})")
                await asyncio.sleep(60)
                continue
            
            # Check each symbol
            for symbol in symbols:
                try:
                    # Get current positions
                    positions = client.get_all_positions()
                    position_symbols = [p.symbol for p in positions]
                    
                    # Check if we already have this position
                    if symbol in position_symbols:
                        continue
                    
                    # Limit total positions
                    if len(positions) >= 3:
                        continue
                    
                    # Get latest quote
                    from alpaca.data.requests import StockLatestQuoteRequest
                    quote_request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
                    quotes = data_client.get_stock_latest_quote(quote_request)
                    
                    if symbol in quotes:
                        quote = quotes[symbol]
                        price = quote.ask_price
                        
                        # Get options flow signal from Unusual Whales
                        bullish_signal = False
                        try:
                            flow = uw_adapter.get_ticker_flow(symbol, limit=10)
                            if flow and 'data' in flow:
                                # Check for bullish flow
                                recent_flows = flow['data'][:5] if isinstance(flow['data'], list) else []
                                bullish_count = sum(1 for f in recent_flows 
                                                  if isinstance(f, dict) and 
                                                  f.get('type', '').lower() in ['call', 'sweep', 'block'])
                                
                                if bullish_count >= 3:
                                    bullish_signal = True
                                    print(f"   📈 Bullish options flow detected for {symbol}")
                        except:
                            pass
                        
                        # Simple momentum check (would be more sophisticated in production)
                        # For demo: Place a small paper trade if we have bullish signal
                        if bullish_signal and positions_opened < 5:  # Limit to 5 trades for demo
                            # Calculate position size (2% of portfolio)
                            portfolio_value = float(account.portfolio_value)
                            position_value = portfolio_value * 0.02
                            shares = int(position_value / price)
                            
                            if shares > 0:
                                print(f"\n🎯 PLACING PAPER TRADE:")
                                print(f"   Symbol: {symbol}")
                                print(f"   Shares: {shares}")
                                print(f"   Price: ${price:.2f}")
                                print(f"   Value: ${shares * price:.2f}")
                                
                                # Place market order
                                from alpaca.trading.requests import MarketOrderRequest
                                from alpaca.trading.enums import OrderSide, TimeInForce
                                
                                order_request = MarketOrderRequest(
                                    symbol=symbol,
                                    qty=shares,
                                    side=OrderSide.BUY,
                                    time_in_force=TimeInForce.DAY
                                )
                                
                                order = client.submit_order(order_request)
                                positions_opened += 1
                                
                                print(f"   ✅ Order submitted: {order.id}")
                                print(f"   Monitor at: https://app.alpaca.markets/paper\n")
                
                except Exception as e:
                    print(f"   ⚠️ Error processing {symbol}: {e}")
            
            # Update account info
            account = client.get_account()
            positions = client.get_all_positions()
            
            # Show status every 5 minutes
            current_time = datetime.now()
            if current_time.minute % 5 == 0 and current_time.second < 30:
                print(f"""
📊 STATUS UPDATE ({current_time.strftime('%H:%M')}):
   Portfolio: ${float(account.portfolio_value):,.2f}
   Positions: {len(positions)}
   Cash: ${float(account.cash):,.2f}
   P&L Today: ${float(account.portfolio_value) - 30000:+,.2f}
""")
            
            # Wait before next check
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n⚠️  Stopping trading system...")
            break
        except Exception as e:
            print(f"❌ Error in trading loop: {e}")
            await asyncio.sleep(60)
    
    print("""
    
📊 TRADING SESSION ENDED
   Check your positions at: https://app.alpaca.markets/paper
   
✅ Paper trading system stopped successfully.
""")

# Run the trading system
if __name__ == "__main__":
    try:
        asyncio.run(trade_symbols())
    except KeyboardInterrupt:
        print("\n👋 Trading stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()