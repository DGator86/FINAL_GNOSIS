#!/usr/bin/env python3
"""
Start Live Trading Bot with Web Dashboard

Runs:
1. FastAPI dashboard server on port 8080
2. Live trading bot with WebSocket integration

Usage:
    python start_with_dashboard.py
    
Then open: http://localhost:8080
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def main():
    """Run dashboard and bot concurrently"""
    
    # Import after path setup
    import uvicorn
    from gnosis.trading.live_bot import LiveTradingBot
    from gnosis.dashboard import (
        app, update_positions, add_trade, update_agent_votes,
        update_regime, update_bar, update_portfolio_stats
    )
    
    print("="*60)
    print("  GNOSIS TRADING SYSTEM")
    print("="*60)
    print("\n🌐 Dashboard: http://localhost:8080")
    print("🤖 Bot: Starting...\n")
    
    # Create bot with dashboard hooks
    bot = LiveTradingBot(
        symbol="SPY",
        bar_interval="1Min",
        enable_memory=True,
        enable_trading=False,  # DRY RUN - set True to actually trade
        paper_mode=True
    )
    
    # Monkey-patch bot to send updates to dashboard
    original_process_bar = bot.process_bar
    
    async def process_bar_with_dashboard(bar):
        """Wrapped process_bar that updates dashboard"""
        # Update dashboard with new bar
        update_bar(bar)
        
        # Call original processing
        await original_process_bar(bar)
        
        # After processing, update dashboard with latest state
        positions_dict = {
            pos.symbol: {
                "symbol": pos.symbol,
                "side": pos.side,
                "size": pos.size,
                "entry_price": pos.entry_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "bars_held": pos.bars_held
            }
            for pos in bot.position_mgr.positions.values()
        }
        update_positions(positions_dict)
        
        # Update portfolio stats
        summary = bot.position_mgr.get_portfolio_summary()
        update_portfolio_stats({
            "capital": 30000.0,
            "equity": 30000.0 * (1 + summary["total_pnl"]),
            "pnl": summary["total_pnl"],
            "daily_pnl": summary["daily_pnl"],
            "total_trades": summary["total_trades"],
            "win_rate": summary.get("win_rate", 0.0)
        })
    
    bot.process_bar = process_bar_with_dashboard
    
    # Monkey-patch close_position to send trade updates
    original_close_position = bot.close_position
    
    async def close_position_with_dashboard(symbol, price, reason):
        """Wrapped close_position that updates dashboard"""
        # Get position before closing
        pos = bot.position_mgr.positions.get(symbol)
        
        # Call original
        await original_close_position(symbol, price, reason)
        
        # Send trade to dashboard
        if pos:
            add_trade({
                "symbol": symbol,
                "side": pos.side,
                "entry_price": pos.entry_price,
                "entry_time": pos.entry_time.isoformat(),
                "exit_price": price,
                "exit_time": bot.position_mgr.trades[-1]["exit_time"].isoformat() if bot.position_mgr.trades else None,
                "exit_reason": reason,
                "realized_pnl": pos.unrealized_pnl,
                "bars_held": pos.bars_held
            })
    
    bot.close_position = close_position_with_dashboard
    
    # Monkey-patch evaluate_agents to send votes
    original_evaluate = bot.evaluate_agents
    
    def evaluate_agents_with_dashboard(features, price):
        """Wrapped evaluate_agents that updates dashboard"""
        agent_views = original_evaluate(features, price)
        
        # Send to dashboard
        update_agent_votes({
            "agents": [
                {
                    "name": av.agent_name,
                    "signal": av.signal,
                    "confidence": av.confidence,
                    "reasoning": av.reasoning
                }
                for av in agent_views
            ],
            "timestamp": str(bot.bars[-1]["t_event"]) if bot.bars else None
        })
        
        return agent_views
    
    bot.evaluate_agents = evaluate_agents_with_dashboard
    
    # Run dashboard server in background
    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="warning")
    server = uvicorn.Server(config)
    
    async def run_server():
        await server.serve()
    
    # Run both concurrently
    await asyncio.gather(
        run_server(),
        bot.run()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        print("="*60)
