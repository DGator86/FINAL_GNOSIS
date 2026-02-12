import os
import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from loguru import logger
from config import load_config
from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
from engines.inputs.synthetic_options_adapter import SyntheticOptionsAdapter
from engines.inputs.stub_adapters import StaticNewsAdapter

from engines.hedge.hedge_engine_v3 import HedgeEngineV3
from engines.liquidity.liquidity_engine_v1 import LiquidityEngineV1
from engines.sentiment.sentiment_engine_v1 import SentimentEngineV1
from engines.sentiment.processors import NewsSentimentProcessor, FlowSentimentProcessor, TechnicalSentimentProcessor
from engines.elasticity.elasticity_engine_v1 import ElasticityEngineV1
from engines.physics.physics_engine import PhysicsEngine

from agents.hedge_agent_v3 import HedgeAgentV3
from agents.liquidity_agent_v1 import LiquidityAgentV1
from agents.sentiment_agent_v1 import SentimentAgentV1
from agents.composer.composer_agent_v1 import ComposerAgentV1
from trade.elite_trade_agent import create_elite_trade_agent

from engines.orchestration.pipeline_runner import PipelineRunner
from ledger.ledger_store import LedgerStore

# Configuration for Deep Backtest
# 1 Year Period (approx) - adjusting to ensure data availability in sandbox environment context
# Assuming we are in late 2025 in this hypothetical scenario or using past data. 
# Let's use a solid 6-month block from 2025-01-01 to 2025-06-30 to capture the "Toxic" Q2 and the lead up.
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 6, 30, tzinfo=timezone.utc)
SYMBOLS = ["SPY", "QQQ"]
INITIAL_CAPITAL = 1_000_000.0

def build_pipeline(symbol, config, market, options):
    # Mock news for speed/reliability in backtest
    news = StaticNewsAdapter()
    
    engines = {
        "hedge": HedgeEngineV3(options, config.engines.hedge.model_dump()),
        "liquidity": LiquidityEngineV1(market, config.engines.liquidity.model_dump()),
        "sentiment": SentimentEngineV1(
            [
                NewsSentimentProcessor(news, config.engines.sentiment.model_dump()),
                FlowSentimentProcessor(config.engines.sentiment.model_dump(), flow_adapter=options),
                TechnicalSentimentProcessor(market, config.engines.sentiment.model_dump()),
            ],
            config.engines.sentiment.model_dump(),
        ),
        "elasticity": ElasticityEngineV1(market, config.engines.elasticity.model_dump()),
        "physics": PhysicsEngine(market, options, config.model_dump()),
    }
    
    primary_agents = {
        "hedge_agent": HedgeAgentV3(config.agents.hedge.model_dump()),
        "liquidity_agent": LiquidityAgentV1(config.agents.liquidity.model_dump()),
        "sentiment_agent": SentimentAgentV1(config.agents.sentiment.model_dump()),
    }
    
    composer = ComposerAgentV1(config.agents.composer.weights, config.agents.composer.model_dump())
    
    trade_agent = create_elite_trade_agent(
        options_adapter=options,
        market_adapter=market,
        broker=None, 
        config=config.agents.trade.model_dump(),
    )
    
    ledger = LedgerStore(Path(f"data/backtest_ledger_{symbol}.jsonl"))
    
    return PipelineRunner(
        symbol=symbol,
        engines=engines,
        primary_agents=primary_agents,
        composer=composer,
        trade_agent=trade_agent,
        ledger_store=ledger,
        config=config.model_dump(),
        auto_execute=False
    )

async def run():
    print(f"🌊 Starting Deep Backtest ({START_DATE.date()} to {END_DATE.date()})")
    print(f"   Symbols: {SYMBOLS}")
    print(f"   Capital: ${INITIAL_CAPITAL:,.2f}")
    
    try:
        market = AlpacaMarketDataAdapter()
        # Using synthetic options for backtest consistency (Black-Scholes pricing)
        options = SyntheticOptionsAdapter(market) 
        print("✅ Data Adapters Initialized")
    except Exception as e:
        print(f"❌ Adapter Init Failed: {e}")
        return

    config = load_config()
    # Optimize Physics for 1H bars
    config.gmm_config.dyn.q0 *= 7.0
    
    # Global Portfolio State
    cash = INITIAL_CAPITAL
    positions = {sym: 0 for sym in SYMBOLS}
    equity_curve = []
    
    # 1. Load Data Batch (Efficiency)
    print("📥 Pre-fetching Market Data...")
    market_data = {}
    timeline = set()
    
    for sym in SYMBOLS:
        bars = market.get_bars(sym, START_DATE, END_DATE, timeframe="1Hour")
        if not bars:
            print(f"⚠️  No data for {sym}")
            continue
        market_data[sym] = {b.timestamp: b for b in bars}
        timeline.update(market_data[sym].keys())
        print(f"   Loaded {len(bars)} bars for {sym}")
        
    sorted_timeline = sorted(list(timeline))
    print(f"✅ Timeline Built: {len(sorted_timeline)} steps")
    
    # 2. Pipeline Instances
    pipelines = {sym: build_pipeline(sym, config, market, options) for sym in SYMBOLS}
    
    # 3. Execution Loop
    print("\n🚀 Running Simulation...")
    trades_count = 0
    
    for i, ts in enumerate(sorted_timeline):
        step_pnl = 0.0
        
        # Current Prices for Equity Calc
        current_prices = {}
        for sym in SYMBOLS:
            if ts in market_data[sym]:
                current_prices[sym] = market_data[sym][ts].close
            elif sym in market_data and market_data[sym]:
                # Fallback to last known if missing bar
                # (Simple forward fill logic not implemented for speed, just skip update)
                pass
        
        # Run Pipelines
        for sym in SYMBOLS:
            if ts not in market_data[sym]: continue
            
            bar = market_data[sym][ts]
            runner = pipelines[sym]
            
            # Execute Pipeline (Async wrap)
            try:
                result = await runner.run_once_async(ts)
            except Exception as e:
                # logger.error(f"Pipeline error {sym} {ts}: {e}")
                continue
                
            # Decision Logic (Physics + Composite)
            # Use Physics Snapshot directly for Regime
            phys = result.physics_snapshot
            if not phys: continue
            
            # SIGNAL LOGIC:
            # 1. Regime Check: Entropy < 1.0 (Stable)
            # 2. Trend: P_up > 0.55 (Long) or < 0.45 (Short)
            
            signal = 0.0
            if phys.entropy < 1.0:
                if phys.p_up > 0.55: signal = 1.0
                elif phys.p_up < 0.45: signal = -1.0
            else:
                # High Entropy -> Mean Reversion / Flat
                # Optional: Fade extreme moves? For now, cash.
                signal = 0.0
                
            # Position Sizing (Fixed fractional 10% per trade)
            target_alloc = 0.10 * signal
            
            price = bar.close
            target_val = equity_curve[-1] if equity_curve else INITIAL_CAPITAL
            target_qty = int((target_val * target_alloc) / price)
            
            current_qty = positions[sym]
            diff = target_qty - current_qty
            
            if diff != 0:
                # Friction
                cost = abs(diff * price) * 0.0005 # 5bps
                cash -= (diff * price + cost)
                positions[sym] = target_qty
                trades_count += 1
                
        # Calculate Equity
        pos_val = sum(positions[s] * current_prices.get(s, 0) for s in SYMBOLS)
        total_eq = cash + pos_val
        equity_curve.append(total_eq)
        
        if i % 100 == 0:
            print(f"   {ts.strftime('%Y-%m-%d')} | Eq: ${total_eq:,.0f} | Trades: {trades_count}")
            
    # 4. Final Stats
    final_eq = equity_curve[-1]
    ret = (final_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    print("-" * 60)
    print(f"💰 Final Equity: ${final_eq:,.2f}")
    print(f"📈 Total Return: {ret:.2%}")
    print(f"🔄 Total Trades: {trades_count}")
    
    # Sharpe
    series = pd.Series(equity_curve)
    rets = series.pct_change().dropna()
    if len(rets) > 0 and rets.std() > 0:
        sharpe = rets.mean() / rets.std() * np.sqrt(252 * 7) # Approx 1H bars
        print(f"📊 Sharpe Ratio: {sharpe:.2f}")
        
    print(f"📉 Max Drawdown: {((series.cummax() - series) / series.cummax()).max():.2%}")

if __name__ == "__main__":
    asyncio.run(run())
