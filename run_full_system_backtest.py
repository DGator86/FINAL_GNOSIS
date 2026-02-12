import os
import sys
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath("."))

from loguru import logger
from config import load_config
from engines.inputs.alpaca_market_adapter import AlpacaMarketDataAdapter
from engines.inputs.synthetic_options_adapter import SyntheticOptionsAdapter
from engines.inputs.news_adapter import NewsAdapter # Will use stub behavior if no real news
from engines.inputs.stub_adapters import StaticNewsAdapter # Force stub to avoid API errors

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

# Q4 2025 Real Data Period
START_DATE = datetime(2025, 10, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 12, 20, tzinfo=timezone.utc)
SYMBOL = "SPY"

def build_backtest_pipeline(symbol, config, market_adapter, options_adapter):
    # News: Use stub for consistency in backtest
    news_adapter = StaticNewsAdapter() 
    
    # Engines
    engines = {
        "hedge": HedgeEngineV3(options_adapter, config.engines.hedge.model_dump()),
        "liquidity": LiquidityEngineV1(market_adapter, config.engines.liquidity.model_dump()),
        "sentiment": SentimentEngineV1(
            [
                NewsSentimentProcessor(news_adapter, config.engines.sentiment.model_dump()),
                # Flow processor needs an adapter that supports get_flow_summary
                FlowSentimentProcessor(config.engines.sentiment.model_dump(), flow_adapter=options_adapter),
                TechnicalSentimentProcessor(market_adapter, config.engines.sentiment.model_dump()),
            ],
            config.engines.sentiment.model_dump(),
        ),
        "elasticity": ElasticityEngineV1(market_adapter, config.engines.elasticity.model_dump()),
        "physics": PhysicsEngine(market_adapter, options_adapter, config.model_dump()),
    }
    
    # Agents
    primary_agents = {
        "hedge_agent": HedgeAgentV3(config.agents.hedge.model_dump()),
        "liquidity_agent": LiquidityAgentV1(config.agents.liquidity.model_dump()),
        "sentiment_agent": SentimentAgentV1(config.agents.sentiment.model_dump()),
    }
    
    composer = ComposerAgentV1(
        config.agents.composer.weights,
        config.agents.composer.model_dump(),
    )
    
    # Trade Agent (Paper Mode)
    trade_agent = create_elite_trade_agent(
        options_adapter=options_adapter,
        market_adapter=market_adapter,
        broker=None, # No broker for pure pipeline simulation (ledger only)
        config=config.agents.trade.model_dump(),
    )
    
    ledger = LedgerStore(Path("data/backtest_ledger.jsonl"))
    
    return PipelineRunner(
        symbol=symbol,
        engines=engines,
        primary_agents=primary_agents,
        composer=composer,
        trade_agent=trade_agent,
        ledger_store=ledger,
        config=config.model_dump(),
        auto_execute=False # We are simulating execution via results analysis
    )

async def run_backtest():
    print(f"🚀 Starting Full Gnosis Backtest for {SYMBOL} ({START_DATE.date()} - {END_DATE.date()})")
    
    # 1. Setup Adapters
    try:
        alpaca = AlpacaMarketDataAdapter()
        print("✅ Real Market Data: Connected")
    except:
        print("❌ Real Market Data: Failed (Check Keys)")
        return
        
    synth_options = SyntheticOptionsAdapter(alpaca)
    print("✅ Synthetic Options: Ready (Black-Scholes)")
    
    config = load_config()
    # Tune Physics for 1H bars
    config.gmm_config.dyn.q0 *= 7.0 
    
    # 2. Build Pipeline
    runner = build_backtest_pipeline(SYMBOL, config, alpaca, synth_options)
    
    # 3. Fetch Timeline
    # We drive the backtest by iterating through 1H bars
    bars = alpaca.get_bars(SYMBOL, START_DATE, END_DATE, timeframe="1Hour")
    if not bars:
        print("❌ No historical bars found.")
        return
        
    print(f"📅 Processing {len(bars)} hourly steps...")
    
    results = []
    
    for i, bar in enumerate(bars):
        ts = bar.timestamp
        
        # Run Pipeline
        # Note: We use run_once_async to allow concurrent engine execution
        try:
            result = await runner.run_once_async(ts)
        except Exception as e:
            logger.error(f"Error running pipeline at {ts}: {e}")
            continue
        
        # Log Summary
        cons = result.consensus
        phys = result.physics_snapshot
        
        # Convert PhysicsSnapshot to dict-like access if it's a model
        entropy = 0.0
        stiffness = 0.0
        p_up = 0.0
        
        if phys:
            entropy = getattr(phys, 'entropy', 0.0)
            stiffness = getattr(phys, 'stiffness', 0.0)
            p_up = getattr(phys, 'p_up', 0.0)
        
        row = {
            "ts": ts,
            "price": bar.close,
            "consensus_dir": cons.get("direction", "neutral") if cons else "neutral",
            "consensus_conf": cons.get("confidence", 0.0) if cons else 0.0,
            "entropy": entropy,
            "stiffness": stiffness,
            "p_up": p_up
        }
        results.append(row)
        
        if i % 20 == 0:
            direction = row['consensus_dir'].upper()
            print(f"   {ts.strftime('%Y-%m-%d %H:%M')} | ${bar.close:<7.2f} | {direction:<8} ({row['consensus_conf']:.2f}) | Ent: {row['entropy']:.2f} | Stiff: {row['stiffness']:.2f}")

    # 4. Simple Performance Analysis of Consensus Signal
    print("-" * 80)
    print("📊 Backtest Performance (Consensus Signal)")
    
    # Calc returns
    df = pd.DataFrame(results)
    df.sort_values("ts", inplace=True)
    df["next_return"] = df["price"].pct_change().shift(-1)
    
    # Signal Logic: Long if Bullish, Short if Bearish
    # Filter by Confidence > 0.6
    df["signal"] = 0.0
    df.loc[(df["consensus_dir"] == "long") & (df["consensus_conf"] > 0.6), "signal"] = 1.0
    df.loc[(df["consensus_dir"] == "short") & (df["consensus_conf"] > 0.6), "signal"] = -1.0
    
    df["strategy_return"] = df["signal"] * df["next_return"]
    
    total_ret = df["strategy_return"].sum()
    win_rate = len(df[df["strategy_return"] > 0]) / len(df[df["signal"] != 0]) if len(df[df["signal"] != 0]) > 0 else 0
    
    print(f"📈 Total Strategy Return (w/o fees): {total_ret:.2%}")
    print(f"🎯 Win Rate: {win_rate:.2%}")
    print(f"🔄 Trades: {len(df[df['signal'] != 0])}")
    
    # Save detailed CSV
    df.to_csv("data/full_gnosis_backtest_results.csv")
    print(f"💾 Detailed results saved to data/full_gnosis_backtest_results.csv")

if __name__ == "__main__":
    asyncio.run(run_backtest())
