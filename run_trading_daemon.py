import time
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath("."))

from loguru import logger
from config import load_config
from engines.inputs.adapter_factory import (
    create_broker_adapter,
    create_market_data_adapter,
    create_news_adapter,
    create_options_adapter,
)
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
from engines.scanner import get_dynamic_universe

def run_daemon():
    logger.info("🚀 Trading Daemon Online (Options Only Mode)")
    
    # 1. Config & Adapters
    config = load_config()
    
    # Enforce Options Preference
    config.agents.trade.min_confidence = 0.6 # High threshold
    
    # Adapters
    try:
        broker = create_broker_adapter(paper=True, prefer_real=True) # Paper for safety initially
        market = create_market_data_adapter(prefer_real=True)
        options = create_options_adapter(prefer_real=True)
        news = create_news_adapter(prefer_real=True)
        
        # Check if Unusal Whales is available for Flow
        flow_adapter = None
        if config.data_sources.unusual_whales_enabled:
            from engines.inputs.unusual_whales_adapter import UnusualWhalesOptionsAdapter
            try:
                flow_adapter = UnusualWhalesOptionsAdapter()
            except: pass
            
    except Exception as e:
        logger.error(f"Failed to init adapters: {e}")
        return

    # 2. Universe
    # Static top list for stability + dynamic scan
    core_universe = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "AMD", "MSFT", "AMZN", "META"]
    
    while True:
        try:
            # Market Hours Check
            clock = broker.get_market_clock()
            if clock and not clock['is_open']: # Check clock is valid
                next_open = clock['next_open']
                if next_open: # Ensure next_open is valid
                    wait_sec = (next_open - datetime.now(timezone.utc)).total_seconds()
                    logger.info(f"Market Closed. Sleeping until {next_open} ({wait_sec/3600:.1f} hours)")
                    time.sleep(min(wait_sec, 3600)) # Wake up hourly to check
                    continue
                else:
                    logger.warning("Market closed but next_open is None. Sleeping 1 hour.")
                    time.sleep(3600)
                    continue
            elif not clock:
                logger.warning("Could not fetch market clock. Retrying in 60s.")
                time.sleep(60)
                continue
                
            logger.info("🟢 Market Open - Starting Scan Cycle")
            logger.info("SYSTEM STATUS: ACTIVE") # Explicit Activation Log for User Assurance
            logger.info("SYSTEM STATUS: ACTIVE") # Explicit Activation Log for User Assurance
            
            # 3. Dynamic Universe Update (Every 30 mins)
            # For now, stick to Core + Top 5
            # universe = core_universe + get_dynamic_universe(config.scanner.model_dump(), 5)
            universe = core_universe # Stick to liquid majors for options
            
            for symbol in universe:
                logger.info(f"Analyzing {symbol}...")
                
                # Build Pipeline
                engines = {
                    "hedge": HedgeEngineV3(options, config.engines.hedge.model_dump()),
                    "liquidity": LiquidityEngineV1(market, config.engines.liquidity.model_dump()),
                    "sentiment": SentimentEngineV1(
                        [
                            NewsSentimentProcessor(news, config.engines.sentiment.model_dump()),
                            FlowSentimentProcessor(config.engines.sentiment.model_dump(), flow_adapter=flow_adapter),
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
                    broker=broker,
                    config=config.agents.trade.model_dump(),
                )
                
                ledger = LedgerStore(config.tracking.ledger_path)
                
                runner = PipelineRunner(
                    symbol=symbol,
                    engines=engines,
                    primary_agents=primary_agents,
                    composer=composer,
                    trade_agent=trade_agent,
                    ledger_store=ledger,
                    config=config.model_dump(),
                    auto_execute=True, # EXECUTE!
                )
                
                # Run Sync (Daemon manages concurrency by symbol if needed, but sequential is safer for rate limits)
                runner.run_once(datetime.now(timezone.utc))
                
            # Cycle Sleep
            time.sleep(300) # 5 minutes between cycles
            
        except Exception as e:
            logger.error(f"Daemon Loop Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_daemon()
