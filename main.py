from __future__ import annotations

"""Command line entrypoint for Super Gnosis pipeline."""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import typer
from dotenv import load_dotenv

from agents.composer.composer_agent_v1 import ComposerAgentV1
from agents.hedge_agent_v3 import HedgeAgentV3
from agents.liquidity_agent_v1 import LiquidityAgentV1
from agents.sentiment_agent_v1 import SentimentAgentV1
from config import AppConfig, load_config
from engines.elasticity.elasticity_engine_v1 import ElasticityEngineV1
from engines.hedge.hedge_engine_v3 import HedgeEngineV3
from engines.inputs.adapter_factory import (
    create_broker_adapter,
    create_market_data_adapter,
    create_news_adapter,
    create_options_adapter,
)
from engines.inputs.market_data_adapter import MarketDataAdapter
from engines.inputs.news_adapter import NewsAdapter
from engines.inputs.options_chain_adapter import OptionsChainAdapter
from engines.liquidity.liquidity_engine_v1 import LiquidityEngineV1
from engines.ml import (
    AnomalyDetector,
    CurriculumRLEvaluator,
    FaissRegimeRetriever,
    KatsForecasterAdapter,
    MLEnhancementEngine,
)
from engines.orchestration.pipeline_runner import PipelineRunner
from engines.sentiment.processors import (
    FlowSentimentProcessor,
    NewsSentimentProcessor,
    TechnicalSentimentProcessor,
)
from engines.sentiment.sentiment_engine_v1 import SentimentEngineV1
from execution.broker_adapters.settings import get_alpaca_paper_setting
from feedback.adaptation_agent import AdaptationAgent
from feedback.tracking_agent import TrackingAgent
from ledger.ledger_store import LedgerStore
from models.features.feature_builder import EnhancedFeatureBuilder, FeatureConfig
from models.lookahead_model import LookaheadModel
from trade.trade_agent_v1 import TradeAgentV1
from universe.watchlist_loader import load_active_watchlist
from watchlist import AdaptiveWatchlist
from cli.result_formatter import format_pipeline_result

# Load environment variables
load_dotenv()

app = typer.Typer(help="Super Gnosis / DHPE Pipeline CLI")

def build_pipeline(
    symbol: str,
    config: AppConfig,
    adapters: Optional[Dict[str, object]] = None,
    watchlist: Optional[AdaptiveWatchlist] = None,
) -> PipelineRunner:
    """Assemble a :class:`PipelineRunner` for ``symbol`` using ``config``."""

    adapters = adapters or {}
    
    # Use adapter factory for automatic fallback
    # Try real adapters first, fall back to stubs if they fail
    options_adapter: OptionsChainAdapter = adapters.get("options") or create_options_adapter(prefer_real=True)
    market_adapter: MarketDataAdapter = adapters.get("market") or create_market_data_adapter(prefer_real=True)
    news_adapter: NewsAdapter = adapters.get("news") or create_news_adapter(prefer_real=False)
    flow_adapter = adapters.get("flow")
    if not flow_adapter and config.data_sources.unusual_whales_enabled:
        try:
            from engines.inputs.unusual_whales_adapter import UnusualWhalesAdapter

            flow_adapter = UnusualWhalesAdapter()
        except Exception:
            flow_adapter = None

    # Create technical processor instance for reuse
    technical_processor = TechnicalSentimentProcessor(market_adapter, config.engines.sentiment.model_dump())

    engines = {
        "hedge": HedgeEngineV3(options_adapter, config.engines.hedge.model_dump()),
        "liquidity": LiquidityEngineV1(market_adapter, config.engines.liquidity.model_dump()),
        "sentiment": SentimentEngineV1(
            [
                NewsSentimentProcessor(news_adapter, config.engines.sentiment.model_dump()),
                FlowSentimentProcessor(config.engines.sentiment.model_dump(), flow_adapter=flow_adapter),
                technical_processor,
            ],
            config.engines.sentiment.model_dump(),
        ),
        "elasticity": ElasticityEngineV1(market_adapter, config.engines.elasticity.model_dump()),
        "mtf": technical_processor,  # For multi-timeframe analysis
    }

    primary_agents = {
        "primary_hedge": HedgeAgentV3(config.agents.hedge.model_dump()),
        "primary_liquidity": LiquidityAgentV1(config.agents.liquidity.model_dump()),
        "primary_sentiment": SentimentAgentV1(config.agents.sentiment.model_dump()),
    }

    composer = ComposerAgentV1(
        config.agents.composer.weights,
        config.agents.composer.model_dump(),
    )
    trade_agent = TradeAgentV1(
        options_adapter,
        market_adapter,
        config.agents.trade.model_dump(),
        broker=adapters.get("broker"),
    )
    active_universe = load_active_watchlist()
    watchlist_universe = list({*active_universe, symbol})

    watchlist = watchlist or AdaptiveWatchlist(
        universe=watchlist_universe,
        min_candidates=3,
        max_candidates=8,
        volume_threshold=10_000_000,
    )

    ledger_path = Path(config.tracking.ledger_path)
    ledger_store = LedgerStore(ledger_path)

    tracking_agent = None
    if config.tracking.enable_position_tracking:
        tracking_agent = TrackingAgent(adapters.get("broker"), enable=True)

    adaptation_agent = None
    if config.adaptation.enabled:
        adaptation_agent = AdaptationAgent(
            state_path=Path(config.adaptation.state_path),
            min_trades=config.adaptation.min_trades_for_update,
            performance_lookback=config.adaptation.performance_lookback,
            min_risk_per_trade=config.adaptation.min_risk_per_trade,
            max_risk_per_trade=config.adaptation.max_risk_per_trade,
        )

    # ML enhancement stack inspired by external repos (Kats, Faiss, RL curriculum)
    forecaster = KatsForecasterAdapter(market_adapter, forecast_horizon=5, min_history=30)
    similarity_retriever = FaissRegimeRetriever(max_history=500, k=5)
    anomaly_detector = AnomalyDetector(contamination=0.05, warmup=25)
    rl_evaluator = CurriculumRLEvaluator()
    ml_engine = MLEnhancementEngine(
        market_adapter=market_adapter,
        forecaster=forecaster,
        similarity_retriever=similarity_retriever,
        anomaly_detector=anomaly_detector,
        rl_evaluator=rl_evaluator,
    )

    return PipelineRunner(
        symbol=symbol,
        engines=engines,
        primary_agents=primary_agents,
        composer=composer,
        trade_agent=trade_agent,
        ledger_store=ledger_store,
        config=config.model_dump(),
        watchlist=watchlist,
        tracking_agent=tracking_agent,
        adaptation_agent=adaptation_agent,
        auto_execute=adapters.get("broker") is not None,
        ml_engine=ml_engine,
    )


@app.command()
def run_once(
    symbol: str = typer.Option("SPY", help="Ticker symbol to evaluate."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry-run mode (no actual execution)"),
) -> None:
    """Run a single pipeline iteration for ``symbol`` and print the results."""
    
    config = load_config()
    runner = build_pipeline(symbol, config)
    now = datetime.now(timezone.utc)
    result = runner.run_once(now)
    
    if dry_run:
        typer.echo("🔍 DRY-RUN MODE: No orders will be executed")

    typer.echo(format_pipeline_result(result))


@app.command()
def live_loop(
    symbol: str = typer.Option("SPY", help="Ticker symbol to trade."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry-run mode (no actual execution)"),
    interval: int = typer.Option(60, "--interval", help="Loop interval in seconds (default: 60)"),
) -> None:
    """
    Run autonomous trading loop with continuous execution.
    
    This command starts the autonomous trading system that:
    1. Runs the full pipeline every {interval} seconds
    2. Generates trade ideas based on current market conditions
    3. Executes trades on Alpaca Paper (if --dry-run not specified)
    4. Tracks predictions and learns from outcomes
    5. Self-optimizes based on performance
    
    Example:
        # Dry-run to preview without execution
        python main.py live-loop --symbol SPY --dry-run
        
        # Start live paper trading
        python main.py live-loop --symbol SPY
        
        # Custom interval (5 minutes)
        python main.py live-loop --symbol SPY --interval 300
    """
    config = load_config()
    paper_mode = get_alpaca_paper_setting()

    # Initialize broker (Alpaca Paper)
    broker = None
    if not dry_run:
        try:
            typer.echo(f"🔌 Connecting to Alpaca {'Paper' if paper_mode else 'Live'} Trading...")
            broker = create_broker_adapter(paper=paper_mode, prefer_real=True)
            account = broker.get_account()
            typer.echo(f"✅ Connected to Alpaca {'Paper' if paper_mode else 'Live'} Trading")
            typer.echo(f"   Account: {account.account_id}")
            typer.echo(f"   Cash: ${account.cash:,.2f}")
            typer.echo(f"   Buying Power: ${account.buying_power:,.2f}")
            typer.echo(f"   Portfolio Value: ${account.portfolio_value:,.2f}")
        except Exception as e:
            typer.echo(f"❌ Failed to connect to Alpaca: {e}", err=True)
            typer.echo("   Make sure ALPACA_API_KEY and ALPACA_SECRET_KEY are set in .env", err=True)
            raise typer.Exit(1)
    else:
        typer.echo("🔍 DRY-RUN MODE: No actual execution")
    
    # Build pipeline with broker
    adapters = {}
    if broker:
        adapters["broker"] = broker
    
    runner = build_pipeline(symbol, config, adapters)
    
    typer.echo("\n" + "="*80)
    typer.echo("🚀 AUTONOMOUS TRADING LOOP STARTED")
    typer.echo("="*80)
    typer.echo(f"   Symbol: {symbol}")
    mode_label = "DRY-RUN" if dry_run else ("PAPER TRADING" if paper_mode else "LIVE TRADING")
    typer.echo(f"   Mode: {mode_label}")
    typer.echo(f"   Interval: {interval} seconds")
    typer.echo(f"   Press Ctrl+C to stop")
    typer.echo("="*80 + "\n")
    
    iteration = 0
    try:
        while True:
            iteration += 1
            now = datetime.now(timezone.utc)
            
            typer.echo(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Iteration #{iteration}")
            typer.echo("-" * 80)
            
            try:
                # Run pipeline
                result = runner.run_once(now)
                
                # Report results
                if hasattr(result, 'trade_ideas'):
                    n_ideas = len(result.trade_ideas) if result.trade_ideas else 0
                    typer.echo(f"   ✓ Generated {n_ideas} trade ideas")
                    
                    # Show top trade idea
                    if n_ideas > 0 and not dry_run:
                        top_idea = result.trade_ideas[0]
                        typer.echo(f"   🎯 Top Idea: {top_idea.strategy_type.value if hasattr(top_idea, 'strategy_type') else 'N/A'}")
                        typer.echo(f"      Confidence: {top_idea.confidence:.2%}" if hasattr(top_idea, 'confidence') else "")
                
                if hasattr(result, 'order_results') and result.order_results:
                    n_orders = len(result.order_results)
                    typer.echo(f"   📊 Executed {n_orders} orders")
                    for order in result.order_results:
                        status = order.status.value if hasattr(order.status, 'value') else order.status
                        typer.echo(f"      {order.symbol}: {status}")
                
                # Check account if live
                if broker and not dry_run:
                    account = broker.get_account()
                    typer.echo(f"   💰 Portfolio: ${account.portfolio_value:,.2f} | Cash: ${account.cash:,.2f}")
                
            except Exception as e:
                typer.echo(f"   ❌ Error in iteration: {e}", err=True)
            
            # Wait for next iteration
            typer.echo(f"   ⏳ Next iteration in {interval} seconds...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        typer.echo("\n\n" + "="*80)
        typer.echo("🛑 AUTONOMOUS TRADING LOOP STOPPED")
        typer.echo("="*80)
        typer.echo(f"   Total Iterations: {iteration}")
        
        if broker and not dry_run:
            try:
                account = broker.get_account()
                typer.echo(f"\n   Final Portfolio Value: ${account.portfolio_value:,.2f}")
                typer.echo(f"   Final Cash: ${account.cash:,.2f}")
                positions = broker.get_positions()
                typer.echo(f"   Open Positions: {len(positions)}")
            except Exception as e:
                logger.warning(f"Could not fetch final account info: {e}")
        
        typer.echo("="*80)


@app.command()
def scan_opportunities(
    top_n: int = typer.Option(25, "--top", help="Number of top opportunities to return"),
    universe: str = typer.Option("default", "--universe", help="Symbol universe: 'default', 'sp500', 'nasdaq100', or comma-separated list"),
    min_score: float = typer.Option(0.5, "--min-score", help="Minimum opportunity score (0-1)"),
    output_file: str = typer.Option(None, "--output", help="Save results to file (JSON)"),
) -> None:
    """
    Scan multiple symbols and rank by trading opportunity quality.
    
    Uses DHPE engines to identify:
    - High energy asymmetry (directional opportunities)
    - Strong liquidity (tradeable)
    - Volatility expansion (breakout potential)
    - Sentiment conviction (directional bias)
    - Active options markets (liquid derivatives)
    
    Example:
        # Scan default universe for top 25 opportunities
        python main.py scan-opportunities
        
        # Top 10 with minimum score 0.6
        python main.py scan-opportunities --top 10 --min-score 0.6
        
        # Custom symbol list
        python main.py scan-opportunities --universe "SPY,QQQ,AAPL,TSLA,NVDA"
        
        # Save to file
        python main.py scan-opportunities --output opportunities.json
    """
    import json

    from engines.scanner import (
        DEFAULT_UNIVERSE,
        OpportunityScanner,
        get_dynamic_universe,
    )
    
    config = load_config()
    get_alpaca_paper_setting()

    # Determine universe
    if universe == "default":
        # Use dynamic top-N ranking system
        typer.echo("📊 Ranking universe using dynamic scanner...")
        symbol_list = get_dynamic_universe(config.scanner.model_dump(), config.scanner.default_top_n)
        typer.echo(f"✅ Selected top {len(symbol_list)} most active options underlyings")
    elif universe == "sp500":
        typer.echo("⚠️ SP500 universe not yet implemented, using default")
        symbol_list = DEFAULT_UNIVERSE
    elif universe == "nasdaq100":
        typer.echo("⚠️ NASDAQ100 universe not yet implemented, using default")
        symbol_list = DEFAULT_UNIVERSE
    else:
        # Custom comma-separated list
        symbol_list = [s.strip().upper() for s in universe.split(',')]
    
    typer.echo("\n" + "="*80)
    typer.echo("🔍 OPPORTUNITY SCANNER")
    typer.echo("="*80)
    typer.echo(f"   Universe: {len(symbol_list)} symbols")
    typer.echo(f"   Top N: {top_n}")
    typer.echo(f"   Min Score: {min_score}")
    typer.echo("="*80 + "\n")
    
    # Build engines for scanner
    typer.echo("Building engines...")
    options_adapter = create_options_adapter(prefer_real=True)
    market_adapter = create_market_data_adapter(prefer_real=True)
    news_adapter = create_news_adapter(prefer_real=False)
    
    hedge_engine = HedgeEngineV3(options_adapter, config.engines.hedge.model_dump())
    liquidity_engine = LiquidityEngineV1(market_adapter, config.engines.liquidity.model_dump())
    sentiment_engine = SentimentEngineV1(
        [
            NewsSentimentProcessor(news_adapter, config.engines.sentiment.model_dump()),
            FlowSentimentProcessor(config.engines.sentiment.model_dump()),
            TechnicalSentimentProcessor(market_adapter, config.engines.sentiment.model_dump()),
        ],
        config.engines.sentiment.model_dump(),
    )
    elasticity_engine = ElasticityEngineV1(market_adapter, config.engines.elasticity.model_dump())
    
    # Create scanner
    scanner = OpportunityScanner(
        hedge_engine=hedge_engine,
        liquidity_engine=liquidity_engine,
        sentiment_engine=sentiment_engine,
        elasticity_engine=elasticity_engine,
        options_adapter=options_adapter,
        market_adapter=market_adapter,
    )
    
    # Run scan
    typer.echo(f"Scanning {len(symbol_list)} symbols...")
    scan_result = scanner.scan(symbol_list, top_n=top_n)
    
    typer.echo(f"✓ Scan complete in {scan_result.scan_duration_seconds:.1f} seconds\n")
    
    # Filter by minimum score
    opportunities = [opp for opp in scan_result.opportunities if opp.score >= min_score]
    
    if not opportunities:
        typer.echo("❌ No opportunities found meeting criteria")
        return
    
    # Display results
    typer.echo("="*80)
    typer.echo(f"🎯 TOP {len(opportunities)} OPPORTUNITIES")
    typer.echo("="*80 + "\n")
    
    for opp in opportunities:
        typer.echo(f"#{opp.rank} {opp.symbol} - Score: {opp.score:.3f}")
        typer.echo(f"   Type: {opp.opportunity_type.upper()}")
        typer.echo(f"   Direction: {opp.direction.upper()} (confidence: {opp.confidence:.1%})")
        typer.echo(f"   Energy: {opp.energy_asymmetry:+.1f} | Movement: {opp.movement_energy:.0f}")
        typer.echo(f"   Liquidity: {opp.liquidity_score:.2f} | Options: {opp.options_score:.2f}")
        typer.echo(f"   {opp.reasoning}")
        typer.echo()
    
    # Summary
    typer.echo("="*80)
    typer.echo("📊 SUMMARY")
    typer.echo("="*80)
    
    by_type = {}
    by_direction = {}
    
    for opp in opportunities:
        by_type[opp.opportunity_type] = by_type.get(opp.opportunity_type, 0) + 1
        by_direction[opp.direction] = by_direction.get(opp.direction, 0) + 1
    
    typer.echo(f"By Type:")
    for opp_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
        typer.echo(f"   {opp_type}: {count}")
    
    typer.echo(f"\nBy Direction:")
    for direction, count in sorted(by_direction.items(), key=lambda x: x[1], reverse=True):
        typer.echo(f"   {direction}: {count}")
    
    # Save to file if requested
    if output_file:
        results_dict = {
            'scan_timestamp': scan_result.scan_timestamp.isoformat(),
            'symbols_scanned': scan_result.symbols_scanned,
            'duration_seconds': scan_result.scan_duration_seconds,
            'opportunities': [
                {
                    'rank': opp.rank,
                    'symbol': opp.symbol,
                    'score': opp.score,
                    'opportunity_type': opp.opportunity_type,
                    'direction': opp.direction,
                    'confidence': opp.confidence,
                    'energy_asymmetry': opp.energy_asymmetry,
                    'movement_energy': opp.movement_energy,
                    'liquidity_score': opp.liquidity_score,
                    'reasoning': opp.reasoning,
                }
                for opp in opportunities
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        typer.echo(f"\n✓ Results saved to {output_file}")
    
    typer.echo("\n" + "="*80)


@app.command()
def multi_symbol_loop(
    top_n: int = typer.Option(5, "--top", help="Number of top symbols to trade simultaneously"),
    scan_interval: int = typer.Option(300, "--scan-interval", help="Seconds between universe scans (default: 300 = 5 min)"),
    trade_interval: int = typer.Option(60, "--trade-interval", help="Seconds between trades per symbol (default: 60)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry-run mode (no actual execution)"),
    universe: str = typer.Option("default", "--universe", help="Symbol universe to scan"),
) -> None:
    """
    Run autonomous trading on multiple symbols simultaneously.
    
    This command:
    1. Scans the universe every {scan_interval} seconds
    2. Identifies top N opportunities
    3. Trades the top N symbols in rotation
    4. Re-scans periodically to adapt to changing conditions
    
    Example:
        # Trade top 5 opportunities, re-scan every 5 minutes
        python main.py multi-symbol-loop
        
        # Top 10, scan every 10 minutes
        python main.py multi-symbol-loop --top 10 --scan-interval 600
        
        # Dry-run mode
        python main.py multi-symbol-loop --dry-run
    """
    from engines.scanner import (
        OpportunityScanner,
        get_dynamic_universe,
    )
    
    config = load_config()
    paper_mode = get_alpaca_paper_setting()

    # Determine universe
    if universe == "default":
        # Use dynamic top-N ranking system
        typer.echo("📊 Ranking universe using dynamic scanner...")
        symbol_list = get_dynamic_universe(config.scanner.model_dump(), top_n)
        typer.echo(f"✅ Selected top {len(symbol_list)} most active options underlyings")
    else:
        symbol_list = [s.strip().upper() for s in universe.split(',')]
    
    # Initialize broker
    broker = None
    if not dry_run:
        try:
            typer.echo(f"🔌 Connecting to Alpaca {'Paper' if paper_mode else 'Live'} Trading...")
            broker = create_broker_adapter(paper=paper_mode, prefer_real=True)
            account = broker.get_account()
            typer.echo(f"✅ Connected to Alpaca {'Paper' if paper_mode else 'Live'} Trading")
            typer.echo(f"   Account: {account.account_id}")
            typer.echo(f"   Portfolio: ${account.portfolio_value:,.2f}")
        except Exception as e:
            typer.echo(f"❌ Failed to connect to Alpaca: {e}", err=True)
            raise typer.Exit(1)
    
    typer.echo("\n" + "="*80)
    typer.echo("🚀 MULTI-SYMBOL AUTONOMOUS TRADING")
    typer.echo("="*80)
    typer.echo(f"   Universe: {len(symbol_list)} symbols")
    typer.echo(f"   Top N: {top_n}")
    typer.echo(f"   Scan Interval: {scan_interval} seconds")
    typer.echo(f"   Trade Interval: {trade_interval} seconds")
    mode_label = "DRY-RUN" if dry_run else ("PAPER TRADING" if paper_mode else "LIVE TRADING")
    typer.echo(f"   Mode: {mode_label}")
    typer.echo(f"   Press Ctrl+C to stop")
    typer.echo("="*80 + "\n")
    
    # Build scanner
    options_adapter = create_options_adapter(prefer_real=True)
    market_adapter = create_market_data_adapter(prefer_real=True)
    news_adapter = create_news_adapter(prefer_real=False)
    
    hedge_engine = HedgeEngineV3(options_adapter, config.engines.hedge.model_dump())
    liquidity_engine = LiquidityEngineV1(market_adapter, config.engines.liquidity.model_dump())
    sentiment_engine = SentimentEngineV1(
        [
            NewsSentimentProcessor(news_adapter, config.engines.sentiment.model_dump()),
            FlowSentimentProcessor(config.engines.sentiment.model_dump()),
            TechnicalSentimentProcessor(market_adapter, config.engines.sentiment.model_dump()),
        ],
        config.engines.sentiment.model_dump(),
    )
    elasticity_engine = ElasticityEngineV1(market_adapter, config.engines.elasticity.model_dump())
    
    scanner = OpportunityScanner(
        hedge_engine=hedge_engine,
        liquidity_engine=liquidity_engine,
        sentiment_engine=sentiment_engine,
        elasticity_engine=elasticity_engine,
        options_adapter=options_adapter,
        market_adapter=market_adapter,
    )
    
    # Trading state
    active_symbols = []
    last_scan_time = 0
    iteration = 0
    
    try:
        while True:
            iteration += 1
            now = datetime.now(timezone.utc)
            current_time = time.time()
            
            typer.echo(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Iteration #{iteration}")
            typer.echo("-" * 80)
            
            # Re-scan universe if interval elapsed
            if current_time - last_scan_time >= scan_interval:
                typer.echo("🔍 Scanning universe for opportunities...")
                scan_result = scanner.scan(symbol_list, top_n=top_n)
                active_symbols = [opp.symbol for opp in scan_result.opportunities[:top_n]]
                last_scan_time = current_time
                
                typer.echo(f"✓ Top {len(active_symbols)} opportunities:")
                for i, sym in enumerate(active_symbols, 1):
                    opp = scan_result.opportunities[i-1]
                    typer.echo(f"   {i}. {sym} - {opp.opportunity_type} ({opp.score:.3f})")
                typer.echo()
            
            # Trade each active symbol
            if active_symbols:
                for symbol in active_symbols:
                    try:
                        typer.echo(f"📊 Trading {symbol}...")
                        
                        # Build pipeline for this symbol
                        adapters = {}
                        if broker:
                            adapters["broker"] = broker
                        
                        runner = build_pipeline(symbol, config, adapters)
                        result = runner.run_once(now)
                        
                        # Report
                        if hasattr(result, 'trade_ideas'):
                            n_ideas = len(result.trade_ideas) if result.trade_ideas else 0
                            if n_ideas > 0:
                                typer.echo(f"   ✓ {symbol}: {n_ideas} trade ideas generated")
                        
                        if hasattr(result, 'order_results') and result.order_results:
                            n_orders = len(result.order_results)
                            typer.echo(f"   📈 {symbol}: {n_orders} orders executed")
                    
                    except Exception as e:
                        typer.echo(f"   ❌ {symbol}: Error - {e}")
                
                # Show portfolio if live
                if broker and not dry_run:
                    account = broker.get_account()
                    positions = broker.get_positions()
                    typer.echo(f"\n   💰 Portfolio: ${account.portfolio_value:,.2f} | Positions: {len(positions)}")
            
            # Wait for next iteration
            typer.echo(f"   ⏳ Next iteration in {trade_interval} seconds...")
            time.sleep(trade_interval)
    
    except KeyboardInterrupt:
        typer.echo("\n\n" + "="*80)
        typer.echo("🛑 MULTI-SYMBOL TRADING STOPPED")
        typer.echo("="*80)
        typer.echo(f"   Total Iterations: {iteration}")
        
        if broker and not dry_run:
            try:
                account = broker.get_account()
                positions = broker.get_positions()
                typer.echo(f"\n   Final Portfolio: ${account.portfolio_value:,.2f}")
                typer.echo(f"   Open Positions: {len(positions)}")
                if positions:
                    for pos in positions:
                        typer.echo(f"      {pos.symbol}: {pos.quantity} @ ${pos.avg_entry_price:.2f} | P&L: ${pos.unrealized_pnl:+,.2f}")
            except Exception as e:
                logger.warning(f"Could not fetch final portfolio info: {e}")
        
    typer.echo("="*80)


@app.command("download-data")
def cli_download_data(
    symbols: str = typer.Option("SPY,QQQ", help="Comma separated symbols"),
    start: str = typer.Option("2020-01-01", help="Start date for downloads"),
    end: str = typer.Option(datetime.utcnow().strftime("%Y-%m-%d"), help="End date"),
    cache_dir: Path = typer.Option(Path("data/historical"), help="Cache root"),
) -> None:
    """Download historical data using Massive.com REST client."""

    from scripts.download_historical import download_bars, download_options

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    symbols_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    for sym in symbols_list:
        download_bars(sym, start_dt, end_dt, cache_dir)
        download_options(sym, cache_dir)


@app.command("backtest-full")
def backtest_full(
    symbol: str = typer.Option("SPY", help="Symbol to backtest"),
    start: str = typer.Option("2020-01-01", help="Start date"),
    end: str = typer.Option(datetime.utcnow().strftime("%Y-%m-%d"), help="End date"),
) -> None:
    """Run full backtest across all engines/agents using cached or live data."""

    config = load_config()
    adapters: Dict[str, object] = {}
    runner = build_pipeline(symbol, config, adapters)

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    day = start_dt
    while day <= end_dt:
        ts = datetime.combine(day.date(), datetime.min.time()).replace(tzinfo=timezone.utc)
        typer.echo(f"Running pipeline for {symbol} on {ts.date()}")
        runner.run_once(ts)
        day += timedelta(days=1)

    typer.echo("Backtest complete; results appended to ledger")


@app.command("train-ml")
def cli_train_ml(
    ledger: Path = typer.Option(Path("data/ledger.jsonl"), help="Ledger path"),
    model_path: Path = typer.Option(Path("data/models/lookahead.pkl"), help="Where to save the model"),
) -> None:
    """Train lookahead model from ledger and persist artifact."""

    builder = EnhancedFeatureBuilder(FeatureConfig())
    df = builder.build_from_ledger(ledger)
    if df.empty:
        typer.echo("Ledger is empty; nothing to train")
        raise typer.Exit(1)

    model = LookaheadModel()
    train_score, test_score = model.train(df)
    model.save(model_path)
    typer.echo(f"Saved model to {model_path} (train={train_score:.3f}, test={test_score:.3f})")


if __name__ == "__main__":
    app()
