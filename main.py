from __future__ import annotations

"""Command line entrypoint for Super Gnosis pipeline."""

import time
from datetime import datetime, timezone
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
from engines.inputs.market_data_adapter import MarketDataAdapter
from engines.inputs.news_adapter import NewsAdapter
from engines.inputs.options_chain_adapter import OptionsChainAdapter
from engines.inputs.stub_adapters import (
    StaticMarketDataAdapter,
    StaticNewsAdapter,
    StaticOptionsAdapter,
)
from engines.liquidity.liquidity_engine_v1 import LiquidityEngineV1
from engines.orchestration.pipeline_runner import PipelineRunner
from engines.sentiment.processors import FlowSentimentProcessor, NewsSentimentProcessor, TechnicalSentimentProcessor
from engines.sentiment.sentiment_engine_v1 import SentimentEngineV1
from ledger.ledger_store import LedgerStore
from trade.trade_agent_v1 import TradeAgentV1

# Load environment variables
load_dotenv()

app = typer.Typer(help="Super Gnosis / DHPE Pipeline CLI")


def build_pipeline(
    symbol: str,
    config: AppConfig,
    adapters: Optional[Dict[str, object]] = None,
) -> PipelineRunner:
    """Assemble a :class:`PipelineRunner` for ``symbol`` using ``config``."""

    adapters = adapters or {}
    options_adapter: OptionsChainAdapter = adapters.get("options") or StaticOptionsAdapter()
    market_adapter: MarketDataAdapter = adapters.get("market") or StaticMarketDataAdapter()
    news_adapter: NewsAdapter = adapters.get("news") or StaticNewsAdapter()

    engines = {
        "hedge": HedgeEngineV3(options_adapter, config.engines.hedge.model_dump()),
        "liquidity": LiquidityEngineV1(market_adapter, config.engines.liquidity.model_dump()),
        "sentiment": SentimentEngineV1(
            [
                NewsSentimentProcessor(news_adapter, config.engines.sentiment.model_dump()),
                FlowSentimentProcessor(config.engines.sentiment.model_dump()),
                TechnicalSentimentProcessor(market_adapter, config.engines.sentiment.model_dump()),
            ],
            config.engines.sentiment.model_dump(),
        ),
        "elasticity": ElasticityEngineV1(market_adapter, config.engines.elasticity.model_dump()),
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
    trade_agent = TradeAgentV1(options_adapter, config.agents.trade.model_dump())

    ledger_path = Path(config.tracking.ledger_path)
    ledger_store = LedgerStore(ledger_path)

    return PipelineRunner(
        symbol=symbol,
        engines=engines,
        primary_agents=primary_agents,
        composer=composer,
        trade_agent=trade_agent,
        ledger_store=ledger_store,
        config=config.model_dump(),
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
    
    typer.echo(result)


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
    from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
    
    config = load_config()
    
    # Initialize broker (Alpaca Paper)
    broker = None
    if not dry_run:
        try:
            typer.echo("🔌 Connecting to Alpaca Paper Trading...")
            broker = AlpacaBrokerAdapter(paper=True)
            account = broker.get_account()
            typer.echo(f"✅ Connected to Alpaca Paper Trading")
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
    typer.echo(f"   Mode: {'DRY-RUN' if dry_run else 'LIVE PAPER TRADING'}")
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
            except:
                pass
        
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
    from engines.scanner import OpportunityScanner, DEFAULT_UNIVERSE
    import json
    
    config = load_config()
    
    # Determine universe
    if universe == "default":
        symbol_list = DEFAULT_UNIVERSE
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
    options_adapter = StaticOptionsAdapter()
    market_adapter = StaticMarketDataAdapter()
    news_adapter = StaticNewsAdapter()
    
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
    from engines.scanner import OpportunityScanner, DEFAULT_UNIVERSE
    from execution.broker_adapters.alpaca_adapter import AlpacaBrokerAdapter
    
    config = load_config()
    
    # Determine universe
    if universe == "default":
        symbol_list = DEFAULT_UNIVERSE
    else:
        symbol_list = [s.strip().upper() for s in universe.split(',')]
    
    # Initialize broker
    broker = None
    if not dry_run:
        try:
            typer.echo("🔌 Connecting to Alpaca Paper Trading...")
            broker = AlpacaBrokerAdapter(paper=True)
            account = broker.get_account()
            typer.echo(f"✅ Connected to Alpaca Paper Trading")
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
    typer.echo(f"   Mode: {'DRY-RUN' if dry_run else 'LIVE PAPER TRADING'}")
    typer.echo(f"   Press Ctrl+C to stop")
    typer.echo("="*80 + "\n")
    
    # Build scanner
    options_adapter = StaticOptionsAdapter()
    market_adapter = StaticMarketDataAdapter()
    news_adapter = StaticNewsAdapter()
    
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
            except:
                pass
        
        typer.echo("="*80)


if __name__ == "__main__":
    app()
