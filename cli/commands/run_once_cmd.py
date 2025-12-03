"""Run-once command for single pipeline execution."""

from datetime import datetime, timezone

import typer
from loguru import logger

from cli.pipeline_builder import build_pipeline
from config import load_config
from engines.inputs.adapter_factory import create_broker_adapter


def run_once(
    symbol: str = typer.Option("SPY", help="Ticker symbol to evaluate."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Dry-run mode (no actual execution)"),
) -> None:
    """Run a single pipeline iteration for symbol and print the results."""
    typer.echo("=" * 80)
    typer.echo(f"🔍 SINGLE RUN: {symbol}")
    typer.echo("=" * 80)

    # Load config
    config = load_config()

    # Create broker adapter if not dry-run
    adapters = {}
    if not dry_run:
        try:
            broker = create_broker_adapter(prefer_real=True)
            adapters["broker"] = broker
            typer.echo(f"✅ Connected to broker (paper trading)")
        except Exception as e:
            logger.warning(f"Could not connect to broker: {e}")
            typer.echo(f"⚠️  Running in dry-run mode (broker unavailable)")

    # Build pipeline
    pipeline = build_pipeline(symbol, config, adapters)

    # Run once
    timestamp = datetime.now(timezone.utc)
    result = pipeline.run_once(timestamp)

    # Display results
    typer.echo("\n📊 RESULTS")
    typer.echo("-" * 80)

    if result.hedge_snapshot:
        typer.echo(f"\n🏛️  Hedge Engine:")
        typer.echo(f"   Elasticity: {result.hedge_snapshot.elasticity:.2f}")
        typer.echo(f"   Movement Energy: {result.hedge_snapshot.movement_energy:.2f}")
        typer.echo(f"   Energy Asymmetry: {result.hedge_snapshot.energy_asymmetry:+.2f}")
        typer.echo(f"   Regime: {result.hedge_snapshot.regime}")

    if result.liquidity_snapshot:
        typer.echo(f"\n💧 Liquidity Engine:")
        typer.echo(f"   Score: {result.liquidity_snapshot.liquidity_score:.2f}")
        typer.echo(f"   Spread: {result.liquidity_snapshot.bid_ask_spread:.4f}")

    if result.sentiment_snapshot:
        typer.echo(f"\n📰 Sentiment Engine:")
        typer.echo(f"   Score: {result.sentiment_snapshot.sentiment_score:+.2f}")
        typer.echo(f"   Confidence: {result.sentiment_snapshot.confidence:.2f}")

    if result.composer_decision:
        typer.echo(f"\n🎯 Composer Decision:")
        typer.echo(f"   Signal: {result.composer_decision.signal.upper()}")
        typer.echo(f"   Confidence: {result.composer_decision.confidence:.2f}")
        typer.echo(f"   Reasoning: {result.composer_decision.reasoning}")

    if result.trade_idea:
        typer.echo(f"\n💡 Trade Idea:")
        typer.echo(f"   Strategy: {result.trade_idea.get('strategy', 'N/A')}")
        typer.echo(f"   Direction: {result.trade_idea.get('direction', 'N/A')}")
        typer.echo(f"   Size: {result.trade_idea.get('size', 'N/A')}")

    typer.echo("\n" + "=" * 80)
