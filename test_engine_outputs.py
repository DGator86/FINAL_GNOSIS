#!/usr/bin/env python3
"""
Interactive test script to inspect outputs from each engine and agent.

Usage:
    python test_engine_outputs.py                    # Test all with stub data
    python test_engine_outputs.py --real             # Test with real API data
    python test_engine_outputs.py --symbol NVDA      # Test specific symbol
    python test_engine_outputs.py --engine hedge     # Test specific engine only
    python test_engine_outputs.py --agent hedge      # Test specific agent only
    python test_engine_outputs.py --full-pipeline    # Run complete pipeline
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_snapshot(name: str, snapshot: Any) -> None:
    """Pretty print a snapshot object."""
    print(f"\n{name}:")
    print("-" * 40)
    if hasattr(snapshot, "model_dump"):
        data = snapshot.model_dump()
    elif hasattr(snapshot, "__dict__"):
        data = snapshot.__dict__
    else:
        data = str(snapshot)

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            elif isinstance(value, datetime):
                print(f"  {key}: {value.isoformat()}")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"  {data}")


def create_stub_adapters():
    """Create stub adapters for testing without API calls."""
    from engines.inputs.stub_adapters import (
        StaticMarketDataAdapter,
        StaticNewsAdapter,
        StaticOptionsAdapter,
    )

    return {
        "options": StaticOptionsAdapter(),
        "market": StaticMarketDataAdapter(),
        "news": StaticNewsAdapter(),
    }


def create_real_adapters():
    """Create real adapters that make API calls."""
    from engines.inputs.adapter_factory import (
        create_market_data_adapter,
        create_news_adapter,
        create_options_adapter,
    )

    return {
        "options": create_options_adapter(prefer_real=True),
        "market": create_market_data_adapter(prefer_real=True),
        "news": create_news_adapter(prefer_real=False),  # News usually stub
    }


def test_hedge_engine(adapters: Dict[str, Any], symbol: str, timestamp: datetime) -> Any:
    """Test Hedge Engine v3 and return snapshot."""
    from engines.hedge.hedge_engine_v3 import HedgeEngineV3

    print_section("HEDGE ENGINE V3")

    config = {
        "enabled": True,
        "lookback_days": 30,
        "min_dte": 7,
        "max_dte": 60,
    }

    engine = HedgeEngineV3(adapters["options"], config)
    snapshot = engine.run(symbol, timestamp)

    print_snapshot("HedgeSnapshot", snapshot)

    # Interpretation
    print("\n📊 Interpretation:")
    if snapshot.energy_asymmetry > 0.3:
        print("  → Bullish bias (call pressure dominant)")
    elif snapshot.energy_asymmetry < -0.3:
        print("  → Bearish bias (put pressure dominant)")
    else:
        print("  → Neutral bias")

    print(f"  → Regime: {snapshot.regime}")
    print(f"  → Movement Energy: {snapshot.movement_energy:.2f}")
    print(f"  → Elasticity (market stiffness): {snapshot.elasticity:.2f}")

    return snapshot


def test_liquidity_engine(adapters: Dict[str, Any], symbol: str, timestamp: datetime) -> Any:
    """Test Liquidity Engine and return snapshot."""
    from engines.liquidity.liquidity_engine_v1 import LiquidityEngineV1

    print_section("LIQUIDITY ENGINE V1")

    config = {
        "enabled": True,
        "lookback_days": 5,
        "volume_threshold": 1_000_000,
    }

    engine = LiquidityEngineV1(adapters["market"], config)
    snapshot = engine.run(symbol, timestamp)

    print_snapshot("LiquiditySnapshot", snapshot)

    # Interpretation
    print("\n📊 Interpretation:")
    if snapshot.liquidity_score > 0.7:
        print("  → High liquidity - good for trading")
    elif snapshot.liquidity_score > 0.4:
        print("  → Moderate liquidity - acceptable")
    else:
        print("  → Low liquidity - caution advised")

    return snapshot


def test_sentiment_engine(adapters: Dict[str, Any], symbol: str, timestamp: datetime) -> Any:
    """Test Sentiment Engine and return snapshot."""
    from engines.sentiment.processors import (
        FlowSentimentProcessor,
        NewsSentimentProcessor,
        TechnicalSentimentProcessor,
    )
    from engines.sentiment.sentiment_engine_v1 import SentimentEngineV1

    print_section("SENTIMENT ENGINE V1")

    config = {
        "enabled": True,
        "news_weight": 0.4,
        "flow_weight": 0.3,
        "technical_weight": 0.3,
    }

    processors = [
        NewsSentimentProcessor(adapters["news"], config),
        FlowSentimentProcessor(config),
        TechnicalSentimentProcessor(adapters["market"], config),
    ]

    engine = SentimentEngineV1(processors, config)
    snapshot = engine.run(symbol, timestamp)

    print_snapshot("SentimentSnapshot", snapshot)

    # Interpretation
    print("\n📊 Interpretation:")
    if snapshot.sentiment_score > 0.3:
        print("  → Bullish sentiment")
    elif snapshot.sentiment_score < -0.3:
        print("  → Bearish sentiment")
    else:
        print("  → Neutral sentiment")

    return snapshot


def test_elasticity_engine(adapters: Dict[str, Any], symbol: str, timestamp: datetime) -> Any:
    """Test Elasticity Engine and return snapshot."""
    from engines.elasticity.elasticity_engine_v1 import ElasticityEngineV1

    print_section("ELASTICITY ENGINE V1")

    config = {
        "enabled": True,
        "volatility_lookback": 20,
    }

    engine = ElasticityEngineV1(adapters["market"], config)
    snapshot = engine.run(symbol, timestamp)

    print_snapshot("ElasticitySnapshot", snapshot)

    # Interpretation
    print("\n📊 Interpretation:")
    print(f"  → Volatility Regime: {snapshot.volatility_regime}")
    if snapshot.volatility > 0.3:
        print("  → High volatility - breakout strategies favored")
    elif snapshot.volatility < 0.15:
        print("  → Low volatility - mean reversion favored")
    else:
        print("  → Moderate volatility")

    return snapshot


def test_hedge_agent(hedge_snapshot: Any, timestamp: datetime) -> Any:
    """Test Hedge Agent v3 and return suggestion."""
    from agents.hedge_agent_v3 import HedgeAgentV3
    from schemas.core_schemas import PipelineResult

    print_section("HEDGE AGENT V3")

    config = {
        "enabled": True,
        "confidence_threshold": 0.3,
        "energy_threshold": 0.5,
    }

    agent = HedgeAgentV3(config)

    # Create minimal pipeline result
    pipeline_result = PipelineResult(
        timestamp=timestamp,
        symbol=hedge_snapshot.symbol,
        hedge_snapshot=hedge_snapshot,
    )

    suggestion = agent.suggest(pipeline_result, timestamp)

    if suggestion:
        print_snapshot("AgentSuggestion", suggestion)
        print("\n📊 Interpretation:")
        print(f"  → Recommended direction: {suggestion.direction.value}")
        print(f"  → Confidence: {suggestion.confidence:.2%}")
        print(f"  → Reasoning: {suggestion.reasoning}")
    else:
        print("  No suggestion generated (below threshold)")

    return suggestion


def test_liquidity_agent(liquidity_snapshot: Any, timestamp: datetime) -> Any:
    """Test Liquidity Agent and return suggestion."""
    from agents.liquidity_agent_v1 import LiquidityAgentV1
    from schemas.core_schemas import PipelineResult

    print_section("LIQUIDITY AGENT V1")

    config = {
        "enabled": True,
        "min_liquidity_score": 0.3,
    }

    agent = LiquidityAgentV1(config)

    pipeline_result = PipelineResult(
        timestamp=timestamp,
        symbol=liquidity_snapshot.symbol,
        liquidity_snapshot=liquidity_snapshot,
    )

    suggestion = agent.suggest(pipeline_result, timestamp)

    if suggestion:
        print_snapshot("AgentSuggestion", suggestion)
        print("\n📊 Interpretation:")
        print(f"  → Recommended direction: {suggestion.direction.value}")
        print(f"  → Confidence: {suggestion.confidence:.2%}")
    else:
        print("  No suggestion generated")

    return suggestion


def test_sentiment_agent(sentiment_snapshot: Any, timestamp: datetime) -> Any:
    """Test Sentiment Agent and return suggestion."""
    from agents.sentiment_agent_v1 import SentimentAgentV1
    from schemas.core_schemas import PipelineResult

    print_section("SENTIMENT AGENT V1")

    config = {
        "enabled": True,
        "sentiment_threshold": 0.2,
    }

    agent = SentimentAgentV1(config)

    pipeline_result = PipelineResult(
        timestamp=timestamp,
        symbol=sentiment_snapshot.symbol,
        sentiment_snapshot=sentiment_snapshot,
    )

    suggestion = agent.suggest(pipeline_result, timestamp)

    if suggestion:
        print_snapshot("AgentSuggestion", suggestion)
        print("\n📊 Interpretation:")
        print(f"  → Recommended direction: {suggestion.direction.value}")
        print(f"  → Confidence: {suggestion.confidence:.2%}")
    else:
        print("  No suggestion generated")

    return suggestion


def test_composer_agent(suggestions: list, timestamp: datetime) -> Any:
    """Test Composer Agent and return consensus."""
    from agents.composer.composer_agent_v1 import ComposerAgentV1

    print_section("COMPOSER AGENT V1")

    weights = {
        "hedge": 0.4,
        "liquidity": 0.2,
        "sentiment": 0.4,
    }
    config = {"enabled": True}

    composer = ComposerAgentV1(weights, config)

    valid_suggestions = [s for s in suggestions if s is not None]

    if not valid_suggestions:
        print("  No suggestions to compose")
        return None

    consensus = composer.compose(valid_suggestions, timestamp)

    print("\n📊 Consensus Result:")
    print("-" * 40)
    for key, value in consensus.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n📊 Interpretation:")
    direction = consensus.get("direction", "neutral")
    confidence = consensus.get("confidence", 0)

    if direction == "long" and confidence > 0.5:
        print("  → STRONG BUY SIGNAL")
    elif direction == "long":
        print("  → Weak bullish signal")
    elif direction == "short" and confidence > 0.5:
        print("  → STRONG SELL SIGNAL")
    elif direction == "short":
        print("  → Weak bearish signal")
    else:
        print("  → NEUTRAL - No clear direction")

    return consensus


def test_trade_agent(consensus: Dict[str, Any], symbol: str, timestamp: datetime, adapters: Dict[str, Any]) -> Any:
    """Test Trade Agent and return trade ideas."""
    from schemas.core_schemas import PipelineResult
    from trade.trade_agent_v1 import TradeAgentV1

    print_section("TRADE AGENT V1")

    config = {
        "enabled": True,
        "max_position_size": 10000.0,
        "risk_per_trade": 0.02,
        "confidence_threshold": 0.1,
    }

    trade_agent = TradeAgentV1(
        options_adapter=adapters["options"],
        market_adapter=adapters["market"],
        config=config,
        broker=None,  # No execution
    )

    pipeline_result = PipelineResult(
        timestamp=timestamp,
        symbol=symbol,
        consensus=consensus,
    )

    trade_ideas = trade_agent.generate_ideas(pipeline_result, timestamp)

    if trade_ideas:
        for i, idea in enumerate(trade_ideas, 1):
            print(f"\n📈 Trade Idea #{i}:")
            print("-" * 40)
            print_snapshot("TradeIdea", idea)

            print("\n📊 Position Details:")
            print(f"  → Size: ${idea.size:,.2f}")
            print(f"  → Strategy: {idea.strategy_type.value}")
            print(f"  → Direction: {idea.direction.value}")
            print(f"  → Confidence: {idea.confidence:.2%}")
    else:
        print("  No trade ideas generated (consensus below threshold)")

    return trade_ideas


def test_full_pipeline(symbol: str, use_real: bool = False):
    """Run the full pipeline and show all outputs."""
    from config import load_config
    from main import build_pipeline

    print_section("FULL PIPELINE TEST")
    print(f"Symbol: {symbol}")
    print(f"Data Source: {'Real APIs' if use_real else 'Stub Data'}")

    config = load_config()

    adapters = {}
    if not use_real:
        # Use stubs
        from engines.inputs.stub_adapters import (
            StaticMarketDataAdapter,
            StaticNewsAdapter,
            StaticOptionsAdapter,
        )
        adapters["options"] = StaticOptionsAdapter()
        adapters["market"] = StaticMarketDataAdapter()
        adapters["news"] = StaticNewsAdapter()

    runner = build_pipeline(symbol, config, adapters)
    timestamp = datetime.now(timezone.utc)

    result = runner.run_once(timestamp)

    # Print all snapshots
    if result.hedge_snapshot:
        print_snapshot("HedgeSnapshot", result.hedge_snapshot)

    if result.liquidity_snapshot:
        print_snapshot("LiquiditySnapshot", result.liquidity_snapshot)

    if result.sentiment_snapshot:
        print_snapshot("SentimentSnapshot", result.sentiment_snapshot)

    if result.elasticity_snapshot:
        print_snapshot("ElasticitySnapshot", result.elasticity_snapshot)

    print_section("AGENT SUGGESTIONS")
    for suggestion in result.suggestions:
        print_snapshot(f"Suggestion from {suggestion.agent_name}", suggestion)

    print_section("CONSENSUS")
    if result.consensus:
        for key, value in result.consensus.items():
            print(f"  {key}: {value}")

    print_section("TRADE IDEAS")
    if result.trade_ideas:
        for idea in result.trade_ideas:
            print_snapshot("TradeIdea", idea)
    else:
        print("  No trade ideas generated")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test engine and agent outputs")
    parser.add_argument("--symbol", default="SPY", help="Symbol to test (default: SPY)")
    parser.add_argument("--real", action="store_true", help="Use real API data instead of stubs")
    parser.add_argument("--engine", choices=["hedge", "liquidity", "sentiment", "elasticity", "all"],
                       default="all", help="Test specific engine")
    parser.add_argument("--agent", choices=["hedge", "liquidity", "sentiment", "composer", "trade", "all"],
                       default="all", help="Test specific agent")
    parser.add_argument("--full-pipeline", action="store_true", help="Run complete pipeline")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if not args.verbose:
        logger.disable("engines")
        logger.disable("agents")
        logger.disable("trade")

    symbol = args.symbol.upper()
    timestamp = datetime.now(timezone.utc)

    print("\n" + "=" * 80)
    print("  GNOSIS ENGINE & AGENT OUTPUT TESTER")
    print("=" * 80)
    print(f"  Symbol: {symbol}")
    print(f"  Timestamp: {timestamp.isoformat()}")
    print(f"  Data Source: {'Real APIs' if args.real else 'Stub Data (deterministic)'}")
    print("=" * 80)

    if args.full_pipeline:
        test_full_pipeline(symbol, use_real=args.real)
        return

    # Create adapters
    adapters = create_real_adapters() if args.real else create_stub_adapters()

    # Test engines
    hedge_snapshot = None
    liquidity_snapshot = None
    sentiment_snapshot = None
    elasticity_snapshot = None

    if args.engine in ["hedge", "all"]:
        hedge_snapshot = test_hedge_engine(adapters, symbol, timestamp)

    if args.engine in ["liquidity", "all"]:
        liquidity_snapshot = test_liquidity_engine(adapters, symbol, timestamp)

    if args.engine in ["sentiment", "all"]:
        sentiment_snapshot = test_sentiment_engine(adapters, symbol, timestamp)

    if args.engine in ["elasticity", "all"]:
        elasticity_snapshot = test_elasticity_engine(adapters, symbol, timestamp)

    # Test agents (need engine outputs)
    suggestions = []

    if args.agent in ["hedge", "all"] and hedge_snapshot:
        suggestion = test_hedge_agent(hedge_snapshot, timestamp)
        if suggestion:
            suggestions.append(suggestion)

    if args.agent in ["liquidity", "all"] and liquidity_snapshot:
        suggestion = test_liquidity_agent(liquidity_snapshot, timestamp)
        if suggestion:
            suggestions.append(suggestion)

    if args.agent in ["sentiment", "all"] and sentiment_snapshot:
        suggestion = test_sentiment_agent(sentiment_snapshot, timestamp)
        if suggestion:
            suggestions.append(suggestion)

    # Composer and Trade Agent
    consensus = None
    if args.agent in ["composer", "all"] and suggestions:
        consensus = test_composer_agent(suggestions, timestamp)

    if args.agent in ["trade", "all"] and consensus:
        test_trade_agent(consensus, symbol, timestamp, adapters)

    print_section("TEST COMPLETE")
    print(f"  Symbol tested: {symbol}")
    print(f"  Engines run: {args.engine}")
    print(f"  Agents run: {args.agent}")
    print(f"  Suggestions generated: {len(suggestions)}")


if __name__ == "__main__":
    main()
