"""Pipeline builder for Super Gnosis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from agents.composer.composer_agent_v1 import ComposerAgentV1
from agents.hedge_agent_v3 import HedgeAgentV3
from agents.liquidity_agent_v1 import LiquidityAgentV1
from agents.sentiment_agent_v1 import SentimentAgentV1
from config import AppConfig
from engines.elasticity.elasticity_engine_v1 import ElasticityEngineV1
from engines.hedge.hedge_engine_v3 import HedgeEngineV3
from engines.inputs.adapter_factory import (
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
from feedback.adaptation_agent import AdaptationAgent
from feedback.tracking_agent import TrackingAgent
from ledger.ledger_store import LedgerStore
from trade.trade_agent_v1 import TradeAgentV1
from universe.watchlist_loader import load_active_watchlist
from watchlist import AdaptiveWatchlist


def build_pipeline(
    symbol: str,
    config: AppConfig,
    adapters: Optional[Dict[str, object]] = None,
    watchlist: Optional[AdaptiveWatchlist] = None,
) -> PipelineRunner:
    """
    Assemble a PipelineRunner for symbol using config.

    Args:
        symbol: Trading symbol
        config: Application configuration
        adapters: Optional pre-configured adapters (options, market, news, broker)
        watchlist: Optional adaptive watchlist

    Returns:
        Configured PipelineRunner instance
    """
    adapters = adapters or {}

    # Use adapter factory for automatic fallback
    # Try real adapters first, fall back to stubs if they fail
    options_adapter: OptionsChainAdapter = adapters.get("options") or create_options_adapter(
        prefer_real=True
    )
    market_adapter: MarketDataAdapter = adapters.get("market") or create_market_data_adapter(
        prefer_real=True
    )
    news_adapter: NewsAdapter = adapters.get("news") or create_news_adapter(prefer_real=False)

    # Build engines
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

    # Build primary agents
    primary_agents = {
        "primary_hedge": HedgeAgentV3(config.agents.hedge.model_dump()),
        "primary_liquidity": LiquidityAgentV1(config.agents.liquidity.model_dump()),
        "primary_sentiment": SentimentAgentV1(config.agents.sentiment.model_dump()),
    }

    # Build composer
    composer = ComposerAgentV1(
        config.agents.composer.weights,
        config.agents.composer.model_dump(),
    )

    # Build trade agent
    trade_agent = TradeAgentV1(
        options_adapter,
        market_adapter,
        config.agents.trade.model_dump(),
        broker=adapters.get("broker"),
    )

    # Build watchlist
    active_universe = load_active_watchlist()
    watchlist_universe = list({*active_universe, symbol})

    watchlist = watchlist or AdaptiveWatchlist(
        universe=watchlist_universe,
        min_candidates=3,
        max_candidates=8,
        volume_threshold=10_000_000,
    )

    # Build ledger
    ledger_path = Path(config.tracking.ledger_path)
    ledger_store = LedgerStore(ledger_path)

    # Build tracking agent
    tracking_agent = None
    if config.tracking.enable_position_tracking:
        tracking_agent = TrackingAgent(adapters.get("broker"), enable=True)

    # Build adaptation agent
    adaptation_agent = None
    if config.adaptation.enabled:
        adaptation_agent = AdaptationAgent(
            state_path=Path(config.adaptation.state_path),
            min_trades=config.adaptation.min_trades_for_update,
            performance_lookback=config.adaptation.performance_lookback,
            min_risk_per_trade=config.adaptation.min_risk_per_trade,
            max_risk_per_trade=config.adaptation.max_risk_per_trade,
        )

    # ML enhancement stack
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

    # Assemble and return pipeline
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
