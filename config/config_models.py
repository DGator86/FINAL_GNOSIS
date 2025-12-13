"""Configuration data models."""

from __future__ import annotations


from typing import Optional

from pydantic import BaseModel, Field


class HedgeEngineConfig(BaseModel):
    """Configuration for Hedge Engine."""
    
    enabled: bool = True
    lookback_days: int = 30
    min_dte: int = 7
    max_dte: int = 60


class LiquidityEngineConfig(BaseModel):
    """Configuration for Liquidity Engine."""
    
    enabled: bool = True
    min_volume: float = 100000.0
    max_spread_pct: float = 0.5


class SentimentEngineConfig(BaseModel):
    """Configuration for Sentiment Engine."""
    
    enabled: bool = True
    news_weight: float = 0.4
    flow_weight: float = 0.3
    technical_weight: float = 0.3


class ElasticityEngineConfig(BaseModel):
    """Configuration for Elasticity Engine."""
    
    enabled: bool = True
    volatility_window: int = 20


class EnginesConfig(BaseModel):
    """Configuration for all engines."""
    
    hedge: HedgeEngineConfig = Field(default_factory=HedgeEngineConfig)
    liquidity: LiquidityEngineConfig = Field(default_factory=LiquidityEngineConfig)
    sentiment: SentimentEngineConfig = Field(default_factory=SentimentEngineConfig)
    elasticity: ElasticityEngineConfig = Field(default_factory=ElasticityEngineConfig)


class HedgeAgentConfig(BaseModel):
    """Configuration for Hedge Agent."""
    
    enabled: bool = True
    min_confidence: float = 0.5


class LiquidityAgentConfig(BaseModel):
    """Configuration for Liquidity Agent."""
    
    enabled: bool = True
    min_liquidity_score: float = 0.3


class SentimentAgentConfig(BaseModel):
    """Configuration for Sentiment Agent."""
    
    enabled: bool = True
    min_sentiment_threshold: float = 0.2


class ComposerWeights(BaseModel):
    """Weights for the composer agent."""
    
    hedge: float = 0.4
    liquidity: float = 0.2
    sentiment: float = 0.4


class ComposerAgentConfig(BaseModel):
    """Configuration for Composer Agent."""
    
    enabled: bool = True
    weights: ComposerWeights = Field(default_factory=ComposerWeights)


class TradeAgentConfig(BaseModel):
    """Configuration for Trade Agent."""
    
    enabled: bool = True
    max_position_size: float = 10000.0
    risk_per_trade: float = 0.02


class AgentsConfig(BaseModel):
    """Configuration for all agents."""
    
    hedge: HedgeAgentConfig = Field(default_factory=HedgeAgentConfig)
    liquidity: LiquidityAgentConfig = Field(default_factory=LiquidityAgentConfig)
    sentiment: SentimentAgentConfig = Field(default_factory=SentimentAgentConfig)
    composer: ComposerAgentConfig = Field(default_factory=ComposerAgentConfig)
    trade: TradeAgentConfig = Field(default_factory=TradeAgentConfig)


class RankingCriteria(BaseModel):
    """Ranking criteria for dynamic scanner."""
    
    options_volume_weight: float = 0.40
    open_interest_weight: float = 0.25
    gamma_exposure_weight: float = 0.20
    liquidity_score_weight: float = 0.10
    unusual_flow_weight: float = 0.05


class ScannerConfig(BaseModel):
    """Configuration for dynamic top N scanner."""
    
    mode: str = "dynamic_top_n"
    default_top_n: int = 25
    ranking_criteria: RankingCriteria = Field(default_factory=RankingCriteria)
    min_daily_options_volume: float = 500000.0
    require_unusual_whales: bool = False
    update_frequency: int = 300  # seconds
    cache_duration: int = 60  # seconds


class TradingConfig(BaseModel):
    """Configuration for trading behavior."""
    
    multi_symbol_default: bool = True
    max_concurrent_positions: int = 10
    position_size_pct: float = 0.04
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05


class TrackingConfig(BaseModel):
    """Configuration for tracking and logging."""

    ledger_path: str = "data/ledger.jsonl"
    log_level: str = "INFO"
    enable_position_tracking: bool = True


class AdaptationConfig(BaseModel):
    """Configuration for adaptive feedback loops."""

    enabled: bool = False
    state_path: str = "data/adaptation_state.json"
    min_trades_for_update: int = 5
    performance_lookback: int = 20
    min_risk_per_trade: float = 0.005
    max_risk_per_trade: float = 0.05


class StorageConfig(BaseModel):
    """Configuration for cloud storage (S3-compatible)."""

    enabled: bool = False
    provider: str = "massive"  # massive, aws, minio, etc.
    endpoint: str = "https://files.massive.com"
    bucket: str = "flatfiles"
    region: str = "us-east-1"
    auto_sync: bool = False  # Automatically sync local data to S3
    sync_features: bool = True  # Sync feature store to S3
    sync_ledger: bool = True  # Sync ledger to S3
    sync_logs: bool = False  # Sync logs to S3
    sync_models: bool = False  # Sync ML models to S3


class BacktestConfig(BaseModel):
    """Configuration for backtesting runs."""

    use_all_components: bool = True
    use_real_data: bool = True
    cache_enabled: bool = True
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class DataSourcesConfig(BaseModel):
    """Primary data source selection and feature flags."""

    primary: str = "massive"
    options_provider: str = "massive"
    unusual_whales_enabled: bool = True
    cache_path: str = "data/historical"


class AppConfig(BaseModel):
    """Main application configuration."""

    engines: EnginesConfig = Field(default_factory=EnginesConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    adaptation: AdaptationConfig = Field(default_factory=AdaptationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    data_sources: DataSourcesConfig = Field(default_factory=DataSourcesConfig)
