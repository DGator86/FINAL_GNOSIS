"""Configuration data models."""

from __future__ import annotations

from typing import Dict

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


class TrackingConfig(BaseModel):
    """Configuration for tracking and logging."""
    
    ledger_path: str = "data/ledger.jsonl"
    log_level: str = "INFO"


class AppConfig(BaseModel):
    """Main application configuration."""
    
    engines: EnginesConfig = Field(default_factory=EnginesConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
