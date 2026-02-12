"""
GNOSIS Centralized Configuration Management V2

This module provides centralized configuration for all GNOSIS components:
- Engine configurations
- Agent configurations  
- Composer configurations
- Trade agent configurations
- Monitor configurations

Author: GNOSIS Trading System
Version: 2.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class TradingMode(str, Enum):
    """Trading mode selection."""
    FULL_GNOSIS = "full_gnosis"  # Full automated trading
    ALPHA_SIGNALS = "alpha_signals"  # Signal-only mode
    PAPER_TRADING = "paper_trading"  # Paper trading mode
    BACKTEST = "backtest"  # Backtesting mode


class RiskLevel(str, Enum):
    """Risk level presets."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class EngineConfig:
    """Configuration for all GNOSIS engines."""
    
    # Hedge Engine V3
    hedge: Dict[str, Any] = field(default_factory=lambda: {
        "regime_components": 3,
        "regime_history": 256,
        "regime_min_samples": 32,
        "gamma_weight": 0.6,
        "vanna_weight": 0.4,
        "lookback_days": 20,
        "lstm_enabled": False,  # LSTM prediction
    })
    
    # Liquidity Engine V5 (PENTA)
    liquidity: Dict[str, Any] = field(default_factory=lambda: {
        "min_volume_threshold": 1_000_000,
        "spread_threshold": 0.5,  # 0.5% spread
        # PENTA sub-engine weights
        "wyckoff_weight": 0.18,
        "ict_weight": 0.18,
        "order_flow_weight": 0.18,
        "supply_demand_weight": 0.18,
        "liquidity_concepts_weight": 0.18,
        "base_weight": 0.10,  # Remaining weight
    })
    
    # Sentiment Engine
    sentiment: Dict[str, Any] = field(default_factory=lambda: {
        "news_weight": 0.4,
        "flow_weight": 0.3,
        "technical_weight": 0.3,
    })


@dataclass
class AgentConfig:
    """Configuration for all GNOSIS agents."""
    
    # Hedge Agent V3
    hedge_agent: Dict[str, Any] = field(default_factory=lambda: {
        "min_confidence": 0.5,
        "energy_threshold": 0.3,
    })
    
    # Liquidity Agent V5
    liquidity_agent: Dict[str, Any] = field(default_factory=lambda: {
        "min_confidence": 0.5,
        "confluence_threshold": 3,  # Min methodologies for confluence
    })
    
    # Sentiment Agent
    sentiment_agent: Dict[str, Any] = field(default_factory=lambda: {
        "min_confidence": 0.5,
        "sentiment_threshold": 0.3,
    })


@dataclass
class ComposerConfig:
    """Configuration for ComposerAgentV4."""
    
    # Agent weights (must sum to 1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "hedge": 0.40,
        "sentiment": 0.40,
        "liquidity": 0.20,
    })
    
    # Consensus thresholds
    bullish_threshold: float = 0.30
    bearish_threshold: float = -0.30
    min_consensus_score: float = 0.50
    
    # PENTA confluence bonuses
    penta_bonuses: Dict[str, float] = field(default_factory=lambda: {
        "PENTA": 0.30,  # All 5 methodologies
        "QUAD": 0.25,   # 4 methodologies
        "TRIPLE": 0.15,  # 3 methodologies
        "DUAL": 0.05,   # 2 methodologies
    })
    
    # Express modes
    enable_0dte_mode: bool = True
    enable_cheap_calls_mode: bool = True


@dataclass
class TradeAgentConfig:
    """Configuration for Trade Agents."""
    
    # Common settings
    min_confidence: float = 0.50
    strong_confidence_threshold: float = 0.80
    moderate_confidence_threshold: float = 0.65
    
    # Risk management
    default_stop_pct: float = 0.03  # 3% stop loss
    default_target_pct: float = 0.05  # 5% take profit
    max_position_size: float = 0.10  # 10% of portfolio
    max_portfolio_risk: float = 0.02  # 2% max risk per trade
    
    # PENTA adjustments
    penta_target_multiplier: float = 1.30  # 30% extended target for PENTA
    penta_position_bonus: float = 0.10  # 10% larger positions for PENTA
    
    # Options
    suggest_options: bool = True
    options_dte_range: List[int] = field(default_factory=lambda: [7, 21])
    
    # Signal validity
    signal_validity_hours: int = 24
    default_holding_days: int = 3


@dataclass
class MonitorConfig:
    """Configuration for Monitoring Agents."""
    
    # GnosisMonitor (Full Trading)
    gnosis_monitor: Dict[str, Any] = field(default_factory=lambda: {
        "initial_equity": 100_000,
        "max_drawdown_threshold": 0.15,  # 15% max drawdown alert
        "daily_loss_limit": 0.02,  # 2% daily loss limit
        "win_rate_threshold": 0.45,  # Alert if win rate drops below
    })
    
    # AlphaMonitor (Signals)
    alpha_monitor: Dict[str, Any] = field(default_factory=lambda: {
        "min_accuracy_threshold": 0.50,  # 50% min signal accuracy
        "signal_tracking_days": 30,
    })


@dataclass
class BacktestConfig:
    """Configuration for Backtesting."""
    
    # Date range
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-01"
    
    # Capital
    initial_capital: float = 100_000
    
    # Symbols
    symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"])
    
    # Risk settings
    max_positions: int = 5
    max_position_pct: float = 0.04  # 4% max per position
    min_confidence: float = 0.50
    
    # Cost modeling
    slippage_bps: float = 5.0
    commission_per_trade: float = 0.0
    
    # Monte Carlo
    monte_carlo_runs: int = 1000


@dataclass
class GnosisConfigV2:
    """
    Master configuration for the entire GNOSIS system.
    
    Usage:
        config = GnosisConfigV2()
        config = GnosisConfigV2.for_risk_level(RiskLevel.CONSERVATIVE)
        config = GnosisConfigV2.for_trading_mode(TradingMode.ALPHA_SIGNALS)
    """
    
    # System mode
    trading_mode: TradingMode = TradingMode.ALPHA_SIGNALS
    risk_level: RiskLevel = RiskLevel.MODERATE
    
    # Component configs
    engines: EngineConfig = field(default_factory=EngineConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    composer: ComposerConfig = field(default_factory=ComposerConfig)
    trade_agent: TradeAgentConfig = field(default_factory=TradeAgentConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    @classmethod
    def for_risk_level(cls, level: RiskLevel) -> "GnosisConfigV2":
        """Create configuration preset for a specific risk level."""
        config = cls(risk_level=level)
        
        if level == RiskLevel.CONSERVATIVE:
            config.trade_agent.min_confidence = 0.70
            config.trade_agent.max_position_size = 0.05
            config.trade_agent.default_stop_pct = 0.02
            config.composer.min_consensus_score = 0.60
        
        elif level == RiskLevel.AGGRESSIVE:
            config.trade_agent.min_confidence = 0.40
            config.trade_agent.max_position_size = 0.15
            config.trade_agent.default_stop_pct = 0.05
            config.composer.min_consensus_score = 0.40
        
        return config
    
    @classmethod
    def for_trading_mode(cls, mode: TradingMode) -> "GnosisConfigV2":
        """Create configuration preset for a specific trading mode."""
        config = cls(trading_mode=mode)
        
        if mode == TradingMode.BACKTEST:
            config.trade_agent.suggest_options = False
            config.engines.hedge["lstm_enabled"] = False
        
        elif mode == TradingMode.PAPER_TRADING:
            config.trade_agent.max_position_size = 0.05
            config.trade_agent.min_confidence = 0.60
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "trading_mode": self.trading_mode.value,
            "risk_level": self.risk_level.value,
            "engines": {
                "hedge": self.engines.hedge,
                "liquidity": self.engines.liquidity,
                "sentiment": self.engines.sentiment,
            },
            "agents": {
                "hedge": self.agents.hedge_agent,
                "liquidity": self.agents.liquidity_agent,
                "sentiment": self.agents.sentiment_agent,
            },
            "composer": {
                "weights": self.composer.weights,
                "thresholds": {
                    "bullish": self.composer.bullish_threshold,
                    "bearish": self.composer.bearish_threshold,
                    "consensus": self.composer.min_consensus_score,
                },
                "penta_bonuses": self.composer.penta_bonuses,
            },
            "trade_agent": {
                "confidence": {
                    "min": self.trade_agent.min_confidence,
                    "strong": self.trade_agent.strong_confidence_threshold,
                    "moderate": self.trade_agent.moderate_confidence_threshold,
                },
                "risk": {
                    "stop_pct": self.trade_agent.default_stop_pct,
                    "target_pct": self.trade_agent.default_target_pct,
                    "max_position": self.trade_agent.max_position_size,
                },
                "penta": {
                    "target_multiplier": self.trade_agent.penta_target_multiplier,
                    "position_bonus": self.trade_agent.penta_position_bonus,
                },
            },
            "backtest": {
                "start_date": self.backtest.start_date,
                "end_date": self.backtest.end_date,
                "initial_capital": self.backtest.initial_capital,
                "symbols": self.backtest.symbols,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GnosisConfigV2":
        """Create configuration from dictionary."""
        config = cls()
        
        if "trading_mode" in data:
            config.trading_mode = TradingMode(data["trading_mode"])
        if "risk_level" in data:
            config.risk_level = RiskLevel(data["risk_level"])
        
        # Update nested configs
        if "engines" in data:
            if "hedge" in data["engines"]:
                config.engines.hedge.update(data["engines"]["hedge"])
            if "liquidity" in data["engines"]:
                config.engines.liquidity.update(data["engines"]["liquidity"])
        
        if "composer" in data:
            if "weights" in data["composer"]:
                config.composer.weights = data["composer"]["weights"]
        
        if "trade_agent" in data:
            if "confidence" in data["trade_agent"]:
                conf = data["trade_agent"]["confidence"]
                config.trade_agent.min_confidence = conf.get("min", 0.50)
        
        return config


# Singleton instance for global access
_default_config: Optional[GnosisConfigV2] = None


def get_config() -> GnosisConfigV2:
    """Get the global GNOSIS configuration."""
    global _default_config
    if _default_config is None:
        _default_config = GnosisConfigV2()
    return _default_config


def set_config(config: GnosisConfigV2) -> None:
    """Set the global GNOSIS configuration."""
    global _default_config
    _default_config = config


# Convenience presets
CONSERVATIVE_CONFIG = GnosisConfigV2.for_risk_level(RiskLevel.CONSERVATIVE)
MODERATE_CONFIG = GnosisConfigV2.for_risk_level(RiskLevel.MODERATE)
AGGRESSIVE_CONFIG = GnosisConfigV2.for_risk_level(RiskLevel.AGGRESSIVE)
BACKTEST_CONFIG = GnosisConfigV2.for_trading_mode(TradingMode.BACKTEST)
