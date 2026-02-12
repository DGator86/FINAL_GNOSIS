"""
Gnosis Alpha Configuration

Configuration management for Gnosis Alpha trading signals.
Supports separate Alpaca credentials from main Gnosis system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

# Load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class AlphaConfig:
    """Configuration for Gnosis Alpha signal generator."""
    
    # Alpaca Paper Trading API (Alpha-specific)
    alpaca_api_key: str = field(default_factory=lambda: os.getenv(
        "ALPHA_ALPACA_API_KEY",
        os.getenv("ALPACA_API_KEY", "")
    ))
    alpaca_secret_key: str = field(default_factory=lambda: os.getenv(
        "ALPHA_ALPACA_SECRET_KEY",
        os.getenv("ALPACA_SECRET_KEY", "")
    ))
    alpaca_base_url: str = field(default_factory=lambda: os.getenv(
        "ALPHA_ALPACA_BASE_URL",
        os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
    ))
    
    # Signal Generation Settings
    min_confidence: float = 0.65  # Minimum confidence to generate signal
    max_holding_days: int = 7  # Maximum holding period in days
    lookback_days: int = 20  # Days of data to analyze
    
    # PDT Settings (Pattern Day Trader rules)
    pdt_enabled: bool = True  # Enable PDT tracking
    max_day_trades: int = 3  # Maximum day trades per 5 rolling days
    pdt_lookback_days: int = 5  # Rolling window for PDT tracking
    account_minimum: float = 25000.0  # PDT threshold
    
    # Position Sizing
    max_position_pct: float = 0.10  # Max 10% of portfolio per position
    max_positions: int = 5  # Maximum concurrent positions
    
    # Risk Management
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    
    # Universe (stocks to scan)
    universe: List[str] = field(default_factory=lambda: [
        # Magnificent 7
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        # Growth Leaders
        "AMD", "CRM", "NFLX", "AVGO", "ADBE",
        # Value Stalwarts
        "JPM", "V", "JNJ", "UNH", "HD",
        # Sector Leaders
        "XOM", "LLY", "MA", "COST", "PEP", "ORCL", "MRK", "BA"
    ])
    
    # Output Settings
    signal_output_dir: str = field(default_factory=lambda: os.getenv(
        "ALPHA_SIGNAL_DIR", "data/alpha_signals"
    ))
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv(
        "ALPHA_LOG_LEVEL", "INFO"
    ))
    
    @classmethod
    def from_env(cls) -> "AlphaConfig":
        """Create config from environment variables."""
        config = cls()
        
        # Override from environment if set
        if min_conf := os.getenv("ALPHA_MIN_CONFIDENCE"):
            config.min_confidence = float(min_conf)
        if max_hold := os.getenv("ALPHA_MAX_HOLDING_DAYS"):
            config.max_holding_days = int(max_hold)
        if max_pos := os.getenv("ALPHA_MAX_POSITIONS"):
            config.max_positions = int(max_pos)
        if universe := os.getenv("ALPHA_UNIVERSE"):
            config.universe = [s.strip() for s in universe.split(",")]
        if stop_loss := os.getenv("ALPHA_STOP_LOSS_PCT"):
            config.stop_loss_pct = float(stop_loss)
        if take_profit := os.getenv("ALPHA_TAKE_PROFIT_PCT"):
            config.take_profit_pct = float(take_profit)
            
        return config
    
    def ensure_output_dir(self) -> Path:
        """Ensure signal output directory exists."""
        path = Path(self.signal_output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if not self.alpaca_api_key:
            issues.append("ALPHA_ALPACA_API_KEY or ALPACA_API_KEY not set")
        if not self.alpaca_secret_key:
            issues.append("ALPHA_ALPACA_SECRET_KEY or ALPACA_SECRET_KEY not set")
        if self.min_confidence < 0 or self.min_confidence > 1:
            issues.append("min_confidence must be between 0 and 1")
        if self.max_holding_days < 1:
            issues.append("max_holding_days must be at least 1")
        if self.stop_loss_pct <= 0 or self.stop_loss_pct > 1:
            issues.append("stop_loss_pct must be between 0 and 1")
        if self.take_profit_pct <= 0 or self.take_profit_pct > 1:
            issues.append("take_profit_pct must be between 0 and 1")
        if not self.universe:
            issues.append("universe cannot be empty")
            
        return issues
    
    def to_dict(self) -> dict:
        """Convert config to dictionary (hiding secrets)."""
        return {
            "alpaca_base_url": self.alpaca_base_url,
            "alpaca_api_key": self.alpaca_api_key[:8] + "..." if self.alpaca_api_key else None,
            "min_confidence": self.min_confidence,
            "max_holding_days": self.max_holding_days,
            "pdt_enabled": self.pdt_enabled,
            "max_day_trades": self.max_day_trades,
            "max_positions": self.max_positions,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "universe_count": len(self.universe),
        }
