"""
Gnosis Alpha - Short-term Directional Trading Signals

A simplified trading signal system optimized for:
- 0-7 day holding periods
- Directional stocks AND options
- PDT-compliant (Pattern Day Trading rules)
- Robinhood/Webull friendly output
- ML-enhanced predictions

Supports:
- Stock signals (BUY/SELL/HOLD)
- Options signals (Long Call, Long Put, Covered Call, Cash-Secured Put)
- ⚡ 0DTE Options (Zero Days to Expiration - HIGH RISK)
- 🤖 Machine Learning enhanced signals

Uses technical analysis and ML models to generate simple, actionable signals.
"""

# Stock trading
from alpha.signal_generator import AlphaSignalGenerator, AlphaSignal, SignalDirection
from alpha.pdt_tracker import PDTTracker
from alpha.alpha_config import AlphaConfig
from alpha.alpha_trader import AlphaTrader

# Options trading
from alpha.options_signal import (
    OptionsSignalGenerator,
    OptionsSignal,
    OptionContract,
    OptionStrategy,
    OptionType,
    OptionSignalDirection,
)
from alpha.options_trader import OptionsTrader

# 0DTE Options (HIGH RISK)
from alpha.zero_dte import (
    ZeroDTEGenerator,
    ZeroDTESignal,
    ZeroDTEStrategy,
    ZeroDTERisk,
    print_0dte_disclaimer,
)

# Machine Learning (lazy import to avoid slow startup)
def _get_ml_module():
    """Lazy import of ML module."""
    from alpha import ml
    return ml

__all__ = [
    # Stock
    "AlphaSignalGenerator",
    "AlphaSignal", 
    "SignalDirection",
    "PDTTracker",
    "AlphaConfig",
    "AlphaTrader",
    # Options
    "OptionsSignalGenerator",
    "OptionsSignal",
    "OptionContract",
    "OptionStrategy",
    "OptionType",
    "OptionSignalDirection",
    "OptionsTrader",
    # 0DTE (HIGH RISK)
    "ZeroDTEGenerator",
    "ZeroDTESignal",
    "ZeroDTEStrategy",
    "ZeroDTERisk",
    "print_0dte_disclaimer",
]

__version__ = "1.3.0"
